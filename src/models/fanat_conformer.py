
# 2021 Ruchao Fan

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding, TextEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks.fanat_conformer_blocks import Encoder
from models.blocks import ConEmbedMapper, ConDecoder, TrfEmbedMapper, TrfDecoder, ConAcExtra, TrfAcExtra
from models.fanat import FaNat
from utils.ctc_prefix import CTCPrefixScore

def make_model(input_size, args):
    c = copy.deepcopy
    # we do not make a comparison with relative and absolute in CASS_NAT
    assert args.pos_type == "relative", "conformer must use relative positional encoding"
    enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.enc_max_relative_len)     
    enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
    enc_conv_module = ConvModule(args.d_model, args.enc_kernel_size, activation=Swish())    
    enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())

    dec_ff = PositionwiseFeedForward(args.d_model, args.d_decff, args.dropout, activation=Swish())
    generator = Generator(args.d_model, args.vocab_size)
    pe = create_pe(args.d_model)

    if args.use_conv_dec:        
        dec_self_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        dec_src_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        dec_conv_module = ConvModule(args.d_model, args.dec_kernel_size, activation=Swish())
        dec_position = RelativePositionalEncoding(args.d_model, args.dropout, args.dec_max_relative_len)
        model = ConFaNat(
                    ConvEmbedding(input_size, args.d_model, args.dropout, enc_position),
                    Encoder(args.d_model, c(enc_ff), enc_attn, enc_conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff),
                    ConAcExtra(args.d_model, c(dec_src_attn), c(dec_ff), dec_position, args.pos_type, args.dropout, args.N_extra),
                    ConEmbedMapper(args.d_model, c(dec_ff), c(dec_self_attn), c(dec_conv_module), c(dec_ff), args.dropout, args.N_map, args.pos_type, args.share_ff),
                    ConDecoder(args.d_model, c(dec_ff), c(dec_self_attn), c(dec_conv_module), c(dec_src_attn), c(dec_ff), args.dropout, args.N_dec, args.pos_type, args.share_ff), 
                    c(generator), c(generator), pe)
    else:
        dec_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        model = ConFaNat(
                    ConvEmbedding(input_size, args.d_model, args.dropout, enc_position),
                    Encoder(args.d_model, c(enc_ff), enc_attn, enc_conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff),
                    TrfAcExtra(args.d_model, c(dec_attn), c(dec_ff), args.dropout, args.N_extra),
                    TrfEmbedMapper(args.d_model, c(dec_attn), c(dec_ff), args.dropout, args.N_map),
                    TrfDecoder(args.d_model, c(dec_attn), c(dec_attn), c(dec_ff), args.dropout, args.N_dec), 
                    c(generator), c(generator), pe)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def create_pe(d_model, max_len=5000):
    pe = torch.zeros(max_len, d_model, requires_grad=False)
    position = torch.arange(0., max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, T=1.0):
        return F.log_softmax(self.proj(x)/T, dim=-1)

class ConFaNat(FaNat):
    def __init__(self, *args):
        super(ConFaNat, self).__init__(*args)

