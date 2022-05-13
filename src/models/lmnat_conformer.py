
# 2021 Ruchao Fan

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding, TextEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks import TrfEncoder, ConEncoder
from models.blocks import ConSAD, ConMAD, TrfSAD, TrfMAD, ConAcExtra, TrfAcExtra
from models.cassnat import CassNAT
from utils.ctc_prefix import CTCPrefixScore

def make_model(input_size, args):
    c = copy.deepcopy
    # we do not make a comparison with relative and absolute in CASS_NAT
    if args.use_conv_enc:
        assert args.pos_type == "relative", "conformer must use relative positional encoding"
        enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.enc_max_relative_len)     
        enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        enc_conv_module = ConvModule(args.d_model, args.enc_kernel_size, activation=Swish())    
        enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())
        encoder = ConEncoder(args.d_model, c(enc_ff), enc_attn, enc_conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff)
    else:
        attn = MultiHeadedAttention(args.n_head, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout)
        enc_position = PositionalEncoding(args.d_model, args.dropout)
        encoder = TrfEncoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc)

    if args.use_conv_dec:
        dec_self_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        dec_src_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        dec_conv_module = ConvModule(args.d_model, args.dec_kernel_size, activation=Swish())
        dec_position = RelativePositionalEncoding(args.d_model, args.dropout, args.dec_max_relative_len)
        dec_ff = PositionwiseFeedForward(args.d_model, args.d_decff, args.dropout, activation=Swish())
        dec_ff_original = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout, activation=Swish())
        Extra = ConAcExtra(args.d_model, c(dec_src_attn), dec_ff_original, dec_position, args.pos_type, args.dropout, args.N_extra)
        Sad = ConSAD(args.d_model, c(dec_ff), c(dec_self_attn), c(dec_conv_module), c(dec_ff), args.dropout, args.N_self_dec, args.pos_type, args.share_ff)
        Mad = ConMAD(args.d_model, c(dec_ff), c(dec_self_attn), c(dec_conv_module), c(dec_src_attn), c(dec_ff), args.dropout, args.N_mix_dec, args.pos_type, args.share_ff)
    else:
        dec_ff = PositionwiseFeedForward(args.d_model, args.d_decff, args.dropout)
        dec_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        Extra = TrfAcExtra(args.d_model, c(dec_attn), c(dec_ff), args.dropout, args.N_extra)
        Sad = TrfSAD(args.d_model, c(dec_attn), c(dec_ff), args.dropout, args.N_self_dec)
        Mad = TrfMAD(args.d_model, c(dec_attn), c(dec_attn), c(dec_ff), args.dropout, args.N_mix_dec)
    
    generator = Generator(args.d_model, args.vocab_size)
    interctc_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interctc_alpha > 0 else None
    interce_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interce_alpha > 0 else None
    pe = create_pe(args.d_model)

    model = ConCassNAT(
                ConvEmbedding(input_size, args.d_model, args.dropout, enc_position),
                encoder, Extra, Sad, Mad, c(generator), c(generator), pe, interctc_gen, interce_gen, args)

    if args.interce_alpha > 0:
        if args.interce_layer <= args.N_self_dec:
            args.selfce_alpha = args.interce_alpha
            args.mixce_alpha = 0
        else:
            args.selfce_alpha = 0
            args.mixce_alpha = args.interce_alpha
            args.interce_layer = (args.interce_layer - args.N_self_dec)
    else:
        args.selfce_alpha = 0
        args.mixce_alpha = 0

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
    def __init__(self, d_model, vocab, add_norm=False):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.add_norm = add_norm
        if add_norm:
            self.norm = LayerNorm(d_model)

    def forward(self, x, T=1.0):
        if self.add_norm:
            x = self.norm(x)
        return F.log_softmax(self.proj(x)/T, dim=-1)

class ConCassNAT(CassNAT):
    def __init__(self, *args):
        super(ConCassNAT, self).__init__(*args)

