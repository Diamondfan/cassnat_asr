
# 2020 Ruchao Fan
# Some transformer-related codes are borrowed from 
# https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, ConvEmbedding, RelativePositionalEncoding, ConvEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks import TrfEncoder, ConEncoder
from models.ctc_transformer import CTCTransformer
from utils.ctc_prefix import CTCPrefixScore, logzero, logone


def make_model(input_size, args):
    c = copy.deepcopy
    if args.use_conv_enc:
        assert args.pos_type == "relative", "conformer must use relative positional encoding"
        enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.enc_max_relative_len)    
        enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        enc_conv_module = ConvModule(args.d_model, args.enc_kernel_size, activation=Swish())    
        enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())
        encoder = ConEncoder(args.d_model, c(enc_ff), enc_attn, enc_conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff)
    else:
        attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        enc_position = PositionalEncoding(args.d_model, args.dropout, max_len=args.max_len)
        encoder = TrfEncoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc)
        
    generator = Generator(args.d_model, args.vocab_size)
    interctc_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interctc_alpha > 0 else None
    model = CTCConformer(
        ConvEmbedding(input_size, args.d_model, args.dropout, enc_position),
        encoder, generator, interctc_gen, args)
        
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

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

class CTCConformer(CTCTransformer):
    def __init__(self, *args):
        super(CTCConformer, self).__init__(*args)

