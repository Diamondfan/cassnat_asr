#!/usr/bin/env python3
# 2021 Ruchao Fan

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding, TextEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks.conformer_blocks import Encoder, Decoder
from models.transformer import Transformer
from utils.ctc_prefix import CTCPrefixScore

def make_model(input_size, args):
    c = copy.deepcopy
    #assert args.pos_type == "relative", "conformer must use relative positional encoding"
    if args.pos_type == "relative":
        enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.enc_max_relative_len)
        enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
    elif args.pos_type == "absolute":
        enc_position = PositionalEncoding(args.d_model, args.dropout)
        enc_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        
    attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
    conv_module = ConvModule(args.d_model, args.enc_kernel_size, activation=Swish())
    enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())
    dec_ff = PositionwiseFeedForward(args.d_model, args.d_decff, args.dropout, activation=Swish())
    position = PositionalEncoding(args.d_model, args.dropout)
    generator = Generator(args.d_model, args.vocab_size)
    
    interctc_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interctc_alpha > 0 else None
    model = Conformer(
        ConvEmbedding(input_size, args.d_model, args.dropout, enc_position),
        Encoder(args.d_model, c(enc_ff), enc_attn, conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff),
        nn.Sequential(TextEmbedding(args.d_model, args.vocab_size), position), 
        Decoder(args.d_model, c(attn), c(attn), dec_ff, args.dropout, args.N_dec),
        c(generator), c(generator), interctc_gen, args)
    
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

class Conformer(Transformer):
    def __init__(self, *args):
        super(Conformer, self).__init__(*args)

