#!/usr/bin/env python

# 2020 Ruchao Fan
# Transformer-based neural language model

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention import MultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, TextEmbedding
from models.blocks.transformer_blocks import Encoder

def make_model(args):
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.n_head, args.d_model)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    generator = Generator(args.d_model, args.vocab_size)
    
    model = TransformerLM(
        nn.Sequential(TextEmbedding(args.d_model, args.vocab_size), c(position)), 
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N),
        c(generator))
        
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, T=1.0):
        return F.log_softmax(self.proj(x)/T, dim=-1)

class TransformerLM(nn.Module):
    def __init__(self, text_embed, encoder, out_gen):
        super(TransformerLM, self).__init__()
        self.text_embed = text_embed
        self.encoder = encoder
        self.out_generator = out_gen

    def forward(self, tgt, tgt_mask):
        tgt = self.text_embed(tgt)
        enc_h = self.encoder(tgt, tgt_mask)
        lm_out = self.out_generator(enc_h)
        return lm_out

