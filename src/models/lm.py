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
        args.d_model, nn.Sequential(TextEmbedding(args.d_model, args.vocab_size), c(position)), 
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
    def __init__(self, hidden_size, text_embed, encoder, out_gen):
        super(TransformerLM, self).__init__()
        self.dim = hidden_size
        self.text_embed = text_embed
        self.encoder = encoder
        self.out_generator = out_gen

    def forward(self, tgt, tgt_mask):
        tgt = self.text_embed(tgt)
        enc_h = self.encoder(tgt, tgt_mask)
        lm_out = self.out_generator(enc_h)
        return lm_out

    def extract_features(self, tgt, tgt_mask):
        tgt_mask_tril = tgt_mask & self._subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        tgt = self.text_embed(tgt)
        enc_text = self.encoder(tgt, tgt_mask_tril)
        return enc_text, tgt_mask

    def forward_backbone(self, input_embed, tgt_mask):
        out_embed = self.encoder(input_embed, tgt_mask)
        return out_embed, tgt_mask

    def _subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)
    
    def _target_mask(self, ys):
        tgt_mask = (ys != 0).unsqueeze(1)
        tgt_mask = tgt_mask & self._subsequent_mask(ys.size(-1)).type_as(tgt_mask)
        return tgt_mask

    def score(self, ys, cache_stats):
        bs = ys.size(0)
        layers = len(self.encoder.layers)
        if cache_stats is None:
            batch_state = None
        else:
            batch_state = [ torch.stack([cache_stats[b][i] for b in range(bs)]) for i in range(layers)]

        h, states = self.encoder.forward_one_step(self.text_embed(ys), self._target_mask(ys), cache=batch_state)
        
        logp = self.out_generator(h[:,-1,:])
        state_list = [[ states[i][b] for i in range(layers)] for b in range(bs)]
        return logp, state_list

    def remove_unused_module(self):
        self.out_generator = None

    def remove_unused_module_aggresive(self):
        self.text_embed = None
        self.out_generator = None

