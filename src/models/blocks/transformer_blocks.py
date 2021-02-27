#!/usr/bin/env python3
# 2020 Ruchao Fan

import torch
import torch.nn as nn
from models.modules.norm import LayerNorm
from models.modules.utils import clones, SublayerConnection

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, cache=None):
        if cache is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask), has_cache=False)
        else:
            x_query = x[:,-1:,:]
            mask = None if mask is None else mask[:,-1:,:]
            x_query = self.sublayer[0].norm(x_query)
            x = self.sublayer[0](x, lambda x: self.self_attn(x_query, x, x, mask), has_cache=True)
        
        x = self.sublayer[1](x, self.feed_forward)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, size, self_attn, feed_forward, dropout, N):
        super(Encoder, self).__init__()
        layer = EncoderLayer(size, self_attn, feed_forward, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.num_layers = N
        
    def forward(self, x, mask, interctc_alpha=0):
        "Pass the input (and mask) through each layer in turn."
        n_layer = 0
        for layer in self.layers:
            x = layer(x, mask)
            if interctc_alpha > 0 and n_layer == int(self.num_layers / 2) - 1:
                inter_out = x
            n_layer += 1
        if interctc_alpha > 0:
            return (self.norm(x), inter_out)
        else:
            return self.norm(x)

    def forward_one_step(self, x, mask, cache=None):
        if cache is None:
            cache = [None for _ in range(len(self.layers))]

        new_cache = []
        for c, layer in zip(cache, self.layers):
            x = layer(x, mask, cache=c)
            new_cache.append(x)
        return self.norm(x), new_cache


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, N):
        super(Decoder, self).__init__()
        layer = DecoderLayer(size, self_attn, src_attn, feed_forward, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

