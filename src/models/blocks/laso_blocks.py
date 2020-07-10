#!/usr/bin/env python3
# 2020 Ruchao Fan

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

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class PDSLayer(nn.Module):
    "Decoder is made of src-attn, and feed forward (defined below)"
    def __init__(self, size, src_attn, feed_forward, dropout):
        super(PDSLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
 
    def forward(self, x, memory, src_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, size, self_attn, feed_forward, dropout, N):
        super(Encoder, self).__init__()
        layer = EncoderLayer(size, self_attn, feed_forward, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class PosDepSummarizer(nn.Module):
    "Position Dependent Summarizer"
    def __init__(self, size, src_attn, feed_forward, dropout, N):
        super(PosDepSummarizer, self).__init__()
        layer = PDSLayer(size, src_attn, feed_forward, dropout)
        self.layers = clones(layer, N)
        
    def forward(self, x, memory, src_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask)
        return x


