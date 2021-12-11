#!/usr/bin/env python3
# 2020 Ruchao Fan

import math
import torch.nn as nn
from models.modules.norm import LayerNorm
from models.modules.utils import clones, SublayerConnection

class SelfAttLayer(nn.Module):
    "Attention block with self-attn and feed forward (defined below)"
    def __init__(self, size, feed_forward1, self_attn, conv_module, feed_forward2, dropout, pos_type, share_ff=False, ff_scale=0.5):
        super(SelfAttLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward1 = feed_forward1
        self.conv_module = conv_module
        if share_ff:
            self.feed_forward2 = feed_forward1
        else:
            self.feed_forward2 = feed_forward2

        self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.size = size
        self.pos_type = pos_type
        self.ff_scale = ff_scale

    def forward(self, x, mask, pos_embed):
        x = self.sublayer[0](x, self.feed_forward1, self.ff_scale)

        if self.pos_type == "absolute":
            x = self.sublayer[1](x, self.conv_module) 
            x = self.sublayer[2](x, lambda x: self.self_attn(x, x, x, mask, pos_embed))
        elif self.pos_type == "relative":
            x = self.sublayer[2](x, lambda x: self.self_attn(x, x, x, mask, pos_embed))
            x = self.sublayer[1](x, self.conv_module)

        x = self.sublayer[3](x, self.feed_forward2, self.ff_scale)
        return x


class SrcAttLayer(nn.Module):
    "Attention block with src-attn and feed forward"
    def __init__(self, size, src_attn, feed_forward, pos_enc, pos_type, dropout):
        super(SrcAttLayer, self).__init__()
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)
        self.pos_enc = pos_enc
        self.pos_type = pos_type
        self.size = size
    
    def forward(self, x, m, src_mask):
        x = self.src_attn(x, m, m, src_mask)
        x = self.pos_enc(x * math.sqrt(self.size))
        if self.pos_type == "relative":
            pos_embed = x[1]
            x = self.sublayer(x[0], self.feed_forward)
            return (x, pos_embed)
        elif self.pos_type == "absolute":
            x = self.sublayer(x, self.feed_forward)
            return x

class MixAttLayer(nn.Module):
    "Attention block with self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, feed_forward1, self_attn, conv_module, src_attn, feed_forward2, dropout, pos_type, share_ff=False, ff_scale=0.5):
        super(MixAttLayer, self).__init__()
        self.size = size
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward1 = feed_forward1
        self.conv_module = conv_module
        if share_ff:
            self.feed_forward2 = feed_forward1
        else:
            self.feed_forward2 = feed_forward2

        self.sublayer = clones(SublayerConnection(size, dropout), 5)
        self.size = size
        self.pos_type = pos_type
        self.ff_scale = ff_scale

    def forward(self, x, memory, src_mask, self_mask, pos_embed):
        x = self.sublayer[0](x, self.feed_forward1, self.ff_scale)

        if self.pos_type == "absolute":
            x = self.sublayer[1](x, self.conv_module) 
            x = self.sublayer[2](x, lambda x: self.self_attn(x, x, x, self_mask, pos_embed))
        elif self.pos_type == "relative":
            x = self.sublayer[2](x, lambda x: self.self_attn(x, x, x, self_mask, pos_embed))
            x = self.sublayer[1](x, self.conv_module)

        m = memory
        x = self.sublayer[3](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayer[4](x, self.feed_forward2, self.ff_scale)
        return x

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, size, feed_forward1, self_attn, conv_module, feed_forward2, dropout, N, pos_type, share_ff=False, ff_scale=0.5):
        super(Encoder, self).__init__()
        layer = SelfAttLayer(size, feed_forward1, self_attn, conv_module, feed_forward2, dropout, pos_type, share_ff, ff_scale)
        self.layers = clones(layer, N)
        self.pos_type = pos_type
        self.norm = LayerNorm(size)
        
    def forward(self, x, mask, interctc_alpha=0, interctc_layer=0):
        "Pass the input (and mask) through each layer in turn."
        if self.pos_type == "relative":
            x, pos_embed = x[0], x[1]
        elif self.pos_type == "absolute":
            pos_embed = None

        n_layer = 0
        for layer in self.layers:
            x = layer(x, mask, pos_embed)
            if interctc_alpha > 0 and n_layer == interctc_layer - 1:
                inter_out = x
            n_layer += 1

        if interctc_alpha > 0:
            return (self.norm(x), inter_out)
        else:
            return self.norm(x)

class AcEmbedExtractor(nn.Module):
    "Extractor sub-unit level acoustic embedding with CTC segments"
    def __init__(self, size, src_attn, feed_forward, pos_enc, pos_type, dropout, N):
        super(AcEmbedExtractor, self).__init__()
        layer = SrcAttLayer(size, src_attn, feed_forward, pos_enc, pos_type, dropout)
        self.layers = clones(layer, N)
        assert N == 1, "if more than 1 layer use, code needs to be modified"

    def forward(self, x, memory, trigger_mask):
        for layer in self.layers:
            x = layer(x, memory, trigger_mask)
        return x

class SelfAttDecoder(nn.Module):
    "Map the acoustic embedding to word embedding, similar with Encoder structure"
    def __init__(self, size, feed_forward1, self_attn, conv_module, feed_forward2, dropout, N, pos_type, share_ff=False, ff_scale=0.5):
        super(SelfAttDecoder, self).__init__()
        layer = SelfAttLayer(size, feed_forward1, self_attn, conv_module, feed_forward2, dropout, pos_type, share_ff, ff_scale)
        self.layers = clones(layer, N)
        self.pos_type = pos_type
        
    def forward(self, x, mask, interce_alpha=0, interce_layer=0):
        "Pass the input (and mask) through each layer in turn."
        if self.pos_type == "relative":
            x, pos_embed = x[0], x[1]
        elif self.pos_type == "absolute":
            pos_embed = None

        n_layer = 0
        for layer in self.layers:
            x = layer(x, mask, pos_embed)
            if interce_alpha > 0 and n_layer == interce_layer - 1:
                interce_out = x
            n_layer += 1

        if interce_alpha > 0:
            return (x, pos_embed, interce_out)
        else:
            return (x, pos_embed)

class MixAttDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, size, feed_forward1, self_attn, conv_module, src_attn, feed_forward2, dropout, N, pos_type, share_ff=False, ff_scale=0.5):
        super(MixAttDecoder, self).__init__()
        layer = MixAttLayer(size, feed_forward1, self_attn, conv_module, src_attn, feed_forward2, dropout, pos_type, share_ff, ff_scale)
        self.layers = clones(layer, N)
        self.pos_type = pos_type
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, interce_alpha=0, interce_layer=0):
        if self.pos_type == "relative":
            x, pos_embed = x[0], x[1]
        elif self.pos_type == "absolute":
            pos_embed = None

        n_layer = 0
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, pos_embed)
            if interce_alpha > 0 and n_layer == interce_layer - 1:
                interce_out = x
            n_layer += 1

        if interce_alpha > 0:
            return (self.norm(x), interce_out)
        else:
            return self.norm(x)

