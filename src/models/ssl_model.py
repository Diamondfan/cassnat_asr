
# 2020 Ruchao Fan
# Some transformer-related codes are borrowed from 
# https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding, TextEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks.transformer_blocks import Encoder as TrfEncoder
from models.blocks.transformer_blocks import Decoder as TfrDecoder
from models.blocks.conformer_blocks import Encoder as ConEncoder
from models.blocks.conformer_blocks import Decoder as ConDecoder

from utils.ctc_prefix import CTCPrefixScore

def make_model(input_size, args):
    c = copy.deepcopy
    if args.model_type == "transformer":
        attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        embed_layer = ConvEmbedding(input_size, args.d_model, args.dropout, c(position), causal=args.causal)
        encoder = TrfEncoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc)

    elif args.model_type == "conformer":
        if args.pos_type == "relative":
            enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.max_relative_len)
            enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        elif args.pos_type == "absolute":
            enc_position = PositionalEncoding(args.d_model, args.dropout)
            enc_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
            
        conv_module = ConvModule(args.d_model, args.kernel_size, activation=Swish())
        enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())
        position = PositionalEncoding(args.d_model, args.dropout)
        embed_layer = ConvEmbedding(input_size, args.d_model, args.dropout, enc_position, causal=args.causal),
        encoder = ConEncoder(args.d_model, c(enc_ff), enc_attn, conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff),
        
    generator = Generator(args.n_generator, args.d_model, args.encoded_size)
    inter_generator = Generator(args.n_generator, args.d_model, args.encoded_size, add_norm=True) if args.inter_alpha > 0 else None
    model = SSLModel(embed_layer, encoder, generator, inter_generator) 

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class Generator(nn.Module):
    def __init__(self, n_generator, d_model, encoded_size, add_norm=False):
        super(Generator, self).__init__()
        self.n_generator = n_generator
        self.projs = nn.ModuleList()
        for i in range(n_generator):
            self.projs.append(nn.Linear(d_model, encoded_size))

        self.add_norm = add_norm
        if add_norm:
            self.norm = LayerNorm(d_model)

    def forward(self, x):
        if self.add_norm:
            x = self.norm(x)
        out = []
        for i in range(self.n_generator):
            out.append(self.projs[i](x))
        return out

class SSLModel(nn.Module):
    def __init__(self, src_embed, encoder, output_gen, inter_gen=None):
        super(SSLModel, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.output_generator = output_gen
        if inter_gen is not None:
            self.inter_generator = inter_gen

    def forward(self, src, src_mask, out_alpha, inter_alpha, inter_layer, args):
        if args.causal:
            src, src_mask = self.get_causal_mask(src, src_mask, args.forward)
        
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask, inter_alpha, inter_layer)
        
        if inter_alpha > 0:
            enc_h, inter_h = enc_h
            output = self.output_generator(enc_h)
            inter_out = self.inter_generator(inter_h)
        else:
            output = self.output_generator(enc_h)
            inter_out = 0

        return output, inter_out, enc_h, x

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)

    def get_causal_mask(self, src, src_mask, forward=True):
        size = src.size(1)
        ret = torch.ones(size, size, dtype=torch.uint8)

        if forward:       
            src_mask = src_mask & torch.tril(ret, out=ret).unsqueeze(0).type_as(src_mask)
        else:
            src_mask = src_mask & torch.triu(ret, out=ret).unsqueeze(0).type_as(src_mask)
        return src, src_mask


