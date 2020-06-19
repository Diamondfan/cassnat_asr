
# 2020 Ruchao Fan
# Some transformer-related codes are borrowed from 
# https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import torch.nn as nn
import torch.nn.functional as F
from model.modules.attention import MultiHeadedAttention
from model.modules.positionff import PositionwiseFeedForward
from model.modules.embedding import PositionalEncoding, ConvEmbedding TextEmbedding
from models.blocks.transformer_blocks import Encoder, Decoder

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, src_embed, encoder, tgt_embed, decoder, ctc_gen, att_gen):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_generator = ctc_gen
        self.att_generator = att_gen

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_h = self.encoder(self.src_embed(src), src_mask)
        #CTC Loss needs log probability as input
        ctc_out = F.log_softmax(self.ctc_generator(enc_h), dim=-1)
        dec_h = self.decoder(self.tgt_embed(tgt), enc_h, src_mask, tgt_mask)
        att_out = self.att_generator(dec_h)
        return ctc_out, att_out

    
def make_model(input_size, args):
    c = copy.deepcopy
    attn = MultiHeadedAttention(arg.n_head, arg.d_model)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    generator = Generator(args.d_model, args.vocab_size)
    
    model = Transformer(
        nn.Sequential(ConvEmbedding(input_size, args.d_model), c(position)),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc),
        nn.Sequential(TextEmbedding(args.d_model, args.vocab_size), c(position)), 
        Decoder(args.d_model, c(attn), c(attn), c(ff), args.dropout, args.N_dec),
        c(generator), c(generator))
        
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


