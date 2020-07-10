
# 2020 Ruchao Fan
# Some transformer-related codes are borrowed from 
# https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby

from models.modules.attention import MultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, ConvEmbedding, TextEmbedding
from models.blocks.laso_blocks import Encoder, PosDepSummarizer
#from utils.ctc_prefix import CTCPrefixScore, logzero, logone

def make_model(input_size, args):
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.n_head, args.d_model)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    generator = Generator(args.d_model, args.vocab_size)
    pe = create_pe(args.d_model)

    model = LASO(
        ConvEmbedding(input_size, args.d_model, args.dropout),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc),
        PosDepSummarizer(args.d_model, c(attn), c(ff), args.dropout, args.N_pds), 
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_dec),
        c(generator), c(generator), pe)
        
    
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
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, T=1.0):
        return F.log_softmax(self.proj(x)/T, dim=-1)

class LASO(nn.Module):
    def __init__(self, src_embed, encoder, pdsummarizer, decoder, ctc_gen, att_gen, pe):
        super(LASO, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.pdsummarizer = pdsummarizer
        self.decoder = decoder
        self.ctc_generator = ctc_gen
        self.att_generator = att_gen
        self.pe = pe

    def forward(self, src, tgt, src_mask, tgt_mask, ctc_alpha):
        # 1. compute ctc output
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask)
        if ctc_alpha > 0:
            ctc_out = self.ctc_generator(enc_h)
        else:
            ctc_out = 0
        
        # 2. position dependent summarizer
        bs, ymax = tgt.size()
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        pds_out = self.pdsummarizer(pe, enc_h, x_mask)
 
        # 4. decoder, output units generation
        dec_h = self.decoder(pds_out, tgt_mask)
        att_out = self.att_generator(dec_h)
        return ctc_out, att_out, enc_h

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)
    
    def beam_decode(self, src, src_mask, src_size, vocab, args):
        """att decoding with rnnlm and ctc out probability

        args.rnnlm: path of rnnlm model
        args.ctc_weight: use ctc out probability for joint decoding when >0.
        """
        bs = src.size(0)
        sos = vocab.word2index['sos']
        eos = vocab.word2index['eos']
        blank = vocab.word2index['blank']

        x, src_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, src_mask)
        ctc_out = self.ctc_generator(enc_h)
        src_size = (src_size * ctc_out.size(1)).long()

        trigger_mask, ylen, ymax = self.best_path_align(ctc_out, src_size, blank)
        trigger_mask = trigger_mask & src_mask
        
        tgt_mask1 = torch.full((bs, ymax), 1).type_as(src_mask)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        tgt_mask2 = tgt_mask1 & self.subsequent_mask(ymax).type_as(tgt_mask1) # uni-direc
        
        # 3. acoustic language model
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        acouh = self.acoustic_extractor(pe, enc_h, trigger_mask[:,1:,:])   # sos + n_unit
        acouh_gen = self.acoustic_lm(acouh, tgt_mask2)
        
        # 4. decoder, output units generation
        dec_h = self.decoder(acouh_gen, tgt_mask1)
        att_out = self.att_generator(dec_h)
        
        #log_probs, best_paths = torch.max(ctc_out, -1)
        #aligned_seq_shift = best_paths.new_zeros(best_paths.size())
        #aligned_seq_shift[:, 1:] = best_paths[:,:-1]
        #dup = best_paths == aligned_seq_shift
        #best_paths.masked_fill_(dup, 0)
        log_probs, best_paths = torch.max(att_out, -1)

        batch_top_seqs = []
        for b in range(bs):
            batch_top_seqs.append([])
            batch_top_seqs[b].append({'hyp': best_paths[b].cpu().numpy()})
        return batch_top_seqs

