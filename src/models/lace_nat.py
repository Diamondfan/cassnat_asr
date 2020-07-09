
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
from models.blocks.lace_nat_blocks import Encoder, Decoder, AcousticExtracter
from utils.ctc_prefix import CTCPrefixScore, logzero, logone

def make_model(input_size, args):
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.n_head, args.d_model)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    generator = Generator(args.d_model, args.vocab_size)
    pe = create_pe(args.d_model)

    model = LaceNat(
        ConvEmbedding(input_size, args.d_model, args.dropout),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc),
        AcousticExtracter(args.d_model, c(attn), c(ff), args.dropout),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, 3),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, 3),
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

class LaceNat(nn.Module):
    def __init__(self, src_embed, encoder, acoustic_extractor, acoustic_lm, decoder, ctc_gen, att_gen, pe):
        super(LaceNat, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.acoustic_extractor = acoustic_extractor
        self.acoustic_lm = acoustic_lm
        self.decoder = decoder
        self.ctc_generator = ctc_gen
        self.att_generator = att_gen
        self.pe = pe

    def forward(self, src, src_mask, src_size, tgt_label, label_sizes, blank, alm_alpha):
        # 1. compute ctc output
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask)
        ctc_out = self.ctc_generator(enc_h)
        
        # 2. obtain segment point and mask for alm, decoder
        src_size = (src_size * ctc_out.size(1)).long()
        trigger_mask, ylen, ymax = self.viterbi_align(ctc_out, x_mask, src_size, tgt_label[:,:-1], label_sizes, blank)
        trigger_mask = trigger_mask & x_mask
        
        bs, _, d_model = enc_h.size()
        tgt_mask1 = torch.full((bs, ymax), 1).type_as(src_mask)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        tgt_mask2 = tgt_mask1 & self.subsequent_mask(ymax).type_as(tgt_mask1) # uni-direc
        
        # 3. acoustic language model
        if alm_alpha > 0:
            pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax+1,:]
            acouh = self.acoustic_extractor(pe, enc_h, trigger_mask)   # sos + n_unit + eos
            acouh_gen = self.acoustic_lm(acouh[:,:-1,:], tgt_mask2)
        else:
            pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
            acouh = self.acoustic_extractor(pe, enc_h, trigger_mask[:,1:,:])   # n_unit + eos
            acouh_gen = self.acoustic_lm(acouh, tgt_mask1)
 
        # 4. decoder, output units generation
        dec_h = self.decoder(acouh_gen, tgt_mask1)
        att_out = self.att_generator(dec_h)
        return ctc_out, acouh[:,1:,:].contiguous(), acouh_gen, att_out

    def viterbi_align(self, ctc_out, src_mask, src_size, ys, ylens, blank):
        """
        ctc_out: log probability of ctc output
        src_mask, src_size: specify the effective length of each sample in a batch
        ys: target label
        ylen: specify the effective label length of each sampel in a batch
        """
        bs, xmax, vocab = ctc_out.size()

        mask = src_mask.transpose(1,2).repeat([1, 1, vocab]) #bs, T, vocab
        log_probs = ctc_out.masked_fill(mask == 0, logzero).transpose(0, 1)
       
        # 1. insert blanks between labels
        ymax = ys.size(1)
        path = ys.new_zeros(ys.size(0), ymax * 2 + 1).fill_(blank).long()
        path[:, 1::2] = ys
        path_lens = 2 * ylens.long() + 1
        max_path_len = path.size(1)

        # 2. keep probabilities in path 
        batch_index = torch.arange(bs).type_as(ylens).unsqueeze(1)
        seq_index = torch.arange(xmax).type_as(ylens).unsqueeze(1).unsqueeze(2)
        log_probs_path = log_probs[seq_index, batch_index, path]
        
        # 3. forward algorithm with max replacing sum
        bp = ys.new_zeros(bs, xmax, max_path_len)
        alpha = log_probs.new_zeros(xmax+1, bs, max_path_len).fill_(logzero)
        alpha[0, :, 0] = logone

        same_transition = (path[:, :-2] == path[:, 2:]) #including blank-2-blank
        index_fix = torch.arange(max_path_len).type_as(ylens)
        outside = index_fix >= path_lens.unsqueeze(1)
        
        for t in range(xmax):
            mat = alpha.new_zeros(3, bs, max_path_len).fill_(logzero)
            mat[0, :, :] = alpha[t]
            mat[1, :, 1:] = alpha[t, :, :-1]
            mat[2, :, 2:] = alpha[t, :, :-2]
            mat[2, :, 2:][same_transition] = logzero   # blank and same label have only two previous nodes
            max_prob, max_indices = torch.max(mat, dim=0)
            max_prob[outside] = logzero
            bp[:,t,:] = index_fix - max_indices
            alpha[t+1,:,:] = max_prob + log_probs_path[t,:,:]
        
        # 4. Compare N-1 and N-2 at t-1, get the path with a higher prob
        #    Then back path tracing, Seems hard to parallelize this part
        aligned_seq = ys.new_zeros((bs, xmax))
        for b in range(bs):
            xb, yb = src_size[b].item(), path_lens[b].item()
            score1, score2 = alpha[xb, b, yb-1], alpha[xb, b, yb-2]
            aligned_seq[b, xb-1] = yb - 1 if score1 > score2 else yb - 2
            for t in range(xb-1, 0, -1):
                aligned_seq[b, t-1] = bp[b,t,aligned_seq[b, t]]
        
        # 5. remove repetition, locate the time step for each label 
        aligned_seq = torch.gather(path, 1, aligned_seq)
        aligned_seq_shift = aligned_seq.new_zeros(aligned_seq.size())
        aligned_seq_shift[:, 1:] = aligned_seq[:,:-1]
        dup = aligned_seq == aligned_seq_shift
        aligned_seq.masked_fill_(dup, 0)
        aligned_seq_shift[:,1:] = aligned_seq[:,:-1]
        
        # 6. transcribe aliged_seq to trigger mask
        trigger_mask = (aligned_seq_shift != blank).cumsum(1).unsqueeze(1).repeat(1, ymax+1, 1)
        trigger_mask = trigger_mask == torch.arange(ymax+1).type_as(trigger_mask).unsqueeze(0).unsqueeze(2)
        sos_mask = trigger_mask.new_zeros((bs,1, xmax))
        sos_mask[:,:,0] = 1
        trigger_mask[:,-1:,:].masked_fill_(src_mask==0, 0)
        trigger_mask[:,-1,:].scatter_(1, src_size.unsqueeze(1)-1, 1)
        trigger_mask = torch.cat([sos_mask, trigger_mask], 1)
        
        ylen = ylens + 1  # +1 for <eos>
        ymax += 1
        return trigger_mask, ylen, ymax

    def best_path_align(self, ctc_out, src_mask, src_size, blank):
        "This is used for decoding, forced alignment is needed for training"
        bs, xmax, _ = ctc_out.size()
        best_paths = ctc_out.argmax(-1)
        best_paths = best_paths.masked_fill(src_mask.squeeze(1)==0, 0)

        aligned_seq_shift = best_paths.new_zeros(best_paths.size())
        aligned_seq_shift[:, 1:] = best_paths[:,:-1]
        dup = best_paths == aligned_seq_shift
        best_paths.masked_fill_(dup, 0)
        aligned_seq_shift[:,1:] = best_paths[:,:-1]
        
        ylen = torch.sum((best_paths != blank), 1)
        ymax = torch.max(ylen).item()
        trigger_mask = (aligned_seq_shift != blank).cumsum(1).unsqueeze(1).repeat(1, ymax+1, 1)
        trigger_mask = trigger_mask == torch.arange(ymax+1).type_as(trigger_mask).unsqueeze(0).unsqueeze(2)
        sos_mask = trigger_mask.new_zeros((bs,1, xmax))
        sos_mask[:,:,0] = 1
        trigger_mask = torch.cat([sos_mask, trigger_mask], 1)
        
        ylen = ylen + 1
        ymax += 1
        return trigger_mask, ylen, ymax

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

        #import pdb
        #pdb.set_trace()
        trigger_mask, ylen, ymax = self.best_path_align(ctc_out, src_mask, src_size, blank)
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

