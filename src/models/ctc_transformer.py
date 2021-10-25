
# 2020 Ruchao Fan
# Some transformer-related codes are borrowed from 
# https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, ConvEmbedding
from models.blocks.transformer_blocks import Encoder
from utils.ctc_prefix import CTCPrefixScore, logzero, logone


def make_model(input_size, args):
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout, max_len=args.max_len)
    generator = Generator(args.d_model, args.vocab_size)
    
    interctc_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interctc_alpha > 0 else None
    model = CTCTransformer(
        ConvEmbedding(input_size, args.d_model, args.dropout, c(position), causal=args.causal),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc),
        generator, interctc_gen)
        
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

class CTCTransformer(nn.Module):
    def __init__(self, src_embed, encoder, ctc_gen, interctc_gen=None):
        super(CTCTransformer, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.ctc_generator = ctc_gen
        if interctc_gen is not None:
            self.interctc_generator = interctc_gen

    def forward(self, src, src_mask, ctc_alpha, interctc_alpha, args):
        if args.causal:
            src, src_mask = self.get_causal_mask(src, src_mask, args.forward)
        
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask, interctc_alpha)
        #CTC Loss needs log probability as input
        
        if interctc_alpha > 0:
            enc_h, inter_h = enc_h
            ctc_out = self.ctc_generator(enc_h)
            inter_out = self.interctc_generator(inter_h)
        else:
            ctc_out = self.ctc_generator(enc_h)
            inter_out = 0

        return ctc_out, inter_out, enc_h

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

    def greedy_decode(self, src, src_mask, src_size, vocab, args):
        bs = src.size(0)
        if args.causal:
            src, src_mask = self.get_causal_mask(src, src_mask, args.forward)

        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask)
        ctc_out = self.ctc_generator(enc_h)
        pred = ctc_out.argmax(-1).cpu().numpy()
        
        batch_top_seqs = [[{"hyp": []}] for b in range(bs)]
        feat_size = (src_size * enc_h.size(1)).long()
        
        for b in range(bs):
            result = []
            for j in range(feat_size[b].item()):
                if pred[b][j] != 0:
                    if j != 0 and pred[b][j] == pred[b][j-1]:
                        continue
                    else:
                        result.append(pred[b][j])
            batch_top_seqs[b][0]["hyp"] = result
        return batch_top_seqs
    
    def beam_decode(self, src, src_mask, src_size, vocab, args, lm_model=None):
        """ctc beam decoding

        args.rnnlm: path of rnnlm model
        args.lm_weight: use lm out probability for joint decoding when >0.
        """
        bs = src.size(0)
        sos = vocab.word2index['sos']
        blank = vocab.word2index['blank']
        beam_width = args.ctc_beam
        pruning_size = args.ctc_pruning
        length_penalty = args.ctc_lp 
        lm_weight = args.lm_weight
        
        if args.causal:
            src, src_mask = self.get_causal_mask(src, src_mask, args.forward)

        x, src_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, src_mask)
        ctc_out = self.ctc_generator(enc_h)
        top_probs, top_indices = torch.topk(ctc_out, pruning_size, dim=-1)
        src_size = (src_size * enc_h.size(1)).long()

        ys = torch.ones(1,1).fill_(sos).type_as(src_mask).long() 
        batch_top_seqs = [ [{'ys': ys, 'p_blk': logone, 'p_nblk': logzero, 'score_ctc': 0.0, 'score_lm': 0.0, 'hyp': []}] for b in range(bs) ]
        
        max_ys = 1
        for t in range(ctc_out.size(1)):
            if lm_model is not None:
                ys, ys_size = [], []
                for b in range(bs):
                    if t > src_size[b].item() or torch.exp(ctc_out[b, t, blank]).item() > 0.95:
                        continue

                    for seq in batch_top_seqs[b]:
                        ys_fill = torch.zeros(1, max_ys).type_as(src_size).long()
                        ys_size.append(seq['ys'].size(1))
                        ys_fill.narrow(1, 0, ys_size[-1]).copy_(seq['ys'])
                        ys.append(ys_fill)

                if len(ys) == 0:
                    continue
                else:
                    max_ys += 1

                ys = torch.cat(ys, dim=0)
                ys_size = torch.Tensor(ys_size).type_as(src_size).long().unsqueeze(1).unsqueeze(2)

                tgt_mask = (ys != args.padding_idx).unsqueeze(1)
                tgt_mask = tgt_mask & model.subsequent_mask(ys.size(-1)).type_as(src_mask)
                lm_prob = lm_model(ys, tgt_mask)
                lm_prob = torch.gather(lm_prob, 1, (ys_size-1).repeat(1,1,lm_prob.size(-1)))[:,-1,:]

            s_idx = -1
            for b in range(bs):
                if t > src_size[b].item(): # or torch.exp(ctc_out[b, t, blank]).item() > 0.95:
                    continue

                new_beam = []
                for seq in batch_top_seqs[b]:
                    s_idx += 1
                    ys, hyp, p_b, p_nb, score_lm = seq['ys'], seq['hyp'], seq['p_blk'], seq['p_nblk'], seq['score_lm']

                    # blank or repetition
                    new_p_nb = p_nb + ctc_out[b, t, hyp[-1]].item() if len(hyp) > 0 else logzero
                    p_temp = ctc_out[b, t, blank].item()
                    new_p_b = np.logaddexp(p_b + p_temp, p_nb + p_temp)
                    p_total = np.logaddexp(new_p_b, new_p_nb)
                    new_beam.append({'ys':ys, 'p_blk': new_p_b, 'p_nblk': new_p_nb, 'score_ctc': p_total, 'score_lm': score_lm, 'hyp': hyp})

                    # add one extra token
                    new_p_b = logzero
                    for c in top_indices[b, t]:
                        if c.item() == blank:
                            continue

                        c_prev = hyp[-1] if len(hyp) > 0 else None
                        p_temp = ctc_out[b, t, c].item() 
                        new_p_nb = np.logaddexp(p_b + p_temp, p_nb + p_temp) if c.item() != c_prev else p_b + p_temp
                        p_total = np.logaddexp(new_p_b, new_p_nb)
                        if lm_model is not None:
                            score_lm += lm_prob[s_idx, c].item() * lm_weight
                            ys = torch.cat([seq['ys'], c.view(-1,1)], dim=1)
                        
                        new_beam.append({'ys':ys, 'p_blk': new_p_b, 'p_nblk': new_p_nb, 'score_ctc': p_total, 'score_lm': score_lm, 'hyp': hyp+[c.item()]})

                sort_f = lambda x: x['score_ctc'] + x['score_lm'] + length_penalty * len(x['hyp']) \
                            if length_penalty is not None else lambda x: x['score_ctc'] + x['score_lm']
                batch_top_seqs[b] = sorted(new_beam, key=sort_f, reverse=True)[:beam_width]
        return batch_top_seqs

