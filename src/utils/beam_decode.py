#!/usr/bin/env python
# 2020 Ruchao Fan

import torch
import numpy as np
from utils.ctc_prefix import logzero, logone

def ctc_beam_decode(model, src, src_mask, src_size, vocab, args, lm_model=None):
    # decoding with lm is not applicable temporarily
    bs = src.size(0)
    sos = vocab.word2index['sos']
    eos = vocab.word2index['eos']
    blank = vocab.word2index['blank']
    beam_width = args.ctc_beam
    pruning_size = args.ctc_pruning
    lm_weight = args.ctc_lm_weight
    length_penalty = args.ctc_lp 

    x, src_mask = model.src_embed(src, src_mask)
    enc_h = model.encoder(x, src_mask)
    src_size = (src_size * enc_h.size(1)).long()
    ctc_out = model.ctc_generator(enc_h)
    top_probs, top_indices = torch.topk(ctc_out, pruning_size, dim=-1)
    ys = torch.ones(1,1).fill_(sos).type_as(src_size).long()
    
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
            if t > src_size[b].item() or torch.exp(ctc_out[b, t, blank]).item() > 0.95:
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


def ctc_beam_decode_cache(model, src, src_mask, src_size, vocab, args, lm_model=None):
    # Decoding batch first and then iteratively with time
    #
    bs = src.size(0)
    sos = vocab.word2index['sos']
    eos = vocab.word2index['eos']
    blank = vocab.word2index['blank']
    beam_width = args.beam_width
    pruning_size = args.ctc_beam
    lm_weight = args.lm_weight

    x, src_mask = model.src_embed(src, src_mask)
    enc_h = model.encoder(x, src_mask)
    src_size = (src_size * enc_h.size(1)).long()
    ctc_out = model.ctc_generator(enc_h)
    
    batch_top_seqs = [ [{'p_blk': logone, 'p_nblk': logzero, 'score_ctc': 0.0, 'score_lm': 0.0, 'hyp': [], 'prev_state': None}] for b in range(bs) ]
    
    best_hyps = []
    for b in range(bs):
        beam = batch_top_seqs[b]
        top_probs, top_indices = torch.topk(ctc_out[b], pruning_size, dim=-1)
        
        for t in range(src_size[b].item()):
            if torch.exp(ctc_out[b, t, blank]).item() > 0.95:
                continue 

            new_beam = []                
            for seq in beam:
                hyp, p_b, p_nb, score_lm, prev_state = seq['hyp'], seq['p_blk'], seq['p_nblk'], seq['score_lm'], seq['prev_state']

                # blank or repetition
                new_p_nb = p_nb + ctc_out[b, t, hyp[-1]].item() if len(hyp) > 0 else logzero
                p_temp = ctc_out[b, t, blank].item()
                new_p_b = np.logaddexp(p_b + p_temp, p_nb + p_temp)
                p_total = np.logaddexp(new_p_b, new_p_nb)
                new_beam.append({'p_blk': new_p_b, 'p_nblk': new_p_nb, 'score_ctc': p_total, 'score_lm': score_lm, 'hyp': hyp, 'prev_state': prev_state})

                if lm_model is not None:
                    ys = [sos] + hyp
                    ys = torch.Tensor(ys).type_as(ctc_out).long().unsqueeze(0)
                    lm_prob, new_states = lm_model.score(ys, prev_state)

                # add one extra token
                new_p_b = logzero
                for c in top_indices[t].cpu().numpy():
                    if c == blank:
                        continue
                    
                    c_prev = hyp[-1] if len(hyp) > 0 else None
                    p_temp = ctc_out[b, t, c].item() 
                    new_p_nb = np.logaddexp(p_b + p_temp, p_nb + p_temp) if c != c_prev else p_b + p_temp
                    p_total = np.logaddexp(new_p_b, new_p_nb)
                    if lm_model is not None:
                        score_lm += lm_prob[0, c].item() * lm_weight
                        current_states = new_states
                    else:
                        current_states = None
                    
                    new_beam.append({'p_blk': new_p_b, 'p_nblk': new_p_nb, 'score_ctc': p_total, 'score_lm': score_lm, 'hyp': hyp+[c], 'prev_state': current_states})

            sort_f = lambda x: x['score_ctc'] + x['score_lm'] + args.length_penalty * len(x['hyp']) \
                        if length_penalty is not None else lambda x: x['score_ctc'] + x['score_lm']
            beam = sorted(new_beam, key=sort_f, reverse=True)[:beam_width]
        batch_top_seqs[b] = beam
    return batch_top_seqs
                        
