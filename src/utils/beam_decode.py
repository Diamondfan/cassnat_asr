#!/usr/bin/env python
# 2020 Ruchao Fan

import torch
import numpy as np
from utils.ctc_prefix import logzero, logone

def ctc_beam_decode(model, src, src_mask, src_size, vocab, args):
        # decoding with lm is not applicable temporarily
        bs = src.size(0)
        sos = vocab.word2index['sos']
        eos = vocab.word2index['eos']
        blank = vocab.word2index['blank']
        beam_width = args.beam_width
        pruning_size = args.ctc_beam
        lm_weight = args.lm_weight
        lm = args.rnnlm

        x, src_mask = model.src_embed(src, src_mask)
        enc_h = model.encoder(x, src_mask)
        src_size = (src_size * enc_h.size(1)).long()
        ctc_out = model.ctc_generator(enc_h)
        
        batch_top_seqs = [ [{'p_blk': logone, 'p_nblk': logzero, 'score': 0.0, 'hyp': [] } ] for b in range(bs) ]
        
        if lm != 'None':
            assert lm_weight > 0
            lm.eval()
            for b in range(bs):
                batch_top_seqs[b][0]['rnnlm_prev'] = lm.init(sos)

        best_hyps = []
        for b in range(bs):
            beam = batch_top_seqs[b]
            #import pdb
            #pdb.set_trace()
            top_probs, top_indices = torch.topk(ctc_out[b], pruning_size, dim=-1)
            
            for t in range(src_size[b].item()):
                if torch.exp(ctc_out[b, t, blank]).item() > 0.95:
                    continue 

                new_beam = []                
                for seq in beam:
                    hyp, p_b, p_nb = seq['hyp'], seq['p_blk'], seq['p_nblk']

                    # blank or repetition
                    new_p_nb = p_nb + ctc_out[b, t, hyp[-1]].item() if len(hyp) > 0 else logzero
                    p_temp = ctc_out[b, t, blank].item()
                    new_p_b = np.logaddexp(p_b + p_temp, p_nb + p_temp)
                    p_total = np.logaddexp(new_p_b, new_p_nb)
                    new_beam.append({'p_blk': new_p_b, 'p_nblk': new_p_nb, 'score': p_total, 'hyp': hyp})

                    # add one extra token
                    new_p_b = logzero
                    for c in top_indices[t].cpu().numpy():
                        if c == blank:
                            continue
                        
                        c_prev = hyp[-1] if len(hyp) > 0 else None
                        p_temp = ctc_out[b, t, c].item() 
                        new_p_nb = np.logaddexp(p_b + p_temp, p_nb + p_temp) if c != c_prev else p_b + p_temp

                        p_total = np.logaddexp(new_p_b, new_p_nb)
                        new_beam.append({'p_blk': new_p_b, 'p_nblk': new_p_nb, 'score': p_total, 'hyp': hyp+[c]})

                sort_f = lambda x: x['score'] + args.length_penalty * len(x['hyp']) \
                            if args.length_penalty is not None else lambda x: x['score']
                beam = sorted(new_beam, key=sort_f, reverse=True)[:beam_width]
            batch_top_seqs[b] = beam
        return batch_top_seqs

