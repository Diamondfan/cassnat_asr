
# 2020 Ruchao Fan
# Some transformer-related codes are borrowed from 
# https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention import MultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, ConvEmbedding, TextEmbedding
from models.blocks.transformer_blocks import Encoder, Decoder
from utils.ctc_prefix import CTCPrefixScore

def make_model(input_size, args):
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.n_head, args.d_model)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    generator = Generator(args.d_model, args.vocab_size)
    
    model = Transformer(
        ConvEmbedding(input_size, args.d_model, args.dropout),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc),
        nn.Sequential(TextEmbedding(args.d_model, args.vocab_size), c(position)), 
        Decoder(args.d_model, c(attn), c(attn), c(ff), args.dropout, args.N_dec),
        c(generator), c(generator))
        
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, T=1.0):
        return F.log_softmax(self.proj(x)/T, dim=-1)

class Transformer(nn.Module):
    def __init__(self, src_embed, encoder, tgt_embed, decoder, ctc_gen, att_gen):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_generator = ctc_gen
        self.att_generator = att_gen

    def forward(self, src, tgt, src_mask, tgt_mask, ctc_alpha):
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask)
        #CTC Loss needs log probability as input
        if ctc_alpha > 0:
            ctc_out = self.ctc_generator(enc_h)
        else:
            ctc_out = 0
        dec_h = self.decoder(self.tgt_embed(tgt), enc_h, x_mask, tgt_mask)
        att_out = self.att_generator(dec_h)
        return ctc_out, att_out, enc_h

    def forward_att(self, src, tgt, src_mask, tgt_mask):
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask)
        dec_h = self.decoder(self.tgt_embed(tgt), enc_h, x_mask, tgt_mask)
        att_out = F.softmax(self.att_generator.proj(dec_h), dim=-1)
        return att_out

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)
        
    def beam_decode(self, src, src_mask, vocab, args, lm_model=None):
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
        if args.ctc_weight > 0:
            ctc_out = self.ctc_generator(enc_h)
            ctc_length = ctc_out.size(1)
            Scorer = CTCPrefixScore(ctc_length, blank, eos)
            ctc_out.masked_fill_(src_mask.transpose(1,2)==0, Scorer.logzero)
            ctc_out[:,:,blank].masked_fill_(src_mask.squeeze(1)==0, 0)
            init_r = Scorer.initial_state(ctc_out)
        else:
            ctc_out = None

        ys = torch.ones(1, 1).fill_(sos).long()
        if args.use_gpu:
            ys = ys.cuda()
        
        batch_top_seqs = [ [{'ys': ys, 'score': 0.0, 'hyp': [sos] } ] for b in range(bs) ]
        
        for b in range(bs):
            if ctc_out is not None:
                batch_top_seqs[b][0]['ctc_prob_prev'] = init_r[b:b+1]
                init_score = torch.Tensor([[0.0]])
                batch_top_seqs[b][0]['ctc_score_prev'] = init_score.cuda() if args.use_gpu else init_score

        max_decode_step = int(args.max_decode_ratio * enc_h.size(1)) if args.max_decode_ratio > 0 else enc_h.size(1)
        for i in range(max_decode_step):
            # batchify the batch and beam
            all_seqs, ys, ench_use, src_mask_use = [], [], [], []
            
            if ctc_out is not None:
                ctc_out_use, ctc_prev_prob, ctc_prev_score = [], [], []
            
            for b in range(bs):
                all_seqs.append([])
                for seq in batch_top_seqs[b]:
                    if seq['hyp'][-1] == eos:
                        all_seqs[b].append(seq)
                        continue
                    ys.append(seq['ys'])
                    ench_use.append(enc_h[b:b+1])
                    src_mask_use.append(src_mask[b:b+1])
                    
                    if ctc_out is not None:
                        ctc_out_use.append(ctc_out[b:b+1])
                        ctc_prev_prob.append(seq['ctc_prob_prev'])
                        ctc_prev_score.append(seq['ctc_score_prev'])
            
            if len(ys) == 0: #if no beam active, end decoding
                break
            # concat and get decoder out probability
            ys = torch.cat(ys, dim=0)
            src_mask_use = torch.cat(src_mask_use, dim=0)
            ench_use = torch.cat(ench_use, dim=0)
            tgt_mask = (ys != args.padding_idx).unsqueeze(1)
            tgt_mask = tgt_mask & self.subsequent_mask(ys.size(-1)).type_as(src_mask_use.data)
            dec_h = self.decoder(self.tgt_embed(ys), ench_use, src_mask_use, tgt_mask)
            att_prob = self.att_generator(dec_h[:, -1, :], T=args.T)
            if args.lm_weight > 0:
                lm_prob = lm_model(ys, tgt_mask)[:,-1,:]
                local_prob = att_prob + args.lm_weight * lm_prob
                local_prob[:,eos] = (1 + args.lm_weight) * att_prob[:,eos]
            else:
                local_prob = att_prob
            
            if ctc_out is not None:
                att_scores, indices = torch.topk(att_prob, args.ctc_beam, dim=-1)
                ctc_out_use = torch.cat(ctc_out_use, dim=0)
                ctc_prev_prob = torch.cat(ctc_prev_prob, dim=0)
                ctc_prev_scores = torch.cat(ctc_prev_score, dim=0).repeat(1, args.ctc_beam)
                
                ctc_scores, ctc_probs = Scorer(ys, indices, ctc_out_use, ctc_prev_prob)
                local_scores = args.ctc_weight * (ctc_scores - ctc_prev_scores) \
                                + (1 - args.ctc_weight) * att_scores
               
                if args.lm_weight > 0:
                    local_scores += args.lm_weight * lm_prob.gather(1, indices)

                local_scores, local_indices = torch.topk(local_scores, args.beam_width, dim=-1)
                indices = torch.gather(indices, 1, local_indices)
            else:
                local_scores, indices = torch.topk(local_prob, args.beam_width, dim=-1)
            
            # distribute scores to corresponding sample and beam
            s_idx = -1
            for b in range(bs):
                for seq in batch_top_seqs[b]:
                    if seq['hyp'][-1] == eos:
                       continue
                    s_idx += 1

                    for j in range(args.beam_width):
                        next_token = indices[s_idx][j]
                        token_score = local_scores[s_idx][j].item()
                        score = seq['score'] + token_score

                        ys = torch.cat([seq['ys'],next_token.view(-1,1)],dim=-1)
                        rs_seq = {'ys':ys, 'score': score, 'hyp': seq['hyp']+ [next_token.item()] } 
                        
                        if ctc_out is not None:
                            true_idx = local_indices[s_idx, j]
                            rs_seq['ctc_prob_prev'] = ctc_probs[s_idx:s_idx+1, true_idx,:,:]
                            rs_seq['ctc_score_prev'] = ctc_scores[s_idx:s_idx+1, true_idx:true_idx+1]
                        all_seqs[b].append(rs_seq)

                sort_f = lambda x:x['score'] + (len(x['hyp'])-1) * args.length_penalty \
                            if args.length_penalty is not None else lambda x:x['score']                
                batch_top_seqs[b] = sorted(all_seqs[b], key=sort_f, reverse=True)[:args.beam_width]
        return batch_top_seqs


