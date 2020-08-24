
# 2020 Ruchao Fan

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import groupby

from models.modules.attention import MultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, ConvEmbedding, TextEmbedding
from models.blocks.fanat_blocks import Encoder, Decoder, AcEmbedExtractor, EmbedMapper
from utils.ctc_prefix import logzero, logone, CTCPrefixScore

def make_model(input_size, args):
    c = copy.deepcopy
    attn = MultiHeadedAttention(args.n_head, args.d_model)
    ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
    position = PositionalEncoding(args.d_model, args.dropout)
    generator = Generator(args.d_model, args.vocab_size)
    pe = create_pe(args.d_model)

    if args.use_src:
        decoder_use = Decoder(args.d_model, c(attn), c(attn), c(ff), args.dropout, args.N_dec)
    else:
        decoder_use = Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_dec)

    model = FaNatTp(
        ConvEmbedding(input_size, args.d_model, args.dropout),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc),
        AcEmbedExtractor(args.d_model, c(attn), c(ff), args.dropout, args.N_extra),
        EmbedMapper(args.d_model, c(attn), c(ff), args.dropout, args.N_map),
        decoder_use,
        c(generator), c(generator), c(generator), pe)
        
    
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

class FaNatTp(nn.Module):
    def __init__(self, src_embed, encoder, acembed_extractor, embed_mapper, decoder, ctc_gen, trig_pred, att_gen, pe):
        super(FaNatTp, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.acembed_extractor = acembed_extractor
        self.embed_mapper = embed_mapper
        self.decoder = decoder
        self.ctc_generator = ctc_gen
        self.trigger_predictor = trig_pred
        self.att_generator = att_gen
        self.pe = pe

    def forward(self, src, src_mask, src_size, tgt_label, ylen, args, tgt=None):
        # 1. compute ctc output
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask)
        tp_out = self.trigger_predictor(enc_h)
        if args.ctc_alpha > 0:
            ctc_out = self.ctc_generator(enc_h)
        else:
            ctc_out = enc_h.new_zeros(enc_h.size())
        
        # 2. prepare different masks,
        if args.online_trigger:
            assert args.ctc_alpha > 0
            src_size = (src_size * ctc_out.size(1)).long()
            blank = args.padding_idx
            with torch.no_grad():
                aligned_seq, trigger_mask, ylen, ymax = self.viterbi_align(ctc_out, x_mask, src_size, tgt_label[:,:-1], ylen, blank, args.sample_dist)
            
            if args.context_trigger > 0:
                trigger_shift_right = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_right[:, :, 1:] = trigger_mask[:,:, :-1]
                #trigger_shift_left = trigger_mask.new_zeros(trigger_mask.size())
                #trigger_shift_left[:,:,:-1] = trigger_mask[:,:,1:]
                trigger_mask = trigger_mask | trigger_shift_right #| trigger_shift_left
            trigger_mask = trigger_mask & x_mask
        else:
            raise NotImplementError

        bs, _, d_model = enc_h.size()
        tgt_mask1 = torch.full((bs, ymax), 1).type_as(src_mask)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        
        # 3. Extract Acoustic embedding and Map it to Word embedding
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        ac_embed = self.acembed_extractor(pe, enc_h, trigger_mask)
        pred_embed = self.embed_mapper(ac_embed, tgt_mask1)

        # 4. decoder, output units generation
        if args.use_unimask:
            true_embed = args.word_embed(tgt)
            true_embed = true_embed.masked_fill(tgt_mask1.transpose(1,2) == 0, 0)
            pred_embed = torch.cat([true_embed[:,0:1,:], pred_embed[:,:-1,:]], dim=1)
            pred_embed = pred_embed.masked_fill(tgt_mask1.transpose(1,2) == 0, 0)
            tgt_mask = tgt_mask1 & self.subsequent_mask(ymax).type_as(tgt_mask1) # uni-direc
        else:
            tgt_mask = tgt_mask1
            true_embed = None

        if args.use_src:
            if args.src_trigger:
                x_mask = trigger_mask
            dec_h = self.decoder(pred_embed, enc_h, x_mask, tgt_mask)
        else:
            dec_h = self.decoder(pred_embed, tgt_mask)
        att_out = self.att_generator(dec_h)
        return ctc_out, tp_out, att_out, tgt_mask1, aligned_seq, x_mask

    def viterbi_align(self, ctc_out, src_mask, src_size, ys, ylens, blank, sample_dist):
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

        if sample_dist > 0:
            orig_pos = torch.nonzero(aligned_seq_shift, as_tuple=True)
            sample_shift = torch.randint(-sample_dist, sample_dist, orig_pos[1].size()).type_as(aligned_seq_shift)
            aligned_seq_shift[orig_pos] = blank
            orig_pos[1].add_(sample_shift)
            aligned_seq_shift[orig_pos] = 1

        # 6. transcribe aliged_seq to trigger mask
        trigger_mask = (aligned_seq_shift != blank).cumsum(1).unsqueeze(1).repeat(1, ymax+1, 1)
        trigger_mask = trigger_mask == torch.arange(ymax+1).type_as(trigger_mask).unsqueeze(0).unsqueeze(2)
        #sos_mask = trigger_mask.new_zeros((bs,1, xmax))
        #sos_mask[:,:,0] = 1
        trigger_mask[:,-1:,:].masked_fill_(src_mask==0, 0)   # remove position with padding_idx
        trigger_mask[:,-1,:].scatter_(1, src_size.unsqueeze(1)-1, 1) # give the last character one to keep at least one active position for eos
        #trigger_mask = torch.cat([sos_mask, trigger_mask], 1)
                 
        ylen = ylens + 1
        ymax += 1
        return aligned_seq, trigger_mask, ylen, ymax

    def best_path_align(self, ctc_out, src_mask, src_size, blank, sample_dist=0, sample_num=0):
        "This is used for decoding, forced alignment is needed for training"
        bs, xmax, _ = ctc_out.size()
        best_paths = ctc_out.argmax(-1)
        best_paths = best_paths.masked_fill(src_mask.squeeze(1)==0, 0)

        aligned_seq_shift = best_paths.new_zeros(best_paths.size())
        aligned_seq_shift[:, 1:] = best_paths[:,:-1]
        dup = best_paths == aligned_seq_shift
        best_paths.masked_fill_(dup, 0)
        aligned_seq_shift[:,1:] = best_paths[:,:-1]

        if sample_dist > 0:
            for b in range(bs // sample_num):
                orig_pos = torch.nonzero(aligned_seq_shift[b*sample_num+1:(b+1)*sample_num,:], as_tuple=True)
                sample_shift = torch.randint(-sample_dist, sample_dist, orig_pos[1].size()).type_as(aligned_seq_shift)
                aligned_seq_shift[b*sample_num+1:(b+1)*sample_num,:][orig_pos] = blank
                orig_pos[1].add_(sample_shift)
                aligned_seq_shift[b*sample_num+1:(b+1)*sample_num,:][orig_pos] = 1
            
        ylen = torch.sum((aligned_seq_shift != blank), 1)
        ymax = torch.max(ylen).item()
        trigger_mask = (aligned_seq_shift != blank).cumsum(1).unsqueeze(1).repeat(1, ymax+1, 1)
        trigger_mask = trigger_mask == torch.arange(ymax+1).type_as(trigger_mask).unsqueeze(0).unsqueeze(2)
        #sos_mask = trigger_mask.new_zeros((bs,1, xmax))
        #sos_mask[:,:,0] = 1
        trigger_mask[:,-1:,:].masked_fill_(src_mask==0, 0)
        trigger_mask[:,-1,:].scatter_(1, src_size.unsqueeze(1)-1, 1)
        #trigger_mask = torch.cat([sos_mask, trigger_mask], 1)
        
        ylen = ylen + 1
        ymax += 1
        return trigger_mask, ylen, ymax

    def beam_path_align(self, ctc_out, src_mask, src_size, blank, ctc_top_seqs, sample_dist):
        # Obatain a better alignment and then trigger mask with languag model
        # This is similar with ctc beam search, but without merging the paths with same output
        bs = ctc_out.size(0)
        tgt_label, ylen = [], []
        for b in range(bs):
            tgt_label.append([])
            ylen.append(0)
            for idx in ctc_top_seqs[b][0]['hyp']:
                tgt_label[-1].append(idx)
                ylen[-1] += 1

        ymax = max(ylen)
        tgt = src_size.new_zeros(bs, ymax).long()
        for b in range(bs):
            tgt[b].narrow(0, 0, ylen[b]).copy_(torch.Tensor(tgt_label[b]).type_as(src_size))
        ylen = torch.Tensor(ylen).type_as(src_size)
        
        trigger_mask, ylen, ymax = self.viterbi_align(ctc_out, src_mask, src_size, tgt, ylen, blank, sample_dist)
        return trigger_mask, ylen, ymax

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)
    
    def beam_decode(self, src, src_mask, src_size, vocab, args, lm_model=None, ctc_top_seqs=None, labels=None, label_sizes=None):
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
        
        if args.use_ctc_trigger:
            ctc_out = self.ctc_generator(enc_h)
            src_size = (src_size * ctc_out.size(1)).long()
            if args.decode_type == 'ctc_att':
                trigger_mask, ylen, ymax = self.beam_path_align(ctc_out, src_mask, src_size, blank, ctc_top_seqs, args.sample_dist)
            elif args.decode_type == 'oracle_att':
                trigger_mask, ylen, ymax = self.viterbi_align(ctc_out, src_mask, src_size, labels[:,1:-1], label_sizes, blank, args.sample_dist)
            else:
                if args.sample_num > 0:
                    ctc_out = ctc_out.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, ctc_out.size(1), ctc_out.size(2))
                    enc_h = enc_h.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, enc_h.size(1), enc_h.size(2))
                    src_mask = src_mask.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, src_mask.size(1), src_mask.size(2))
                    src_size = src_size.unsqueeze(1).repeat(1, args.sample_num).reshape(-1)
                trigger_mask, ylen, ymax = self.best_path_align(ctc_out, src_mask, src_size, blank, args.sample_dist, args.sample_num)

            if args.context_trigger > 0:
                trigger_shift_right = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_right[:, :, 1:] = trigger_mask[:,:, :-1]
                #trigger_shift_left = trigger_mask.new_zeros(trigger_mask.size())
                #trigger_shift_left[:,:,:-1] = trigger_mask[:,:,1:]
                trigger_mask = trigger_mask | trigger_shift_right #| trigger_shift_left
            trigger_mask = trigger_mask & src_mask
        else:
            tp_out = self.trigger_predictor(enc_h)
            src_size = (src_size * tp_out.size(1)).long()
            trigger_mask, ylen, ymax = self.best_path_align(tp_out, src_mask, src_size, blank, args.sample_dist, args.sample_num)

        bs, _, d_model = enc_h.size()
        tgt_mask1 = torch.full((bs, ymax), 1).type_as(src_mask)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        
        # 3. Extract Acoustic embedding and Map it to Word embedding
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        ac_embed = self.acembed_extractor(pe, enc_h, trigger_mask)
        pred_embed = self.embed_mapper(ac_embed, tgt_mask1)

        # 4. decoder, output units generation
        if args.use_unimask:
            sos_input = src_size.new_zeros(bs, 1).fill_(sos).long()
            sos_embed = args.word_embed(sos_input)
            pred_embed = torch.cat([sos_embed, pred_embed[:,:-1,:]], dim=1)
            pred_embed = pred_embed.masked_fill(tgt_mask1.transpose(1,2) == 0, 0)
            tgt_mask = tgt_mask1 & self.subsequent_mask(ymax).type_as(tgt_mask1) # uni-direc
        else:
            tgt_mask = tgt_mask1

        if args.use_src:
            if args.src_trigger:
                src_mask = trigger_mask
            dec_h = self.decoder(pred_embed, enc_h, src_mask, tgt_mask)
        else:
            dec_h = self.decoder(pred_embed, tgt_mask)
        att_out = self.att_generator(dec_h)
        
        if args.sample_num > 0:
            _, seql, dim = att_out.size()
            att_out = att_out.reshape(-1, args.sample_num, seql, dim).masked_fill(tgt_mask.reshape(-1, args.sample_num, tgt_mask.size(-2), tgt_mask.size(-1)).transpose(2,3)==0, 0)
            max_prob = att_out.max(-1)[0]
            prob_sum = max_prob.sum(-1) / (max_prob != 0).sum(-1).float()
            max_indices = prob_sum.max(-1, keepdim=True)[1]
            att_out = torch.gather(att_out, 1, max_indices.unsqueeze(2).unsqueeze(3).repeat(1,1,seql,dim)).squeeze(1)
            bs = att_out.size(0)
            ylen = ylen.reshape(bs, args.sample_num)[:,0]
        #best_scores, best_indices = torch.topk(att_out, args.beam_width, dim=-1)
        #log_probs, best_paths = torch.max(ctc_out, -1)
        #aligned_seq_shift = best_paths.new_zeros(best_paths.size())
        #aligned_seq_shift[:, 1:] = best_paths[:,:-1]
        #dup = best_paths == aligned_seq_shift
        #best_paths.masked_fill_(dup, 0)
        
        ys = torch.ones(1, 1).fill_(sos).long()
        if args.use_gpu:
            ys = ys.cuda()
        
        batch_top_seqs = [ [{'ys': ys, 'score': 0.0, 'hyp': [sos] } ] for b in range(bs) ]
        
        for i in range(ymax):
            # batchify the batch and beam
            all_seqs, ys, att_prob = [], [], []
            
            for b in range(bs):
                all_seqs.append([])
                for seq in batch_top_seqs[b]:
                    if i > ylen[b].item():
                        all_seqs[b].append(seq)
                        continue
            
                    att_prob.append(att_out[b,i:i+1,:])
                    if args.lm_weight > 0:
                        ys.append(seq['ys'])

            if len(att_prob) == 0: #if no beam active, end decoding
                break
            # concat and get decoder out probability
            att_prob = torch.cat(att_prob, dim=0)
       
            if args.lm_weight > 0:
                ys = torch.cat(ys, dim=0)

            if args.lm_weight > 0:
                tgt_mask = (ys != args.padding_idx).unsqueeze(1)
                tgt_mask = tgt_mask & self.subsequent_mask(ys.size(-1)).type_as(src_mask)
                lm_prob = lm_model(ys, tgt_mask)[:,-1,:]
                local_prob = att_prob + args.lm_weight * lm_prob
            else:
                local_prob = att_prob
            
            local_scores, indices = torch.topk(local_prob, args.beam_width, dim=-1)
            
            # distribute scores to corresponding sample and beam
            s_idx = -1
            for b in range(bs):
                for seq in batch_top_seqs[b]:
                    if i > ylen[b].item():
                       continue
                    s_idx += 1

                    for j in range(args.beam_width):
                        next_token = indices[s_idx][j]
                        token_score = local_scores[s_idx][j].item()
                        score = seq['score'] + token_score

                        if args.lm_weight > 0:
                            ys = torch.cat([seq['ys'],next_token.view(-1,1)],dim=-1)
                        else:
                            ys = seq['ys']

                        rs_seq = {'ys':ys, 'score': score, 'hyp': seq['hyp']+ [next_token.item()] } 
                        all_seqs[b].append(rs_seq)

                sort_f = lambda x:x['score'] + (len(x['hyp'])-1) * args.length_penalty \
                            if args.length_penalty is not None else lambda x:x['score']                
                batch_top_seqs[b] = sorted(all_seqs[b], key=sort_f, reverse=True)[:args.beam_width]

        #batch_top_seqs = []
        #for b in range(bs):
        #    batch_top_seqs.append([])
        #    batch_top_seqs[b].append({'hyp': best_paths[b].cpu().numpy()})
        return batch_top_seqs
