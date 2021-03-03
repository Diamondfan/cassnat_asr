
# 2020 Ruchao Fan

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
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

    interctc_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interctc_alpha > 0 else None
    interce_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interce_alpha > 0 else None
    if args.use_src:
        decoder_use = Decoder(args.d_model, c(attn), c(attn), c(ff), args.dropout, args.N_dec)
    else:
        decoder_use = Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_dec)

    model = FaNat(
        ConvEmbedding(input_size, args.d_model, args.dropout, c(position)),
        Encoder(args.d_model, c(attn), c(ff), args.dropout, args.N_enc),
        AcEmbedExtractor(args.d_model, c(attn), c(ff), args.dropout, args.N_extra),
        EmbedMapper(args.d_model, c(attn), c(ff), args.dropout, args.N_map),
        decoder_use,
        c(generator), c(generator), pe, interctc_gen, interce_gen)
        
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

class FaNat(nn.Module):
    def __init__(self, src_embed, encoder, acembed_extractor, embed_mapper, decoder, ctc_gen, att_gen, pe, interctc_gen=None, interce_gen=None):
        super(FaNat, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.acembed_extractor = acembed_extractor
        self.embed_mapper = embed_mapper
        self.decoder = decoder
        self.ctc_generator = ctc_gen
        self.att_generator = att_gen
        self.pe = pe
        if interctc_gen is not None:
            self.interctc_generator = interctc_gen
        if interce_gen is not None:
            self.interce_generator = interce_gen

    def forward(self, src, src_mask, src_size, tgt_label, ylen, args):
        # 1. compute ctc output
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask, args.interctc_alpha)
        if args.ctc_alpha > 0 and args.interctc_alpha == 0:
            ctc_out = self.ctc_generator(enc_h)
            interctc_out = 0
        elif args.ctc_alpha > 0 and args.interctc_alpha > 0:
            enc_h, inter_h = enc_h[0], enc_h[1]
            ctc_out = self.ctc_generator(enc_h)
            interctc_out = self.interctc_generator(inter_h)
        else:
            ctc_out = enc_h.new_zeros(enc_h.size())
            interctc_out = 0
        
        # 2. prepare different masks,
        if args.use_trigger:
            assert args.ctc_alpha > 0
            src_size = (src_size * ctc_out.size(1)).long()
            blank = args.padding_idx
            if args.use_best_path:
                trigger_mask, ylen, ymax = self.best_path_align(ctc_out, x_mask, src_size, blank)
            else:
                aligned_seq_shift, ylen, ymax = self.viterbi_align(ctc_out, x_mask, src_size, tgt_label[:,:-1], ylen, blank, args.sample_dist, args.sample_topk)
                trigger_mask, ylen, ymax = self.align_to_mask(aligned_seq_shift, ylen, ymax, x_mask, src_size, blank)

            if args.right_trigger > 0:
                trigger_shift_right = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_right[:, :, 1:] = trigger_mask[:,:, :-1]
                trigger_mask = trigger_mask | trigger_shift_right
            if args.left_trigger > 0:
                trigger_shift_left = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_left[:,:,:-1] = trigger_mask[:,:,1:]
                trigger_mask = trigger_mask | trigger_shift_left
            trigger_mask = trigger_mask & x_mask
        else:
            trigger_mask = x_mask
            ylen = ylen + 1
            ymax = ylen.max().item()

        bs, _, d_model = enc_h.size()
        tgt_mask_bidi = torch.full((bs, ymax), 1).type_as(src_mask)
        tgt_mask_bidi = tgt_mask_bidi.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask_bidi = tgt_mask_bidi.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        
        # 3. Extract Acoustic embedding and Map it to Word embedding
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        ac_embed = self.acembed_extractor(pe, enc_h, trigger_mask)
        pred_embed = self.embed_mapper(ac_embed, tgt_mask_bidi)

        # 4. decoder, output units generation
        if args.use_unimask:
            pred_embed = torch.cat([args.sos_embed, pred_embed[:,:-1,:]], dim=1)
            tgt_mask = tgt_mask_bidi & self.subsequent_mask(ymax).type_as(tgt_mask_bidi) # uni-direc
        else:
            tgt_mask = tgt_mask_bidi
            true_embed = None

        if args.use_src:
            if args.src_trigger:
                x_mask = trigger_mask
            dec_h = self.decoder(pred_embed, enc_h, x_mask, tgt_mask, args.interce_alpha)
            if args.interce_alpha > 0:
                dec_h, interce_h = dec_h[0], dec_h[1]
                interce_out = self.interce_generator(interce_h)
            else:
                interce_out = 0
        else:
            dec_h = self.decoder(pred_embed, tgt_mask)
        att_out = self.att_generator(dec_h)
        return ctc_out, att_out, pred_embed, tgt_mask_bidi, interctc_out, interce_out

    def viterbi_align(self, ctc_out, src_mask, src_size, ys, ylens, blank, sample_dist, sample_topk):
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
        
        if sample_topk > 1:
            t_sample = torch.randint(1, xmax, (1, sample_topk)).numpy().tolist()
        
        for t in range(xmax):
            mat = alpha.new_zeros(3, bs, max_path_len).fill_(logzero)
            mat[0, :, :] = alpha[t]
            mat[1, :, 1:] = alpha[t, :, :-1]
            mat[2, :, 2:] = alpha[t, :, :-2]
            mat[2, :, 2:][same_transition] = logzero   # blank and same label have only two previous nodes
            if sample_topk > 1 and t in t_sample[0]:
                topk_prob, topk_indices = torch.topk(mat, 2, dim=0)
                max_prob = topk_prob[1]
                max_prob[:,0] = topk_prob[0,:,0]   # the first position has only one prefix
                max_indices = topk_indices[1]
                max_indices[:,0] = topk_indices[0,:,0]
            else:
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
            for b in range(bs):
                i = 0
                while i < 3:
                    i += 1
                    orig_pos = torch.nonzero(aligned_seq_shift[b])
                    sample_shift = torch.randint(-sample_dist, sample_dist+1, orig_pos.size()).type_as(aligned_seq_shift)
                    new_pos = orig_pos + sample_shift
                    if new_pos.size(0) > 1:
                        new_pos[-2] = min(new_pos[-2], xmax-1)  # useful when sample_dist = 2
                    new_pos[-1] = min(new_pos[-1], xmax-1)
                    new_aligned_seq = aligned_seq_shift[b].clone()
                    new_aligned_seq[orig_pos] = blank
                    new_aligned_seq[new_pos] = 1
                    if torch.sum(new_aligned_seq) == ylens[b]:
                        aligned_seq_shift[b] = new_aligned_seq
                        break
        return aligned_seq_shift, ylens, ymax

    def align_to_mask(self, aligned_seq_shift, ylens, ymax, src_mask, src_size, blank):
        # 6. transcribe aliged_seq to trigger mask
        trigger_mask = (aligned_seq_shift != blank).cumsum(1).unsqueeze(1).repeat(1, ymax+1, 1)
        trigger_mask = trigger_mask == torch.arange(ymax+1).type_as(trigger_mask).unsqueeze(0).unsqueeze(2)
        trigger_mask.masked_fill_(src_mask==0, 0)   # remove position with padding_idx
        trigger_mask[torch.arange(trigger_mask.size(0)).cuda(), ylens, src_size-1] = 1 # give the last character one to keep at least one active position for eos
        
        ylen = ylens + 1
        ymax += 1
        return trigger_mask, ylen, ymax

    def best_path_align(self, ctc_out, src_mask, src_size, blank, sample_dist=0, sample_num=0):
        "This is used for decoding, forced alignment is needed for training"
        bs, xmax, _ = ctc_out.size()
        if sample_dist == 0 and sample_num > 1:
            mask = (ctc_out.max(-1)[0].exp() < 0.9).unsqueeze(-1)
            topk = ctc_out.topk(2, -1)[1]
            select = torch.randint(0, 2, (topk.size(0), topk.size(1), 1)).type_as(topk).masked_fill(mask==0, 0)
            select.index_fill_(0, torch.arange(0, bs, sample_num).type_as(select), 0)
            best_paths = topk.gather(-1, select).squeeze(-1)
        else:
            best_paths = ctc_out.argmax(-1)
        
        best_paths = best_paths.masked_fill(src_mask.squeeze(1)==0, 0)
        aligned_seq_shift = best_paths.new_zeros(best_paths.size())
        aligned_seq_shift[:, 1:] = best_paths[:,:-1]
        dup = best_paths == aligned_seq_shift
        best_paths.masked_fill_(dup, 0)
        aligned_seq_shift[:,1:] = best_paths[:,:-1]
        ylen = torch.sum((aligned_seq_shift != blank), 1)
        ymax = torch.max(ylen).item()

        if sample_dist > 0:
            for b in range(bs // sample_num):
                i = 1
                while i < sample_num:
                    idx = b * sample_num + i
                    orig_pos = torch.nonzero(aligned_seq_shift[idx,:])
                    sample_shift = torch.randint(-sample_dist, sample_dist+1, orig_pos.size()).type_as(aligned_seq_shift)
                    new_pos = orig_pos + sample_shift
                    new_pos[-1] = min(new_pos[-1], xmax-1)
                    new_aligned_seq = aligned_seq_shift[idx,:].clone()
                    new_aligned_seq[orig_pos] = blank
                    new_aligned_seq[new_pos] = 1
                    if torch.sum(new_aligned_seq) == ylen[idx]:
                        aligned_seq_shift[idx,:] = new_aligned_seq
                        i += 1
        return aligned_seq_shift, ylen, ymax

    def beam_path_align(self, ctc_out, src_mask, src_size, blank, ctc_top_seqs, sample_num):
        # Obatain a better alignment and then trigger mask with languag model
        # This is similar with ctc beam search, but without merging the paths with same output
        bs = int(ctc_out.size(0) /  sample_num)
        tgt_label, ylen = [], []
        for b in range(bs):
            for i in range(sample_num):
                tgt_label.append([])
                ylen.append(0)
                for idx in ctc_top_seqs[b][i]['hyp']:
                    tgt_label[-1].append(idx)
                    ylen[-1] += 1

        ymax = max(ylen)
        tgt = src_size.new_zeros(bs*sample_num, ymax).long()
        for b in range(bs*sample_num):
            tgt[b].narrow(0, 0, ylen[b]).copy_(torch.Tensor(tgt_label[b]).type_as(src_size))
        ylen = torch.Tensor(ylen).type_as(src_size)
        
        if ymax == 0:
            aligned_seq_shift = ctc_out.new_zeros(ctc_out.size(0), ctc_out.size(1)).long()
        else:
            aligned_seq_shift, ylen, ymax = self.viterbi_align(ctc_out, src_mask, src_size, tgt, ylen, blank, 0, 0)
        return aligned_seq_shift, ylen, ymax

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)
    
    def beam_decode(self, src, x_mask, src_size, vocab, args, lm_model=None, ctc_top_seqs=None, labels=None, label_sizes=None):
        """att decoding with rnnlm and ctc out probability

        args.rnnlm: path of rnnlm model
        args.ctc_weight: use ctc out probability for joint decoding when >0.
        """
        bs = src.size(0)
        sos = vocab.word2index['sos']
        eos = vocab.word2index['eos']
        blank = vocab.word2index['blank']

        x, src_mask = self.src_embed(src, x_mask)
        enc_h = self.encoder(x, src_mask)
        ctc_out = self.ctc_generator(enc_h)

        if args.use_trigger:
            src_size = (src_size * ctc_out.size(1)).long()
            #used to include oracle path in sampe path
            if args.test_hitrate:
                aligned_seq_shift1, ylen1, ymax1 = self.viterbi_align(ctc_out, src_mask, src_size, labels[:,1:-1], label_sizes, blank, args.sample_dist, 0)

            if args.sample_num > 1:
                ctc_out = ctc_out.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, ctc_out.size(1), ctc_out.size(2))
                enc_h = enc_h.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, enc_h.size(1), enc_h.size(2))
                src_mask = src_mask.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, src_mask.size(1), src_mask.size(2))
                src_size = src_size.unsqueeze(1).repeat(1, args.sample_num).reshape(-1)
            
            if args.decode_type == 'ctc_att':
                aligned_seq_shift, ylen, ymax = self.beam_path_align(ctc_out, src_mask, src_size, blank, ctc_top_seqs, args.sample_num)
            elif args.decode_type == 'oracle_att':
                aligned_seq_shift, ylen, ymax = self.viterbi_align(ctc_out, src_mask, src_size, labels[:,1:-1], label_sizes, blank, args.sample_dist, 0)
            else:             
                aligned_seq_shift, ylen, ymax = self.best_path_align(ctc_out, src_mask, src_size, blank, args.sample_dist, args.sample_num)
            
            if args.test_hitrate and args.sample_num < 2:
                args.total += (aligned_seq_shift1 != 0).sum().item()
                aligned_seq_shift = aligned_seq_shift.masked_fill((aligned_seq_shift != 0), 1)
                aligned_seq_shift1 = aligned_seq_shift1.masked_fill((aligned_seq_shift1 != 0), 1)
                mask = (aligned_seq_shift == 0) & (aligned_seq_shift1 == 0)
                args.num_correct += (aligned_seq_shift == aligned_seq_shift1).masked_fill(mask, 0).sum(-1).item()
                args.length_total += 1
                num = 1 if ((aligned_seq_shift != 0).sum().item() == (aligned_seq_shift1 != 0).sum().item()) else 0
                args.length_correct += num
                #args.err = (aligned_seq_shift != 0).sum().item() - (aligned_seq_shift1 != 0).sum().item()

            trigger_mask, ylen, ymax = self.align_to_mask(aligned_seq_shift, ylen, ymax, src_mask, src_size, blank)
                 
            if args.right_trigger > 0:
                trigger_shift_right = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_right[:, :, 1:] = trigger_mask[:,:, :-1]
                trigger_mask = trigger_mask | trigger_shift_right
            if args.left_trigger > 0:
                trigger_shift_left = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_left[:,:,:-1] = trigger_mask[:,:,1:]
                trigger_mask = trigger_mask | trigger_shift_left
            trigger_mask = trigger_mask & src_mask
        else:
            trigger_mask = src_mask
            src_size = (src_size * ctc_out.size(1)).long()
            _, ylen, ymax = self.best_path_align(ctc_out, src_mask, src_size, blank)
            ymax = ylen.max().item()

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
        
        if args.sample_num > 1:
            _, seql, dim = att_out.size()
            att_pred = att_out.argmax(-1)
            lm_input = torch.cat([att_out.new_zeros(att_out.size(0), 1).fill_(sos).long(), att_pred[:,:-1]], 1)
            lm_tgt_mask = tgt_mask1 & self.subsequent_mask(ymax).type_as(tgt_mask1)
            lm_out = lm_model(lm_input, lm_tgt_mask)
            # this part use ast baseline to do the score part
            #x, src_mask = lm_model.src_embed(src, x_mask)
            #enc_h = lm_model.encoder(x, src_mask)
            #enc_h = enc_h.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, enc_h.size(1), enc_h.size(2))
            #src_mask = src_mask.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, src_mask.size(1), src_mask.size(2))
            #lm_out = lm_model.forward_att(enc_h, lm_input, src_mask, lm_tgt_mask)
            
            lm_score = torch.gather(lm_out, -1, att_pred.unsqueeze(-1)).squeeze(-1)
            lm_score = lm_score.reshape(-1, args.sample_num, seql).masked_fill(tgt_mask.reshape(-1, args.sample_num, tgt_mask.size(-1))==0, 0)
            att_out = att_out.reshape(-1, args.sample_num, seql, dim).masked_fill(tgt_mask.reshape(-1, args.sample_num, tgt_mask.size(-2), tgt_mask.size(-1)).transpose(2,3)==0, 0)
            prob_sum = lm_score.sum(-1) / (lm_score != 0).sum(-1).float()
            max_indices = prob_sum.max(-1, keepdim=True)[1]
            att_out = torch.gather(att_out, 1, max_indices.unsqueeze(2).unsqueeze(3).repeat(1,1,seql,dim)).squeeze(1)
            bs = att_out.size(0)
            ylen = ylen.reshape(bs, args.sample_num).gather(1, max_indices)
            ymax = torch.max(ylen).item()

            if args.test_hitrate:
                args.total += (aligned_seq_shift1 != 0).sum(-1).item()
                aligned_seq_shift = aligned_seq_shift.gather(0, max_indices.repeat(1, aligned_seq_shift.size(-1)))
                aligned_seq_shift = aligned_seq_shift.masked_fill((aligned_seq_shift != 0), 1)
                aligned_seq_shift1 = aligned_seq_shift1.masked_fill((aligned_seq_shift1 != 0), 1)
                mask = (aligned_seq_shift == 0) & (aligned_seq_shift1 == 0)
                args.num_correct += (aligned_seq_shift == aligned_seq_shift1).masked_fill(mask, 0).sum(-1).item()
                args.length_total += 1
                num = 1 if ((aligned_seq_shift != 0).sum().item() == (aligned_seq_shift1 != 0).sum().item()) else 0
                args.length_correct += num
                #args.err = (aligned_seq_shift != 0).sum().item() - (aligned_seq_shift1 != 0).sum().item()
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
        return batch_top_seqs, args

    def beam_decode_adapt_num(self, src, x_mask, src_size, vocab, args, lm_model=None, ctc_top_seqs=None, labels=None, label_sizes=None):
        """
        Use batch size 1 to do adpative path samping, not useful.
        """
        sos = vocab.word2index['sos']
        eos = vocab.word2index['eos']
        blank = vocab.word2index['blank']

        x, src_mask = self.src_embed(src, x_mask)
        enc_h = self.encoder(x, src_mask)
        ctc_out = self.ctc_generator(enc_h)
        bs, xmax, _ = ctc_out.size()
        src_size = (src_size * xmax).long()
        mask = (ctc_out.max(-1)[0].exp() < 0.9).unsqueeze(-1)
        topk = ctc_out.topk(2, -1)[1]
        sample_num = min(args.sample_num, 2 ** mask.sum().item())
        
        topk = topk.repeat(sample_num, 1, 1)
        mask = mask.repeat(sample_num, 1, 1)
        enc_h = enc_h.repeat(sample_num, 1, 1)
        src_mask = src_mask.repeat(sample_num, 1, 1)
        src_size = src_size.repeat(sample_num).reshape(-1)

        select = torch.randint(0, 2, (topk.size(0), topk.size(1), 1)).type_as(topk).masked_fill(mask==0, 0)
        select.index_fill_(0, torch.arange(0, bs, sample_num).type_as(select), 0)
        best_paths = topk.gather(-1, select).squeeze(-1)
        
        best_paths = best_paths.masked_fill(src_mask.squeeze(1)==0, 0)
        aligned_seq_shift = best_paths.new_zeros(best_paths.size())
        aligned_seq_shift[:, 1:] = best_paths[:,:-1]
        dup = best_paths == aligned_seq_shift
        best_paths.masked_fill_(dup, 0)
        aligned_seq_shift[:,1:] = best_paths[:,:-1]
        ylen = torch.sum((aligned_seq_shift != blank), 1)
        ymax = torch.max(ylen).item()

        trigger_mask = (aligned_seq_shift != blank).cumsum(1).unsqueeze(1).repeat(1, ymax+1, 1)
        trigger_mask = trigger_mask == torch.arange(ymax+1).type_as(trigger_mask).unsqueeze(0).unsqueeze(2)
        trigger_mask[:,-1:,:].masked_fill_(src_mask==0, 0)
        trigger_mask[:,-1,:].scatter_(1, src_size.unsqueeze(1)-1, 1)
        
        ylen = ylen + 1
        ymax += 1
        trigger_mask = trigger_mask & src_mask

        bs, _, d_model = enc_h.size()
        tgt_mask1 = torch.full((bs, ymax), 1).type_as(src_mask)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        
        # 3. Extract Acoustic embedding and Map it to Word embedding
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        ac_embed = self.acembed_extractor(pe, enc_h, trigger_mask)
        pred_embed = self.embed_mapper(ac_embed, tgt_mask1)
        tgt_mask = tgt_mask1

        if args.use_src:
            if args.src_trigger:
                src_mask = trigger_mask
            dec_h = self.decoder(pred_embed, enc_h, src_mask, tgt_mask)
        else:
            dec_h = self.decoder(pred_embed, tgt_mask)
        att_out = self.att_generator(dec_h)
        
        _, seql, dim = att_out.size()
        att_pred = att_out.argmax(-1)
        lm_input = torch.cat([att_out.new_zeros(att_out.size(0), 1).fill_(sos).long(), att_pred[:,:-1]], 1)
        lm_tgt_mask = tgt_mask1 & self.subsequent_mask(ymax).type_as(tgt_mask1)
        lm_out = lm_model(lm_input, lm_tgt_mask)
        lm_score = torch.gather(lm_out, -1, att_pred.unsqueeze(-1)).squeeze(-1)
        lm_score = lm_score.masked_fill(tgt_mask.squeeze(1)==0, 0)
        att_out = att_out.masked_fill(tgt_mask.transpose(1,2)==0, 0)
        prob_sum = lm_score.sum(-1) / (lm_score != 0).sum(-1).float()
        max_indices = prob_sum.max(-1, keepdim=True)[1]
        att_out = torch.gather(att_out, 0, max_indices.unsqueeze(1).unsqueeze(2).repeat(1,seql,dim))
        bs = att_out.size(0)
        ylen = ylen.gather(0, max_indices)
        
        ys = torch.ones(1, 1).fill_(sos).long()
        if args.use_gpu:
            ys = ys.cuda()
        
        batch_top_seqs = [{'ys': ys, 'score': 0.0, 'hyp': [sos] } ]
        
        for i in range(ymax):
            # batchify the batch and beam
            all_seqs, ys, att_prob = [], [], []
            
            for seq in batch_top_seqs:
                if i > ylen[0].item():
                    all_seqs.append(seq)
                    continue
        
                att_prob.append(att_out[0,i:i+1,:])
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
            for seq in batch_top_seqs:
                if i > ylen[0].item():
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
                    all_seqs.append(rs_seq)

            sort_f = lambda x:x['score'] + (len(x['hyp'])-1) * args.length_penalty \
                        if args.length_penalty is not None else lambda x:x['score']                
            batch_top_seqs = sorted(all_seqs, key=sort_f, reverse=True)[:args.beam_width]

        return [batch_top_seqs]

