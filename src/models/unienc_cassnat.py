
# 2023 Ruchao Fan

import copy
import math
import editdistance as ed
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding, TextEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks import TrfEncoder, ConEncoder, TrfAcExtra
from models.blocks.hubert_blocks import HubertModel
from utils.ctc_prefix import logzero, logone, CTCPrefixScore
from utils.loss import LabelSmoothing, KLDivLoss
from models.modules.ssl_util import compute_mask_indices

def make_model(input_size, args):
    c = copy.deepcopy

    if args.model_type != "hubert":
        if args.use_conv_enc:
            assert args.pos_type == "relative", "conformer must use relative positional encoding"
            enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.enc_max_relative_len)     
            enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
            enc_conv_module = ConvModule(args.d_model, args.enc_kernel_size, activation=Swish())    
            enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())
            encoder = ConEncoder(args.d_model, c(enc_ff), enc_attn, enc_conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff)
        else:
            assert args.model_type == "transformer"
            enc_attn = MultiHeadedAttention(args.n_head, args.d_model)
            enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout)
            enc_position = PositionalEncoding(args.d_model, args.dropout)
            encoder = TrfEncoder(args.d_model, enc_attn, enc_ff, args.dropout, args.N_enc)
        src_embed = ConvEmbedding(input_size, args.d_model, args.dropout, enc_position)
        projection_layer, back_projection_layer, inter_projection_layer = None, None, None
    else:
        src_embed = None
        encoder = HubertModel(args)
        projection_layer = nn.Linear(args.encoder_embed_dim, args.d_model)
        back_projection_layer = nn.Linear(args.d_model, args.encoder_embed_dim)
        inter_projection_layer = nn.Linear(args.encoder_embed_dim, args.d_model) if args.interctc_alpha > 0 else None

    dec_src_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
    dec_ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout, activation=Swish())
    Extra = TrfAcExtra(args.d_model, c(dec_src_attn), c(dec_ff), args.dropout, args.N_extra)
    
    generator = Generator(args.d_model, args.vocab_size)
    interctc_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interctc_alpha > 0 else None
    interce_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interce_alpha > 0 else None
    mlm_gen = Generator(args.d_model, args.vocab_size) if args.use_mlm else None
    pe = create_pe(args.d_model)

    model = UECassNAT(src_embed, encoder, projection_layer, back_projection_layer, inter_projection_layer,
                        Extra, c(generator), c(generator), pe, interctc_gen, interce_gen, mlm_gen ,args)

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

class UECassNAT(nn.Module):
    def __init__(self, src_embed, encoder, proj_layer, back_proj_layer, inter_proj_layer, acembed_extractor, ctc_gen, att_gen, pe, interctc_gen, interce_gen, mlm_gen, args):
        super(UECassNAT, self).__init__()
        self.model_type = args.model_type
        self.src_embed = src_embed
        self.encoder = encoder
        self.projection_layer = proj_layer
        self.acembed_extractor = acembed_extractor
        self.back_projection_layer = back_proj_layer
        self.ctc_generator = ctc_gen
        self.att_generator = att_gen
        self.pe = pe
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
        self.att_loss = LabelSmoothing(args.vocab_size, args.text_padding_idx, args.label_smooth)
        if args.apply_mask and args.use_mask_embed:
            self.mask_embed = nn.Parameter(torch.FloatTensor(args.d_model).uniform_())
        
        if args.use_mlm:    
            self.mlm_loss = LabelSmoothing(args.vocab_size, args.text_padding_idx, 0)
            self.mlm_generator = mlm_gen

        if interctc_gen is not None:
            self.interctc_generator = interctc_gen
            self.inter_projection_layer = inter_proj_layer
            self.interctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
        if interce_gen is not None:
            self.interce_generator = interce_gen
            self.interce_loss = LabelSmoothing(args.vocab_size, args.text_padding_idx, args.label_smooth)

    def forward(self, src, src_mask, src_size, tgt_label, label_sizes, args):
        # 1. compute ctc output
        if args.model_type != "hubert":
            conv_fea, fea_padding_mask = self.src_embed(src, src_mask)
            enc_h = self.encoder(conv_fea, fea_padding_mask, args.interctc_alpha, args.interctc_layer)
            inter_h = enc_h[1] if args.interctc_alpha > 0 else None
            enc_h = enc_h[0] if args.interctc_alpha > 0 else None
            conv_fea = conv_fea[0] if args.model_type == "conformer" else conv_fea
            x_mask = fea_padding_mask
            fea_padding_mask = fea_padding_mask.squeeze(1)
            src_size = (src_size * enc_h.size(1)).long()
        else:
            conv_fea, fea_padding_mask = self.encoder.forward_features(src, padding_mask=src_mask, mask_prob=args.mask_prob)
            enc_h, x_mask, layer_results = self.encoder.forward_encoder(conv_fea, fea_padding_mask)
            enc_h = self.projection_layer(enc_h)
            x_mask = ~x_mask.unsqueeze(1)
            src_size = (x_mask != 0).squeeze(1).sum(-1)
            if args.interctc_alpha > 0:
                inter_h = layer_results[args.interctc_layer-1][0].transpose(0,1)
                inter_h = self.inter_projection_layer(inter_h)
            else:
                inter_h = None

        loss = 0
        if args.ctc_alpha > 0:
            ctc_out = self.ctc_generator(enc_h)
            ctc_loss = self.ctc_loss(ctc_out.transpose(0,1), tgt_label, src_size, label_sizes)
            loss += args.ctc_alpha * ctc_loss
        else:
            ctc_loss = torch.Tensor([0])

        if args.interctc_alpha > 0:
            interctc_out = self.interctc_generator(inter_h)
            interctc_loss = self.interctc_loss(interctc_out.transpose(0,1), tgt_label, src_size, label_sizes)
            loss += args.interctc_alpha * interctc_loss
        else:
            interctc_loss = torch.Tensor([0])

        # 2. prepare different masks,
        ylen = label_sizes
        if args.use_trigger:
            assert args.ctc_alpha > 0
            blank = 0
            aligned_seq_shift, ylen, ymax = self.viterbi_align(ctc_out, x_mask, src_size, tgt_label[:,:-1], ylen, blank, args.sample_topk)
            trigger_mask, ylen, ymax = self.align_to_mask(aligned_seq_shift, ylen, ymax, x_mask, src_size, blank)
            trigger_mask = self.expand_trigger_mask(trigger_mask, args.left_trigger, args.right_trigger)
            trigger_mask = trigger_mask & x_mask
        else:
            trigger_mask = x_mask
            ylen = ylen + 1
            ymax = ylen.max().item()

        # 3. Extract Acoustic embedding and Map it to Word embedding
        bs, _, d_model = enc_h.size()
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        ac_embed = self.acembed_extractor(pe, enc_h, trigger_mask)

        tgt_mask_bidi = torch.full((bs, ymax), 1).type_as(x_mask)
        tgt_mask_bidi = tgt_mask_bidi.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask_bidi = tgt_mask_bidi.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        ae_padding_mask = (~tgt_mask_bidi.squeeze(1).bool())

        # mask ac_embed and use mlm loss
        if args.apply_mask:
            B, T, _ = ac_embed.size()
            mask_indices = compute_mask_indices((B, T), ae_padding_mask, args.mask_prob, args.mask_length, 
                                args.mask_selection, args.mask_other, min_masks=args.min_masks, no_overlap=args.no_mask_overlap, 
                                min_space=args.mask_min_space, require_same_masks=args.require_same_masks, 
                                mask_dropout=args.mask_dropout)
            mask_indices = torch.from_numpy(mask_indices).to(ac_embed.device)
            if args.use_mask_embed:
                ac_embed[mask_indices] = self.mask_embed        
            else:
                ac_embed[mask_indices] = ac_embed.mean(1, keepdim=True).repeat(1,ac_embed.size(1),1)[mask_indices]

        # 4. decoder, output units generation
        if self.model_type != "hubert":
            ae_padding_mask = ~ae_padding_mask

        if self.model_type == "hubert":
            ac_embed = self.back_projection_layer(ac_embed)

        dec_input = torch.cat([conv_fea, ac_embed], dim=1)
        padding_mask = torch.cat([fea_padding_mask, ae_padding_mask], dim=-1)
        fea_len = fea_padding_mask.size(1)

        if self.model_type != "hubert":
            dec_input = self.src_embed.pos_enc(dec_input)
            dec_h = self.encoder(dec_input, padding_mask.unsqueeze(1), args.interce_alpha, args.interce_layer)
            interce_h = dec_h[1] if args.interce_alpha > 0 else None
            dec_h = dec_h[0] if args.interce_alpha > 0 else dec_h
        else:
            dec_h, dec_mask, dec_layer_results = self.encoder.forward_encoder(dec_input, padding_mask)  
            dec_h = self.projection_layer(dec_h)
            interce_h = dec_layer_results[args.interce_layer-1][0].transpose(0,1)  
            interce_h = self.inter_projection_layer(interce_h)
        
        att_out = self.att_generator(dec_h[:,fea_len:,:])
        att_loss = self.att_loss(att_out.view(-1, att_out.size(-1)), tgt_label.view(-1))
        loss += args.att_alpha * att_loss 

        if args.interce_alpha > 0:
            interce_out = self.interce_generator(interce_h[:,fea_len:,:])
            interce_loss = self.interce_loss(interce_out.view(-1, interce_out.size(-1)), tgt_label.view(-1))
            loss += args.interce_alpha * interce_loss
        else:
            interce_loss = torch.Tensor([0])

        if args.multi_ctc:
            if args.ctc_alpha > 0:
                ctc_out2 = self.ctc_generator(dec_h[:,:fea_len,:])
                ctc_loss2 = self.ctc_loss(ctc_out2.transpose(0,1), tgt_label, src_size, label_sizes)
                loss += args.multi_ctc_alpha * args.ctc_alpha * ctc_loss2
            else:
                ctc_loss2 = torch.Tensor([0])

            if args.interctc_alpha > 0:
                interctc_out2 = self.interctc_generator(interce_h[:,:fea_len,:])
                interctc_loss2 = self.interctc_loss(interctc_out2.transpose(0,1), tgt_label, src_size, label_sizes)
                loss += args.multi_ctc_alpha * args.interctc_alpha * interctc_loss2
            else:
                interctc_loss2 = torch.Tensor([0])
        else:
            ctc_out2 = None
            ctc_loss2 = torch.Tensor([0])
            interctc_loss2 = torch.Tensor([0])
    
        return ctc_out, ctc_out2, att_out, loss, ctc_loss, ctc_loss2, att_loss, src_size

    def expand_trigger_mask(self, trigger_mask, left_trigger, right_trigger):
        if isinstance(right_trigger, int):
            if right_trigger > 0:
                trigger_shift_right = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_right[:, :, 1:] = trigger_mask[:,:, :-1]
                trigger_mask = trigger_mask | trigger_shift_right
            if left_trigger > 0:
                trigger_shift_left = trigger_mask.new_zeros(trigger_mask.size())
                trigger_shift_left[:,:,:-1] = trigger_mask[:,:,1:]
                trigger_mask = trigger_mask | trigger_shift_left
        if isinstance(right_trigger, str):
            trigger_shift_right = torch.cat([trigger_mask[:,1:,:], trigger_mask[:,-1:,:]], dim=1)        
            trigger_shift_left = torch.cat([trigger_mask[:,:1,:], trigger_mask[:,:-1,:]], dim=1)
            trigger_mask = trigger_mask | trigger_shift_right | trigger_shift_left

        return trigger_mask

    def viterbi_align(self, ctc_out, src_mask, src_size, ys, ylens, blank, sample_topk):
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

    def best_path_align(self, ctc_out, src_mask, src_size, blank, sample_num=0, threshold=0.9, include_best=True):
        "This is used for decoding, forced alignment is needed for training"
        bs, xmax, _ = ctc_out.size()
        if sample_num > 1:
            mask = (ctc_out.max(-1)[0].exp() < threshold).unsqueeze(-1)
            topk = ctc_out.topk(2, -1)[1]
            select = torch.randint(0, 2, (topk.size(0), topk.size(1), 1)).type_as(topk).masked_fill(mask==0, 0)
            if include_best:
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
    
    def beam_decode(self, src, x_mask, src_size, tokenizer, args, lm_model=None, ctc_top_seqs=None, labels=None, label_sizes=None):
        """att decoding with rnnlm and ctc out probability

        args.rnnlm: path of rnnlm model
        args.ctc_weight: use ctc out probability for joint decoding when >0.
        """
        bs = src.size(0)
        sos = tokenizer.vocab['sos']
        eos = tokenizer.vocab['eos']
        blank = tokenizer.vocab['blank']

        if args.model_type != "hubert":
            conv_fea, fea_padding_mask = self.src_embed(src, x_mask)
            enc_h = self.encoder(conv_fea, fea_padding_mask)
            conv_fea = conv_fea[0] if args.model_type == "conformer" else conv_fea
            src_mask = fea_padding_mask
            src_size = (src_size * enc_h.size(1)).long()
        else:
            conv_fea, fea_padding_mask = self.encoder.forward_features(src, padding_mask=x_mask, mask_prob=0)
            enc_h, src_mask, layer_results = self.encoder.forward_encoder(conv_fea, fea_padding_mask)
            enc_h = self.projection_layer(enc_h)
            src_mask = ~src_mask.unsqueeze(1)
            src_size = (src_mask != 0).squeeze(1).sum(-1)

        ctc_out = self.ctc_generator(enc_h)

        if args.use_trigger:
            #used to include oracle path in sampe path
            if args.test_hitrate:
                aligned_seq_shift1, ylen1, ymax1 = self.viterbi_align(ctc_out, src_mask, src_size, labels[:,1:-1], label_sizes, blank, 0)

            if args.sample_num > 1:
                conv_fea = conv_fea.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, conv_fea.size(1), conv_fea.size(2))
                ctc_out = ctc_out.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, ctc_out.size(1), ctc_out.size(2))
                enc_h = enc_h.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, enc_h.size(1), enc_h.size(2))
                src_mask = src_mask.unsqueeze(1).repeat(1, args.sample_num, 1, 1).reshape(-1, src_mask.size(1), src_mask.size(2))
                src_size = src_size.unsqueeze(1).repeat(1, args.sample_num).reshape(-1)
            
            if args.decode_type == 'ctc_att':
                aligned_seq_shift, ylen, ymax = self.beam_path_align(ctc_out, src_mask, src_size, blank, ctc_top_seqs, args.sample_num)
            elif args.decode_type == 'oracle_att':
                aligned_seq_shift, ylen, ymax = self.viterbi_align(ctc_out, src_mask, src_size, labels[:,1:-1], label_sizes, blank, 0)
            else:             
                aligned_seq_shift, ylen, ymax = self.best_path_align(ctc_out, src_mask, src_size, blank, args.sample_num, args.threshold)
            
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
            
            trigger_mask = self.expand_trigger_mask(trigger_mask, args.left_trigger, args.right_trigger)
            trigger_mask = trigger_mask & src_mask
        else:
            trigger_mask = src_mask
            src_size = (src_size * ctc_out.size(1)).long()
            _, ylen, ymax = self.best_path_align(ctc_out, src_mask, src_size, blank)
            ymax = ylen.max().item()

        bs, _, d_model = enc_h.size()
        pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
        ac_embed = self.acembed_extractor(pe, enc_h, trigger_mask)

        tgt_mask1 = torch.full((bs, ymax), 1).type_as(src_mask)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
        tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
        ae_padding_mask = (~tgt_mask1.squeeze(1).bool())

        # 4. decoder, output units generation
        if self.model_type == "hubert":
            ac_embed = self.back_projection_layer(ac_embed)

        dec_input = torch.cat([conv_fea, ac_embed], dim=1)
        padding_mask = torch.cat([~src_mask.squeeze(1), ae_padding_mask], dim=-1)
        fea_len = src_mask.size(-1)

        if self.model_type != "hubert":
            dec_input = self.src_embed.pos_enc(dec_input)
            dec_h = self.encoder(dec_input, ~padding_mask.unsqueeze(1))
        else:
            dec_h, dec_mask, dec_layer_results = self.encoder.forward_encoder(dec_input, padding_mask)  
            dec_h = self.projection_layer(dec_h)

        torch.cuda.empty_cache()
        for i in range(args.num_iters - 1):
            ctc_out_new = self.ctc_generator(dec_h[:,:fea_len,:])
            if args.sample_num2 > 1:
                conv_fea = conv_fea.unsqueeze(1).repeat(1, args.sample_num2, 1, 1).reshape(-1, conv_fea.size(1), conv_fea.size(2))
                ctc_out_new = ctc_out_new.unsqueeze(1).repeat(1, args.sample_num2, 1, 1).reshape(-1, ctc_out_new.size(1), ctc_out_new.size(2))
                enc_h = dec_h[:,:fea_len,:]
                enc_h = enc_h.unsqueeze(1).repeat(1, args.sample_num2, 1, 1).reshape(-1, enc_h.size(1), enc_h.size(2))
                src_mask = src_mask.unsqueeze(1).repeat(1, args.sample_num2, 1, 1).reshape(-1, src_mask.size(1), src_mask.size(2))
                src_size = src_size.unsqueeze(1).repeat(1, args.sample_num2).reshape(-1)
            else:
                enc_h = dec_h[:,:fea_len,:]
            
            aligned_seq_shift, ylen, ymax = self.best_path_align(ctc_out_new, src_mask, src_size, blank, args.sample_num2, args.threshold)
            trigger_mask, ylen, ymax = self.align_to_mask(aligned_seq_shift, ylen, ymax, src_mask, src_size, blank)
            trigger_mask = self.expand_trigger_mask(trigger_mask, args.left_trigger, args.right_trigger)
            trigger_mask = trigger_mask & src_mask

            bs, _, d_model = enc_h.size()
            pe = self.pe.type_as(src).unsqueeze(0).repeat(bs, 1, 1)[:,:ymax,:]
            ac_embed = self.acembed_extractor(pe, enc_h, trigger_mask)

            tgt_mask1 = torch.full((bs, ymax), 1).type_as(src_mask)
            tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 0).cumprod(1)
            tgt_mask1 = tgt_mask1.scatter(1, ylen.unsqueeze(1)-1, 1).unsqueeze(1)
            ae_padding_mask = (~tgt_mask1.squeeze(1).bool())

            if self.model_type == "hubert":
                ac_embed = self.back_projection_layer(ac_embed)

            dec_input = torch.cat([conv_fea, ac_embed], dim=1)
            padding_mask = torch.cat([~src_mask.squeeze(1), ae_padding_mask], dim=-1)
            fea_len = src_mask.size(-1)

            if self.model_type != "hubert":
                dec_input = self.src_embed.pos_enc(dec_input)
                dec_h = self.encoder(dec_input, ~padding_mask.unsqueeze(1))
            else:
                dec_h, dec_mask, dec_layer_results = self.encoder.forward_encoder(dec_input, padding_mask) 
                dec_h = self.projection_layer(dec_h)
            torch.cuda.empty_cache()

        att_out = self.att_generator(dec_h[:,fea_len:,:])

        # LM ranking
        torch.cuda.empty_cache()
        sample_num = args.sample_num * args.sample_num2 ** (args.num_iters - 1)
        if sample_num > 1:
            _, seql, dim = att_out.size() 
            att_pred = att_out.argmax(-1)
            if args.rank_model != 'n-gram':
                lm_input = torch.cat([att_out.new_zeros(att_out.size(0), 1).fill_(sos).long(), att_pred[:,:-1]], 1)
                lm_tgt_mask = tgt_mask1 & self.subsequent_mask(ymax).type_as(tgt_mask1)
            
                if args.rank_model == 'lm':
                    # this part used for lm rescore
                    lm_out = lm_model(lm_input, lm_tgt_mask)
                if args.rank_model == 'at_baseline':
                    # this part use ast baseline to do the score part
                    if self.model_type != "hubert":
                        x, src_mask = lm_model.src_embed(src, x_mask)
                        enc_h = lm_model.encoder(x, src_mask)
                        enc_h = enc_h.unsqueeze(1).repeat(1, sample_num, 1, 1).reshape(-1, enc_h.size(1), enc_h.size(2))
                        src_mask = src_mask.unsqueeze(1).repeat(1, sample_num, 1, 1).reshape(-1, src_mask.size(1), src_mask.size(2))
                        lm_out = lm_model.forward_decoder(enc_h, lm_input, src_mask, lm_tgt_mask)
                    else:
                        enc_h, src2_mask, layer_results = lm_model.encoder(src, padding_mask=x_mask, mask_prob=args.mask_prob)
                        enc_h = lm_model.projection_layer(enc_h)
                        src2_mask = src2_mask.unsqueeze(1)
                        src2_mask = ~src2_mask
                        enc_h = enc_h.unsqueeze(1).repeat(1, sample_num, 1, 1).reshape(-1, enc_h.size(1), enc_h.size(2))
                        src2_mask = src2_mask.unsqueeze(1).repeat(1, sample_num, 1, 1).reshape(-1, src2_mask.size(1), src2_mask.size(2))
                        lm_out = lm_model.forward_decoder(enc_h, lm_input, src2_mask, lm_tgt_mask)

                lm_score = torch.gather(lm_out, -1, att_pred.unsqueeze(-1)).squeeze(-1)
                lm_score = lm_score.reshape(-1, sample_num, seql).masked_fill(tgt_mask1.reshape(-1, sample_num, tgt_mask1.size(-1))==0, 0)
                prob_sum = lm_score.sum(-1) / (lm_score != 0).sum(-1).float()
                max_indices = prob_sum.max(-1, keepdim=True)[1]
            elif args.rank_model == 'n-gram':
                prob_sum = att_pred.new_zeros(att_pred.size(0)).float()
                tgt_len = tgt_mask1.sum(-1)
                for i in range(att_pred.size(0)):
                    sentence = []
                    for j in range(tgt_len[i]):
                        index = att_pred[i][j].item()
                        if index != 2:
                            sentence.append(tokenizer.ids_to_tokens[index])
                    score = lm_model.score(''.join(sentence).replace('▁', ' ').strip())
                    prob_sum[i] = score / tgt_len[i]
                max_indices = prob_sum.reshape(-1, args.sample_num).max(-1, keepdim=True)[1]
            else:
                sample_num = args.sample_num
            
            att_out = att_out.reshape(-1, sample_num, seql, dim).masked_fill(tgt_mask1.reshape(-1, sample_num, tgt_mask1.size(-2), tgt_mask1.size(-1)).transpose(2,3)==0, 0)
            att_out = torch.gather(att_out, 1, max_indices.unsqueeze(2).unsqueeze(3).repeat(1,1,seql,dim)).squeeze(1)
            if args.save_embedding:
                ac_embed = ac_embed[0].reshape(-1, sample_num, ac_embed[0].size(-2), ac_embed[0].size(-1))
                ac_embed = torch.gather(ac_embed, 1, max_indices.unsqueeze(2).unsqueeze(3).repeat(1,1,ac_embed.size(-2),ac_embed.size(-1))).squeeze(1)
                args.ac_embed = ac_embed.cpu()
                
            bs = att_out.size(0)
            ylen = ylen.reshape(bs, sample_num).gather(1, max_indices)
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
            
        torch.cuda.empty_cache()
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

