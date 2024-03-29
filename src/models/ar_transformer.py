
# 2020-2023 Ruchao Fan
# Some transformer-related codes are borrowed from 
# https://nlp.seas.harvard.edu/2018/04/03/attention.html

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding, TextEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks.transformer_blocks import Encoder as TrfEncoder
from models.blocks.conformer_blocks import Encoder as ConEncoder
from models.blocks.conformer_blocks import Decoder
from models.blocks.hubert_blocks import HubertModel
from utils.loss import LabelSmoothing
from utils.ctc_prefix import CTCPrefixScore

def make_model(input_size, args):
    c = copy.deepcopy

    # encoder settings: transformer, conformer and hubert encoder
    if args.model_type == "transformer":
        enc_position = PositionalEncoding(args.d_model, args.dropout, args.max_len)
        src_embed = ConvEmbedding(input_size, args.d_model, args.dropout, enc_position)
        enc_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        enc_ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout, activation=Swish())
        encoder = TrfEncoder(args.d_model, enc_attn, enc_ff, args.dropout, args.N_enc)
        projection_layer, interctc_projection_layer = None, None

    elif args.model_type == "conformer":
        if args.pos_type == "relative":
            enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.enc_max_relative_len)
            enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        elif args.pos_type == "absolute":
            enc_position = PositionalEncoding(args.d_model, args.dropout)
            enc_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)

        src_embed = ConvEmbedding(input_size, args.d_model, args.dropout, enc_position)
        conv_module = ConvModule(args.d_model, args.enc_kernel_size, activation=Swish())
        enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())
        encoder = ConEncoder(args.d_model, c(enc_ff), enc_attn, conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff)
        projection_layer, interctc_projection_layer = None, None

    elif args.model_type == "hubert":
        src_embed = None
        encoder = HubertModel(args)
        projection_layer = nn.Linear(args.encoder_embed_dim, args.d_model)
        interctc_projection_layer = nn.Linear(args.encoder_embed_dim, args.d_model) if args.interctc_alpha > 0 else None

    # decoder settings
    dec_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
    dec_ff = PositionwiseFeedForward(args.d_model, args.d_decff, args.dropout, activation=Swish())
    dec_position = PositionalEncoding(args.d_model, args.dropout, args.max_len)
    decoder = Decoder(args.d_model, c(dec_attn), c(dec_attn), dec_ff, args.dropout, args.N_dec)
    
    generator = Generator(args.d_model, args.vocab_size)
    interctc_gen = Generator(args.d_model, args.vocab_size, add_norm=True) if args.interctc_alpha > 0 else None
    
    model = AR_Transformer(
        src_embed, encoder, projection_layer, interctc_projection_layer, 
        nn.Sequential(TextEmbedding(args.d_model, args.vocab_size), dec_position), 
        decoder, c(generator), c(generator), interctc_gen, args)
        
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

class AR_Transformer(nn.Module):
    def __init__(self, src_embed, encoder, projection_layer, interctc_projection_layer,
                    tgt_embed, decoder, ctc_gen, att_gen, interctc_gen, args):
        super(AR_Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.ctc_generator = ctc_gen
        self.att_generator = att_gen
        self.model_type = args.model_type
        
        self.ctc_alpha = args.ctc_alpha
        self.interctc_alpha = args.interctc_alpha
        self.interctc_layer = args.interctc_layer
        self.ctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
        self.att_loss = LabelSmoothing(args.vocab_size, args.text_padding_idx, args.label_smooth)

        if interctc_gen is not None:
            self.interctc_generator = interctc_gen
            self.interctc_projection_layer = interctc_projection_layer
            self.interctc_loss = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    def forward(self, src, tgt, src_mask, tgt_mask, feat_sizes, label_sizes, tgt_label, mask_prob=0.0):
        if self.model_type != "hubert":
            x, x_mask = self.src_embed(src, src_mask)
            enc_h = self.encoder(x, x_mask, self.interctc_alpha, self.interctc_layer)
            inter_h = enc_h[1] if self.interctc_alpha > 0 else None
            enc_h = enc_h if self.interctc_alpha == 0 else enc_h[0]
            max_feat_size = enc_h.size(1)
            feat_size = (feat_sizes * max_feat_size).long()
        else:
            enc_h, x_mask, layer_results = self.encoder(src, padding_mask=src_mask, mask_prob=mask_prob)
            enc_h = self.projection_layer(enc_h)
            x_mask = x_mask.unsqueeze(1)
            x_mask = ~x_mask
            feat_size = (x_mask != 0).squeeze(1).sum(-1)
            if self.interctc_alpha > 0:
                inter_h = layer_results[self.interctc_layer-1][0]
                inter_h = self.interctc_projection_layer(inter_h.transpose(0,1))

        if self.interctc_alpha > 0:
            inter_out = self.interctc_generator(inter_h)
            interctc_loss = self.interctc_loss(inter_out.transpose(0,1), tgt_label, feat_size, label_sizes)
        else:
            interctc_loss = torch.Tensor([0]) 

        if self.ctc_alpha > 0:
            ctc_out = self.ctc_generator(enc_h)
            ctc_loss = self.ctc_loss(ctc_out.transpose(0,1), tgt_label, feat_size, label_sizes)
        else:
            ctc_loss = torch.Tensor([0])

        dec_h = self.decoder(self.tgt_embed(tgt), enc_h, x_mask, tgt_mask)
        att_out = self.att_generator(dec_h)
        att_loss = self.att_loss(att_out.view(-1, att_out.size(-1)), tgt_label.view(-1))        

        loss = att_loss + self.ctc_alpha * ctc_loss
        if self.interctc_alpha > 0:
            loss += self.interctc_alpha * interctc_loss

        return ctc_out, att_out, loss, att_loss, ctc_loss, feat_size

    def forward_att(self, src, tgt, src_mask, tgt_mask):
        if self.model_type != "hubert":
            x, x_mask = self.src_embed(src, src_mask)
            enc_h = self.encoder(x, x_mask)
        else:
            enc_h, x_mask, layer_results = self.encoder(src, padding_mask=src_mask)
            enc_h = self.projection_layer(enc_h)
            x_mask = x_mask.unsqueeze(1)
            x_mask = ~x_mask

        dec_h = self.decoder(self.tgt_embed(tgt), enc_h, x_mask, tgt_mask)
        att_out = F.softmax(self.att_generator.proj(dec_h), dim=-1)
        return att_out

    def forward_decoder(self, enc_h, tgt, x_mask, tgt_mask):
        dec_h = self.decoder(self.tgt_embed(tgt), enc_h, x_mask, tgt_mask)
        att_out = F.softmax(self.att_generator.proj(dec_h), dim=-1)
        return att_out

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)
        
    def beam_decode(self, src, src_mask, tokenizer, args, lm_model=None):
        """att decoding with rnnlm and ctc out probability

        args.rnnlm: path of rnnlm model
        args.ctc_weight: use ctc out probability for joint decoding when >0.
        """
        bs = src.size(0)
        sos = tokenizer.vocab['sos']
        eos = tokenizer.vocab['eos']
        blank = tokenizer.vocab['blank']

        if self.model_type != "hubert":
            x, src_mask = self.src_embed(src, src_mask)
            enc_h = self.encoder(x, src_mask)
        else:
            enc_h, x_mask, _ = self.encoder(src, padding_mask=src_mask, mask_prob=args.mask_prob)
            enc_h = self.projection_layer(enc_h)
            x_mask = x_mask.unsqueeze(1)
            x_mask = ~x_mask
            src_mask = x_mask

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
            tgt_mask = (ys != args.text_padding_idx).unsqueeze(1)
            tgt_mask = tgt_mask & self.subsequent_mask(ys.size(-1)).type_as(src_mask_use.data)
            dec_h = self.decoder(self.tgt_embed(ys), ench_use, src_mask_use, tgt_mask)
            att_prob = self.att_generator(dec_h[:, -1, :], T=args.T)
            if args.lm_weight > 0:
                lm_prob = lm_model(ys, tgt_mask)[:,-1,:]
                local_prob = att_prob + args.lm_weight * lm_prob
                # need to be experimented for eos probability
                #local_prob[:,eos] = (1 + args.lm_weight) * att_prob[:,eos]
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

    def fast_decode_with_ctc(self, src, src_mask, tokenizer, args, lm_model=None):
        """
        Take CTC output as the decoder input for decoder. Regard the decoder as 
        a correction model.
        """
        bs = src.size(0)
        sos = tokenizer.vocab['sos']
        eos = tokenizer.vocab['eos']
        blank = tokenizer.vocab['blank']

        if self.model_type != "hubert":
            x, src_mask = self.src_embed(src, src_mask)
            enc_h = self.encoder(x, src_mask)
        else:
            enc_h, x_mask, _ = self.encoder(src, padding_mask=src_mask, mask_prob=args.mask_prob)
            enc_h = self.projection_layer(enc_h)
            x_mask = x_mask.unsqueeze(1)
            x_mask = ~x_mask
            src_mask = x_mask
            
        ctc_out = self.ctc_generator(enc_h)
        bs, xmax, _ = ctc_out.size()
        best_paths = ctc_out.argmax(-1)
        best_paths = best_paths.masked_fill(src_mask.squeeze(1)==0, 0)

        aligned_seq_shift = best_paths.new_zeros(best_paths.size())
        aligned_seq_shift[:, 1:] = best_paths[:,:-1]
        dup = best_paths == aligned_seq_shift
        best_paths.masked_fill_(dup, 0)
        ctc_greedy, length = [], []
        for b in range(bs):
            ctc_greedy.append(best_paths[b][best_paths[b].nonzero()])
            length.append(ctc_greedy[-1].size(0))
        max_length = max(length)
        tgt_input = best_paths.new_zeros(bs, max_length+1).fill_(args.text_padding_idx)
        tgt_input[:,0] = sos
        for b in range(bs):
            tgt_input[b].narrow(0, 1, length[b]).copy_(ctc_greedy[b][:,0])
        tgt_mask = (tgt_input != args.text_padding_idx).unsqueeze(1)
        tgt_mask = tgt_mask & self.subsequent_mask(tgt_input.size(-1)).type_as(tgt_mask)
        dec_h = self.decoder(self.tgt_embed(tgt_input), enc_h, src_mask, tgt_mask)
        att_out = self.att_generator(dec_h)

        ys = torch.ones(1, 1).fill_(sos).long()
        if args.use_gpu:
            ys = ys.cuda()
        
        batch_top_seqs = [ [{'ys': ys, 'score': 0.0, 'hyp': [sos] } ] for b in range(bs) ]
        
        for i in range(max_length+1):
            # batchify the batch and beam
            all_seqs, ys, att_prob = [], [], []
            
            for b in range(bs):
                all_seqs.append([])
                for seq in batch_top_seqs[b]:
                    if i > length[b]:
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
                tgt_mask = (ys != args.text_padding_idx).unsqueeze(1)
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
                    if i > length[b]:
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

                        hyp = seq['hyp'] + [next_token.item()] if next_token.item() != eos else seq['hyp']
                        rs_seq = {'ys':ys, 'score': score, 'hyp': hyp } 
                        all_seqs[b].append(rs_seq)

                sort_f = lambda x:x['score'] + (len(x['hyp'])-1) * args.length_penalty \
                            if args.length_penalty is not None else lambda x:x['score']                
                batch_top_seqs[b] = sorted(all_seqs[b], key=sort_f, reverse=True)[:args.beam_width]
        return batch_top_seqs


