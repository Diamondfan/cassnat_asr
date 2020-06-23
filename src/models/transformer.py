
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

    def forward(self, src, tgt, src_mask, tgt_mask):
        x, x_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, x_mask)
        #CTC Loss needs log probability as input
        ctc_out = self.ctc_generator(enc_h)
        dec_h = self.decoder(self.tgt_embed(tgt), enc_h, x_mask, tgt_mask)
        att_out = self.att_generator(dec_h)
        return ctc_out, att_out, enc_h

    def subsequent_mask(self, size):
        ret = torch.ones(size, size, dtype=torch.uint8)
        return torch.tril(ret, out=ret).unsqueeze(0)

    def beam_decode(self, src, src_mask, vocab, args):
        # rnnlm is not supported at present
        x, src_mask = self.src_embed(src, src_mask)
        enc_h = self.encoder(x, src_mask)
        bs = src.size(0)
        sos = vocab.word2index['sos']
        eos = vocab.word2index['eos']
        ys = torch.ones(1, 1).fill_(sos).long()
        if args.use_gpu:
            ys = ys.cuda()
        
        if args.rnnlm != 'None':
            batch_top_seqs = [ [{'ys': ys, 'score': 0.0, 'hyp': [sos], 'rnnlm_prev': lm.init(sos)} ] for b in range(bs)]
        else:
            batch_top_seqs = [ [{'ys': ys, 'score': 0.0, 'hyp': [sos] } ] for b in range(bs) ]
        
        for i in range(args.max_decode_step-1):
            # concat input of all samples in a batch and their braches for parallel computation
            all_seqs, ys, ench_use, src_mask_use = [], [], [], []
            if args.rnnlm != 'None':
                lm_input, lm_hidden0, lm_hidden1 = [], [], []

            for b in range(bs):
                all_seqs.append([])
                for seq in batch_top_seqs[b]:
                    if seq['hyp'][-1] == eos:
                        all_seqs[b].append(seq)
                        continue
                    ys.append(seq['ys'])
                    ench_use.append(enc_h[b:b+1])
                    src_mask_use.append(src_mask[b:b+1])
                    if args.rnnlm != 'None':
                        lm_input.append([seq['hyp'][-1]-1])
                        lm_hidden0.append(seq['rnnlm_prev'][0])
                        lm_hidden1.append(seq['rnnlm_prev'][1])
            
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
            
            if args.rnnlm != 'None':
                lm_input = torch.LongTensor(lm_input).cuda()
                lm_hidden = (torch.cat(lm_hidden0, dim=1), torch.cat(lm_hidden1, dim=1))
                lm_output, hidden_out = lm(lm_input, lm_hidden)
                lm_output = F.pad(lm_output[:,-1,:], pad=[1,0])  #output is different with transformer
                prob += lm.lm_alpha * lm_output  
            
            att_scores, indices = torch.topk(att_prob, args.beam_width, dim=-1)
            
            # distribute scores to corresponding sample and beam
            s_idx = -1
            for b in range(bs):
                for seq in batch_top_seqs[b]:
                    if seq['hyp'][-1] == eos:
                        continue
                    s_idx += 1

                    for j in range(args.beam_width):
                        next_token = indices[s_idx][j]
                        token_score = att_scores[s_idx][j].item()
                        score = seq['score'] + token_score
                        if args.rnnlm != 'None':
                            lm_score = seq[3] + lm.get_ngram_prob(wid)
                            score += lm.lm_alpha * lm_score

                        ys = torch.cat([seq['ys'],next_token.view(-1,1)],dim=-1)
                        rs_seq = {'ys':ys, 'score': score, 'hyp': seq['hyp']+ [next_token.item()] } 
                        if args.rnnlm != 'None':
                            rs_seq['rnnlm_prev'] = (hidden_out[0][:,s_idx:s_idx+1,:], hidden_out[1][:,s_idx:s_idx+1,:])

                        all_seqs[b].append(rs_seq)
                if args.length_penalty != 'None':
                    all_seqs[b] = sorted(all_seqs[b], key=lambda x:(x['score']+args.length_penalty*len(x['hyp'])), reverse=True)
                else:
                    all_seqs[b] = sorted(all_seqs[b], key=lambda x:(x['score']), reverse=True)
            batch_top_seqs = [all_seqs[b][:args.beam_width] for b in range(bs)]
        return batch_top_seqs

    
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


