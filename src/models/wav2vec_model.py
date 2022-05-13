#!/usr/bin/env python3
# 2021 Ruchao Fan
# Part of codes are borrowed from Fairseq (https://github.com/pytorch/fairseq)

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.norm import LayerNorm
from models.modules.attention import MultiHeadedAttention, RelMultiHeadedAttention
from models.modules.positionff import PositionwiseFeedForward
from models.modules.embedding import PositionalEncoding, RelativePositionalEncoding, ConvEmbedding
from models.modules.conformer_related import Swish, ConvModule
from models.blocks.conformer_blocks import Encoder as ConEncoder
from models.blocks.transformer_blocks import Encoder as TrfEncoder
from models.modules.ssl_util import compute_mask_indices, buffered_arange, init_bert_params
from models.modules.gumbel_vector_quantizer import GumbelVectorQuantizer
from utils.loss import Wav2vecLoss

def make_model(input_size, args):
    c = copy.deepcopy
    if args.model_type == "transformer":
        enc_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        enc_ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        enc_position = PositionalEncoding(args.d_model, args.dropout)
        encoder = TrfEncoder(args.d_model, enc_attn, enc_ff, args.dropout, args.N_enc)

    elif args.model_type == "conformer": 
        #assert args.pos_type == "relative", "conformer must use relative positional encoding"
        if args.pos_type == "relative":
            enc_position = RelativePositionalEncoding(args.d_model, args.dropout, args.enc_max_relative_len)
            enc_attn = RelMultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        elif args.pos_type == "absolute":
            enc_position = PositionalEncoding(args.d_model, args.dropout)
            enc_attn = MultiHeadedAttention(args.n_head, args.d_model, args.dropout)
        conv_module = ConvModule(args.d_model, args.enc_kernel_size, activation=Swish())
        enc_ff = PositionwiseFeedForward(args.d_model, args.d_encff, args.dropout, activation=Swish())
        encoder = ConEncoder(args.d_model, c(enc_ff), enc_attn, conv_module, c(enc_ff), args.dropout, args.N_enc, args.pos_type, args.share_ff)

    else:
        raise NotImplementedError

    final_dim = args.d_model
    model = Wav2vec2(
        ConvEmbedding(input_size, args.d_model, args.dropout, enc_position),
        encoder, args)
    
    for name, p in model.named_parameters():
        if p.dim() > 1 and name.split(".")[0] not in ["quantizer", "mask_embed"]:
            nn.init.xavier_uniform_(p)
    return model


class Wav2vec2(nn.Module):
    def __init__(self, src_embed, encoder, args):
        super(Wav2vec2, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        
        # arguments
        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.mask_dropout = args.mask_dropout
        self.mask_min_space = args.mask_min_space
        self.require_same_masks = args.require_same_masks
        self.no_mask_overlap = args.no_mask_overlap
    
        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_before = args.mask_channel_before
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.n_negatives = args.num_negatives
        self.cross_sample_negatives = args.cross_sample_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere

        self.logit_temp = args.logit_temp
        self.final_dim = args.final_dim
        
        if args.quantize_targets:
            vq_dim = args.latent_dim if args.latent_dim > 0 else args.final_dim
            self.quantizer = GumbelVectorQuantizer(dim=args.d_model, num_vars=args.latent_vars, 
                                                    temp=args.latent_temp, groups=args.latent_groups, 
                                                    combine_groups=False, vq_dim=vq_dim, 
                                                    time_first=True, weight_proj_depth=args.quantizer_depth, 
                                                    weight_proj_factor=args.quantizer_factor)

            self.project_q = nn.Linear(vq_dim, self.final_dim)
        else:
            self.quantizer = None
            self.project_q = nn.Linear(args.d_model, self.final_dim)

        #self.layer_norm = LayerNorm(args.d_model)
        self.mask_embed = nn.Parameter(torch.FloatTensor(args.d_model).uniform_())
        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_feature = nn.Dropout(args.dropout_features)

        self.final_proj = nn.Linear(args.d_model, self.final_dim)
    
    def forward(self, src, src_mask, num_updates, mask=True, mask_indices=None, mask_channel_indices=None, padding_count=None):

        features, features_mask = self.src_embed(src, src_mask)
        if isinstance(features, tuple):
            features, pos_embed = features
        else:
            pos_embed = None

        padding_mask = (~features_mask).squeeze(1)
        #features = self.layer_norm(features)
        unmasked_features = features.clone()

        features = self.dropout_input(features)
        unmasked_features = self.dropout_feature(unmasked_features)
    
        num_vars, code_ppl, prob_ppl, curr_temp = None, None, None, None

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, mask_indices=mask_indices, mask_channel_indices=mask_channel_indices)
            y = unmasked_features[mask_indices].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        if pos_embed is not None:
            x = (x, pos_embed)

        x = self.encoder(x, features_mask)

        if self.quantizer:

            self.quantizer.set_num_updates(num_updates)

            if self.negatives_from_everywhere:
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)

                negs, _ = self.sample_negatives(y, mask_indices[0].sum(), padding_count=padding_count)
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q(y)
                
                negs, _ = self.sample_negatives(y, y.size(1), padding_count=padding_count)

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(y.size(0) * y.size(1), self.codebook_negatives)
                cb_negs = cb_negs.view(self.codebook_negatives, y.size(0), y.size(1), -1)
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    mask_indices[0].sum(),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )
    
        x = x[mask_indices].view(x.size(0), -1, x.size(-1))
        x = self.final_proj(x)
        x = self.compute_pred(x, y, negs)

        result = {
            "x": x,
            "padding_mask": padding_mask,
        }

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result
    
    def compute_pred(self, x, y, negatives):
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)
    
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        logits = logits / self.logit_temp
        logits = logits.type_as(x)                

        if neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                self._inftensor = float("-inf")
            logits[1:][neg_is_pos] = self._inftensor

        return logits

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        return pen

    def apply_mask(self, x, padding_mask, mask_indices=None, mask_channel_indices=None):

        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices((B, C), None, self.mask_channel_prob,
                                    self.mask_channel_length, self.mask_channel_selection,
                                    self.mask_channel_other, no_overlap=self.no_mask_channel_overlap,
                                    min_sapce=self.mask_channel_min_space,)

            mask_channel_indices = (torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1))
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices((B, T), padding_mask, self.mask_prob, self.mask_length, 
                                self.mask_selection, self.mask_other, min_masks=2, no_overlap=self.no_mask_overlap, 
                                min_space=self.mask_min_space, require_same_masks=self.require_same_masks, 
                                mask_dropout=self.mask_dropout)
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_embed
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices((B, C), None, self.mask_channel_prob,
                                        self.mask_channel_length, self.mask_channel_selection,
                                        self.mask_channel_other, no_overlap=self.no_mask_channel_overlap,
                                        min_space=self.mask_channel_min_space,)

                mask_channel_indices = (torch.from_numpy(mask_channel_indices).to(x.device)
                                        .unsqueeze(1).expand(-1, T, -1))

            x[mask_channel_indices] = 0
        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )

                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def remove_pretraining_modules(self, last_layer=None):
        self.quantizer = None
        self.project_q = None
        self.final_proj = None 

