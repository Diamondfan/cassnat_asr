#!/usr/bin/env python3
# 2020 Ruchao Fan

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.utils import clones

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
        scores = scores.masked_fill(mask == 0, min_value)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """ Multi-Head Attention Layer
    
    :param int h: number of heads
    :param int d_model: dimensions in the model
    :dropout float dropout: dropout rate after weight computation 
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, pos_embed=None):
        """ Take cross attention as an example
        query: batch x U x d
        key: batch x T x d
        value: batch x T x d
        mask: batch x U x T      
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class RelMultiHeadedAttention(nn.Module):
    """ Multi-Head Attention Layer
    
    :param int h: number of heads
    :param int d_model: dimensions in the model
    :dropout float dropout: dropout rate after weight computation 
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(RelMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, pos_embed=None):
        """ Take cross attention as an example
        query: batch x U x d
        key: batch x T x d
        value: batch x T x d
        mask: batch x U x T      
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
             for l, x in zip(self.linears, (query, key, value))]
        t_q = query.size(1)
        pos_embed = self.linear_pos(pos_embed).unsqueeze(0).repeat(nbatches,1,
                        1).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  #(nb, h, 2*t_q-1, d_k)
        
        # 2) Apply attention on all the projected vectors in batch. 
        
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)       #(nb, h, t_q, d_k)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)       #(nb, h, t_q, d_k)

        scores_ac = torch.matmul(q_with_bias_u, key.transpose(1, 2).transpose(-2, -1))   #(nb, h, t_q, t_k)
        scores_bd = torch.matmul(q_with_bias_v, pos_embed.transpose(-2, -1))     #(nb, h, t_q, 2*t_q -1)   
        
        # select corresponding relative embeddings (out of memory)
        #ones = torch.ones(t_q, t_q).type_as(query)
        #mask_embed = (torch.cat([torch.triu(ones), torch.tril(ones)[:,1:]], dim=1) == 1)
        #scores_bd = scores_bd.masked_select(mask_embed).view(nbatches, self.h, t_q, -1)  #(nb, h, t_q, t_k)
        #scores = (scores_ac + torch.flip(scores_bd, [-1])) / math.sqrt(self.d_k)

        zero_pad = torch.zeros((*scores_bd.size()[:3], 1)).type_as(scores_bd)
        padded = torch.cat([zero_pad, scores_bd], dim=-1)
        padded = padded.view(*scores_bd.size()[:2], scores_bd.size(3) + 1, scores_bd.size(2))
        scores_bd = padded[:, :, 1:].view_as(scores_bd)[:,:,:,:t_q]
        scores = (scores_ac + scores_bd) / math.sqrt(self.d_k)

        if mask is not None:
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask == 0, min_value)
        p_attn = F.softmax(scores, dim = -1).masked_fill(mask==0, 0.0)
        
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value.transpose(1, 2))
        self.attn = p_attn
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)



