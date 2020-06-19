#!/usr/bin/env python3

import copy
import torch.nn as nn

from models.modules.norm import LayerNorm

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size.
        Using pre-norm for stable training at early stage
        sublayer can be self attention and feedforward
        """
        return x + self.dropout(sublayer(self.norm(x)))

class StatsPoolingLayer(nn.Module):
    def __init__(self):
        super(StatsPoolingLayer, self).__init__()

    def forward(self, x, num_frs=None):
        mean = []
        std = []
        for e, l in enumerate(x):
            size = num_frs[e].item()
            mean.append(torch.mean(x[e:e+1,:size,:], dim=1))
            std.append(torch.std(x[e:e+1,:size,:], dim=1))
        mean = torch.cat(mean, dim=0)
        std = torch.cat(std, dim=0)
        #mean = torch.mean(x, dim=1)
        #std = torch.std(x, dim=1)
        return torch.cat((mean, std), dim=1)


