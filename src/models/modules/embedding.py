#!/usr/bin/env python3
# 2020 Ruchao Fan

import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    "Implement absolute position embedding."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class TextEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(TextEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class ConvEmbedding(nn.Module):
    """
    Mapping input features to embeddings and downsample with 4.
    """
    def __init__(self, input_size, d_model, dropout):
        super(ConvEmbedding, self).__init__()

        self.conv = nn.Sequential(
                    nn.Conv2d(1, d_model, 3, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(d_model, d_model, 3, 2, 1),
                    nn.ReLU(), )
        
        self.linear_out = nn.Sequential(
                            nn.Linear(d_model * (((input_size-1)//2) // 2 + 1), d_model),
                            PositionalEncoding(d_model, dropout) )

    def forward(self, x, mask):
        "mask needs to be revised to downsample version"
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, d = x.size()
        x = self.linear_out(x.transpose(1,2).contiguous().view(b, t, c*d))
        mask = mask[:, :, ::2][:, :, ::2]
        return x, mask


