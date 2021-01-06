#!/usr/bin/env python3
# 2020 Ruchao Fan

import torch
import torch.nn as nn

import torch.nn.functional as F

# swish activation for positionff and conv blocks
class Swish(nn.Module):    
    def forward(self, x):
        return x * torch.sigmoid(x)

# conv blocks
class ConvModule(nn.Module):
    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super(ConvModule, self).__init__()
        
        self.pointwise_conv1 = nn.Conv1d(channels, 2*channels, kernel_size=1, 
                                         stride=1, padding=0, bias=bias)

        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size, stride=1,
                                        padding=padding, groups=channels, bias=bias)

        self.norm = nn.GroupNorm(channels, channels)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1,
                                         stride=1, padding=0, bias=bias)

        self.activation = activation

    def forward(self, x):
        x = x.transpose(1, 2)

        # pointwise1 + GLU
        x = F.glu(self.pointwise_conv1(x), dim=1)
    
        # depthwise conv + batchnorm + swish
        x = self.activation(self.norm(self.depthwise_conv(x)))

        # pointwise2
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)

