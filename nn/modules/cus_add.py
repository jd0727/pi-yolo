from typing import Sequence, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.modules import C2fPSA
from nn.modules.block import Bottleneck, Conv, C3k, PSABlock, Attention


class MBA(nn.Module):

    def __init__(self, c1, c2, n=1, e=0.5):
        nn.Module.__init__(self)
        self.num_repeat = n
        self.out_channels = c2
        self.in_channels = c1
        self.sp_channels = c2
        self.inner_channels = self.sp_channels // 2
        _channels = self.sp_channels // self.num_repeat
        self.blks = nn.ModuleList([])
        for i in range(self.num_repeat):
            last_channels = _channels + self.inner_channels if i > 0 else _channels
            self.blks.append(nn.Sequential(
                Conv(last_channels, self.inner_channels, 1, 1, ),
                AttentionNoProj(self.inner_channels, attn_ratio=0.5, num_heads=self.inner_channels // 64),
                Conv(self.inner_channels, self.inner_channels, 1, 1, ),
            ))
        self.inner = Conv(self.in_channels, self.sp_channels, 1, 1, )
        self.outer = Conv(self.sp_channels + self.inner_channels * self.num_repeat, self.out_channels, 1, 1, )

    def forward(self, x):
        x = self.inner(x)
        xs = x.chunk(self.num_repeat, dim=1)
        ys = [x]
        buffer = None
        for i in range(self.num_repeat):
            if i == 0:
                buffer = xs[i]
            else:
                buffer = torch.cat([buffer, xs[i]], dim=1)
            buffer = self.blks[i](buffer)
            ys.append(buffer)
        y = torch.cat(ys, dim=1)
        return self.outer(y)


class AttentionNoProj(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_key = int(self.dim * attn_ratio)
        self.head_dim = dim // num_heads
        self.head_dim_key = int(self.head_dim * attn_ratio)
        self.scale = self.head_dim_key ** -0.5

        self.qkv = Conv(dim, self.dim + self.dim_key * 2, 1, act=False)
        # self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.size()
        N = H * W
        q, k, v = self.qkv(x).split([self.dim_key, self.dim_key, self.dim], dim=1)
        q = q.view(B, self.num_heads, self.head_dim_key, N).transpose(-1, -2)
        k = k.view(B, self.num_heads, self.head_dim_key, N).transpose(-1, -2)
        pe = self.pe(v)
        v = v.view(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

        out = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return out + pe


class MBC(nn.Module):
    def __init__(self, c1, c2, n=4, c3k=False, e=0.5, g=1, shortcut=True):
        nn.Module.__init__(self)
        # print(c2)
        self.blks = nn.ModuleList([])
        self.in_channels = c1
        self.out_channels = c2
        self.num_repeat = n * 2
        self.sp_channels = int(self.out_channels * e * 2)
        self.inner_channels = self.sp_channels // 4
        _channels = self.sp_channels // self.num_repeat
        for i in range(self.num_repeat):
            last_channels = _channels + self.inner_channels if i > 0 else _channels
            self.blks.append(nn.Sequential(
                Conv(last_channels, self.inner_channels, 1, 1, ),
                Conv(self.inner_channels, self.inner_channels, 3, 1, ),
                Conv(self.inner_channels, self.inner_channels, 3, 1, ),
            ))
        self.inner = Conv(self.in_channels, self.sp_channels, 1, 1, )
        self.outer = Conv(self.sp_channels + self.inner_channels * self.num_repeat, self.out_channels, 1, 1, )

    def forward(self, x):
        # print(x.size())
        x = self.inner(x)
        xs = x.chunk(self.num_repeat, dim=1)
        ys = [x]
        buffer = None
        for i in range(self.num_repeat):
            if i == 0:
                buffer = xs[i]
            else:
                buffer = torch.cat([buffer, xs[i]], dim=1)
            buffer = self.blks[i](buffer)
            ys.append(buffer)
        y = torch.cat(ys, dim=1)
        return self.outer(y)


StageCus = MBC
C2PSACus3 = MBA
