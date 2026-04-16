from typing import Sequence, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.modules.block import Bottleneck, Conv, C3k, PSABlock, Attention, C2f
from nn.modules.conv import autopad, DWConv, GhostConv


# <editor-fold desc='FINet'>
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BottleneckSE(nn.Module):
    """
    添加了SENet的bottleneck
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckSE, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c_, c2, 3, 1, g=g)

        self.conv = nn.Conv2d(c_, c2, 3, 1, autopad(3, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.add = shortcut and c1 == c2
        if self.add:
            self.se = SELayer(c2, reduction=16)
        else:
            self.se = None

    def forward(self, x):
        residual = x
        x = self.cv1(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.add:
            x = self.se(x)
            x = residual + x
        x = self.act(x)

        return x
    # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3SE(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(BottleneckSE(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# </editor-fold>


# <editor-fold desc='CACSYV'>
class CACS_C2f(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            CACS_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CACS_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        self.add = shortcut and c1 == c2
        c1 >>= 1
        c2 >>= 1

        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)

        self.conv = nn.Conv2d(c_, c2, k[1], 1, autopad(3, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        self.ca = CoordAttlayer(c2, c2, reduction=32)

        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.cv1(x2)
        x2 = self.conv(x2)
        x2 = self.bn(x2)

        x2 = self.ca(x2)
        out = torch.cat((x2, x1), dim=1)
        out = self.act(out)

        return channel_shuffle(out, 2)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttlayer(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttlayer, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


# </editor-fold>

# <editor-fold desc='FYV8'>
class FYV8Concat(nn.Module):
    def __init__(self, c1s, c2):
        nn.Module.__init__(self, )
        c1 = c1s[0]
        self.ws = nn.Parameter(torch.ones(size=(len(c1s),), dtype=torch.float))
        self.fuse = Conv(sum(c1s) + c1, c2)

    def forward(self, xs):
        ws_nm = self.ws / (torch.sum(self.ws) + 1e-7)
        ys = [xs[0]]
        for x, w in zip(xs, ws_nm):
            ys.append(x * w)
        y = torch.cat(ys, dim=1)
        return self.fuse(y)


# </editor-fold>

# <editor-fold desc='MCI-GLA'>

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PSABottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(PSABottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = PSAModule(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class PSAC3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[PSABottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SAtt(nn.Module):
    def __init__(self, c1, ):
        super(SAtt, self).__init__()
        self.cv1 = nn.Conv2d(c1, c1 // 16, kernel_size=1, )
        self.cv2 = nn.Conv2d(c1 // 16, c1 // 16, kernel_size=3, padding=1)
        self.cv3 = nn.Conv2d(c1 // 16, c1 // 16, kernel_size=5, padding=2)
        self.cv4 = nn.Conv2d(c1 // 16, c1 // 16, kernel_size=7, padding=3)
        self.cv5 = nn.Conv2d(c1 // 4, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        x3 = self.cv3(x1)
        x4 = self.cv4(x1)
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.cv5(feats)
        out = self.sigmoid(out)
        return out


class CoT(nn.Module):
    # Contextual Transformer Networks https://arxiv.org/abs/2107.12292
    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class GAL(nn.Module):
    def __init__(self, c1, c2, ):
        super(GAL, self).__init__()
        self.cv1 = SAtt(c1)
        self.cv2 = CoT(c1)

    def forward(self, x):
        k1 = self.cv1(x)
        k1 = k1 * x
        k2 = self.cv2(x)
        return k1 + k2


class MCI_GLA(nn.Module):
    def __init__(self, c1):
        nn.Module.__init__(self, )
        self.mci = PSAC3(c1, c1)
        self.gal = GAL(c1, c1)

    def forward(self, x):
        x = self.mci(x)
        x = self.gal(x)
        return x


# </editor-fold>

# <editor-fold desc='Lite-YV-ID'>

# class GhostConv(nn.Module):
#     # Ghost Convolution https://github.com/huawei-noah/ghostnet
#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
#         super().__init__()
#         c_ = c2 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
#         self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
#
#     def forward(self, x):
#         y = self.cv1(x)
#         return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class C2fGhost(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        # print('C2f', c1, c2)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(GhostBottleneck(self.c, self.c) for _ in range(n))


# </editor-fold>

# <editor-fold desc='MFI-YV'>

class MSAGhostConv(GhostConv):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        GhostConv.__init__(self, c1, c2, k=k, s=s, g=g, act=act)
        self.branches = nn.ModuleList([])
        for k in [3, 5, 7]:
            branch = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=(1, k), stride=(1, 1), padding=(0, k // 2)),
                nn.ReLU(),
                nn.Conv2d(c2, c2, kernel_size=(k, 1), stride=(1, 1), padding=(k // 2, 0))
            )
            self.branches.append(branch)
        self.proj = nn.Conv2d(c2 * 3, c2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        out = GhostConv.forward(self, x)
        x_down = F.avg_pool2d(x, stride=2, kernel_size=2)
        bouts = []
        for b in self.branches:
            bouts.append(b(x_down))
        x_bout = self.proj(torch.cat(bouts, dim=1))
        attn = F.upsample_nearest(torch.sigmoid(x_bout), scale_factor=2)

        if not out.size() == attn.size():
            attn = F.pad(attn, (0, out.size(3) - attn.size(3), 0, out.size(2) - attn.size(2)))
        return out * attn


class MSAGhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            MSAGhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s),
            MSAGhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(
            DWConv(c1, c1, k, s, act=False),
            Conv(c1, c2, 1, 1)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class C2fMSAGhost(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        # print('C2f', c1, c2)
        self.c = int(c2 * e)  # hidden channels
        self.m = nn.ModuleList(MSAGhostBottleneck(self.c, self.c) for _ in range(n))
# </editor-fold>
