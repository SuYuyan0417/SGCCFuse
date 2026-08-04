
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.module_util import initialize_weights


class AttentionBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 4), bias=False),
            nn.GELU(),
            nn.Linear(max(dim // reduction, 4), dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RestormBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.norm1 = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 2, 1, 1, 0, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True)
        self.attn = AttentionBlock(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn_conv1 = nn.Conv2d(dim, dim * 2, 1, 1, 0, bias=True)
        self.ffn_act = nn.GELU()
        self.ffn_conv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True)
        initialize_weights(
            [self.dwconv, self.pwconv1, self.pwconv2,
             self.ffn_conv1, self.ffn_conv2], 0.1
        )

    def forward(self, x):
        residual = x
        y = self.dwconv(x)
        y = y.permute(0, 2, 3, 1)
        y = self.norm1(y)
        y = y.permute(0, 3, 1, 2)
        y = self.pwconv2(self.act(self.pwconv1(y)))
        y = self.attn(y)
        x = residual + y
        residual = x
        y = x.permute(0, 2, 3, 1)
        y = self.norm2(y)
        y = y.permute(0, 3, 1, 2)
        y = self.ffn_conv2(self.ffn_act(self.ffn_conv1(y)))
        x = residual + y
        return x


class SharedEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, num_blocks=4):
        super().__init__()
        self.dim = dim
        self.shallow = nn.Conv2d(in_channels, dim, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RestormBlock(dim) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        initialize_weights([self.shallow, self.tail], 0.1)

    def forward(self, x):
        feat = self.shallow(x)  # 3→dim 浅层特征, 全分辨率
        feat = self.body(feat)  # 堆叠 Restormer 块提取语义
        feat = self.tail(feat)
        return feat
