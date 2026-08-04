
import torch
import torch.nn as nn


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt v2)。"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ScaleBlock(nn.Module):


    def __init__(self, dim, exp_ratio=4, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, stride=1, padding=padding, groups=dim, bias=True
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * exp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)  # 深度可分离卷积提取局部特征
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)  # LayerNorm (通道维)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)  # 全局响应归一化
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = residual + self.gamma * x  # LayerScale 门控残差
        return x
