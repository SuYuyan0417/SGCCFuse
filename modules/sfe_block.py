
import torch
import torch.nn as nn


class SFEBlock(nn.Module):
    

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        self.conv_f1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.conv_f2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        hidden = max(dim * 2 // reduction, 4)
        self.fc1 = nn.Linear(dim * 2, hidden)
        self.act = nn.GELU()
        self.fc_s = nn.Linear(hidden, dim * 2)
        self.spatial_conv = nn.Conv2d(1, 1, 1, 1, 0, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        f1 = self.conv_f1(x)  # 分支1 线性变换
        f2 = self.conv_f2(x)  # 分支2 线性变换
        cat = torch.cat([f1, f2], dim=1)
        gap_ch = cat.mean(dim=[2, 3])  # 通道全局平均池化
        s = self.fc_s(self.act(self.fc1(gap_ch)))  # 通道显著性权重 (SE 风格)
        s = s.view(B, 2 * C, 1, 1)
        S1 = s[:, :C, :, :]
        S2 = s[:, C:, :, :]
        gap_sp = cat.mean(dim=1, keepdim=True)  # 空间平均特征
        gmp_sp = cat.max(dim=1, keepdim=True)[0]  # 空间最大特征
        diff = gap_sp - gmp_sp
        M = torch.sigmoid(self.spatial_conv(diff))  # 空间显著性掩码
        u1 = f1 * S2  # 交叉通道加权: 分支1 用分支2 的通道权重
        u2 = f2 * S1
        fused = (u1 + u2) * M  # 空间掩码调制
        out = x + fused  # 残差连接
        return out
