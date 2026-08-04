"""SGCCFuse: Saliency-Guided Cycle-Consistent Cross-Attention Fuse (显著性引导的循环一致交叉注意力融合网络), 面向红外与可见光图像融合的双分支多尺度融合网络。"""
import torch
import torch.nn as nn
from modules.shared_encoder import SharedEncoder
from modules.scale_block import ScaleBlock
from modules.sfe_block import SFEBlock
from modules.cpca_block import CPCABlock


class PatchEmbed(nn.Module):
    """Patch Embedding: 2x2 patch → token, 空间 /2, 通道 ×2。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        return self.proj(x)


class SGCCFuse(nn.Module):
    """SGCCFuse: Saliency-Guided Cycle-Consistent Cross-Attention Fuse (显著性引导的循环一致交叉注意力融合网络)。"""

    def __init__(self, in_channels=3, out_channels=3, dim=64,
                 enc_blocks=4, num_scales=3, **kwargs):
        super(SGCCFuse, self).__init__()
        self.in_c1 = in_channels
        self.in_c2 = in_channels
        self.dim = dim
        self.num_scales = num_scales
        dims = [dim * (2 ** i) for i in range(num_scales + 1)]
        self.shared_encoder = SharedEncoder(
            in_channels=in_channels, dim=dim, num_blocks=enc_blocks
        )
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(dims[i], dims[i + 1]) for i in range(num_scales)
        ])
        self.enc_blocks_list = nn.ModuleList([
            ScaleBlock(dims[i + 1]) for i in range(num_scales)
        ])
        self.feat_extract_blocks = nn.ModuleList([
            SFEBlock(dims[i + 1]) for i in range(num_scales)
        ])
        self.cross_attn_blocks = nn.ModuleList([
            CPCABlock(dims[i + 1]) for i in range(num_scales)
        ])
        self.dec1_deep = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[3] * 2, dims[2], kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dims[2], dims[2], kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.dec1_shallow = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dims[2], dims[2], kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.dec1_refine = ScaleBlock(dims[2])
        self.dec2_deep = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[2], dims[1], kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dims[1], dims[1], kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.dec2_shallow = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dims[1], dims[1], kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.dec2_refine = ScaleBlock(dims[1])
        self.dec_out_deep = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dims[1], dims[0], kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.stem_skip = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.dec_out_refine = ScaleBlock(dims[0])
        self.out_proj = nn.Conv2d(dims[0], out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x1 = x.narrow(1, 0, self.in_c1)  # 前3通道 = 红外 IR
        x2 = x.narrow(1, self.in_c1, self.in_c2)  # 后3通道 = 可见光 VI
        feat_v = self.shared_encoder(x1)
        feat_r = self.shared_encoder(x2)
        cur_v, cur_r = feat_v, feat_r
        enhs = []
        # 三尺度串行下采样 + 各自 SFE 显著性提取 + CPCA 跨模态交叉注意力
        for i in range(self.num_scales):
            cur_v = self.patch_embeds[i](cur_v)  # 空间 /2, 通道 ×2
            cur_r = self.patch_embeds[i](cur_r)
            cur_v = self.enc_blocks_list[i](cur_v)
            cur_r = self.enc_blocks_list[i](cur_r)
            f1_v = self.feat_extract_blocks[i](cur_v)  # 创新点1 SFE
            f1_r = self.feat_extract_blocks[i](cur_r)
            enh_v, enh_r = self.cross_attn_blocks[i](f1_v, f1_r)  # 创新点2 CPCA
            enhs.append((enh_v, enh_r))
        # 解码器由深到浅逐级上采样融合, 每级聚合深/浅两支并 refine
        deep_v, deep_r = enhs[2]
        deep = self.dec1_deep(torch.cat([deep_v, deep_r], dim=1))  # 尺度2 深支 (H/8→H/4)
        shallow_v, shallow_r = enhs[1]
        shallow = self.dec1_shallow(torch.cat([shallow_v, shallow_r], dim=1))  # 尺度1 浅支 skip
        fused = self.dec1_refine(deep + shallow)
        deep = self.dec2_deep(fused)  # 尺度1 深支 (H/4→H/2)
        shallow_v, shallow_r = enhs[0]
        shallow = self.dec2_shallow(torch.cat([shallow_v, shallow_r], dim=1))  # 尺度0 浅支 skip
        fused = self.dec2_refine(deep + shallow)
        deep = self.dec_out_deep(fused)  # 输出深支 (H/2→H)
        skip = self.stem_skip(torch.cat([feat_v, feat_r], dim=1))  # 全分辨率 stem 浅层 skip 恢复高频细节
        fused = self.dec_out_refine(deep + skip)
        out = self.out_proj(fused)  # 输出 3 通道融合图
        return out
