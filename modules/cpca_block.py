
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCABlock(nn.Module):

    def __init__(self, dim, num_heads=8, num_iters=3, qkv_dim_ratio=0.5,
                 max_kv_tokens=1024, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须能被 num_heads {num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_iters = max(num_iters, 1)
        self.max_kv_tokens = max_kv_tokens
        qkv_dim = int(dim * qkv_dim_ratio)
        qkv_dim = max((qkv_dim // num_heads) * num_heads, num_heads)
        self.qkv_dim = qkv_dim
        self.qkv_head_dim = qkv_dim // num_heads
        self.q_a = nn.Linear(dim, qkv_dim, bias=True)
        self.k_a = nn.Linear(dim, qkv_dim, bias=True)
        self.v_a = nn.Linear(dim, dim, bias=True)
        self.proj_a = nn.Linear(dim, dim)
        self.q_b = nn.Linear(dim, qkv_dim, bias=True)
        self.k_b = nn.Linear(dim, qkv_dim, bias=True)
        self.v_b = nn.Linear(dim, dim, bias=True)
        self.proj_b = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.gamma_v = nn.Parameter(torch.full((dim,), 0.1))  # IR 分支门控残差, 渐进学习的初始权重
        self.gamma_r = nn.Parameter(torch.full((dim,), 0.1))  # VI 分支门控残差

    def _cycle_consistent_mask(self, attn):
        # 循环一致掩码: 若 i 的最强匹配为 j 且 j 的最强匹配回指 i, 则视为可靠对应
        with torch.no_grad():
            a = attn.float()
            i_star = a.argmax(dim=-1)  # 每个 query 匹配到的 key 索引
            j_star = a.argmax(dim=-2)  # 每个 key 反向匹配到的 query 索引
            N_q = a.size(-2)
            q_idx = torch.arange(N_q, device=a.device).view(1, 1, N_q)
            j_at_i = torch.gather(j_star, -1, i_star)  # j_star[i_star]
            consistent_q = (j_at_i == q_idx).float()  # 闭合一致 (j[i]==i) 置 1
            mask = consistent_q.unsqueeze(-1).expand_as(attn)
        return mask

    def _cross_attn_direction(self, q_feat, kv_feat, q_proj, k_proj, v_proj, out_proj, gamma):
        B, C, H, W = q_feat.shape
        N_q = H * W
        q = q_feat.flatten(2).transpose(1, 2)  # query 来自源模态 (IR 或 VI)
        if N_q > self.max_kv_tokens:  # 超大分辨率时对 KV 下采样, 控制注意力复杂度
            target = int(self.max_kv_tokens ** 0.5)
            kv_ds = F.adaptive_avg_pool2d(kv_feat, (target, target))
        else:
            kv_ds = kv_feat
        Hk, Wk = kv_ds.shape[-2:]
        N_k = Hk * Wk
        kv = kv_ds.flatten(2).transpose(1, 2)
        q = q_proj(q).reshape(B, N_q, self.num_heads, self.qkv_head_dim).permute(0, 2, 1, 3)
        k = k_proj(kv).reshape(B, N_k, self.num_heads, self.qkv_head_dim).permute(0, 2, 1, 3)
        v = v_proj(kv).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        sim = (q @ k.transpose(-2, -1)) * self.scale
        bias = torch.zeros_like(sim)
        # 迭代式循环一致优化: 每轮抑制不一致位置 (mask=0) 的注意力, 强化一致对应
        for _ in range(self.num_iters):
            attn = (sim + bias).softmax(dim=-1)
            mask = self._cycle_consistent_mask(attn)
            bias = (-self.tau * (1.0 - mask)).to(sim.dtype)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N_q, C)
        out = self.proj_drop(out_proj(out))
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return q_feat + gamma.view(1, -1, 1, 1) * out  # 门控残差: 渐进注入跨模态信息

    def forward(self, feat_v, feat_r):
        # 双向循环一致交叉注意力: IR→VI 与 VI→IR 各自独立优化, 形成对称互补
        out_v = self._cross_attn_direction(
            feat_v, feat_r, self.q_a, self.k_a, self.v_a, self.proj_a, self.gamma_v
        )
        out_r = self._cross_attn_direction(
            feat_r, feat_v, self.q_b, self.k_b, self.v_b, self.proj_b, self.gamma_r
        )
        return out_v, out_r
