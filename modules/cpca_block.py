
import torch
import torch.nn as nn
import torch.nn.functional as F


class CPCABlock(nn.Module):

    def __init__(self, dim, num_heads=8, num_iters=5, qkv_dim_ratio=0.5,
                 max_kv_tokens=2048, attn_drop=0., proj_drop=0.,
                 early_stop=True, converge_tol=1e-3, score_high=0.95):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须能被 num_heads {num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_iters = max(num_iters, 1)
        self.max_kv_tokens = max_kv_tokens
        # 早停相关: 根据循环一致率自动判断是否继续迭代
        self.early_stop = early_stop
        self.converge_tol = converge_tol      # 连续两轮一致率变化 < tol 视为收敛
        self.score_high = score_high          # 一致率超过此阈值视为匹配已足够好
        qkv_dim = int(dim * qkv_dim_ratio)
        qkv_dim = max((qkv_dim // num_heads) * num_heads, num_heads)
        self.qkv_dim = qkv_dim
        self.qkv_head_dim = qkv_dim // num_heads
        # q/k 投影后做 L2 归一化, 使 sim = q·k / (|q||k|) ∈ [-1, 1],
        # 不受网络初始化尺度影响, 保证注意力有区分度
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
        # 余弦相似度的温度: sim = (q·k)/(|q||k|) / sqrt(qkv_head_dim), 范围 ±1/sqrt(head_dim)≈±0.35
        self.scale = self.qkv_head_dim ** -0.5
        # bias 幅值: 固定绝对值, 不依赖 sim_std (sim 归一化后尺度固定, 无需自适应)
        self.tau = nn.Parameter(torch.tensor(2.0))
        self.gamma_v = nn.Parameter(torch.full((dim,), 0.1))  # IR 分支门控残差, 渐进学习的初始权重
        self.gamma_r = nn.Parameter(torch.full((dim,), 0.1))  # VI 分支门控残差

    def _cycle_consistent_mask(self, attn):
        # 循环一致掩码: 一致的 top-1 匹配的 (query,key) 对 = 1, 其余 = 0
        with torch.no_grad():
            a = attn.float()
            i_star = a.argmax(dim=-1)          # (B, h, N_q) 每个 query 匹配到的 key 索引
            j_star = a.argmax(dim=-2)          # (B, h, N_k) 每个 key 反向匹配到的 query 索引
            N_q = a.size(-2)
            q_idx = torch.arange(N_q, device=a.device).view(1, 1, N_q)
            # 取 j_star 在 i_star 位置的值: 即 query i 的 top-1 key j, 再看 key j 的 top-1 query 是谁
            j_at_i = torch.gather(j_star, -1, i_star)  # (B,h,N_q), 因为 i_star 在 N_q 维上索引 j_star 的 N_k 维
            consistent_q = (j_at_i == q_idx).float()   # 闭合一致置 1
            score = consistent_q.mean()
            # 构建逐 (query,key) 的掩码: 只有 query i 与其 top-1 key j 配对且循环一致时, 该位置 = 1
            mask = torch.zeros_like(attn)
            mask.scatter_(-1, i_star.unsqueeze(-1), consistent_q.unsqueeze(-1))
        return mask, score

    def _cross_attn_direction(self, q_feat, kv_feat, q_proj, k_proj, v_proj, out_proj, gamma):
        B, C, H, W = q_feat.shape
        N_q = H * W
        # 目标: N_q * N_k <= max_kv_tokens, 同时尽量保留空间信息
        max_tokens = self.max_kv_tokens
        target_side = int(max_tokens ** 0.5)  # 默认 32 -> 32x32=1024

        q_2d = q_feat       # (B, C, H, W)
        kv_2d = kv_feat     # (B, C, Hk, Wk)

        # 对 Q 下采样: 若 H*W > target_side^2, 池化到 target_side x target_side
        if N_q > target_side * target_side:
            q_ds = F.adaptive_avg_pool2d(q_2d, (target_side, target_side))
        else:
            q_ds = q_2d
        # 对 KV 下采样: 同样控制
        if kv_2d.shape[-2] * kv_2d.shape[-1] > target_side * target_side:
            kv_ds = F.adaptive_avg_pool2d(kv_2d, (target_side, target_side))
        else:
            kv_ds = kv_2d

        Hq, Wq = q_ds.shape[-2:]
        Hk, Wk = kv_ds.shape[-2:]
        N_q_ds = Hq * Wq
        N_k = Hk * Wk
        q = q_ds.flatten(2).transpose(1, 2)   # (B, N_q_ds, C)
        kv = kv_ds.flatten(2).transpose(1, 2) # (B, N_k, C)
        q = q_proj(q).reshape(B, N_q_ds, self.num_heads, self.qkv_head_dim).permute(0, 2, 1, 3)
        k = k_proj(kv).reshape(B, N_k, self.num_heads, self.qkv_head_dim).permute(0, 2, 1, 3)
        v = v_proj(kv).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # L2 归一化 q/k, 使 sim = 余弦相似度 ∈ [-1, 1], 与网络初始化尺度无关
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        sim = (q @ k.transpose(-2, -1)) * self.scale
        del q, k
        bias = torch.zeros_like(sim)
        prev_attn = None
        for it in range(self.num_iters):
            attn = (sim + bias).softmax(dim=-1)
            mask, score = self._cycle_consistent_mask(attn)
            # 不一致位置 -tau, 固定幅值强制锐化
            bias = (self.tau * (2.0 * mask - 1.0)).to(sim.dtype)
            # 早停: 一致率足够高, 或 attn 变化极小
            if self.early_stop:
                if score.item() >= self.score_high:
                    break
                if prev_attn is not None:
                    attn_change = (attn - prev_attn).abs().max().item()
                    if it >= 2 and attn_change < self.converge_tol:
                        break
                prev_attn = attn.detach()
            del attn, mask
        attn = (sim + bias).softmax(dim=-1)
        del sim, bias, prev_attn
        attn = self.attn_drop(attn)
        out = attn @ v   # (B, h, N_q_ds, head_dim)
        del attn
        out = out.transpose(1, 2).reshape(B, N_q_ds, C)
        out = self.proj_drop(out_proj(out))
        out = out.transpose(1, 2).reshape(B, C, Hq, Wq)
        # 将下采样后的输出上采样回原分辨率, 与 q_feat 做门控残差
        if (Hq, Wq) != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
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
