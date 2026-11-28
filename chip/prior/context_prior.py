"""Stage II 第二步：上下文条件技能先验 p_θ(q_t | s^prop_{t-H:t}, U_t)（论文 §3.4，附录 B.3，表 6）。

输入：
    prop_hist  (N, H=5, P)   最近 5 帧本体感知
    contexts   dict[name -> (N, D_m)]   各模态的原始信号（高度图 16×16、BPS 512、关节到物体距离 96、目标 10 维、文本 512 等）
    avail      dict[name -> (N,) bool]  每个模态在这条数据里有没有（TRUMANS 没有物体，OMOMO 没有场景）
输出：
    logits (N, K=1024)   对 1024 个技能 token 的分布

结构：每帧本体 → 一个 token；每个模态 → 一个 token（模态各自的 MLP tokenizer）；再加一个可学的 [CLS]；
      一起送进 4 层 Transformer encoder（d=512, 8 头, FFN 1536, GELU）；[CLS] 的输出接线性层出 logits。
      缺失的模态用 key_padding_mask 遮掉（不是补零——补零会让模型以为"有一个全零的场景"）。
训练：交叉熵 vs. VQ-VAE 给的 teacher token；训练时每个可用模态以 p=0.3 独立丢弃（modality masking），
      这样下游缺模态或者想做 classifier-free guidance（Stage III 的 CFG）时模型都见过。

和下游的关系：Stage III 用它初始化任务策略并做 KL 正则；我们的 MPPI 用它当"动作词典上的先验"，
在它给的分布里采样 token，再用 VQ 解码器变成动作。

纯 torch，本地可测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PriorConfig:
    prop_dim: int                                   # 单帧本体维度
    history: int = 5                                # H
    context_dims: dict[str, int] = field(default_factory=lambda: {"height": 256, "bps": 512, "prox": 96, "goal_human": 10, "goal_object": 3, "text": 512})
    codebook_size: int = 1024
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 1536
    dropout: float = 0.0
    tokenizer_hidden: int = 512
    modality_mask_prob: float = 0.3                 # 训练时每个可用模态被丢弃的概率


class ModalityTokenizer(nn.Module):
    """Tok_m：一个模态的原始向量 → d_model 的 token。"""

    def __init__(self, in_dim: int, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x):
        return self.net(x)


class ContextPrior(nn.Module):
    def __init__(self, cfg: PriorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.modalities = list(cfg.context_dims.keys())
        # 序列布局：[CLS] + H 帧本体 + M 个模态
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.prop_tok = ModalityTokenizer(cfg.prop_dim, d, cfg.tokenizer_hidden)
        self.prop_pos = nn.Parameter(torch.zeros(1, cfg.history, d))            # 本体帧的位置编码
        self.ctx_tok = nn.ModuleDict({m: ModalityTokenizer(n, d, cfg.tokenizer_hidden) for m, n in cfg.context_dims.items()})
        self.ctx_type = nn.Parameter(torch.zeros(1, len(self.modalities), d))   # 模态类型编码
        layer = nn.TransformerEncoderLayer(d, cfg.num_heads, cfg.ffn_dim, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.num_layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.codebook_size)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.prop_pos, std=0.02)
        nn.init.normal_(self.ctx_type, std=0.02)

    # ------------------------------------------------------------------
    def _build_sequence(self, prop_hist: torch.Tensor, contexts: dict[str, torch.Tensor],
                        avail: dict[str, torch.Tensor] | None, drop_modalities: bool):
        N, H, P = prop_hist.shape
        assert H == self.cfg.history, f"history 长度应为 {self.cfg.history}，收到 {H}"
        dev = prop_hist.device
        toks = [self.cls.expand(N, -1, -1), self.prop_tok(prop_hist) + self.prop_pos]
        pad = [torch.zeros(N, 1 + H, dtype=torch.bool, device=dev)]        # False = 保留
        for j, m in enumerate(self.modalities):
            if m in contexts:
                x = contexts[m]
                a = avail[m].bool() if (avail is not None and m in avail) else torch.ones(N, dtype=torch.bool, device=dev)
            else:
                x = torch.zeros(N, self.cfg.context_dims[m], device=dev)
                a = torch.zeros(N, dtype=torch.bool, device=dev)
            if drop_modalities and self.cfg.modality_mask_prob > 0:
                keep = torch.rand(N, device=dev) >= self.cfg.modality_mask_prob
                a = a & keep
            toks.append((self.ctx_tok[m](x) + self.ctx_type[:, j]).unsqueeze(1))
            pad.append(~a.unsqueeze(1))
        return torch.cat(toks, dim=1), torch.cat(pad, dim=1)

    def forward(self, prop_hist: torch.Tensor, contexts: dict[str, torch.Tensor] | None = None,
                avail: dict[str, torch.Tensor] | None = None, drop_modalities: bool | None = None) -> torch.Tensor:
        """返回 logits (N, K)。drop_modalities 默认 = self.training。"""
        contexts = contexts or {}
        if drop_modalities is None:
            drop_modalities = self.training
        seq, pad = self._build_sequence(prop_hist, contexts, avail, drop_modalities)
        h = self.encoder(seq, src_key_padding_mask=pad)
        return self.head(self.norm(h[:, 0]))

    def loss(self, prop_hist, contexts, avail, teacher_token: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self(prop_hist, contexts, avail)
        ce = F.cross_entropy(logits, teacher_token)
        with torch.no_grad():
            acc = (logits.argmax(-1) == teacher_token).float().mean()
            top5 = (logits.topk(5, dim=-1).indices == teacher_token[:, None]).any(-1).float().mean()
        return ce, {"ce": float(ce.detach()), "acc": float(acc), "top5": float(top5)}

    @torch.no_grad()
    def sample(self, prop_hist, contexts=None, avail=None, temperature: float = 1.0, num_samples: int = 1) -> torch.Tensor:
        """按先验采样 token：(N, num_samples)。MPPI 的候选来源。"""
        logits = self(prop_hist, contexts, avail, drop_modalities=False) / max(temperature, 1e-6)
        p = logits.softmax(-1)
        return torch.multinomial(p, num_samples, replacement=True)

    @torch.no_grad()
    def guided_logits(self, prop_hist, contexts, avail, guidance_scale: float = 1.5) -> torch.Tensor:
        """Classifier-free guidance（论文 Stage III 用的 w=1.5）：
        logits = (1-w)·无条件 + w·有条件。无条件 = 把所有模态都遮掉。"""
        cond = self(prop_hist, contexts, avail, drop_modalities=False)
        none = {m: torch.zeros_like(a) for m, a in (avail or {}).items()} if avail else {m: torch.zeros(prop_hist.shape[0], dtype=torch.bool, device=prop_hist.device) for m in contexts or {}}
        uncond = self(prop_hist, contexts, none, drop_modalities=False)
        return (1 - guidance_scale) * uncond + guidance_scale * cond
