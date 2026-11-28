"""Stage II 第一步：把专家的"动作"压缩成离散技能 token 的 VQ-VAE（论文 §3.4，附录表 6）。

三个部件：
    Encoder  E_η : o^E_t (专家的模仿观测：本体 + 未来参考)  → z_t ∈ R^64
    Quantizer Q  : z_t → 最近的码本向量 z^(q_t)，q_t ∈ {1..K}，K=1024；码本用 EMA 更新
    Decoder  D_ψ : (s^prop_t, z^(q_t)) → 动作分布（高斯，输出 PD 目标）
损失：L = NLL(a^E_t | D) + β·‖z_t − sg[z^(q_t)]‖²，β=0.25；码本不吃梯度，走 EMA。
梯度用直通估计 z̃ = z + sg[z_q − z] 穿过量化器回到编码器。

为什么要这样做（给自己的备忘）：
- 训练数据是 Stage I 专家在仿真里跑出来的 (o^E, s^prop, a^E) 三元组，不是人体数据。专家的动作已经是
  物理上可行的，所以学出来的 token 都是"能做出来"的动作。
- 解码器只看 s^prop 和 token，不看未来参考。这样下游（Stage III / MPPI）只要挑 token，不需要参考动作。
- K=1024 个 token、每个 64 维，是论文的设置；先验（下一步的 Transformer）输出的就是这 1024 类上的分布。

这个文件只有 torch，没有 IsaacLab 依赖，本地可测。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: list[int], out_dim: int, activation: str = "relu") -> nn.Sequential:
    act = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU, "silu": nn.SiLU}[activation]
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@dataclass
class VQVAEConfig:
    obs_dim: int                      # o^E 维度（编码器输入）
    prop_dim: int                     # s^prop 维度（解码器输入的一部分）
    action_dim: int = 29
    codebook_size: int = 1024
    code_dim: int = 64
    encoder_hidden: list[int] = field(default_factory=lambda: [1024, 1024, 768, 512])
    decoder_hidden: list[int] = field(default_factory=lambda: [1024, 1024, 768, 512])
    activation: str = "relu"
    ema_decay: float = 0.99
    commitment_beta: float = 0.25
    ema_eps: float = 1e-5
    dead_code_threshold: float = 1.0   # EMA 计数低于此的码字视为"死码"，用随机编码器输出重置
    learn_log_std: bool = True         # 解码器高斯的 log σ 是否可学（与状态无关的参数）
    init_log_std: float = -1.0


class EMACodebook(nn.Module):
    """VQ 码本，EMA 更新（Van den Oord 2017 附录 / Razavi 2019）。不参与反向传播。"""

    def __init__(self, size: int, dim: int, decay: float = 0.99, eps: float = 1e-5, dead_threshold: float = 1.0):
        super().__init__()
        self.size, self.dim, self.decay, self.eps, self.dead_threshold = size, dim, decay, eps, dead_threshold
        embed = torch.randn(size, dim) * 0.1
        self.register_buffer("embed", embed)                        # (K, D)
        self.register_buffer("ema_count", torch.ones(size))         # N_k
        self.register_buffer("ema_sum", embed.clone())              # m_k
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _init_from(self, z: torch.Tensor):
        """第一批数据到来时用它的随机样本初始化码本，比纯随机收敛快得多。"""
        n = z.shape[0]
        idx = torch.randint(0, n, (self.size,), device=z.device)
        self.embed.copy_(z[idx])
        self.ema_sum.copy_(self.embed)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z (N, D) → (z_q (N, D), idx (N,))。训练模式下顺便做 EMA 更新。"""
        if self.training and not bool(self.initialized):
            self._init_from(z.detach())
        # 最近邻：‖z‖² − 2 z·e + ‖e‖²
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.embed.t() + self.embed.pow(2).sum(1)[None, :])
        idx = d.argmin(dim=1)
        z_q = F.embedding(idx, self.embed)
        if self.training:
            self._ema_update(z.detach(), idx)
        return z_q, idx

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, idx: torch.Tensor):
        onehot = F.one_hot(idx, self.size).type_as(z)               # (N, K)
        count = onehot.sum(0)                                       # (K,)
        summ = onehot.t() @ z                                       # (K, D)
        self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(summ, alpha=1 - self.decay)
        # Laplace 平滑，避免除零
        n = self.ema_count.sum()
        count_s = (self.ema_count + self.eps) / (n + self.size * self.eps) * n
        self.embed.copy_(self.ema_sum / count_s[:, None])
        # 死码重置：长期没人用的码字换成当前批次里的随机 z
        dead = self.ema_count < self.dead_threshold
        if dead.any():
            k = int(dead.sum())
            repl = z[torch.randint(0, z.shape[0], (k,), device=z.device)]
            self.embed[dead] = repl
            self.ema_sum[dead] = repl
            self.ema_count[dead] = 1.0

    @torch.no_grad()
    def usage(self) -> dict[str, float]:
        p = self.ema_count / self.ema_count.sum()
        perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum())
        return {"perplexity": float(perplexity), "active_codes": int((self.ema_count > self.dead_threshold).sum())}


class SkillVQVAE(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = mlp(cfg.obs_dim, cfg.encoder_hidden, cfg.code_dim, cfg.activation)
        self.codebook = EMACodebook(cfg.codebook_size, cfg.code_dim, cfg.ema_decay, cfg.ema_eps, cfg.dead_code_threshold)
        self.decoder = mlp(cfg.prop_dim + cfg.code_dim, cfg.decoder_hidden, cfg.action_dim, cfg.activation)
        if cfg.learn_log_std:
            self.log_std = nn.Parameter(torch.full((cfg.action_dim,), cfg.init_log_std))
        else:
            self.register_buffer("log_std", torch.full((cfg.action_dim,), cfg.init_log_std))

    # ---- 各部件 ----
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (z_st 直通后的向量, z_q 码本向量, idx)。z_st 用于解码器，让梯度回到编码器。"""
        z_q, idx = self.codebook(z)
        z_st = z + (z_q - z).detach()
        return z_st, z_q, idx

    def decode_mean(self, prop: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([prop, code], dim=-1))

    def decode_token(self, prop: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """下游用：给 token 下标直接出动作均值（确定性）。"""
        return self.decode_mean(prop, F.embedding(token, self.codebook.embed))

    # ---- 训练 ----
    def forward(self, obs: torch.Tensor, prop: torch.Tensor):
        z = self.encode(obs)
        z_st, z_q, idx = self.quantize(z)
        mean = self.decode_mean(prop, z_st)
        return {"z": z, "z_q": z_q, "idx": idx, "mean": mean}

    def loss(self, obs: torch.Tensor, prop: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        out = self(obs, prop)
        std = self.log_std.exp()
        nll = (0.5 * ((action - out["mean"]) / std).pow(2) + self.log_std + 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(-1).mean()
        commit = F.mse_loss(out["z"], out["z_q"].detach(), reduction="none").sum(-1).mean()
        total = nll + self.cfg.commitment_beta * commit
        with torch.no_grad():
            mse = F.mse_loss(out["mean"], action)
            u = self.codebook.usage()
        return total, {"nll": float(nll.detach()), "commit": float(commit.detach()), "action_mse": float(mse), **u}

    @torch.no_grad()
    def tokenize(self, obs: torch.Tensor) -> torch.Tensor:
        """给专家观测，返回 token 下标（先验训练的标签）。会临时切到 eval，避免 EMA 更新。"""
        was_training = self.training
        self.eval()
        try:
            _, idx = self.codebook(self.encode(obs))
        finally:
            self.train(was_training)
        return idx
