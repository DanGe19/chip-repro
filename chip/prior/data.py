"""读取 collect_expert_rollouts.py 的输出，组装 VQ-VAE / 先验的训练批次。

RolloutDataset
    把所有 chunk 拼成 (num_envs, T_total, ...) 的大数组（float16 省内存，取批次时转 float32）。
    sample_vq(B)     → obs_E, prop, action                      给 VQ-VAE
    sample_prior(B)  → prop_hist (B,H,P), obs_E(当前帧), ...     给先验；历史窗口不跨越 done
    for_prior_targets(vqvae) 预先把每一帧的 teacher token 算好存起来，先验训练时直接查表。

上下文（高度图 / 目标等）这里先不接：走路专家 + TRUMANS 的第一版先验只用本体历史。
接场景上下文时在 sample_prior 里按 (clip_idx, ref_time) 去 chip.envs.geometry 查高度图即可。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class RolloutDataset:
    def __init__(self, root: str | Path, history: int = 5, device: str | torch.device = "cpu", max_chunks: int | None = None):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        files = sorted(root.glob("chunk_*.npz"))
        if max_chunks:
            files = files[:max_chunks]
        if not files:
            raise FileNotFoundError(f"{root} 下没有 chunk_*.npz")
        obs, act, done, clip, rt = [], [], [], [], []
        for f in files:
            d = np.load(f)
            obs.append(d["obs"]); act.append(d["action"]); done.append(d["done"]); clip.append(d["clip_idx"]); rt.append(d["ref_time"])
        self.obs = torch.as_tensor(np.concatenate(obs, 1))          # (N, T, O) float16
        self.action = torch.as_tensor(np.concatenate(act, 1))       # (N, T, A) float16
        self.done = torch.as_tensor(np.concatenate(done, 1))        # (N, T) bool
        self.clip_idx = torch.as_tensor(np.concatenate(clip, 1))
        self.ref_time = torch.as_tensor(np.concatenate(rt, 1))
        self.N, self.T = self.done.shape
        self.H = history
        self.prop_dim = int(self.meta["prop_dim"])
        self.device = torch.device(device)
        self.tokens: torch.Tensor | None = None

        # 先验用：哪些 (n, t) 的 5 帧历史 [t-H+1, t] 不跨越 done。
        # done[n, s] = True 表示 s 之后重置，所以窗口内 s ∈ [t-H+1, t-1] 不能有 done。
        d = self.done.float()
        cs = torch.cumsum(torch.nn.functional.pad(d, (1, 0)), dim=1)        # cs[:, t] = done[:, :t].sum()
        t_idx = torch.arange(self.T)
        lo = (t_idx - self.H + 1).clamp(min=0)
        n_done_in_window = cs[:, t_idx] - cs[:, lo]                           # done 在 [lo, t-1] 内的个数
        valid = (t_idx >= self.H - 1)[None, :] & (n_done_in_window == 0)
        self.valid_prior = valid.nonzero(as_tuple=False)                      # (M, 2) 的 (n, t)

    def __len__(self):
        return self.N * self.T

    # ---- VQ-VAE 批次 ----
    def sample_vq(self, batch: int, generator: torch.Generator | None = None):
        n = torch.randint(0, self.N, (batch,), generator=generator)
        t = torch.randint(0, self.T, (batch,), generator=generator)
        o = self.obs[n, t].float().to(self.device)
        return o, o[:, : self.prop_dim], self.action[n, t].float().to(self.device)

    # ---- 先验批次 ----
    @torch.no_grad()
    def for_prior_targets(self, vqvae, batch: int = 65536):
        """用训好的 VQ-VAE 给每一帧打 teacher token，存到 self.tokens (N, T)。"""
        vqvae.eval()
        flat = self.obs.reshape(-1, self.obs.shape[-1])
        toks = torch.empty(flat.shape[0], dtype=torch.long)
        for i in range(0, flat.shape[0], batch):
            toks[i : i + batch] = vqvae.tokenize(flat[i : i + batch].float().to(self.device)).cpu()
        self.tokens = toks.view(self.N, self.T)
        return self.tokens

    def sample_prior(self, batch: int, generator: torch.Generator | None = None):
        assert self.tokens is not None, "先调用 for_prior_targets(vqvae)"
        pick = torch.randint(0, self.valid_prior.shape[0], (batch,), generator=generator)
        n, t = self.valid_prior[pick].unbind(1)
        offs = torch.arange(-self.H + 1, 1)
        tt = t[:, None] + offs[None, :]                                        # (B, H)
        hist = self.obs[n[:, None], tt][..., : self.prop_dim].float().to(self.device)   # (B, H, P)
        return {
            "prop_hist": hist,
            "token": self.tokens[n, t].to(self.device),
            "clip_idx": self.clip_idx[n, t].to(self.device),
            "ref_time": self.ref_time[n, t].to(self.device),
        }

    def summary(self) -> str:
        return (f"RolloutDataset: {self.N} env × {self.T} 步 = {len(self):,} 样本, obs {self.obs.shape[-1]} (prop {self.prop_dim}), "
                f"先验可用窗口 {self.valid_prior.shape[0]:,}, 重置 {int(self.done.sum()):,} 次")
