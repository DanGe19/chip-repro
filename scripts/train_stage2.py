"""Stage II 训练：VQ-VAE（技能 token 化）→ 上下文先验（预测 token）。纯 torch，不需要 IsaacLab。

    # Pod 上（GPU）：
    cd /data/chip-repro && /workspace/isaaclab/isaaclab.sh -p scripts/train_stage2.py \\
        --data /data/rollouts/walk_v1 --out /data/stage2/walk_v1 --vq-steps 20000 --prior-steps 20000
    # 看曲线：tensorboard --logdir /data/stage2

两个阶段都按论文表 6：AdamW 2e-4，minibatch 8192，grad clip 1.0。
第一版先验只用本体历史（没有场景/目标上下文）；接上下文时在 RolloutDataset.sample_prior 里加。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from chip.prior.context_prior import ContextPrior, PriorConfig
from chip.prior.data import RolloutDataset
from chip.prior.vqvae import SkillVQVAE, VQVAEConfig


def train_vq(ds: RolloutDataset, out: Path, steps: int, batch: int, lr: float, device, log, codebook_size=1024, code_dim=64):
    cfg = VQVAEConfig(obs_dim=ds.obs.shape[-1], prop_dim=ds.prop_dim, action_dim=ds.action.shape[-1],
                      codebook_size=codebook_size, code_dim=code_dim)
    model = SkillVQVAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    for it in range(1, steps + 1):
        obs, prop, act = ds.sample_vq(batch)
        loss, info = model.loss(obs, prop, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 100 == 0 or it == 1:
            for k, v in info.items():
                log.add_scalar(f"vq/{k}", v, it)
            if it % 1000 == 0 or it == 1:
                print(f"[vq {it}/{steps}] nll {info['nll']:.3f} mse {info['action_mse']:.4f} commit {info['commit']:.3f} "
                      f"perp {info['perplexity']:.0f} active {info['active_codes']}  {time.time() - t0:.0f}s", flush=True)
        if it % 5000 == 0 or it == steps:
            torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict()}, out / "vqvae.pt")
    model.eval()
    return model


def train_prior(ds: RolloutDataset, out: Path, steps: int, batch: int, lr: float, device, log, codebook_size=1024, history=5):
    cfg = PriorConfig(prop_dim=ds.prop_dim, history=history, context_dims={}, codebook_size=codebook_size)
    model = ContextPrior(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    for it in range(1, steps + 1):
        b = ds.sample_prior(batch)
        loss, info = model.loss(b["prop_hist"], {}, {}, b["token"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % 100 == 0 or it == 1:
            for k, v in info.items():
                log.add_scalar(f"prior/{k}", v, it)
            if it % 1000 == 0 or it == 1:
                print(f"[prior {it}/{steps}] ce {info['ce']:.3f} acc {info['acc']:.3f} top5 {info['top5']:.3f}  {time.time() - t0:.0f}s", flush=True)
        if it % 5000 == 0 or it == steps:
            torch.save({"cfg": {**cfg.__dict__}, "state_dict": model.state_dict()}, out / "prior.pt")
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vq-steps", type=int, default=20000)
    ap.add_argument("--prior-steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--codebook-size", type=int, default=1024)
    ap.add_argument("--code-dim", type=int, default=64)
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--max-chunks", type=int, default=None, help="调试：只读前几个 chunk")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-vq", action="store_true", help="已有 out/vqvae.pt 时跳过 VQ 训练")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter

    log = SummaryWriter(str(out / "tb"))
    ds = RolloutDataset(a.data, history=a.history, device=a.device, max_chunks=a.max_chunks)
    print(ds.summary(), flush=True)

    if a.skip_vq and (out / "vqvae.pt").exists():
        ck = torch.load(out / "vqvae.pt", map_location=a.device)
        vq = SkillVQVAE(VQVAEConfig(**ck["cfg"])).to(a.device)
        vq.load_state_dict(ck["state_dict"]); vq.eval()
        print("[vq] 已加载", out / "vqvae.pt", flush=True)
    else:
        vq = train_vq(ds, out, a.vq_steps, a.batch, a.lr, a.device, log, a.codebook_size, a.code_dim)

    toks = ds.for_prior_targets(vq)
    hist = torch.bincount(toks.flatten(), minlength=a.codebook_size).float()
    used = int((hist > 0).sum())
    print(f"[tokens] 用到 {used}/{a.codebook_size} 个码字，最常见的 10 个占 {hist.topk(10).values.sum() / hist.sum():.1%}", flush=True)
    (out / "token_stats.json").write_text(json.dumps({"used_codes": used, "hist": hist.tolist()}))

    train_prior(ds, out, a.prior_steps, a.batch, a.lr, a.device, log, a.codebook_size, a.history)
    log.close()
    print("[done]", out, flush=True)


if __name__ == "__main__":
    main()
