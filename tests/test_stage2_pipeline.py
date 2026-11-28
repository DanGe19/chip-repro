"""用假的 rollout 数据跑通 Stage II 整条链：数据集 → VQ-VAE → token → 先验。"""
import json
import sys

import numpy as np
import pytest
import torch

from chip.prior.data import RolloutDataset

PROP, OBS, ACT = 12, 20, 6


def make_rollouts(root, num_envs=8, T=60, chunk=20, seed=0):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    # 合成"专家"：观测是平滑随机游走，动作 = tanh(线性函数(obs))；每 15–25 步随机重置一次
    W = rng.normal(size=(OBS, ACT)) * 0.5
    obs = np.zeros((num_envs, T, OBS), np.float32)
    done = np.zeros((num_envs, T), bool)
    x = rng.normal(size=(num_envs, OBS))
    for t in range(T):
        x = 0.9 * x + 0.1 * rng.normal(size=(num_envs, OBS))
        obs[:, t] = x
        reset = rng.random(num_envs) < 0.06
        done[:, t] = reset
        x[reset] = rng.normal(size=(reset.sum(), OBS))
    act = np.tanh(obs @ W)
    for c, s in enumerate(range(0, T, chunk)):
        sl = slice(s, s + chunk)
        np.savez(root / f"chunk_{c:04d}.npz", obs=obs[:, sl].astype(np.float16), action=act[:, sl].astype(np.float16),
                 clip_idx=np.zeros((num_envs, chunk), np.int32), ref_time=np.tile(np.arange(chunk, dtype=np.float32) * 0.02, (num_envs, 1)),
                 done=done[:, sl])
    (root / "meta.json").write_text(json.dumps({"obs_dim": OBS, "prop_dim": PROP, "action_dim": ACT, "num_envs": num_envs, "steps": T}))
    return obs, act, done


def test_dataset_windows_do_not_cross_resets(tmp_path):
    obs, act, done = make_rollouts(tmp_path / "r")
    ds = RolloutDataset(tmp_path / "r", history=5)
    assert ds.N == 8 and ds.T == 60 and len(ds) == 480
    o, p, a = ds.sample_vq(16)
    assert o.shape == (16, OBS) and p.shape == (16, PROP) and a.shape == (16, ACT)
    assert torch.allclose(o[:, :PROP], p)
    # 所有有效窗口 [t-4, t-1] 内都没有 done
    for n, t in ds.valid_prior.tolist():
        assert t >= 4 and not done[n, t - 4 : t].any()
    # 反过来：随便挑一个跨 done 的窗口，不该在 valid 里
    n0, t0 = np.argwhere(done)[0]
    if t0 + 2 < 60:
        bad = (int(n0), int(t0) + 2)
        assert bad not in set(map(tuple, ds.valid_prior.tolist()))


@pytest.mark.slow
def test_full_stage2_pipeline_on_fake_data(tmp_path):
    make_rollouts(tmp_path / "r", num_envs=16, T=100, chunk=25)
    sys.path.insert(0, "scripts")
    import train_stage2

    out = tmp_path / "out"
    train_stage2.main(["--data", str(tmp_path / "r"), "--out", str(out), "--vq-steps", "300", "--prior-steps", "150",
                       "--batch", "256", "--lr", "2e-3", "--codebook-size", "16", "--code-dim", "8", "--device", "cpu"])
    assert (out / "vqvae.pt").exists() and (out / "prior.pt").exists()
    stats = json.loads((out / "token_stats.json").read_text())
    assert stats["used_codes"] >= 2
    # 加载回来能用：解码 token 得到动作形状
    from chip.prior.vqvae import SkillVQVAE, VQVAEConfig
    ck = torch.load(out / "vqvae.pt")
    vq = SkillVQVAE(VQVAEConfig(**ck["cfg"])); vq.load_state_dict(ck["state_dict"]); vq.eval()
    ds = RolloutDataset(tmp_path / "r")
    o, p, a = ds.sample_vq(32)
    tok = vq.tokenize(o)
    assert vq.decode_token(p, tok).shape == (32, ACT)
