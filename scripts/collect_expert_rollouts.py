"""Stage II 的数据来源：让 Stage I 专家在物理仿真里跑，记录 (o^E, s^prop, a^E, 场景上下文)。

每个并行环境从随机片段/随机时刻开始（和训练一样的 RSI），策略**确定性**地（取均值）跑 horizon 步，
摔倒或参考播完就重置。记录每一步：
    obs_E   (…, obs_dim)   专家完整观测 = VQ 编码器输入
    prop    (…, PROP_DIM)  本体感知    = VQ 解码器 / 先验输入（obs_E 的前 PROP_DIM 维）
    action  (…, 29)        专家动作（均值）= 行为克隆目标
    clip_idx, ref_time     方便之后按片段切、对齐场景上下文
    done                   这一步之后是否重置（先验的 5 帧历史不能跨越它）
存成分块的 .npz（每块 ~ num_envs × horizon 步），外加 meta.json。

用法（Pod）::
    cd /data/chip-repro && PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/collect_expert_rollouts.py \\
        --task Chip-G1-Imitation-Walk-v0 --headless --num_envs 4096 --steps 2000 --out /data/rollouts/walk_v1
    # 4096 env × 2000 步 = 820 万条样本，约 1.5 GB（float16），A40 上 5–10 分钟

论文：4096 环境、horizon 32、按动作类别选对应专家（我们先只有走路专家，所以单模型）。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"), "scripts", "reinforcement_learning", "rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Chip-G1-Imitation-Walk-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=2000, help="每个环境走多少步")
parser.add_argument("--chunk", type=int, default=100, help="每多少步落一个 .npz 文件")
parser.add_argument("--out", required=True)
parser.add_argument("--noise-std", type=float, default=0.0, help="给专家动作加噪声（论文没加；留着做 DAgger 风格扩充）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import functools  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import chip.envs.isaaclab  # noqa: F401, E402
from chip.envs.isaaclab.imitation_env import PROP_DIM  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    ckpt = args.checkpoint or get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)), ".*", "model_.*.pt")
    print("[collect] 模型:", ckpt)

    env = gym.make(args.task, cfg=env_cfg)
    uenv = env.unwrapped
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=uenv.device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    venv.reset()
    obs = venv.get_observations()
    # rsl_rl 的 wrapper 返回 TensorDict（按观测组），取 "policy" 这一组的张量
    def as_tensor(o):
        return o["policy"] if hasattr(o, "keys") else o
    print(f"[collect] 环境就绪：{args.num_envs} env，obs {tuple(as_tensor(obs).shape)}，开始收集 {args.steps} 步")
    N = args.num_envs
    buf = {k: [] for k in ("obs", "action", "clip_idx", "ref_time", "done")}
    chunk_id, n_total, t0 = 0, 0, time.time()

    def flush():
        nonlocal chunk_id, n_total
        if not buf["obs"]:
            return
        arrs = {k: np.stack(v, axis=1) for k, v in buf.items()}          # (N, T, ...)
        np.savez(out / f"chunk_{chunk_id:04d}.npz",
                 obs=arrs["obs"].astype(np.float16), action=arrs["action"].astype(np.float16),
                 clip_idx=arrs["clip_idx"].astype(np.int32), ref_time=arrs["ref_time"].astype(np.float32),
                 done=arrs["done"].astype(bool))
        n_total += arrs["obs"].shape[0] * arrs["obs"].shape[1]
        print(f"[collect] chunk {chunk_id}: {arrs['obs'].shape[1]} 步 × {N} env，累计 {n_total:,} 样本，{time.time() - t0:.0f}s")
        chunk_id += 1
        for v in buf.values():
            v.clear()

    for t in range(args.steps):
        with torch.inference_mode():
            a = policy(obs)
            if args.noise_std > 0:
                a = a + args.noise_std * torch.randn_like(a)
            buf["obs"].append(as_tensor(obs).cpu().numpy())
            buf["action"].append(a.cpu().numpy())
            buf["clip_idx"].append(uenv.clip_idx.cpu().numpy())
            buf["ref_time"].append(uenv.ref_time.cpu().numpy())
            obs, _, dones, _ = venv.step(a)
            buf["done"].append(dones.bool().cpu().numpy())
        if (t + 1) % args.chunk == 0:
            flush()
    flush()

    meta = {
        "task": args.task, "checkpoint": str(ckpt), "num_envs": N, "steps": args.steps,
        "obs_dim": int(as_tensor(obs).shape[1]), "prop_dim": PROP_DIM, "action_dim": 29,
        "clip_ids": uenv.motion.clip_ids, "fps": uenv.motion.fps,
        "layout": "每个 chunk 的数组形状 (num_envs, T, ...)；done[i,t]=True 表示 t 之后重置，历史窗口不要跨过它",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"[collect] 完成：{n_total:,} 样本 → {out}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001  任何异常（含 SystemExit）都打出来，别被看门狗静默吞掉
        import traceback
        traceback.print_exc()
        print(f"[collect] 异常退出: {type(e).__name__}: {e}", flush=True)
    finally:
        sys.stdout.flush()
        import threading
        threading.Timer(20.0, lambda: os._exit(0)).start()
        app.close()
        os._exit(0)
