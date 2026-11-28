"""用训练好的 Stage I 策略在物理仿真里跑一遍，把机器人轨迹存成 RobotMotion，方便用 MuJoCo 渲染对比。

RunPod 容器里 Isaac Sim 不能录像，所以走这条路：IsaacLab 物理仿真 → 记录关节角/根部位姿 → MuJoCo 画图。

    cd /data/chip-repro && PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/rollout_stage1.py \\
        --task Chip-G1-Imitation-Walk-v0 --headless            # 默认用 logs/ 下最新一次训练的最新模型
    # 然后渲染：
    MUJOCO_GL=egl /workspace/isaaclab/isaaclab.sh -p -m chip.retarget.inspect_robot_motion \\
        --motion /data/inspect/rollouts/<名字>.npz --gmr-root /data/GMR --out /data/inspect/rollouts/<名字>

为了看清楚策略的真实水平，录像时关掉提前终止（摔了也继续录），并且每次都从参考片段开头开始。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"), "scripts", "reinforcement_learning", "rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Chip-G1-Imitation-Walk-v0")
parser.add_argument("--checkpoint", default=None, help="model_*.pt 路径；默认 logs/rsl_rl/<experiment>/ 下最新的")
parser.add_argument("--num_envs", type=int, default=4, help="同时跑几个（各自独立，存成 _env0.._envN）")
parser.add_argument("--out", default="/data/inspect/rollouts")
parser.add_argument("--keep-termination", action="store_true", help="保留提前终止（默认关掉，摔倒也继续录）")
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
from chip.retarget.robot_motion import G1_DOF_NAMES, RobotMotion  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.reset_time_mode = "start"
    if not args.keep_termination:
        env_cfg.early_termination = False
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    ckpt = args.checkpoint or get_checkpoint_path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)), ".*", "model_.*.pt")
    print("[rollout] 模型:", ckpt)

    env = gym.make(args.task, cfg=env_cfg)
    uenv = env.unwrapped
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=uenv.device)

    # IsaacLab 受控关节顺序 → GMR/RobotMotion 顺序
    to_gmr = torch.tensor([uenv.ctrl_names.index(n) for n in G1_DOF_NAMES], device=uenv.device)
    origins = uenv.scene.env_origins

    venv.reset()
    obs = venv.get_observations()
    N = args.num_envs
    # 只记录到参考播完为止（参考播完环境会超时重置，再记就把"瞬移回起点"也记进去了）
    T_max = int(uenv.motion.durations.max().item() / uenv.step_dt) - 1
    rp, rq, dq = [], [], []
    alive = torch.ones(N, dtype=torch.bool, device=uenv.device)
    fell_at = torch.full((N,), -1, dtype=torch.long, device=uenv.device)
    for t in range(T_max):
        with torch.inference_mode():
            obs, _, dones, _ = venv.step(policy(obs))
        d = uenv.robot.data
        rp.append((d.root_pos_w - origins).cpu().numpy())
        q = d.root_quat_w
        rq.append(torch.cat([q[:, 1:], q[:, :1]], dim=-1).cpu().numpy())
        dq.append(d.joint_pos[:, uenv.ctrl_ids][:, to_gmr].cpu().numpy())
        low = (d.root_pos_w[:, 2] < 0.4) & (fell_at < 0)
        fell_at[low] = t
        if dones.any():
            alive &= ~dones.bool()
        if not alive.any():
            break

    rp, rq, dq = np.stack(rp, 1), np.stack(rq, 1), np.stack(dq, 1)   # (N,T,·)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = Path(ckpt).parent.name + "_" + Path(ckpt).stem
    clip_name = uenv.motion.clip_ids[0].replace("/", "__")
    for i in range(N):
        T = rp.shape[1]
        rm = RobotMotion(clip_id=f"rollout/{run}/{clip_name}_env{i}", source="rollout", fps=1.0 / uenv.step_dt,
                         root_pos=rp[i].astype(np.float32), root_rot=rq[i].astype(np.float32), dof_pos=dq[i].astype(np.float32))
        p = out / f"{run}__{clip_name}_env{i}.npz"
        rm.save(p)
        fa = int(fell_at[i])
        print(f"[rollout] env{i}: {T} 帧 ({T * uenv.step_dt:.1f} s)，" + (f"第 {fa * uenv.step_dt:.1f} s 骨盆低于 0.4 m（摔倒）" if fa >= 0 else "全程没摔") + f" → {p}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        import threading
        threading.Timer(20.0, lambda: os._exit(0)).start()
        app.close()
        os._exit(0)
