"""环境冒烟测试：建环境、随机动作跑几百步，打印观测形状、奖励各项、终止率。不训练，1–2 分钟。

    cd /data/chip-repro && /workspace/isaaclab/isaaclab.sh -p scripts/smoke_env.py --task Chip-G1-Imitation-Walk-v0 --num_envs 64 --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Chip-G1-Imitation-Walk-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--zero-actions", action="store_true", help="动作全 0（= 保持默认姿态），看 PD 能不能站住")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import chip.envs.isaaclab  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg)
obs, _ = env.reset()
print("obs shape:", obs["policy"].shape, " action dim:", env.action_space.shape)
uenv = env.unwrapped
rew_sum = torch.zeros(args.num_envs, device=uenv.device)
n_done = 0
for i in range(args.steps):
    a = torch.zeros(args.num_envs, 29, device=uenv.device) if args.zero_actions else 0.3 * torch.randn(args.num_envs, 29, device=uenv.device)
    obs, rew, term, trunc, info = env.step(a)
    rew_sum += rew
    n_done += int((term | trunc).sum())
    if i % 50 == 0:
        print(f"step {i:4d}  reward 均值 {rew.mean().item():.4f}  终止 {int(term.sum())}  超时 {int(trunc.sum())}  "
              f"骨盆高度均值 {uenv.robot.data.root_pos_w[:, 2].mean().item():.3f}")
print("累计 episode 结束次数:", n_done, " 平均每步奖励:", (rew_sum.sum() / (args.steps * args.num_envs)).item())
if "log" in uenv.extras:
    for k, v in sorted(uenv.extras["log"].items()):
        print(f"  {k}: {float(v):.4f}")
env.close()
app.close()
