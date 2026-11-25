"""IsaacLab 相关：回放、Stage I 环境、训练配置。只能在 IsaacLab 的 python 里真正运行。

import 本包会把环境注册到 gymnasium（纯字符串入口，不会在这里 import isaaclab）：
    Chip-G1-Imitation-v0        全部 3179 条参考片段
    Chip-G1-Imitation-Walk-v0   调试用：一条 TRUMANS 走路片段（见 WALK_CLIP）
"""

try:
    import gymnasium as gym
except ImportError:  # 本地没装 gymnasium 时（跑单元测试）跳过注册
    gym = None

WALK_CLIP = "trumans/2023-02-20@21-32-45@001762"   # 9.8 s 走了 3.3 m（按位移挑出来的走路片段）

if gym is not None:
    gym.register(
        id="Chip-G1-Imitation-v0",
        entry_point=f"{__name__}.imitation_env:G1ImitationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.imitation_env:G1ImitationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
        },
    )
    gym.register(
        id="Chip-G1-Imitation-Walk-v0",
        entry_point=f"{__name__}.imitation_env:G1ImitationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.imitation_env_cfgs:G1ImitationWalkEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1ImitationPPORunnerCfg",
        },
    )
