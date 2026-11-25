"""环境配置的几个变体。"""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from chip.envs.isaaclab import WALK_CLIP
from chip.envs.isaaclab.imitation_env import G1ImitationEnvCfg


@configclass
class G1ImitationWalkEnvCfg(G1ImitationEnvCfg):
    """调试/冒烟：只用一条走路片段，1024 个环境。"""
    motion_clip_ids = (WALK_CLIP,)
    # configclass 的字段要整个重新声明，不能 .replace() 父类的
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    # 第 2 轮（2026-11-24）：第 1 轮 500 iter×1024 env 局长只到 2.4 s、从头回放 1 s 内扑倒。
    # 抖动惩罚占了总分的 ~15%，先降到 1/10；终止阈值放宽到 1 m，给前期更多试错空间。
    action_rate_scale = 0.1
    termination_pos_thr = 1.0


@configclass
class G1ImitationOmomoEnvCfg(G1ImitationEnvCfg):
    motion_sources = ("omomo",)


@configclass
class G1ImitationTrumansEnvCfg(G1ImitationEnvCfg):
    motion_sources = ("trumans",)
