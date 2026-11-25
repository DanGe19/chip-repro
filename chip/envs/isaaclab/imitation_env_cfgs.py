"""环境配置的几个变体。"""

from isaaclab.utils import configclass

from chip.envs.isaaclab import WALK_CLIP
from chip.envs.isaaclab.imitation_env import G1ImitationEnvCfg


@configclass
class G1ImitationWalkEnvCfg(G1ImitationEnvCfg):
    """调试/冒烟：只用一条走路片段，1024 个环境。"""
    motion_clip_ids = (WALK_CLIP,)
    scene = G1ImitationEnvCfg.scene.replace(num_envs=1024)


@configclass
class G1ImitationOmomoEnvCfg(G1ImitationEnvCfg):
    motion_sources = ("omomo",)


@configclass
class G1ImitationTrumansEnvCfg(G1ImitationEnvCfg):
    motion_sources = ("trumans",)
