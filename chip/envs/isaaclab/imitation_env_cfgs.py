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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)


@configclass
class G1ImitationOmomoEnvCfg(G1ImitationEnvCfg):
    motion_sources = ("omomo",)


@configclass
class G1ImitationTrumansEnvCfg(G1ImitationEnvCfg):
    motion_sources = ("trumans",)
