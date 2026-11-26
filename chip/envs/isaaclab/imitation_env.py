"""Stage I：G1 动作模仿环境（IsaacLab DirectRLEnv）。

每个 episode：从 MotionLib 随机挑一条参考片段和起始时刻（RSI），把机器人摆成参考姿态，然后让策略
每 20 ms 输出 29 个关节的目标角（PD 控制），奖励 = 当前姿态和参考姿态的接近程度（rewards.py）。
偏离太远或摔倒提前终止；参考片段播完算正常结束。

观测（policy）：
    本体：根高度(1) 根朝向四元数 xyzw(4) 根线速度·基座系(3) 根角速度·基座系(3) 关节角(29) 关节速度(29) 上一步动作(29)
    参考（未来 1 帧，论文表 5）：关节角差(29) 参考根位置·基座系(3) 参考根朝向相对量(4) 参考关键刚体位置·基座系(B×3)
动作：29 个受控关节的目标角偏移（乘 action_scale 后加到默认姿态上）。手指关节不控制，目标恒 0。

只在 IsaacLab 的 python 里能 import。
"""

from __future__ import annotations

import csv
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_29DOF_CFG  # isort: skip

from chip.envs.motion_lib import MotionLib
from chip.envs.rewards import (
    RewardCfg, imitation_reward, kinematic_termination, load_reward_cfg, quat_conj, quat_mul, quat_rotate_inverse,
)
from chip.retarget.robot_motion import G1_DOF_NAMES

# 奖励里要跟踪的关键刚体（GMR 的 MuJoCo 模型和 IsaacLab 的 USD 里都有这些名字）
DEFAULT_TRACKED_BODIES = (
    "pelvis", "torso_link",
    "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
)
# 允许接触地面的刚体（其它刚体碰到东西算 undesired contact）
CONTACT_OK_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")


@configclass
class G1ImitationEnvCfg(DirectRLEnvCfg):
    # ---- 时间 ----
    episode_length_s = 10.0
    decimation = 4                      # 200 Hz 物理 / 50 Hz 控制（论文）
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200, render_interval=4,
        physx=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    robot: ArticulationCfg = G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ---- 空间（observation_space 在 __init__ 里按刚体数重算）----
    action_space = 29
    # 本体 98 + 每个未来时刻 (29 + 3 + 4 + 3B)；在 __post_init__ 里按 ref_lookahead_s 和 tracked_bodies 重算
    observation_space = 98 + 4 * (29 + 3 + 4 + 3 * len(DEFAULT_TRACKED_BODIES))

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = 98 + len(self.ref_lookahead_s) * (29 + 3 + 4 + 3 * len(self.tracked_bodies))
    state_space = 0
    action_scale = 0.5                 # 动作 ∈ [-1,1] → ±0.5 rad 偏移

    # ---- 参考动作 ----
    motion_dir: str = "/data/robot_clips"
    motion_sources: tuple[str, ...] | None = None     # ("omomo",) / ("trumans",) / None=全部
    motion_clip_ids: tuple[str, ...] | None = None    # 只用这些片段（调试：一条走路片段）
    motion_limit: int | None = None
    tracked_bodies: tuple[str, ...] = DEFAULT_TRACKED_BODIES
    ref_lookahead_s: tuple[float, ...] = (0.02, 0.1, 0.3, 0.6)   # 观测里给的未来参考时刻（秒）
    anchor_pos_sigma: float | None = 0.15                       # 覆盖 yaml 里的 anchor_pos σ（None=不覆盖）
    anchor_lin_vel_weight: float = 0.5                          # 新增根部速度跟踪项（σ=0.5 m/s）；0=关闭

    # ---- 奖励 / 终止 ----
    reward_cfg_path: str = str(Path(__file__).resolve().parents[3] / "configs" / "stage1_imitation.yaml")
    termination_pos_thr: float = 0.5
    action_rate_scale: float = 1.0     # 乘到论文的 action_rate 惩罚上；前期可调小（课程），学会走路后再调回 1
    termination_height: float = 0.3
    early_termination: bool = True
    reset_time_mode: str = "random"    # "random" = RSI（训练）；"start" = 总从片段开头开始（评估/录像）


class G1ImitationEnv(DirectRLEnv):
    cfg: G1ImitationEnvCfg

    def __init__(self, cfg: G1ImitationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        joint_names = list(self.robot.joint_names)
        # 受控的 29 个关节在 IsaacLab 关节表里的下标（按 IsaacLab 顺序排列）
        self.ctrl_ids = torch.tensor([i for i, n in enumerate(joint_names) if n in G1_DOF_NAMES], device=self.device)
        self.ctrl_names = [joint_names[i] for i in self.ctrl_ids.tolist()]
        assert len(self.ctrl_names) == 29, f"只找到 {len(self.ctrl_names)} 个受控关节"
        self.hand_ids = torch.tensor([i for i, n in enumerate(joint_names) if n not in G1_DOF_NAMES], device=self.device)

        body_names = list(self.robot.body_names)
        self.tracked_body_ids = torch.tensor([body_names.index(n) for n in cfg.tracked_bodies], device=self.device)
        self.pelvis_id = body_names.index("pelvis")
        self.contact_bad_ids = torch.tensor(
            [i for i, n in enumerate(body_names) if n not in CONTACT_OK_BODIES and n != "pelvis"], device=self.device)

        # 参考动作库：关节顺序直接用 IsaacLab 受控关节的顺序
        files = self._collect_motion_files()
        self.motion = MotionLib(files, target_fps=1.0 / (cfg.sim.dt * cfg.decimation), device=self.device,
                                dof_names=self.ctrl_names, body_names=list(cfg.tracked_bodies))
        print("[G1Imitation]", self.motion.summary())

        self.reward_cfg: RewardCfg = load_reward_cfg(cfg.reward_cfg_path)
        self.reward_cfg.dt = cfg.sim.dt * cfg.decimation
        if "action_rate" in self.reward_cfg.penalties:
            self.reward_cfg.penalties["action_rate"] *= cfg.action_rate_scale
        if cfg.anchor_pos_sigma is not None and "anchor_pos" in self.reward_cfg.terms:
            self.reward_cfg.terms["anchor_pos"].sigma = cfg.anchor_pos_sigma
        if cfg.anchor_lin_vel_weight > 0:
            from chip.envs.rewards import Term
            self.reward_cfg.terms["anchor_lin_vel"] = Term(cfg.anchor_lin_vel_weight, 0.5)

        N = self.num_envs
        self.clip_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        self.ref_time = torch.zeros(N, device=self.device)
        self.actions = torch.zeros(N, 29, device=self.device)
        self.prev_actions = torch.zeros(N, 29, device=self.device)
        self.default_ctrl_pos = self.robot.data.default_joint_pos[:, self.ctrl_ids].clone()
        self.joint_limits = self.robot.data.soft_joint_pos_limits[0, self.ctrl_ids]       # (29,2)
        self._episode_sums: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ 场景
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane("/World/ground", GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)))
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    def _collect_motion_files(self) -> list[Path]:
        root = Path(self.cfg.motion_dir)
        rows = list(csv.DictReader(open(root / "index.csv", newline="")))
        if self.cfg.motion_sources:
            rows = [r for r in rows if r["source"] in self.cfg.motion_sources]
        if self.cfg.motion_clip_ids:
            want = set(self.cfg.motion_clip_ids)
            rows = [r for r in rows if r["clip_id"] in want]
        if self.cfg.motion_limit:
            rows = rows[: self.cfg.motion_limit]
        if not rows:
            raise ValueError("没有匹配的参考片段")
        return [root / r["path"] for r in rows]

    # ------------------------------------------------------------------ 动作
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions
        self.actions = actions.clamp(-1.0, 1.0).clone()

    def _apply_action(self):
        target = self.default_ctrl_pos + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target(target, joint_ids=self.ctrl_ids)
        self.robot.set_joint_position_target(torch.zeros(self.num_envs, len(self.hand_ids), device=self.device), joint_ids=self.hand_ids)

    # ------------------------------------------------------------------ 状态拼装
    def _ref_state(self, dt_ahead: float = 0.0) -> dict:
        ref = self.motion.get_state(self.clip_idx, self.ref_time + dt_ahead)
        # 参考是在"机器人世界原点"定义的；每个并行环境有自己的原点偏移
        origins = self.scene.env_origins
        return {
            "anchor_pos": ref["root_pos"] + origins,
            "anchor_rot": ref["root_rot"],
            "body_pos": ref["body_pos"] + origins[:, None, :],
            "body_lin_vel": ref["body_vel"],
            "dof_pos": ref["dof_pos"],
            "dof_vel": ref["dof_vel"],
            "root_lin_vel": ref["root_lin_vel"],
            "root_ang_vel": ref["root_ang_vel"],
        }

    def _cur_state(self) -> dict:
        d = self.robot.data
        q_wxyz = d.root_quat_w
        q = torch.cat([q_wxyz[:, 1:], q_wxyz[:, :1]], dim=-1)   # → xyzw
        bq = d.body_quat_w[:, self.tracked_body_ids]
        return {
            "anchor_pos": d.root_pos_w,
            "anchor_rot": q,
            "body_pos": d.body_pos_w[:, self.tracked_body_ids],
            "body_lin_vel": d.body_lin_vel_w[:, self.tracked_body_ids],
            "body_rot": torch.cat([bq[..., 1:], bq[..., :1]], dim=-1),
            "dof_pos": d.joint_pos[:, self.ctrl_ids],
            "dof_vel": d.joint_vel[:, self.ctrl_ids],
            "root_lin_vel": d.root_lin_vel_w,
            "root_ang_vel": d.root_ang_vel_w,
        }

    # ------------------------------------------------------------------ 观测 / 奖励 / 终止
    def _get_observations(self) -> dict:
        cur = self._cur_state()
        q = cur["anchor_rot"]
        q_inv = quat_conj(q)
        # 本体（基座系）
        lin_b = quat_rotate_inverse(q, cur["root_lin_vel"])
        ang_b = quat_rotate_inverse(q, cur["root_ang_vel"])
        parts = [cur["anchor_pos"][:, 2:3], q, lin_b, ang_b, cur["dof_pos"], cur["dof_vel"] * 0.1, self.actions]
        # 未来多个时刻的参考，全部表达在当前基座系里："目标在我前方多远、朝向差多少、各部位该在哪"
        for dt_ahead in self.cfg.ref_lookahead_s:
            ref = self._ref_state(dt_ahead)
            ref_root_b = quat_rotate_inverse(q, ref["anchor_pos"] - cur["anchor_pos"])
            ref_rot_rel = quat_mul(q_inv, ref["anchor_rot"])
            ref_body_b = quat_rotate_inverse(q[:, None, :], ref["body_pos"] - cur["anchor_pos"][:, None, :])
            parts += [ref["dof_pos"] - cur["dof_pos"], ref_root_b, ref_rot_rel, ref_body_b.flatten(1)]
        obs = torch.cat(parts, dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-50, 50)}

    def _get_rewards(self) -> torch.Tensor:
        cur = self._cur_state()
        ref = self._ref_state(0.0)
        cur.pop("body_rot")      # 参考没有刚体朝向，这一项跳过；接触惩罚等加了 ContactSensor 再上
        total, terms = imitation_reward(
            cur, ref, self.reward_cfg,
            action=self.actions, prev_action=self.prev_actions,
            joint_pos=cur["dof_pos"], joint_limits=self.joint_limits,
        )
        for k, v in terms.items():
            self._episode_sums[k] = self._episode_sums.get(k, torch.zeros(self.num_envs, device=self.device)) + v
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.ref_time += self.reward_cfg.dt
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out |= self.ref_time >= self.motion.durations[self.clip_idx]       # 参考播完
        if self.cfg.early_termination:
            cur = self._cur_state()
            ref = self._ref_state(0.0)
            died = kinematic_termination(cur, ref, pos_thr=self.cfg.termination_pos_thr, height_thr=self.cfg.termination_height)
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    # ------------------------------------------------------------------ 重置（RSI）
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # 日志：各奖励项的"每步平均值"，以及平均局长。必须在 super()._reset_idx 之前读局长（它会清零）
        if self._episode_sums:
            ep_len = torch.clamp(self.episode_length_buf[env_ids].float(), min=1)
            self.extras.setdefault("log", {})
            for k, v in self._episode_sums.items():
                self.extras["log"][f"rew/{k}"] = (v[env_ids] / ep_len).mean()
                v[env_ids] = 0.0
            self.extras["log"]["episode/len_s"] = (ep_len * self.reward_cfg.dt).mean()

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)

        clip = self.motion.sample_clips(n)
        if self.cfg.reset_time_mode == "start":
            t = torch.zeros(n, device=self.device)
        else:
            t = self.motion.sample_times(clip, truncate=1.0)
        self.clip_idx[env_ids] = clip
        self.ref_time[env_ids] = t
        ref = self.motion.get_state(clip, t)

        root_pos = ref["root_pos"] + self.scene.env_origins[env_ids]
        root_pos[:, 2] += 0.02                                   # 抬 2 cm 防止脚插地
        q = ref["root_rot"]
        root_quat_wxyz = torch.cat([q[:, 3:4], q[:, :3]], dim=-1)
        self.robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat_wxyz], dim=-1), env_ids)
        # 参考速度做个保险钳位：数据毛刺会把机器人"砸"向地面
        root_vel = torch.cat([ref["root_lin_vel"].clamp(-2.0, 2.0), ref["root_ang_vel"].clamp(-4.0, 4.0)], dim=-1)
        self.robot.write_root_com_velocity_to_sim(root_vel, env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.ctrl_ids] = ref["dof_pos"]
        joint_vel[:, self.ctrl_ids] = ref["dof_vel"].clamp(-10.0, 10.0)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
