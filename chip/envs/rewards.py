"""Stage I 模仿奖励（论文附录表 8），纯 torch 函数，不依赖 IsaacLab。

约定
----
所有输入都是按环境并行的张量，形状前缀 (N, ...)：
- ``cur`` / ``ref`` 两个 dict，键：
    anchor_pos (N,3)   骨盆世界位置
    anchor_rot (N,4)   骨盆世界朝向，四元数 **xyzw**
    body_pos   (N,B,3) 各刚体世界位置
    body_rot   (N,B,4) 各刚体世界朝向 xyzw（可选；没有就跳过 body_rot 项）
    body_lin_vel (N,B,3), body_ang_vel (N,B,3)（可选）
    object_pos (N,3), object_rot (N,4), object_active (N,) bool（可选，HOI 专家）
- 每一项先算误差 e，再做 RBF 核 K_σ(e) = exp(-e²/σ²)，乘权重后求和。
- "相对身体"项先把刚体量变换到骨盆坐标系：p_rel = R_anchorᵀ (p_body − p_anchor)。
  这样"整体走慢半步"只扣锚点分，不连坐全身姿态分。

用法::

    cfg = load_reward_cfg("configs/stage1_imitation.yaml")
    total, terms = imitation_reward(cur, ref, cfg)        # total (N,), terms: {name: (N,)}
    done = kinematic_termination(cur, ref, pos_thr=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# 四元数工具（xyzw，torch，批量）
# ---------------------------------------------------------------------------


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([-1.0, -1.0, -1.0, 1.0])


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q) v；q (...,4) 与 v (...,3) 可广播。"""
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(q), v)


def quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """两个旋转之间的测地角 (…,)，范围 [0, π]。"""
    dot = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    """只保留绕 z 的分量（航向），xyzw。"""
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return torch.stack([torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)], dim=-1)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class Term:
    weight: float
    sigma: float


@dataclass
class RewardCfg:
    terms: dict[str, Term]
    penalties: dict[str, float] = field(default_factory=dict)
    relative_frame: str = "full"      # "full" = 用骨盆完整朝向对齐；"yaw" = 只按航向对齐
    dt: float = 0.02                  # 论文：整体奖励乘以 Δt

    @classmethod
    def from_dict(cls, d: dict, dt: float = 0.02) -> "RewardCfg":
        terms = {k: Term(float(v["weight"]), float(v["sigma"])) for k, v in d.items() if k != "penalties"}
        return cls(terms=terms, penalties={k: float(v) for k, v in d.get("penalties", {}).items()}, dt=dt)


def load_reward_cfg(path: str | Path) -> RewardCfg:
    import yaml

    with open(path) as f:
        full = yaml.safe_load(f)
    dt = 1.0 / float(full.get("sim", {}).get("control_hz", 50))
    return RewardCfg.from_dict(full["reward"], dt=dt)


def rbf(err: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-(err * err) / (sigma * sigma))


# ---------------------------------------------------------------------------
# 奖励
# ---------------------------------------------------------------------------


def _to_anchor_frame(state: dict, mode: str) -> dict:
    """把 body_* 变换到骨盆坐标系。返回新 dict（不改原来的）。"""
    q = state["anchor_rot"]
    if mode == "yaw":
        q = yaw_quat(q)
    q_inv = quat_conj(q)[:, None, :]            # (N,1,4)
    out = {}
    if "body_pos" in state:
        out["body_pos"] = quat_rotate(q_inv, state["body_pos"] - state["anchor_pos"][:, None, :])
    if "body_rot" in state:
        out["body_rot"] = quat_mul(q_inv.expand_as(state["body_rot"]), state["body_rot"])
    if "body_lin_vel" in state:
        out["body_lin_vel"] = quat_rotate(q_inv, state["body_lin_vel"])
    if "body_ang_vel" in state:
        out["body_ang_vel"] = quat_rotate(q_inv, state["body_ang_vel"])
    if "object_pos" in state:
        out["object_pos"] = quat_rotate(q_inv[:, 0], state["object_pos"] - state["anchor_pos"])
    return out


def imitation_reward(cur: dict, ref: dict, cfg: RewardCfg,
                     action: Optional[torch.Tensor] = None, prev_action: Optional[torch.Tensor] = None,
                     joint_pos: Optional[torch.Tensor] = None, joint_limits: Optional[torch.Tensor] = None,
                     undesired_contact: Optional[torch.Tensor] = None,
                     ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 (total (N,), terms)。terms 里每一项都是 (N,) 已乘权重（惩罚为负），方便直接记日志。

    action / prev_action      (N,A)   用于 action_rate 惩罚
    joint_pos / joint_limits  (N,D) / (D,2) 用于 joint_limit 惩罚（超限量的 L1）
    undesired_contact         (N,)  bool 或 0/1，不该接触的部位是否有接触
    """
    N = cur["anchor_pos"].shape[0]
    dev = cur["anchor_pos"].device
    terms: dict[str, torch.Tensor] = {}
    T = cfg.terms

    # ---- 锚点 ----
    if "anchor_pos" in T:
        e = (cur["anchor_pos"] - ref["anchor_pos"]).norm(dim=-1)
        terms["anchor_pos"] = T["anchor_pos"].weight * rbf(e, T["anchor_pos"].sigma)
    if "anchor_rot" in T:
        e = quat_angle(cur["anchor_rot"], ref["anchor_rot"])
        terms["anchor_rot"] = T["anchor_rot"].weight * rbf(e, T["anchor_rot"].sigma)

    # ---- 相对身体 ----
    c_rel = _to_anchor_frame(cur, cfg.relative_frame)
    r_rel = _to_anchor_frame(ref, cfg.relative_frame)
    if "body_pos" in T and "body_pos" in c_rel and "body_pos" in r_rel:
        e = (c_rel["body_pos"] - r_rel["body_pos"]).norm(dim=-1).mean(dim=-1)        # 各刚体平均
        terms["body_pos"] = T["body_pos"].weight * rbf(e, T["body_pos"].sigma)
    if "body_rot" in T and "body_rot" in c_rel and "body_rot" in r_rel:
        e = quat_angle(c_rel["body_rot"], r_rel["body_rot"]).mean(dim=-1)
        terms["body_rot"] = T["body_rot"].weight * rbf(e, T["body_rot"].sigma)
    if "body_lin_vel" in T and "body_lin_vel" in c_rel and "body_lin_vel" in r_rel:
        e = (c_rel["body_lin_vel"] - r_rel["body_lin_vel"]).norm(dim=-1).mean(dim=-1)
        terms["body_lin_vel"] = T["body_lin_vel"].weight * rbf(e, T["body_lin_vel"].sigma)
    if "body_ang_vel" in T and "body_ang_vel" in c_rel and "body_ang_vel" in r_rel:
        e = (c_rel["body_ang_vel"] - r_rel["body_ang_vel"]).norm(dim=-1).mean(dim=-1)
        terms["body_ang_vel"] = T["body_ang_vel"].weight * rbf(e, T["body_ang_vel"].sigma)

    # ---- 物体（HOI 专家）----
    if "object_pos" in cur and "object_pos" in ref:
        active = ref.get("object_active", torch.ones(N, dtype=torch.bool, device=dev)).float()
        if "object_rel_pos" in T:
            e = (c_rel["object_pos"] - r_rel["object_pos"]).norm(dim=-1)
            terms["object_rel_pos"] = T["object_rel_pos"].weight * rbf(e, T["object_rel_pos"].sigma) * active
        if "object_rot" in T and "object_rot" in cur and "object_rot" in ref:
            e = quat_angle(cur["object_rot"], ref["object_rot"])
            terms["object_rot"] = T["object_rot"].weight * rbf(e, T["object_rot"].sigma) * active

    # ---- 惩罚 ----
    P = cfg.penalties
    if "action_rate" in P and action is not None and prev_action is not None:
        terms["action_rate"] = P["action_rate"] * (action - prev_action).square().sum(dim=-1)
    if "joint_limit" in P and joint_pos is not None and joint_limits is not None:
        lo, hi = joint_limits[:, 0], joint_limits[:, 1]
        over = (lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)
        terms["joint_limit"] = P["joint_limit"] * over.sum(dim=-1)
    if "undesired_contact" in P and undesired_contact is not None:
        terms["undesired_contact"] = P["undesired_contact"] * undesired_contact.float()

    total = torch.stack(list(terms.values()), dim=0).sum(dim=0) * cfg.dt
    return total, terms


def kinematic_termination(cur: dict, ref: dict, pos_thr: float = 0.5, body_thr: Optional[float] = None,
                          height_thr: Optional[float] = 0.3) -> torch.Tensor:
    """偏离参考太远就提前终止 (N,) bool。

    pos_thr    骨盆世界位置误差阈值（m）
    body_thr   任一刚体世界位置误差阈值（m），None 不启用
    height_thr 骨盆高度低于此值视为摔倒，None 不启用
    """
    done = (cur["anchor_pos"] - ref["anchor_pos"]).norm(dim=-1) > pos_thr
    if body_thr is not None and "body_pos" in cur and "body_pos" in ref:
        done |= ((cur["body_pos"] - ref["body_pos"]).norm(dim=-1).max(dim=-1).values > body_thr)
    if height_thr is not None:
        done |= cur["anchor_pos"][:, 2] < height_thr
    return done
