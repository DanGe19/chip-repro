"""参考动作库（Motion Library）：Stage I 模仿训练的数据源。

把一批 ``RobotMotion`` 文件一次性加载到内存/显存，统一重采样到训练频率（论文 50 Hz 控制），
算好速度，拼成大张量，然后支持两类查询：

- ``sample_clips`` / ``sample_times``：随机挑片段和起始时刻（RSI, reference state initialization，
  即每个 episode 从参考动作的随机时刻开始，而不是总从第 0 帧开始）。
- ``get_state(clip_idx, t)``：任意时刻的参考状态（线性插值），作为奖励里的"目标"和观测里的"未来几帧"。

和 IsaacLab 无关，纯 torch，可以在没有 GPU 的机器上测试。

数据约定（和 RobotMotion 一致）：
- 根部四元数存成 (x, y, z, w)；需要 (w, x, y, z) 时用 ``root_rot_wxyz``。
- ``body_pos`` 若存在，是 GMR 给的 *局部* 位置（根在原点、无旋转时各刚体的位置）。
  这里把它变换到世界坐标：p_w = root_pos + R(root_rot) · p_local，奖励里要的是世界坐标。
- 所有长度单位米、角度弧度、时间秒。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from chip.retarget.robot_motion import G1_DOF_NAMES, RobotMotion


# ---------------------------------------------------------------------------
# 小工具：四元数（numpy，加载阶段用）
# ---------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _quat_conj_xyzw(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0], dtype=q.dtype)


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """q: (..., 4) xyzw, v: (..., 3) → R(q) v。"""
    xyz, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """逐元素 slerp。alpha: (N,) 或 (N,1)。"""
    alpha = np.asarray(alpha).reshape(-1, 1)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)          # 取短弧
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    w0 = np.where(small, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / np.where(small, 1.0, sin_t))
    w1 = np.where(small, alpha, np.sin(alpha * theta) / np.where(small, 1.0, sin_t))
    return _quat_normalize(w0 * q0 + w1 * q1)


def _quat_to_angvel_xyzw(q: np.ndarray, dt: float) -> np.ndarray:
    """相邻帧四元数 → 世界系角速度 (T,3)。ω ≈ 2 · vec(q_{t+1} ⊗ q_t⁻¹) / dt（小角近似足够）。"""
    dq = _quat_mul_xyzw(q[1:], _quat_conj_xyzw(q[:-1]))
    dq = np.where(dq[..., 3:4] < 0, -dq, dq)
    w = 2.0 * dq[..., :3] / dt
    return np.concatenate([w, w[-1:]], axis=0)   # 最后一帧复制前一帧，保持长度 T


def _resample_linear(x: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    """沿第 0 维对任意形状数组做线性插值。"""
    flat = x.reshape(x.shape[0], -1)
    out = np.stack([np.interp(dst_t, src_t, flat[:, j]) for j in range(flat.shape[1])], axis=1)
    return out.reshape((len(dst_t),) + x.shape[1:])


def _resample_quat(q: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(src_t, dst_t, side="right") - 1
    idx = idx.clip(0, len(src_t) - 2)
    alpha = (dst_t - src_t[idx]) / (src_t[idx + 1] - src_t[idx])
    return _slerp_xyzw(q[idx], q[idx + 1], alpha.clip(0.0, 1.0))


# ---------------------------------------------------------------------------
# 单条片段的重采样结果
# ---------------------------------------------------------------------------

@dataclass
class _Clip:
    clip_id: str
    source: str
    num_frames: int
    root_pos: np.ndarray     # (T,3)
    root_rot: np.ndarray     # (T,4) xyzw
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    dof_pos: np.ndarray      # (T,D)
    dof_vel: np.ndarray
    body_pos: Optional[np.ndarray]   # (T,B,3) 世界坐标
    body_vel: Optional[np.ndarray]


def _prepare_clip(rm: RobotMotion, target_fps: float, dof_index: np.ndarray, body_index: Optional[np.ndarray]) -> _Clip:
    T0 = rm.num_frames
    src_t = np.arange(T0) / rm.fps
    duration = src_t[-1]
    T = max(2, int(np.floor(duration * target_fps)) + 1)
    dst_t = np.arange(T) / target_fps
    dst_t = np.minimum(dst_t, duration)
    dt = 1.0 / target_fps

    root_pos = _resample_linear(rm.root_pos.astype(np.float64), src_t, dst_t)
    root_rot = _resample_quat(_quat_normalize(rm.root_rot.astype(np.float64)), src_t, dst_t)
    dof_pos = _resample_linear(rm.dof_pos.astype(np.float64)[:, dof_index], src_t, dst_t)

    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = _quat_to_angvel_xyzw(root_rot, dt)
    dof_vel = np.gradient(dof_pos, dt, axis=0)

    body_pos = body_vel = None
    if rm.body_pos is not None and body_index is not None:
        local = rm.body_pos.astype(np.float64)[:, body_index]            # (T0,B,3) 局部
        local = _resample_linear(local, src_t, dst_t)                      # (T,B,3)
        world = root_pos[:, None, :] + _quat_rotate_xyzw(root_rot[:, None, :], local)
        body_pos = world
        body_vel = np.gradient(world, dt, axis=0)

    return _Clip(rm.clip_id, rm.source, T, root_pos, root_rot, root_lin_vel, root_ang_vel,
                 dof_pos, dof_vel, body_pos, body_vel)


# ---------------------------------------------------------------------------
# 动作库
# ---------------------------------------------------------------------------

class MotionLib:
    """把多条 RobotMotion 装进大张量，支持随机采样和按时间插值查询。"""

    def __init__(
        self,
        files: Sequence[str | Path],
        target_fps: float = 50.0,
        device: str | torch.device = "cpu",
        dof_names: Sequence[str] = G1_DOF_NAMES,
        body_names: Optional[Sequence[str]] = None,
        min_duration: float = 0.5,
    ):
        """
        files        RobotMotion .npz 路径列表
        target_fps   重采样频率（论文控制频率 50 Hz）
        dof_names    输出关节顺序（默认 GMR 顺序；给 IsaacLab 的 joint_names 就直接按环境顺序输出）
        body_names   要保留的刚体子集（None = 不加载 body_pos）
        min_duration 短于此的片段丢弃
        """
        self.device = torch.device(device)
        self.fps = float(target_fps)
        self.dt = 1.0 / self.fps
        self.dof_names = tuple(dof_names)
        self.body_names = tuple(body_names) if body_names else None

        clips: list[_Clip] = []
        skipped = []
        for f in files:
            rm = RobotMotion.load(f)
            if rm.num_frames / rm.fps < min_duration:
                skipped.append((rm.clip_id, "too_short"))
                continue
            dof_index = np.array([rm.dof_names.index(n) for n in self.dof_names])
            body_index = None
            if self.body_names:
                if rm.body_names is None or any(n not in rm.body_names for n in self.body_names):
                    skipped.append((rm.clip_id, "missing_body_pos"))
                    continue
                body_index = np.array([rm.body_names.index(n) for n in self.body_names])
            clips.append(_prepare_clip(rm, self.fps, dof_index, body_index))
        if not clips:
            raise ValueError("没有可用片段")
        self.skipped = skipped

        # 拼接
        self.clip_ids = [c.clip_id for c in clips]
        self.sources = [c.source for c in clips]
        lengths = np.array([c.num_frames for c in clips])
        self.num_clips = len(clips)
        self._lengths = torch.as_tensor(lengths, device=self.device)
        self._starts = torch.as_tensor(np.concatenate([[0], np.cumsum(lengths)[:-1]]), device=self.device)
        self.durations = ((self._lengths - 1).double() / self.fps).float()   # 可查询的最大时间

        def cat(name):
            return torch.as_tensor(np.concatenate([getattr(c, name) for c in clips], axis=0), dtype=torch.float32, device=self.device)

        self.root_pos = cat("root_pos")
        self.root_rot = cat("root_rot")            # xyzw
        self.root_lin_vel = cat("root_lin_vel")
        self.root_ang_vel = cat("root_ang_vel")
        self.dof_pos = cat("dof_pos")
        self.dof_vel = cat("dof_vel")
        self.has_body = clips[0].body_pos is not None
        self.body_pos = cat("body_pos") if self.has_body else None
        self.body_vel = cat("body_vel") if self.has_body else None
        self.total_frames = int(lengths.sum())

    # ---- 构造 ----
    @classmethod
    def from_dir(cls, root: str | Path, sources: Optional[Iterable[str]] = None, limit: Optional[int] = None, **kw) -> "MotionLib":
        """从 run_gmr 的输出目录加载（优先读 index.csv，没有就扫 *.npz）。"""
        root = Path(root)
        files: list[Path] = []
        idx = root / "index.csv"
        if idx.exists():
            with open(idx, newline="") as f:
                for row in csv.DictReader(f):
                    if sources and row["source"] not in set(sources):
                        continue
                    files.append(root / row["path"])
        else:
            for p in sorted(root.rglob("*.npz")):
                if sources and p.parent.name not in set(sources):
                    continue
                files.append(p)
        if limit:
            files = files[:limit]
        return cls(files, **kw)

    # ---- 采样 ----
    def sample_clips(self, n: int, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """按时长加权（默认）随机挑 n 条片段，返回片段下标 (n,)。"""
        w = self.durations if weights is None else weights.to(self.device)
        return torch.multinomial(w / w.sum(), n, replacement=True)

    def sample_times(self, clip_idx: torch.Tensor, truncate: float = 0.0) -> torch.Tensor:
        """在 [0, duration - truncate] 均匀采时间。truncate 用来保证后面还有 truncate 秒可看。"""
        max_t = (self.durations[clip_idx] - truncate).clamp(min=0.0)
        return torch.rand_like(max_t) * max_t

    # ---- 查询 ----
    def _locate(self, clip_idx: torch.Tensor, t: torch.Tensor):
        t = t.clamp(min=0.0)
        t = torch.minimum(t, self.durations[clip_idx])
        f = t * self.fps
        f0 = torch.floor(f).long()
        last = self._lengths[clip_idx] - 1
        f0 = torch.minimum(f0, last - 1).clamp(min=0)
        f1 = torch.minimum(f0 + 1, last)
        alpha = (f - f0.float()).clamp(0.0, 1.0).unsqueeze(-1)
        base = self._starts[clip_idx]
        return base + f0, base + f1, alpha

    def get_state(self, clip_idx: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """任意时刻的参考状态（帧间线性插值；四元数用归一化线性插值，相邻帧夹角很小时与 slerp 等价）。

        返回 dict：root_pos (N,3), root_rot (N,4) xyzw, root_lin_vel, root_ang_vel, dof_pos (N,D), dof_vel,
        以及（若加载了）body_pos (N,B,3), body_vel。
        """
        i0, i1, a = self._locate(clip_idx, t)

        def lerp(x):
            return x[i0] * (1 - a) + x[i1] * a

        q0, q1 = self.root_rot[i0], self.root_rot[i1]
        q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0, -q1, q1)
        q = q0 * (1 - a) + q1 * a
        q = q / q.norm(dim=-1, keepdim=True)

        out = {
            "root_pos": lerp(self.root_pos),
            "root_rot": q,
            "root_lin_vel": lerp(self.root_lin_vel),
            "root_ang_vel": lerp(self.root_ang_vel),
            "dof_pos": lerp(self.dof_pos),
            "dof_vel": lerp(self.dof_vel),
        }
        if self.has_body:
            a3 = a.unsqueeze(-1)
            out["body_pos"] = self.body_pos[i0] * (1 - a3) + self.body_pos[i1] * a3
            out["body_vel"] = self.body_vel[i0] * (1 - a3) + self.body_vel[i1] * a3
        return out

    def get_future(self, clip_idx: torch.Tensor, t: torch.Tensor, steps: Sequence[float]) -> dict[str, torch.Tensor]:
        """未来若干时刻的参考（观测里的"接下来要去哪"）。steps 是相对 t 的秒数列表，返回每项形状 (N, K, ...)。"""
        outs = [self.get_state(clip_idx, t + s) for s in steps]
        return {k: torch.stack([o[k] for o in outs], dim=1) for k in outs[0]}

    @staticmethod
    def root_rot_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
        return torch.cat([q_xyzw[..., 3:4], q_xyzw[..., :3]], dim=-1)

    def summary(self) -> str:
        total_s = float(self.durations.sum())
        by_src: dict[str, float] = {}
        for s, d in zip(self.sources, self.durations.tolist()):
            by_src[s] = by_src.get(s, 0.0) + d
        parts = ", ".join(f"{k}: {v / 3600:.2f} h" for k, v in by_src.items())
        return (f"MotionLib: {self.num_clips} 条片段, {self.total_frames} 帧 @ {self.fps:g} Hz, "
                f"共 {total_s / 3600:.2f} h ({parts}); 跳过 {len(self.skipped)} 条; "
                f"dof {len(self.dof_names)}, body {len(self.body_names) if self.body_names else 0}")
