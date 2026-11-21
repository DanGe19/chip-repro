"""重定向后的机器人动作格式 RobotMotion（Unitree G1，29 DoF）。

和 MotionClip 的关系：
    MotionClip（SMPL-X 人体）──重定向──► RobotMotion（G1 关节）
场景、物体、文本这些上下文不参与重定向，原样从 MotionClip 搬过来。

关节顺序固定为 GMR 的 g1_mocap_29dof.xml 顺序（见 G1_DOF_NAMES）。
IsaacLab 里的 G1 关节顺序可能不同，到 envs 模块里按名字重排，不要在这里假设。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from chip.data.schema import ObjectTrack

G1_DOF_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
G1_NUM_DOF = len(G1_DOF_NAMES)   # 29


@dataclass
class RobotMotion:
    clip_id: str
    source: str
    fps: float
    root_pos: np.ndarray                 # (T, 3)   世界坐标，Z 朝上
    root_rot: np.ndarray                 # (T, 4)   四元数 (x, y, z, w)
    dof_pos: np.ndarray                  # (T, 29)  关节角，顺序见 G1_DOF_NAMES
    dof_names: tuple[str, ...] = G1_DOF_NAMES

    # 可选：GMR 顺带输出的 body 位置，方便可视化和算奖励
    body_pos: Optional[np.ndarray] = None        # (T, B, 3)
    body_names: Optional[tuple[str, ...]] = None

    # 从 MotionClip 原样搬过来的上下文
    scene_name: Optional[str] = None
    scene_mesh_path: Optional[str] = None
    objects: list[ObjectTrack] = field(default_factory=list)
    text: Optional[str] = None
    bad_frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # 人类世界 → 机器人世界 的相似变换：p_robot = world_scale * p_human + world_offset。
    # GMR 会按身高缩放人的轨迹、贴地、并把起点归零；物体/场景必须跟着做同样的变换
    # 才能和机器人对得上。scene_mesh_path 指向的是人类世界里的网格，加载时要套这个变换。
    world_scale: float = 1.0
    world_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    def validate(self) -> None:
        T = self.num_frames
        assert self.root_pos.shape == (T, 3), self.root_pos.shape
        assert self.root_rot.shape == (T, 4), self.root_rot.shape
        assert self.dof_pos.shape == (T, len(self.dof_names)), self.dof_pos.shape
        norms = np.linalg.norm(self.root_rot, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "root_rot 不是单位四元数"
        if self.body_pos is not None:
            assert self.body_pos.shape[0] == T and self.body_pos.shape[2] == 3
        for o in self.objects:
            assert o.pose.shape == (T, 4, 4), (o.name, o.pose.shape)

    # ---- 存取 ----
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "root_pos": self.root_pos,
            "root_rot": self.root_rot,
            "dof_pos": self.dof_pos,
            "bad_frames": self.bad_frames,
        }
        if self.body_pos is not None:
            arrays["body_pos"] = self.body_pos
        for i, o in enumerate(self.objects):
            arrays[f"obj{i}_pose"] = o.pose
            arrays[f"obj{i}_scale"] = o.scale
            arrays[f"obj{i}_active"] = o.active
        meta = {
            "clip_id": self.clip_id, "source": self.source, "fps": self.fps,
            "dof_names": list(self.dof_names),
            "body_names": list(self.body_names) if self.body_names else None,
            "scene_name": self.scene_name, "scene_mesh_path": self.scene_mesh_path,
            "text": self.text,
            "objects": [{"name": o.name, "mesh_path": o.mesh_path} for o in self.objects],
            "world_scale": float(self.world_scale),
            "world_offset": [float(x) for x in np.asarray(self.world_offset).ravel()],
        }
        np.savez_compressed(path, meta=np.array(meta, dtype=object), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "RobotMotion":
        d = np.load(path, allow_pickle=True)
        m = d["meta"].item()
        objects = [
            ObjectTrack(name=om["name"], mesh_path=om["mesh_path"],
                        pose=d[f"obj{i}_pose"], scale=d[f"obj{i}_scale"], active=d[f"obj{i}_active"])
            for i, om in enumerate(m["objects"])
        ]
        rm = cls(
            clip_id=m["clip_id"], source=m["source"], fps=float(m["fps"]),
            root_pos=d["root_pos"], root_rot=d["root_rot"], dof_pos=d["dof_pos"],
            dof_names=tuple(m["dof_names"]),
            body_pos=d["body_pos"] if "body_pos" in d else None,
            body_names=tuple(m["body_names"]) if m["body_names"] else None,
            scene_name=m["scene_name"], scene_mesh_path=m["scene_mesh_path"],
            objects=objects, text=m["text"], bad_frames=d["bad_frames"],
            world_scale=float(m.get("world_scale", 1.0)),
            world_offset=np.asarray(m.get("world_offset", [0.0, 0.0, 0.0]), dtype=np.float32),
        )
        rm.validate()
        return rm


# ---- GMR 的世界变换 ----

def gmr_human_scale(human_height: float, pelvis_scale: float = 0.9, height_assumption: float = 1.8) -> float:
    """GMR 对人体根部位置的缩放系数（见 GMR motion_retarget.py: human_scale_table['pelvis'] * ratio）。"""
    return pelvis_scale * human_height / height_assumption


def fit_gmr_world_transform(robot_root_pos: np.ndarray, human_transl: np.ndarray, human_height: float
                            ) -> tuple[float, np.ndarray, float]:
    """拟合 p_robot ≈ s * p_human + c。

    GMR 的处理（scripts/smplx_to_robot_dataset.py）：根部位置乘以 s；整体上下平移使机器人最低点
    贴地；再减去第 0 帧的 XY。所以 s 按公式算，c 的 XY 用逐帧均值拟合（机器人根≈缩放后的人骨盆关节，
    骨盆关节和 transl 差一个随体型变化的小常量，拟合能把它吸收掉），c 的 Z 取 0（地面保持是地面）。

    返回 (s, c(3,), xy 残差的 RMS)；残差应在几厘米量级，大了说明帧没对齐。
    """
    T = min(robot_root_pos.shape[0], human_transl.shape[0])
    s = gmr_human_scale(float(human_height))
    r = np.asarray(robot_root_pos[:T, :2], dtype=np.float64)
    h = np.asarray(human_transl[:T, :2], dtype=np.float64)
    c_xy = (r - s * h).mean(axis=0)
    resid = np.sqrt(((r - (s * h + c_xy)) ** 2).sum(axis=1).mean())
    return s, np.array([c_xy[0], c_xy[1], 0.0], dtype=np.float32), float(resid)


def transform_object_track(o: ObjectTrack, s: float, c: np.ndarray) -> ObjectTrack:
    """把人类世界里的物体轨迹搬到机器人世界：平移缩放+偏移，尺寸乘以 s，旋转不变。"""
    pose = np.array(o.pose, dtype=np.float32, copy=True)
    pose[:, :3, 3] = s * pose[:, :3, 3] + c
    scale = np.asarray(o.scale, dtype=np.float32) * s
    return ObjectTrack(name=o.name, mesh_path=o.mesh_path, pose=pose, scale=scale, active=o.active)


def load_gmr_pickle(path: str | Path) -> dict:
    """读 GMR 的 smplx_to_robot(_dataset).py 输出的 pickle。

    GMR 输出字段（scripts/smplx_to_robot.py）：
        fps            int
        root_pos       (T, 3)
        root_rot       (T, 4)  已经是 (x, y, z, w)
        dof_pos        (T, 29)
        local_body_pos (T, B, 3) 或 None
        link_body_list list[str]
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"GMR pickle {path} 缺字段: {missing}")
    return d


def robot_motion_from_gmr(gmr: dict, clip_meta: dict, objects: list[ObjectTrack] | None = None,
                          bad_frames: np.ndarray | None = None,
                          human_transl: np.ndarray | None = None, human_height: float | None = None,
                          ) -> RobotMotion:
    """GMR 输出 + MotionClip 的上下文 → RobotMotion。

    clip_meta 是 MotionClip.summary() 以外还需要的字段：
        clip_id, source, scene_name, scene_mesh_path, text
    给了 human_transl (T,3) 和 human_height 时，会拟合 GMR 的世界变换并套到物体上
    （否则物体保持人类世界坐标，和机器人对不上——只适合没有物体的片段）。
    """
    root_rot = np.asarray(gmr["root_rot"], dtype=np.float32)
    root_rot /= np.linalg.norm(root_rot, axis=1, keepdims=True)
    T = root_rot.shape[0]

    world_scale, world_offset = 1.0, np.zeros(3, dtype=np.float32)
    if human_transl is not None and human_height is not None:
        world_scale, world_offset, resid = fit_gmr_world_transform(
            np.asarray(gmr["root_pos"]), np.asarray(human_transl), float(human_height))
        if resid > 0.15:
            import warnings
            warnings.warn(f"{clip_meta.get('clip_id')}: GMR 世界变换拟合残差 {resid:.3f} m 偏大，物体可能对不准")
        objects = [transform_object_track(o, world_scale, world_offset) for o in (objects or [])]

    def _cut_scale(scale, T0, n):
        # 只有逐帧的缩放 (T,) / (T,3) 才需要裁剪；(1,) 和 (3,) 是全局的
        per_frame = scale.ndim == 2 or (scale.ndim == 1 and scale.shape[0] == T0 and T0 > 3)
        return scale[:n] if per_frame else scale

    objs = []
    for o in objects or []:
        # 重定向可能丢帧（GMR 偶尔跳过失败帧），按长度对齐
        T0 = o.pose.shape[0]
        n = min(T, T0)
        objs.append(ObjectTrack(
            name=o.name, mesh_path=o.mesh_path,
            pose=o.pose[:n], scale=_cut_scale(o.scale, T0, n),
            active=o.active[:n],
        ))
    if objs and any(o.pose.shape[0] != T for o in objs):
        T = min(T, *(o.pose.shape[0] for o in objs))

    body_pos = gmr.get("local_body_pos")
    rm = RobotMotion(
        clip_id=clip_meta["clip_id"],
        source=clip_meta["source"],
        fps=float(gmr["fps"]),
        root_pos=np.asarray(gmr["root_pos"], dtype=np.float32)[:T],
        root_rot=root_rot[:T],
        dof_pos=np.asarray(gmr["dof_pos"], dtype=np.float32)[:T],
        body_pos=np.asarray(body_pos, dtype=np.float32)[:T] if body_pos is not None else None,
        body_names=tuple(gmr["link_body_list"]) if gmr.get("link_body_list") else None,
        scene_name=clip_meta.get("scene_name"),
        scene_mesh_path=clip_meta.get("scene_mesh_path"),
        objects=[ObjectTrack(o.name, o.mesh_path, o.pose[:T], _cut_scale(o.scale, o.pose.shape[0], T), o.active[:T]) for o in objs],
        text=clip_meta.get("text"),
        bad_frames=np.asarray([b for b in (bad_frames if bad_frames is not None else []) if b < T], dtype=np.int64),
        world_scale=float(world_scale),
        world_offset=np.asarray(world_offset, dtype=np.float32),
    )
    rm.validate()
    return rm
