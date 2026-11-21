"""统一的动作片段格式。

两个数据集（TRUMANS、OMOMO）格式完全不同，这里定义一个中间格式 MotionClip，
各自的加载器把原始数据翻译成它。后续所有阶段（重定向、Stage I/II/III）只认这个格式。

约定：
- 坐标系：Z 轴朝上（IsaacLab 的习惯）。TRUMANS 原始是 Y 轴朝上，加载时转换。
- 人体：SMPL-X 参数。root_orient / body_pose 是轴角（axis-angle）。
- 物体：可选。每个物体有名字、网格路径、逐帧 4x4 位姿。
- 帧率：统一记录在 fps 里，不在加载时重采样（留给后面按需处理）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# SMPL-X 身体关节数（不含根、手、脸）
SMPLX_NUM_BODY_JOINTS = 21


@dataclass
class ObjectTrack:
    """一个物体在整段序列里的轨迹。"""

    name: str                       # 物体名，例如 "plasticbox"
    mesh_path: str                  # OBJ 网格的路径（相对数据根目录）
    pose: np.ndarray                # (T, 4, 4) 世界坐标系下的齐次变换
    scale: np.ndarray               # (T,) 或 (1,)，网格缩放；OMOMO 有逐帧缩放
    active: np.ndarray              # (T,) bool，该帧物体是否参与交互（论文的 m_{o,t}）


@dataclass
class MotionClip:
    """一条动作序列。"""

    # 身份信息
    clip_id: str                    # 全局唯一，例如 "omomo/sub16_plasticbox_000"
    source: str                     # "trumans" | "omomo" | "custom"
    fps: float

    # SMPL-X 人体参数，T 帧
    transl: np.ndarray              # (T, 3)
    root_orient: np.ndarray         # (T, 3)   轴角
    body_pose: np.ndarray           # (T, 21, 3) 轴角
    betas: np.ndarray               # (B,)     体型，B=10 或 16
    gender: str = "neutral"

    # 可选的手部姿态（TRUMANS 有，OMOMO 没有）
    left_hand_pose: Optional[np.ndarray] = None    # (T, 15, 3)
    right_hand_pose: Optional[np.ndarray] = None   # (T, 15, 3)

    # 上下文
    scene_name: Optional[str] = None               # TRUMANS 场景名；OMOMO 为 None
    scene_mesh_path: Optional[str] = None          # 场景 OBJ 路径（相对数据根目录）
    objects: list[ObjectTrack] = field(default_factory=list)
    text: Optional[str] = None                     # 动作文本描述（OMOMO 有）

    # 质量标记
    bad_frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # ---- 便捷属性 ----
    @property
    def num_frames(self) -> int:
        return int(self.transl.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    @property
    def has_object(self) -> bool:
        return len(self.objects) > 0

    @property
    def has_scene(self) -> bool:
        return self.scene_mesh_path is not None

    def validate(self) -> None:
        """检查各数组形状是否一致。加载器最后都应调用一次。"""
        T = self.num_frames
        assert self.transl.shape == (T, 3), self.transl.shape
        assert self.root_orient.shape == (T, 3), self.root_orient.shape
        assert self.body_pose.shape == (T, SMPLX_NUM_BODY_JOINTS, 3), self.body_pose.shape
        assert self.betas.ndim == 1
        for hp in (self.left_hand_pose, self.right_hand_pose):
            if hp is not None:
                assert hp.shape == (T, 15, 3), hp.shape
        for o in self.objects:
            assert o.pose.shape == (T, 4, 4), (o.name, o.pose.shape)
            assert o.active.shape == (T,), (o.name, o.active.shape)
            assert o.scale.shape in ((T,), (1,)), (o.name, o.scale.shape)
        if self.bad_frames.size:
            assert self.bad_frames.min() >= 0 and self.bad_frames.max() < T

    # ---- 存取 ----
    def save(self, path: str | Path) -> None:
        """存成一个 .npz。物体和标量字段打包进 meta。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "transl": self.transl,
            "root_orient": self.root_orient,
            "body_pose": self.body_pose,
            "betas": self.betas,
            "bad_frames": self.bad_frames,
        }
        if self.left_hand_pose is not None:
            arrays["left_hand_pose"] = self.left_hand_pose
        if self.right_hand_pose is not None:
            arrays["right_hand_pose"] = self.right_hand_pose
        for i, o in enumerate(self.objects):
            arrays[f"obj{i}_pose"] = o.pose
            arrays[f"obj{i}_scale"] = o.scale
            arrays[f"obj{i}_active"] = o.active
        meta = {
            "clip_id": self.clip_id,
            "source": self.source,
            "fps": self.fps,
            "gender": self.gender,
            "scene_name": self.scene_name,
            "scene_mesh_path": self.scene_mesh_path,
            "text": self.text,
            "objects": [{"name": o.name, "mesh_path": o.mesh_path} for o in self.objects],
        }
        np.savez_compressed(path, meta=np.array(meta, dtype=object), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "MotionClip":
        d = np.load(path, allow_pickle=True)
        meta = d["meta"].item()
        objects = []
        for i, om in enumerate(meta["objects"]):
            objects.append(
                ObjectTrack(
                    name=om["name"],
                    mesh_path=om["mesh_path"],
                    pose=d[f"obj{i}_pose"],
                    scale=d[f"obj{i}_scale"],
                    active=d[f"obj{i}_active"],
                )
            )
        clip = cls(
            clip_id=meta["clip_id"],
            source=meta["source"],
            fps=float(meta["fps"]),
            transl=d["transl"],
            root_orient=d["root_orient"],
            body_pose=d["body_pose"],
            betas=d["betas"],
            gender=meta["gender"],
            left_hand_pose=d["left_hand_pose"] if "left_hand_pose" in d else None,
            right_hand_pose=d["right_hand_pose"] if "right_hand_pose" in d else None,
            scene_name=meta["scene_name"],
            scene_mesh_path=meta["scene_mesh_path"],
            objects=objects,
            text=meta["text"],
            bad_frames=d["bad_frames"],
        )
        clip.validate()
        return clip

    def summary(self) -> dict:
        """给索引表用的一行摘要。"""
        return {
            "clip_id": self.clip_id,
            "source": self.source,
            "num_frames": self.num_frames,
            "duration_s": round(self.duration_s, 2),
            "has_scene": self.has_scene,
            "objects": [o.name for o in self.objects],
            "num_bad_frames": int(self.bad_frames.size),
        }
