"""MotionClip → GMR 能读的 SMPL-X .npz。

GMR 的 load_smplx_file（general_motion_retargeting/utils/smpl.py）用 np.load(allow_pickle=True) 读文件，
要求这些字段：
    gender            str
    betas             (B,)        B=10 或 16
    root_orient       (T, 3)      轴角
    pose_body         (T, 63)     21 关节 × 3
    trans             (T, 3)
    mocap_frame_rate  标量

GMR 假设 Z 轴朝上，和 MotionClip 的约定一致，不用再转。
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from chip.data.schema import MotionClip

log = logging.getLogger(__name__)


def clip_to_gmr_npz(clip: MotionClip, out_path: str | Path, num_betas: int | None = None) -> Path:
    """写一个 GMR 格式的 npz。num_betas 不为 None 时把 betas 补零或截断到该长度。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    betas = np.asarray(clip.betas, dtype=np.float32).reshape(-1)
    if num_betas is not None:
        if betas.shape[0] < num_betas:
            betas = np.concatenate([betas, np.zeros(num_betas - betas.shape[0], np.float32)])
        else:
            betas = betas[:num_betas]

    np.savez(
        out_path,
        gender=np.array(clip.gender),
        betas=betas,
        root_orient=np.asarray(clip.root_orient, dtype=np.float32),
        pose_body=np.asarray(clip.body_pose, dtype=np.float32).reshape(clip.num_frames, -1),
        trans=np.asarray(clip.transl, dtype=np.float32),
        mocap_frame_rate=np.array(clip.fps),
    )
    return out_path


def safe_name(clip_id: str) -> str:
    """'omomo/sub16_plasticbox_000' → 'omomo__sub16_plasticbox_000'，GMR 的批处理按文件名输出。"""
    return clip_id.replace("/", "__")


def export_clips_for_gmr(clip_paths: Iterable[Path], src_folder: str | Path,
                         num_betas: int | None = None) -> list[dict]:
    """把一批 MotionClip .npz 导出到 GMR 的 src_folder。

    返回一个清单，记录每条 clip 的上下文（场景、物体、文本），重定向完了要拼回去。
    清单同时写成 src_folder/manifest.csv。
    """
    src_folder = Path(src_folder)
    src_folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    for p in clip_paths:
        clip = MotionClip.load(p)
        name = safe_name(clip.clip_id)
        clip_to_gmr_npz(clip, src_folder / f"{name}.npz", num_betas=num_betas)
        manifest.append({
            "name": name,
            "clip_id": clip.clip_id,
            "clip_path": str(p),
            "num_frames": clip.num_frames,
            "fps": clip.fps,
        })
    with open(src_folder / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["name"])
        w.writeheader()
        w.writerows(manifest)
    log.info("导出 %d 条到 %s", len(manifest), src_folder)
    return manifest
