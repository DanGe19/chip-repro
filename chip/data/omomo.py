"""OMOMO 数据集加载器 → MotionClip。

OMOMO 原始格式（来自官方仓库 manip/data/hand_foot_dataset.py）：
    data/
      train_diffusion_manip_seq_joints24.p   # joblib pickle: {idx: seq_dict}
      test_diffusion_manip_seq_joints24.p
      captured_objects/<obj>_cleaned_simplified.obj
      omomo_text_anno_json_data/             # 可选，文本标注

每个 seq_dict 的字段：
    seq_name     str      例如 "sub16_plasticbox_000"，物体名在中间
    betas        (1, 16)
    gender       str
    trans        (T, 3)
    root_orient  (T, 3)   轴角
    pose_body    (T, 63)  = 21 关节 × 3，轴角
    obj_trans    (T, 3, 1)
    obj_rot      (T, 3, 3)
    obj_scale    (T,)
    obj_com_pos  (T, 3)   物体质心

论文只用了 4 个物体：trashcan、plasticbox、largetable、woodchair。

坐标系：OMOMO 为 Z 轴朝上，和我们的约定一致，不做转换。
    TODO(验证)：拿到真实数据后跑 sanity_check()，确认 transl[:, 2] 在 0.8–1.1 m 附近。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .schema import MotionClip, ObjectTrack, SMPLX_NUM_BODY_JOINTS

log = logging.getLogger(__name__)

OMOMO_FPS = 30.0

# 论文附录 A 用的子集。不在这里的物体一律跳过。
PAPER_OBJECTS = ("trashcan", "plasticbox", "largetable", "woodchair")


def _object_name_from_seq(seq_name: str) -> str:
    # "sub16_plasticbox_000" → "plasticbox"
    parts = seq_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"无法从 seq_name 解析物体名: {seq_name!r}")
    return parts[1]


def _load_text_annotations(anno_dir: Optional[Path]) -> dict[str, str]:
    """读 omomo_text_anno_json_data/ 下的文本标注。格式不确定，做了宽松解析。"""
    if anno_dir is None or not anno_dir.exists():
        return {}
    out: dict[str, str] = {}
    for p in sorted(anno_dir.glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:  # noqa: BLE001
            log.warning("读取文本标注失败 %s: %s", p, e)
            continue
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str):
                    out[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], str):
                    out[k] = v[0]
                elif isinstance(v, dict) and "text" in v:
                    out[k] = str(v["text"])
    return out


def iter_omomo_clips(
    data_root: str | Path,
    split: str = "train",
    objects: tuple[str, ...] = PAPER_OBJECTS,
    min_frames: int = 30,
) -> Iterator[MotionClip]:
    """逐条产出 OMOMO 序列。

    Args:
        data_root: OMOMO 的 data/ 目录。
        split: "train" 或 "test"。
        objects: 只保留这些物体。传 None 表示全部。
        min_frames: 太短的序列跳过。
    """
    import joblib  # 延迟导入，纯 numpy 环境不装也能 import 本模块

    data_root = Path(data_root)
    pkl = data_root / f"{split}_diffusion_manip_seq_joints24.p"
    if not pkl.exists():
        raise FileNotFoundError(pkl)

    text_map = _load_text_annotations(data_root / "omomo_text_anno_json_data")
    data_dict = joblib.load(pkl)

    n_total, n_kept = 0, 0
    for idx in sorted(data_dict.keys()):
        n_total += 1
        s = data_dict[idx]
        seq_name = str(s["seq_name"])
        obj_name = _object_name_from_seq(seq_name)
        if objects is not None and obj_name not in objects:
            continue

        T = s["trans"].shape[0]
        if T < min_frames:
            continue

        body_pose = np.asarray(s["pose_body"], dtype=np.float32).reshape(T, SMPLX_NUM_BODY_JOINTS, 3)

        # 物体位姿 → (T, 4, 4)
        obj_trans = np.asarray(s["obj_trans"], dtype=np.float32).reshape(T, 3)
        obj_rot = np.asarray(s["obj_rot"], dtype=np.float32).reshape(T, 3, 3)
        pose = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        pose[:, :3, :3] = obj_rot
        pose[:, :3, 3] = obj_trans

        obj = ObjectTrack(
            name=obj_name,
            mesh_path=f"captured_objects/{obj_name}_cleaned_simplified.obj",
            pose=pose,
            scale=np.asarray(s["obj_scale"], dtype=np.float32).reshape(T),
            active=np.ones(T, dtype=bool),   # OMOMO 整段都是物体交互
        )

        clip = MotionClip(
            clip_id=f"omomo/{seq_name}",
            source="omomo",
            fps=OMOMO_FPS,
            transl=np.asarray(s["trans"], dtype=np.float32),
            root_orient=np.asarray(s["root_orient"], dtype=np.float32),
            body_pose=body_pose,
            betas=np.asarray(s["betas"], dtype=np.float32).reshape(-1),
            gender=str(s.get("gender", "neutral")),
            objects=[obj],
            text=text_map.get(seq_name),
        )
        clip.validate()
        n_kept += 1
        yield clip

    log.info("OMOMO %s: 共 %d 条，保留 %d 条（物体=%s）", split, n_total, n_kept, objects)


def sanity_check(clip: MotionClip) -> list[str]:
    """拿到真实数据后跑一次，返回可疑点列表。空列表表示正常。"""
    issues = []
    z = clip.transl[:, 2]
    if not (0.5 < float(np.median(z)) < 1.3):
        issues.append(f"骨盆高度中位数 {np.median(z):.2f} m，不像 Z 朝上；检查坐标系")
    if clip.objects:
        oz = clip.objects[0].pose[:, 2, 3]
        if float(np.median(oz)) < -0.5 or float(np.median(oz)) > 2.5:
            issues.append(f"物体高度中位数 {np.median(oz):.2f} m，可疑")
    step = np.linalg.norm(np.diff(clip.transl, axis=0), axis=1)
    if step.size and float(step.max()) > 0.5:
        issues.append(f"单帧位移最大 {step.max():.2f} m，可能有跳帧")
    return issues
