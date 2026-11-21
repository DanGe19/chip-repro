"""TRUMANS 数据集加载器 → MotionClip。

TRUMANS 原始格式（来自官方仓库 README）：所有动作片段按帧拼成一个大数组（约 379 万帧，30 fps）。
    Data_release/
      human_pose.npy        (N, 63)   SMPL-X body_pose，轴角
      human_orient.npy      (N, 3)    global_orient
      human_transl.npy      (N, 3)
      left_hand_pose.npy    (N, 45)
      right_hand_pose.npy   (N, 45)
      seg_name.npy          (N,)      每帧属于哪个片段
      frame_id.npy          (N,)      帧在原片段里的编号
      scene_list.npy        场景名列表
      scene_flag.npy        (N,)      每帧的场景索引
      object_list.npy       35 个物体名
      object_flag.npy       (N, 35)   -1 表示不存在，>=1 表示存在且是 object_mat 第二维的索引
      object_mat.npy        (N, 5, 4, 4)
      bad_frames.npy        有问题的帧（全局索引）
      Scene_mesh/<scene>.obj
      Object_all/Object_mesh/<obj>.obj
      Action/               每个片段的文本动作标注（可选）

坐标系：TRUMANS 是 Y 轴朝上，这里转成 Z 轴朝上：(x, y, z)_Yup → (x, -z, y)_Zup。
    场景网格不在这里转（它是文件），由 envs 模块加载网格时用同一个矩阵转。

"idle 子集"：论文附录 A 用了 TRUMANS 的 idle 子集但没写筛选规则。
    这里的默认规则是：整段没有任何物体存在（object_flag 全为 -1）。可以通过参数换规则。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from .schema import MotionClip, ObjectTrack, SMPLX_NUM_BODY_JOINTS

log = logging.getLogger(__name__)

TRUMANS_FPS = 30.0

# Y-up → Z-up 的旋转矩阵
YUP_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]], dtype=np.float32
)
YUP_TO_ZUP_4x4 = np.eye(4, dtype=np.float32)
YUP_TO_ZUP_4x4[:3, :3] = YUP_TO_ZUP


def yup_to_zup_points(p: np.ndarray) -> np.ndarray:
    """(…, 3) 点坐标转换。"""
    return p @ YUP_TO_ZUP.T


def yup_to_zup_axis_angle(aa: np.ndarray) -> np.ndarray:
    """(…, 3) 轴角旋转转换：R_new = M · R_old。"""
    shape = aa.shape
    rot = R.from_rotvec(aa.reshape(-1, 3))
    rot_new = R.from_matrix(YUP_TO_ZUP) * rot
    return rot_new.as_rotvec().reshape(shape).astype(np.float32)


def yup_to_zup_pose(T44: np.ndarray) -> np.ndarray:
    """(…, 4, 4) 齐次变换转换。"""
    return YUP_TO_ZUP_4x4 @ T44


# ---------------------------------------------------------------------------
# 片段切分
# ---------------------------------------------------------------------------

def _segment_ranges(seg_name: np.ndarray) -> list[tuple[str, int, int]]:
    """把逐帧的片段名数组切成 [(name, start, end), …]，end 不含。"""
    n = seg_name.shape[0]
    if n == 0:
        return []
    change = np.nonzero(seg_name[1:] != seg_name[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    return [(str(seg_name[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def is_idle_segment(object_flag_seg: np.ndarray) -> bool:
    """默认的 idle 规则：整段没有任何物体。"""
    return bool(np.all(object_flag_seg < 0))


# ---------------------------------------------------------------------------
# 主加载器
# ---------------------------------------------------------------------------

def iter_trumans_clips(
    data_root: str | Path,
    subset_filter: Optional[Callable[[np.ndarray], bool]] = is_idle_segment,
    min_frames: int = 30,
    load_hands: bool = True,
    drop_bad_segments: bool = False,
) -> Iterator[MotionClip]:
    """逐条产出 TRUMANS 片段。

    Args:
        data_root: Data_release/ 目录。
        subset_filter: 接收该片段的 object_flag (T, 35)，返回是否保留。
            默认只保留 idle（无物体）片段。传 None 保留全部。
        min_frames: 太短的片段跳过。
        load_hands: 是否读手部姿态。
        drop_bad_segments: True 则含 bad_frames 的片段整条丢掉；False 则保留并记录 bad_frames。
    """
    root = Path(data_root)

    def ld(name, mmap=True):
        p = root / name
        if not p.exists():
            raise FileNotFoundError(p)
        return np.load(p, mmap_mode="r" if mmap else None, allow_pickle=not mmap)

    seg_name = np.asarray(ld("seg_name.npy", mmap=False))
    scene_list = [str(x) for x in ld("scene_list.npy", mmap=False)]
    object_list = [str(x) for x in ld("object_list.npy", mmap=False)]
    scene_flag = ld("scene_flag.npy")
    object_flag = ld("object_flag.npy")
    object_mat = ld("object_mat.npy")
    human_pose = ld("human_pose.npy")
    human_orient = ld("human_orient.npy")
    human_transl = ld("human_transl.npy")
    lhand = ld("left_hand_pose.npy") if load_hands and (root / "left_hand_pose.npy").exists() else None
    rhand = ld("right_hand_pose.npy") if load_hands and (root / "right_hand_pose.npy").exists() else None
    # betas.npy：README 没列，但 Drive 文件夹里有。形状未知，可能是 (N, B) 逐帧或 (B,) 全局。
    betas_all = None
    if (root / "betas.npy").exists():
        betas_all = ld("betas.npy")
    bad_frames_all = (
        np.asarray(ld("bad_frames.npy", mmap=False)).astype(np.int64).reshape(-1)
        if (root / "bad_frames.npy").exists() else np.zeros(0, dtype=np.int64)
    )
    bad_set = set(bad_frames_all.tolist())

    action_texts = _load_action_texts(root / "Action")

    n_total, n_kept = 0, 0
    for name, s, e in _segment_ranges(seg_name):
        n_total += 1
        T = e - s
        if T < min_frames:
            continue

        of = np.asarray(object_flag[s:e])
        if subset_filter is not None and not subset_filter(of):
            continue

        bad_local = np.array(sorted(f - s for f in range(s, e) if f in bad_set), dtype=np.int64)
        if drop_bad_segments and bad_local.size:
            continue

        # 场景：取片段第一帧的 scene_flag（同一片段内不会换场景）
        sid = int(scene_flag[s])
        scene_name = scene_list[sid] if 0 <= sid < len(scene_list) else None

        # 物体：对 35 个物体，看该片段里是否出现过
        objects = []
        present_cols = np.nonzero(np.any(of >= 0, axis=0))[0]
        for col in present_cols:
            flags = of[:, col]                       # (T,)，-1 或 >=1
            active = flags >= 0
            pose = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
            om = np.asarray(object_mat[s:e])         # (T, 5, 4, 4)
            idx = np.clip(flags, 0, om.shape[1] - 1)
            pose[active] = om[np.arange(T)[active], idx[active]]
            pose = yup_to_zup_pose(pose)
            obj_name = object_list[col] if col < len(object_list) else f"obj{col}"
            objects.append(
                ObjectTrack(
                    name=obj_name,
                    mesh_path=f"Object_all/Object_mesh/{obj_name}.obj",
                    pose=pose,
                    scale=np.ones(1, dtype=np.float32),
                    active=active,
                )
            )

        clip = MotionClip(
            clip_id=f"trumans/{name}",
            source="trumans",
            fps=TRUMANS_FPS,
            transl=yup_to_zup_points(np.asarray(human_transl[s:e], dtype=np.float32)),
            root_orient=yup_to_zup_axis_angle(np.asarray(human_orient[s:e], dtype=np.float32)),
            body_pose=np.asarray(human_pose[s:e], dtype=np.float32).reshape(T, SMPLX_NUM_BODY_JOINTS, 3),
            betas=_segment_betas(betas_all, s),
            gender="neutral",
            left_hand_pose=np.asarray(lhand[s:e], dtype=np.float32).reshape(T, 15, 3) if lhand is not None else None,
            right_hand_pose=np.asarray(rhand[s:e], dtype=np.float32).reshape(T, 15, 3) if rhand is not None else None,
            scene_name=scene_name,
            scene_mesh_path=f"Scene_mesh/{scene_name}.obj" if scene_name else None,
            objects=objects,
            text=action_texts.get(name),
            bad_frames=bad_local,
        )
        clip.validate()
        n_kept += 1
        yield clip

    log.info("TRUMANS: 共 %d 个片段，保留 %d 个", n_total, n_kept)


def _segment_betas(betas_all, start: int) -> np.ndarray:
    """从 betas.npy 取该片段的体型参数。没有文件就用平均体型（全零）。"""
    if betas_all is None:
        return np.zeros(10, dtype=np.float32)
    b = np.asarray(betas_all)
    if b.ndim == 1:                      # (B,) 全局一套
        return b.astype(np.float32)
    if b.ndim == 2 and b.shape[0] > start:   # (N, B) 逐帧，同一片段内应相同，取首帧
        return np.asarray(b[start], dtype=np.float32)
    return np.asarray(b.reshape(-1)[:16], dtype=np.float32)


def _load_action_texts(action_dir: Path) -> dict[str, str]:
    """Action/ 目录下每个片段一个文本文件。格式未确认，按"文件名=片段名、内容=描述"读。"""
    if not action_dir.exists():
        return {}
    out = {}
    for p in sorted(action_dir.iterdir()):
        if p.is_file():
            try:
                out[p.stem] = p.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                pass
    return out


def sanity_check(clip: MotionClip) -> list[str]:
    """拿到真实数据后跑一次，返回可疑点列表。"""
    issues = []
    z = clip.transl[:, 2]
    if not (0.3 < float(np.median(z)) < 1.3):
        issues.append(f"骨盆高度中位数 {np.median(z):.2f} m，Y→Z 转换可能有误")
    # 转换后"上"方向应该是 +Z：人站着时根关节的局部 Y 轴（SMPL 的躯干方向）应大致指向 +Z
    rot = R.from_rotvec(clip.root_orient[: min(100, clip.num_frames)])
    up = rot.apply(np.array([0.0, 1.0, 0.0]))
    if float(np.median(up[:, 2])) < 0.5:
        issues.append(f"根关节朝向转换后不朝上（median up_z={np.median(up[:, 2]):.2f}）")
    return issues
