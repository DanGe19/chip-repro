"""检查 TRUMANS 场景网格和人/机器人是否对齐（排查"机器人穿墙"）。

对一条片段画三张俯视/斜视图（matplotlib，不需要 MuJoCo）：
  A. 原始数据：人体关节 (Y-up, human_joints.npy) + 原始场景 .obj（都不转）     → 数据本身对不对
  B. 我们的 MotionClip：transl (Z-up) + 场景按 Y-up→Z-up 转                   → 加载/转轴对不对
  C. RobotMotion：机器人根轨迹 + 场景 转轴 + 乘 world_scale + 加 world_offset   → 重定向后的世界对不对

每张图：场景顶点投到 XY 平面（俯视），人/机器人轨迹画成折线，起点●终点■。
人在原始数据里不穿墙，B、C 应该也不穿；哪一张开始穿，问题就在哪一环。

用法（Pod）::
    /workspace/isaaclab/isaaclab.sh -p scripts/check_scene_alignment.py \
        --clip trumans/2023-02-13@21-34-28@002133 --out /data/inspect/scene_check.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from chip.data.schema import MotionClip  # noqa: E402
from chip.data.trumans import YUP_TO_ZUP, yup_to_zup_points  # noqa: E402
from chip.retarget.robot_motion import RobotMotion  # noqa: E402


def _panel(ax, verts_xy, traj_xy, title, extra=None):
    # 场景顶点太多，抽样画散点；只画高度在 0.05–2.0 m 的（去掉地板和天花板，看墙和家具）
    ax.scatter(verts_xy[:, 0], verts_xy[:, 1], s=0.3, c="0.6", alpha=0.5, rasterized=True)
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "r-", lw=1.5)
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "go", ms=7)
    ax.plot(traj_xy[-1, 0], traj_xy[-1, 1], "bs", ms=7)
    if extra is not None:
        ax.plot(extra[:, 0], extra[:, 1], "m.", ms=3)
    ax.set_aspect("equal")
    c = traj_xy.mean(axis=0)
    ax.set_xlim(c[0] - 3, c[0] + 3)
    ax.set_ylim(c[1] - 3, c[1] + 3)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def _wall_verts(mesh: trimesh.Trimesh, up_axis: int, lo=0.05, hi=2.0):
    v = mesh.vertices
    h = v[:, up_axis]
    h = h - np.percentile(h, 1)       # 地板≈0
    keep = (h > lo) & (h < hi)
    v = v[keep]
    if len(v) > 60000:
        v = v[np.random.default_rng(0).choice(len(v), 60000, replace=False)]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="如 trumans/2023-02-13@21-34-28@002133")
    ap.add_argument("--clips-dir", default="/data/clips")
    ap.add_argument("--robot-dir", default="/data/robot_clips")
    ap.add_argument("--trumans", default="/data/datasets/trumans")
    ap.add_argument("--out", default="/data/inspect/scene_check.png")
    a = ap.parse_args()

    src, name = a.clip.split("/", 1)
    clip = MotionClip.load(Path(a.clips_dir) / src / f"{name}.npz")
    rm = RobotMotion.load(Path(a.robot_dir) / src / f"{name}.npz")
    root = Path(a.trumans)
    scene_file = root / clip.scene_mesh_path
    print("片段:", clip.clip_id, " 场景:", clip.scene_mesh_path, " 帧数:", clip.num_frames)
    mesh = trimesh.load(scene_file, force="mesh")
    print("场景网格: %d 顶点, bbox min %s max %s (原始坐标)" % (len(mesh.vertices), mesh.bounds[0].round(2), mesh.bounds[1].round(2)))

    # --- A. 原始数据（Y-up）：human_joints 里取骨盆（关节 0），帧范围从 clip_id 的 @start 解析 ---
    seg, start = name.rsplit("@", 1)
    start = int(start)
    seg_names = np.load(root / "seg_name.npy", allow_pickle=True)
    idx = np.where(seg_names == seg)[0]
    s0 = int(idx[0]) + start
    joints = np.load(root / "human_joints.npy", mmap_mode="r")
    pelvis_raw = np.asarray(joints[s0 : s0 + clip.num_frames, 0, :], dtype=np.float64)   # Y-up
    print("A 原始骨盆 Y 高度: 均值 %.2f" % pelvis_raw[:, 1].mean())
    print("   原始 transl 与 human_joints 骨盆差: %.3f m" % np.linalg.norm(
        yup_to_zup_points(pelvis_raw) - clip.transl, axis=1).mean())

    # --- B. MotionClip (Z-up) + 场景转轴 ---
    v_zup = mesh.vertices @ YUP_TO_ZUP.T
    # --- C. RobotMotion + 场景 转轴·缩放·平移 ---
    s, c = float(rm.world_scale), np.asarray(rm.world_offset, dtype=np.float64)
    v_robot = v_zup * s + c
    print("C 世界变换: scale %.3f offset %s" % (s, c.round(3)))
    print("   机器人根 Z 均值 %.2f, 人 transl Z 均值 %.2f" % (rm.root_pos[:, 2].mean(), clip.transl[:, 2].mean()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # A：原始 Y-up，俯视 = (x, z)，墙高用 y
    wa = _wall_verts(mesh, up_axis=1)
    _panel(axes[0], wa[:, [0, 2]], pelvis_raw[:, [0, 2]], "A. 原始 Y-up：human_joints 骨盆 + 原始场景 (俯视 x–z)")
    # B：Z-up
    wb = _wall_verts(trimesh.Trimesh(vertices=v_zup, faces=mesh.faces, process=False), up_axis=2)
    _panel(axes[1], wb[:, :2], clip.transl[:, :2], "B. MotionClip Z-up：transl + 场景转轴 (俯视 x–y)")
    # C：机器人世界
    wc = _wall_verts(trimesh.Trimesh(vertices=v_robot, faces=mesh.faces, process=False), up_axis=2)
    obj_xy = None
    if rm.objects:
        obj_xy = np.asarray(rm.objects[0].pose)[:, :2, 3]
    _panel(axes[2], wc[:, :2], rm.root_pos[:, :2], f"C. RobotMotion：根轨迹 + 场景×{s:.3f}+offset (俯视 x–y)", extra=obj_xy)
    fig.suptitle(f"{clip.clip_id}  ● 起点  ■ 终点  灰=0.05–2 m 高的场景顶点  紫=物体", fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("已写出", a.out)


if __name__ == "__main__":
    main()
