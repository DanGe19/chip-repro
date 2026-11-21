"""离屏检查重定向结果：数值体检 + 渲染视频/拼图。

GMR 自带的 ``vis_robot_motion.py`` 需要弹出 MuJoCo 窗口，在没有显示器的
云端 Pod 上用不了。这个脚本用 EGL 离屏渲染，顺便算几项"肉眼之外"的指标：

- 脚底最低高度（负得多 = 穿地；正得多 = 悬空）
- 脚在"着地"时的水平速度（大 = 滑步）
- 关节角是否超出 URDF 限位
- 手腕与物体中心的最近距离（搬东西的片段，手应该贴近物体）

用法（在 Pod 上，GMR 装在 IsaacLab 的 python 里）::

    MUJOCO_GL=egl /workspace/isaaclab/isaaclab.sh -p -m chip.retarget.inspect_robot_motion \
        --motion /data/robot_clips_test/omomo/sub11_trashcan_000.npz \
        --gmr-root /data/GMR --out /data/inspect/sub11_trashcan_000

输出：``<out>.mp4``（有 ffmpeg 时）、``<out>_sheet.png``（12 帧拼图）、``<out>_report.txt``。
也可以直接给 GMR 的 ``.pkl``。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 必须在 import mujoco 之前设置，否则会尝试打开 GLFW 窗口
os.environ.setdefault("MUJOCO_GL", "egl")

FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
HAND_BODIES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
FOOT_SOLE_OFFSET = 0.035  # ankle_roll_link 原点到脚底的大致距离（m）


def _load_motion(path: Path):
    """返回 (fps, root_pos (T,3), root_rot_xyzw (T,4), dof_pos (T,29), objects, meta)。"""
    if path.suffix == ".pkl":
        import pickle

        with open(path, "rb") as f:
            d = pickle.load(f)
        return (
            float(d["fps"]),
            np.asarray(d["root_pos"], dtype=np.float64),
            np.asarray(d["root_rot"], dtype=np.float64),
            np.asarray(d["dof_pos"], dtype=np.float64),
            [],
            {"source": "gmr_pkl"},
        )
    from chip.retarget.robot_motion import RobotMotion

    rm = RobotMotion.load(path)
    objects = [(o.name, np.asarray(o.pose, dtype=np.float64)) for o in rm.objects]
    meta = {"source": rm.source, "clip_id": rm.clip_id}
    print(f"世界变换: scale {rm.world_scale:.3f}, offset {np.round(rm.world_offset, 3).tolist()}"
          + ("  ← 未套 GMR 变换，物体会对不上" if rm.world_scale == 1.0 and rm.objects else ""))
    return float(rm.fps), rm.root_pos.astype(np.float64), rm.root_rot.astype(np.float64), rm.dof_pos.astype(np.float64), objects, meta


def _body_positions(model, data, root_pos, root_rot_xyzw, dof_pos, names):
    import mujoco as mj

    ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) for n in names]
    T = root_pos.shape[0]
    out = np.zeros((T, len(ids), 3))
    for t in range(T):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_xyzw[t][[3, 0, 1, 2]]  # mujoco 是 wxyz
        data.qpos[7:] = dof_pos[t]
        mj.mj_kinematics(model, data)
        out[t] = data.xpos[ids]
    return out


def numeric_report(model, data, fps, root_pos, root_rot, dof_pos, objects) -> str:
    lines = []
    T = root_pos.shape[0]
    lines.append(f"帧数 {T}, fps {fps:g}, 时长 {T / fps:.2f}s")

    # 1) 脚
    feet = _body_positions(model, data, root_pos, root_rot, dof_pos, FOOT_BODIES)
    sole_z = feet[:, :, 2] - FOOT_SOLE_OFFSET
    lines.append(f"脚底最低高度: 左 {sole_z[:, 0].min():+.3f} m, 右 {sole_z[:, 1].min():+.3f} m  (理想 ≈ 0, < -0.03 算穿地)")
    lines.append(f"脚底中位高度: 左 {np.median(sole_z[:, 0]):+.3f} m, 右 {np.median(sole_z[:, 1]):+.3f} m")
    vel = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * fps  # (T-1, 2) 水平速度
    contact = sole_z[1:] < 0.05
    for i, side in enumerate(["左", "右"]):
        v = vel[contact[:, i], i]
        if v.size:
            lines.append(f"{side}脚着地时水平速度: 中位 {np.median(v):.3f} m/s, 95% {np.percentile(v, 95):.3f} m/s  (> 0.3 算明显滑步)")

    # 2) 关节限位
    lo = model.jnt_range[1:, 0]
    hi = model.jnt_range[1:, 1]
    over = np.maximum(dof_pos - hi, lo - dof_pos)
    worst = over.max(axis=0)
    bad = np.where(worst > 1e-3)[0]
    if bad.size:
        import mujoco as mj

        names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j + 1) for j in bad]
        lines.append("超出限位的关节: " + ", ".join(f"{n}(+{worst[j]:.3f} rad)" for n, j in zip(names, bad)))
    else:
        lines.append("关节限位: 全部在范围内")

    # 3) 根部
    lines.append(f"骨盆高度: 均值 {root_pos[:, 2].mean():.3f} m, 最低 {root_pos[:, 2].min():.3f} m  (G1 站直 ≈ 0.79)")
    lines.append(f"水平位移: {np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]):.2f} m")

    # 4) 手与物体
    if objects:
        hands = _body_positions(model, data, root_pos, root_rot, dof_pos, HAND_BODIES)
        for name, pose in objects:
            if pose is None or pose.ndim != 3:
                continue
            c = pose[:T, :3, 3]
            d = np.linalg.norm(hands - c[:, None, :], axis=-1)  # (T, 2)
            lines.append(f"物体 {name}: 左手最近 {d[:, 0].min():.3f} m, 右手最近 {d[:, 1].min():.3f} m, 物体中心高度 {c[:, 2].min():.2f}~{c[:, 2].max():.2f} m")
    return "\n".join(lines)


def render(model, data, fps, root_pos, root_rot, dof_pos, out_stem: Path, width=640, height=480, max_frames=600):
    import mujoco as mj

    T = root_pos.shape[0]
    step = max(1, int(np.ceil(T / max_frames)))
    renderer = mj.Renderer(model, height=height, width=width)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.azimuth = 135
    cam.elevation = -15
    frames = []
    for t in range(0, T, step):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot[t][[3, 0, 1, 2]]
        data.qpos[7:] = dof_pos[t]
        mj.mj_forward(model, data)
        cam.lookat[:] = [root_pos[t, 0], root_pos[t, 1], 0.6]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    renderer.close()

    written = []
    # 拼图：12 帧
    try:
        from PIL import Image

        idx = np.linspace(0, len(frames) - 1, min(12, len(frames))).astype(int)
        cols = 4
        rows = int(np.ceil(len(idx) / cols))
        sheet = Image.new("RGB", (cols * width, rows * height), "white")
        for k, i in enumerate(idx):
            sheet.paste(Image.fromarray(frames[i]), ((k % cols) * width, (k // cols) * height))
        p = out_stem.with_name(out_stem.name + "_sheet.png")
        sheet.save(p)
        written.append(p)
    except Exception as e:  # noqa: BLE001
        print("拼图失败:", e, file=sys.stderr)
    # 视频
    try:
        import imageio

        p = out_stem.with_suffix(".mp4")
        with imageio.get_writer(p, fps=max(1, int(round(fps / step))), codec="libx264", quality=7) as w:
            for f in frames:
                w.append_data(f)
        written.append(p)
    except Exception as e:  # noqa: BLE001
        print("视频写入失败（没有 ffmpeg 很正常，看拼图即可）:", e, file=sys.stderr)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion", required=True, type=Path, help="RobotMotion .npz 或 GMR .pkl")
    ap.add_argument("--gmr-root", type=Path, default=Path(os.environ.get("GMR_ROOT", "/data/GMR")))
    ap.add_argument("--robot", default="unitree_g1")
    ap.add_argument("--out", type=Path, default=None, help="输出前缀（不带后缀）")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.gmr_root))
    import mujoco as mj
    from general_motion_retargeting import ROBOT_XML_DICT

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[args.robot]))
    data = mj.MjData(model)

    fps, root_pos, root_rot, dof_pos, objects, _ = _load_motion(args.motion)
    assert dof_pos.shape[1] == model.nq - 7, f"dof 数不匹配: {dof_pos.shape[1]} vs {model.nq - 7}"

    out = args.out or args.motion.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = numeric_report(model, data, fps, root_pos, root_rot, dof_pos, objects)
    print(report)
    out.with_name(out.name + "_report.txt").write_text(report + "\n", encoding="utf-8")
    if not args.no_render:
        for p in render(model, data, fps, root_pos, root_rot, dof_pos, out):
            print("已写出", p)


if __name__ == "__main__":
    main()
