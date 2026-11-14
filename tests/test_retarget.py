"""重定向模块测试：不需要 GMR 环境，用伪造的 GMR 输出。"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from chip.data.schema import MotionClip, ObjectTrack
from chip.retarget.export_smplx import clip_to_gmr_npz, export_clips_for_gmr, safe_name
from chip.retarget.robot_motion import (
    G1_DOF_NAMES, G1_NUM_DOF, RobotMotion, load_gmr_pickle, robot_motion_from_gmr,
)


def make_clip(T=40, with_object=True, clip_id="omomo/sub1_plasticbox_000"):
    rng = np.random.default_rng(0)
    objects = []
    if with_object:
        pose = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        pose[:, 0, 3] = np.linspace(0, 1, T)
        objects.append(ObjectTrack("plasticbox", "captured_objects/plasticbox.obj",
                                   pose, np.ones(T, np.float32), np.ones(T, bool)))
    c = MotionClip(
        clip_id=clip_id, source="omomo", fps=30.0,
        transl=rng.normal(size=(T, 3)).astype(np.float32),
        root_orient=rng.normal(scale=0.1, size=(T, 3)).astype(np.float32),
        body_pose=rng.normal(scale=0.1, size=(T, 21, 3)).astype(np.float32),
        betas=rng.normal(size=16).astype(np.float32),
        gender="male", objects=objects, text="push the box", bad_frames=np.array([3, T - 2]),
    )
    c.validate()
    return c


def make_fake_gmr_pickle(path: Path, T: int, fps=30):
    rng = np.random.default_rng(1)
    q = rng.normal(size=(T, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    d = {
        "fps": fps,
        "root_pos": rng.normal(size=(T, 3)),
        "root_rot": q,
        "dof_pos": rng.normal(scale=0.3, size=(T, G1_NUM_DOF)),
        "local_body_pos": rng.normal(size=(T, 38, 3)),
        "link_body_list": [f"body{i}" for i in range(38)],
    }
    with open(path, "wb") as f:
        pickle.dump(d, f)


def test_g1_dof_table():
    assert G1_NUM_DOF == 29
    assert G1_DOF_NAMES[0] == "left_hip_pitch_joint" and G1_DOF_NAMES[-1] == "right_wrist_yaw_joint"
    assert len(set(G1_DOF_NAMES)) == 29


def test_export_npz_has_gmr_keys(tmp_path):
    c = make_clip()
    p = clip_to_gmr_npz(c, tmp_path / "x.npz", num_betas=16)
    d = np.load(p, allow_pickle=True)
    assert str(d["gender"]) == "male"
    assert d["betas"].shape == (16,)
    assert d["pose_body"].shape == (40, 63)
    assert d["root_orient"].shape == (40, 3) and d["trans"].shape == (40, 3)
    assert float(d["mocap_frame_rate"]) == 30.0


def test_export_pads_betas(tmp_path):
    c = make_clip(); c.betas = np.zeros(10, np.float32)
    d = np.load(clip_to_gmr_npz(c, tmp_path / "y.npz", num_betas=16), allow_pickle=True)
    assert d["betas"].shape == (16,)


def test_safe_name():
    assert safe_name("trumans/seg_001") == "trumans__seg_001"


def test_gmr_to_robot_motion_carries_context(tmp_path):
    c = make_clip(T=40)
    make_fake_gmr_pickle(tmp_path / "g.pkl", T=40)
    gmr = load_gmr_pickle(tmp_path / "g.pkl")
    rm = robot_motion_from_gmr(
        gmr, {"clip_id": c.clip_id, "source": c.source, "scene_name": None,
              "scene_mesh_path": None, "text": c.text},
        objects=c.objects, bad_frames=c.bad_frames,
    )
    assert rm.num_frames == 40
    assert rm.dof_pos.shape == (40, 29)
    assert rm.objects[0].name == "plasticbox"
    assert rm.text == "push the box"
    assert rm.bad_frames.tolist() == [3, 38]
    assert rm.body_names[0] == "body0" and rm.body_pos.shape == (40, 38, 3)


def test_gmr_dropped_frames_truncates_objects(tmp_path):
    """GMR 输出比输入少几帧时，物体轨迹和坏帧都要对齐截断。"""
    c = make_clip(T=40)
    make_fake_gmr_pickle(tmp_path / "g.pkl", T=37)
    rm = robot_motion_from_gmr(load_gmr_pickle(tmp_path / "g.pkl"),
                               {"clip_id": c.clip_id, "source": c.source},
                               objects=c.objects, bad_frames=c.bad_frames)
    assert rm.num_frames == 37
    assert rm.objects[0].pose.shape == (37, 4, 4)
    assert rm.bad_frames.tolist() == [3]        # 38 超出范围被丢掉


def test_robot_motion_roundtrip(tmp_path):
    c = make_clip()
    make_fake_gmr_pickle(tmp_path / "g.pkl", T=40)
    rm = robot_motion_from_gmr(load_gmr_pickle(tmp_path / "g.pkl"),
                               {"clip_id": c.clip_id, "source": c.source}, objects=c.objects)
    rm.save(tmp_path / "rm.npz")
    rm2 = RobotMotion.load(tmp_path / "rm.npz")
    assert np.allclose(rm2.dof_pos, rm.dof_pos)
    assert rm2.dof_names == G1_DOF_NAMES
    assert rm2.objects[0].name == "plasticbox"


def test_run_gmr_skip_mode_end_to_end(tmp_path):
    """用 --skip-gmr 走完整条流水线：导出 → (伪造 GMR 输出) → 收集。"""
    from chip.retarget.run_gmr import main
    clips = tmp_path / "clips"
    for i in range(2):
        make_clip(T=30 + i, clip_id=f"omomo/seq_{i}").save(clips / "omomo" / f"seq_{i}.npz")
    work = tmp_path / "work"
    # 先导出，拿到 GMR 会用的文件名，再伪造对应输出
    manifest = export_clips_for_gmr(sorted(clips.rglob("*.npz")), work / "gmr_src")
    (work / "gmr_out").mkdir()
    for m in manifest[:1]:                       # 只给第一条造输出，第二条模拟失败
        make_fake_gmr_pickle(work / "gmr_out" / f"{m['name']}.pkl", T=m["num_frames"])
    out = tmp_path / "robot"
    main(["--clips", str(clips), "--out", str(out), "--work", str(work), "--skip-gmr"])
    assert (out / "omomo" / "seq_0.npz").exists()
    assert not (out / "omomo" / "seq_1.npz").exists()
    assert "seq_0" in (out / "index.csv").read_text()
