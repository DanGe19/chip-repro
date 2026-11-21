"""用伪造的小数据集测试两个加载器。不需要真实数据，不需要 GPU。"""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from chip.data import MotionClip
from chip.data.omomo import iter_omomo_clips, sanity_check as omomo_check
from chip.data.trumans import (
    YUP_TO_ZUP,
    is_idle_segment,
    iter_trumans_clips,
    parse_object_entry,
    sanity_check as trumans_check,
    yup_to_zup_axis_angle,
    yup_to_zup_points,
)

joblib = pytest.importorskip("joblib")


# ---------------------------------------------------------------------------
# 伪造数据
# ---------------------------------------------------------------------------

def make_fake_omomo(root: Path, n_frames=60):
    """按官方格式造 3 条序列：两条论文物体、一条不在子集里的物体。"""
    rng = np.random.default_rng(0)
    data = {}
    for i, obj in enumerate(["plasticbox", "woodchair", "clothesstand"]):
        T = n_frames
        trans = np.zeros((T, 3), np.float32)
        trans[:, 0] = np.linspace(0, 1, T)
        trans[:, 2] = 0.95                            # Z 朝上，骨盆高 0.95 m
        data[i] = {
            "seq_name": f"sub16_{obj}_00{i}",
            "betas": rng.normal(size=(1, 16)).astype(np.float32),
            "gender": "male",
            "trans": trans,
            "root_orient": np.tile(R.from_euler("x", 90, degrees=True).as_rotvec(), (T, 1)).astype(np.float32),
            "pose_body": rng.normal(scale=0.1, size=(T, 63)).astype(np.float32),
            "obj_trans": np.tile(np.array([[0.5], [0.0], [0.4]], np.float32), (T, 1, 1)),
            "obj_rot": np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
            "obj_scale": np.ones(T, np.float32),
            "obj_com_pos": np.tile(np.array([0.5, 0.0, 0.4], np.float32), (T, 1)),
        }
    root.mkdir(parents=True)
    joblib.dump(data, root / "train_diffusion_manip_seq_joints24.p")


def make_fake_trumans(root: Path):
    """按官方格式造 3 个片段：idle 60 帧、有物体 50 帧、idle 但太短 10 帧。"""
    rng = np.random.default_rng(1)
    segs = [("seg_idle", 60, False), ("seg_obj", 50, True), ("seg_short", 10, False)]
    N = sum(t for _, t, _ in segs)
    seg_name = np.concatenate([[n] * t for n, t, _ in segs]).astype("<U16")
    frame_id = np.concatenate([np.arange(t) for _, t, _ in segs])

    # Y 朝上：骨盆高度放在 y 分量
    transl = np.zeros((N, 3), np.float32)
    transl[:, 1] = 0.9
    transl[:, 2] = np.linspace(0, 2, N)
    orient = np.zeros((N, 3), np.float32)             # Y 朝上下 identity 就是站直
    pose = rng.normal(scale=0.1, size=(N, 63)).astype(np.float32)
    hands = np.zeros((N, 45), np.float32)

    object_flag = -np.ones((N, 35), np.int64)
    object_mat = np.tile(np.eye(4, dtype=np.float32), (N, 5, 1, 1))
    s = 60
    object_flag[s:s + 50, 3] = 1                       # 第 3 个物体，用 object_mat[:, 1]
    object_mat[s:s + 50, 1, :3, 3] = [0.3, 0.7, 0.2]   # Y 朝上，物体高 0.7

    # 动作标签 (N,10)：seg_obj 的后半段标成"正在用手机"；人体关节 (N,24,3)，Y 朝上，站着（骨盆比脚高 0.9）
    action = np.zeros((N, 10), np.float64)
    action[s + 25:s + 50, 5] = 1.0
    joints = np.zeros((N, 24, 3), np.float32)
    joints[:, 0, 1] = 0.9
    joints[100:110, 0, 1] = 0.4                        # seg_obj 末尾 10 帧"坐下"

    root.mkdir(parents=True)
    np.save(root / "action_label.npy", action)
    np.save(root / "human_joints.npy", joints)
    np.save(root / "seg_name.npy", seg_name)
    np.save(root / "frame_id.npy", frame_id)
    np.save(root / "human_transl.npy", transl)
    np.save(root / "human_orient.npy", orient)
    np.save(root / "human_pose.npy", pose)
    np.save(root / "left_hand_pose.npy", hands)
    np.save(root / "right_hand_pose.npy", hands)
    np.save(root / "scene_list.npy", np.array(["sceneA", "sceneB"]))
    np.save(root / "scene_flag.npy", np.zeros(N, np.int64))
    np.save(root / "object_list.npy", np.array([f"obj{i}" for i in range(35)]))
    np.save(root / "object_flag.npy", object_flag)
    np.save(root / "object_mat.npy", object_mat)
    np.save(root / "bad_frames.npy", np.array([5, 7]))


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

def test_omomo_loader_filters_to_paper_objects(tmp_path):
    make_fake_omomo(tmp_path / "omomo")
    clips = list(iter_omomo_clips(tmp_path / "omomo"))
    assert [c.objects[0].name for c in clips] == ["plasticbox", "woodchair"]
    c = clips[0]
    assert c.num_frames == 60 and c.fps == 30
    assert c.body_pose.shape == (60, 21, 3)
    assert c.objects[0].pose.shape == (60, 4, 4)
    assert np.allclose(c.objects[0].pose[0, :3, 3], [0.5, 0.0, 0.4])
    assert omomo_check(c) == []


def test_trumans_idle_filter_and_axis_conversion(tmp_path):
    make_fake_trumans(tmp_path / "trumans")
    clips = list(iter_trumans_clips(tmp_path / "trumans", mode="segments", subset_filter=is_idle_segment, min_frames=30))
    assert [c.clip_id for c in clips] == ["trumans/seg_idle"]   # 有物体的和太短的被过滤
    c = clips[0]
    assert c.num_frames == 60
    assert np.allclose(c.transl[0], [0, 0, 0.9], atol=1e-6)      # y=0.9 → z=0.9
    assert c.scene_mesh_path == "Scene_mesh/sceneA.obj"
    assert c.left_hand_pose.shape == (60, 15, 3)
    assert c.bad_frames.tolist() == [5, 7]
    assert trumans_check(c) == []


def test_trumans_all_segments_keeps_object(tmp_path):
    make_fake_trumans(tmp_path / "trumans")
    clips = {c.clip_id: c for c in iter_trumans_clips(tmp_path / "trumans", mode="segments", subset_filter=None, min_frames=30)}
    c = clips["trumans/seg_obj"]
    assert len(c.objects) == 1 and c.objects[0].name == "obj3"
    assert c.objects[0].active.all()
    # 物体位置 (0.3, 0.7, 0.2)_Yup → (0.3, -0.2, 0.7)_Zup
    assert np.allclose(c.objects[0].pose[0, :3, 3], [0.3, -0.2, 0.7], atol=1e-6)


def test_yup_to_zup_is_a_proper_rotation():
    assert np.allclose(YUP_TO_ZUP @ YUP_TO_ZUP.T, np.eye(3))
    assert np.isclose(np.linalg.det(YUP_TO_ZUP), 1.0)
    assert np.allclose(yup_to_zup_points(np.array([0, 1, 0.0])), [0, 0, 1])
    # 轴角转换后，原本的"上"（+Y）要变成 +Z
    aa = R.from_euler("y", 30, degrees=True).as_rotvec()[None]
    up = R.from_rotvec(yup_to_zup_axis_angle(aa)).apply([0, 1, 0])
    assert np.allclose(up, [0, 0, 1], atol=1e-6)


def test_motion_clip_roundtrip(tmp_path):
    make_fake_omomo(tmp_path / "omomo")
    c = next(iter_omomo_clips(tmp_path / "omomo"))
    p = tmp_path / "x.npz"
    c.save(p)
    c2 = MotionClip.load(p)
    assert c2.clip_id == c.clip_id
    assert np.allclose(c2.body_pose, c.body_pose)
    assert c2.objects[0].name == "plasticbox"
    assert np.allclose(c2.objects[0].pose, c.objects[0].pose)


def test_build_index_cli(tmp_path):
    from chip.data.build_index import main
    make_fake_omomo(tmp_path / "omomo")
    make_fake_trumans(tmp_path / "trumans")
    out = tmp_path / "clips"
    main(["--omomo", str(tmp_path / "omomo"), "--trumans", str(tmp_path / "trumans"), "--out", str(out),
          "--trumans-min-frames", "20"])
    assert (out / "index.csv").exists()
    assert (out / "omomo" / "sub16_plasticbox_000.npz").exists()
    assert (out / "trumans" / "seg_idle@000000.npz").exists()
    stats = (out / "index_stats.txt").read_text()
    assert "omomo" in stats and "trumans" in stats


def test_is_idle_segment():
    assert is_idle_segment(-np.ones((10, 35)))
    f = -np.ones((10, 35)); f[3, 0] = 1
    assert not is_idle_segment(f)


def test_parse_object_entry():
    name, scale = parse_object_entry("movable_chair_seat_01(0.87,1.23,1.17)")
    assert name == "movable_chair_seat_01"
    assert np.allclose(scale, [0.87, 1.23, 1.17])
    name2, scale2 = parse_object_entry("bottle_03")
    assert name2 == "bottle_03" and np.allclose(scale2, 1.0)


def test_trumans_idle_runs_mode(tmp_path):
    """默认模式：seg_idle 整段 60 帧是一段；seg_obj 前 25 帧无标签且站立是一段，
    25–50 帧有动作标签被排除，剩下 0 帧...; seg_short 只有 10 帧不够 min_frames。"""
    make_fake_trumans(tmp_path / "trumans")
    clips = list(iter_trumans_clips(tmp_path / "trumans", mode="idle_runs", min_frames=20))
    ids = [c.clip_id for c in clips]
    assert "trumans/seg_idle@000000" in ids
    assert "trumans/seg_obj@000000" in ids           # 前 25 帧
    lens = {c.clip_id: c.num_frames for c in clips}
    assert lens["trumans/seg_idle@000000"] == 60 and lens["trumans/seg_obj@000000"] == 25
    assert all("seg_short" not in i for i in ids)


def test_trumans_idle_runs_max_hours(tmp_path):
    make_fake_trumans(tmp_path / "trumans")
    # 预算只够 ~1 秒（30 帧）：只能装下 seg_obj 的 25 帧那段，装不下 60 帧的
    clips = list(iter_trumans_clips(tmp_path / "trumans", mode="idle_runs", min_frames=20, max_hours=30 / 30 / 3600))
    assert sum(c.num_frames for c in clips) <= 30
