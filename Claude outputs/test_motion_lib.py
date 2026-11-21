import numpy as np
import pytest
import torch

from chip.envs.motion_lib import MotionLib
from chip.retarget.robot_motion import G1_DOF_NAMES, G1_NUM_DOF, RobotMotion


def _yaw_quat_xyzw(yaw):
    return np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)], axis=-1)


def make_motion(path, clip_id, T=61, fps=30.0, vx=0.5, yaw_rate=0.3, with_body=True):
    """匀速前进 + 匀速转身 + 关节正弦摆动，速度有解析解。"""
    t = np.arange(T) / fps
    root_pos = np.stack([vx * t, np.zeros(T), np.full(T, 0.75)], axis=1).astype(np.float32)
    root_rot = _yaw_quat_xyzw(yaw_rate * t).astype(np.float32)
    dof = np.zeros((T, G1_NUM_DOF), np.float32)
    dof[:, 0] = 0.3 * np.sin(2 * np.pi * 0.5 * t)
    body_pos = body_names = None
    if with_body:
        # 两个刚体：一个在根上方 0.3 m，一个在根前方 0.2 m（局部坐标）
        body_pos = np.tile(np.array([[0, 0, 0.3], [0.2, 0, 0]], np.float32), (T, 1, 1))
        body_names = ("up_link", "front_link")
    rm = RobotMotion(clip_id=clip_id, source=clip_id.split("/")[0], fps=fps, root_pos=root_pos, root_rot=root_rot,
                     dof_pos=dof, body_pos=body_pos, body_names=body_names)
    rm.save(path)
    return rm


@pytest.fixture
def lib_files(tmp_path):
    files = [tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"]
    make_motion(files[0], "omomo/a", T=61)
    make_motion(files[1], "trumans/b", T=91, vx=0.0, yaw_rate=0.0)
    make_motion(files[2], "trumans/c", T=5)   # 0.13 s，应被 min_duration 过滤
    return files


def test_load_resample_and_summary(lib_files):
    lib = MotionLib(lib_files, target_fps=50.0, body_names=["up_link", "front_link"])
    assert lib.num_clips == 2 and lib.skipped == [("trumans/c", "too_short")]
    # 2 s @30fps → 2 s @50fps = 101 帧
    assert lib._lengths.tolist() == [101, 151]
    assert lib.root_pos.shape == (252, 3) and lib.dof_pos.shape == (252, G1_NUM_DOF)
    assert lib.body_pos.shape == (252, 2, 3)
    assert "2 条片段" in lib.summary()


def test_velocities_match_analytic(lib_files):
    lib = MotionLib(lib_files, body_names=["up_link", "front_link"])
    idx = torch.zeros(3, dtype=torch.long)
    t = torch.tensor([0.3, 1.0, 1.7])
    s = lib.get_state(idx, t)
    assert torch.allclose(s["root_lin_vel"], torch.tensor([[0.5, 0.0, 0.0]]).expand(3, 3), atol=2e-2)
    assert torch.allclose(s["root_ang_vel"], torch.tensor([[0.0, 0.0, 0.3]]).expand(3, 3), atol=2e-2)
    # 关节 0：0.3 sin(πt) → 速度 0.3π cos(πt)
    expect = 0.3 * np.pi * np.cos(np.pi * t.numpy())
    assert np.allclose(s["dof_vel"][:, 0].numpy(), expect, atol=0.05)
    # 根位置插值
    assert torch.allclose(s["root_pos"][:, 0], 0.5 * t, atol=1e-3)


def test_body_pos_is_world_frame(lib_files):
    lib = MotionLib(lib_files, body_names=["up_link", "front_link"])
    idx = torch.zeros(1, dtype=torch.long)
    t = torch.tensor([1.0])
    s = lib.get_state(idx, t)
    yaw = 0.3 * 1.0
    # up_link：根 + (0,0,0.3)
    assert torch.allclose(s["body_pos"][0, 0], torch.tensor([0.5, 0.0, 1.05]), atol=1e-3)
    # front_link：根 + R_yaw·(0.2,0,0)
    expect = torch.tensor([0.5 + 0.2 * np.cos(yaw), 0.2 * np.sin(yaw), 0.75], dtype=torch.float32)
    assert torch.allclose(s["body_pos"][0, 1], expect, atol=1e-3)


def test_sampling_and_future(lib_files):
    torch.manual_seed(0)
    lib = MotionLib(lib_files)
    c = lib.sample_clips(1000)
    # 时长加权：片段 1 是 3 s，片段 0 是 2 s
    frac = (c == 1).float().mean().item()
    assert 0.5 < frac < 0.7
    t = lib.sample_times(c, truncate=0.5)
    assert torch.all(t >= 0) and torch.all(t <= lib.durations[c] - 0.5 + 1e-6)
    fut = lib.get_future(c[:4], t[:4], steps=[0.0, 0.1, 0.2])
    assert fut["dof_pos"].shape == (4, 3, G1_NUM_DOF)
    # 越界时间被夹到片段末尾，不报错
    s_end = lib.get_state(torch.tensor([0]), torch.tensor([99.0]))
    assert torch.allclose(s_end["root_pos"][0, 0], torch.tensor(1.0), atol=1e-3)


def test_dof_reorder_to_env_names(lib_files):
    env_order = list(reversed(G1_DOF_NAMES))
    lib = MotionLib(lib_files, dof_names=env_order)
    s = lib.get_state(torch.tensor([0]), torch.tensor([0.5]))
    # 原来的关节 0 现在在最后一位
    assert abs(s["dof_pos"][0, -1].item() - 0.3 * np.sin(np.pi * 0.5)) < 1e-2
    assert abs(s["dof_pos"][0, 0].item()) < 1e-6


def test_from_dir_reads_index(tmp_path):
    out = tmp_path / "robot_clips"
    (out / "omomo").mkdir(parents=True)
    make_motion(out / "omomo" / "x.npz", "omomo/x")
    make_motion(out / "omomo" / "y.npz", "omomo/y")
    (out / "index.csv").write_text("clip_id,source,path\nomomo/x,omomo,omomo/x.npz\nomomo/y,omomo,omomo/y.npz\n")
    lib = MotionLib.from_dir(out, limit=1)
    assert lib.num_clips == 1 and lib.clip_ids == ["omomo/x"]
    with pytest.raises(ValueError):
        MotionLib.from_dir(out, sources=["trumans"])   # 没有该来源的片段
