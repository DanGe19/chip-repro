"""几何编码和格式转换的测试。纯 CPU。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

trimesh = pytest.importorskip("trimesh")

from chip.envs.conversions import (
    JointReorder, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz, rot_z, yaw_from_quat_xyzw,
)
from chip.envs.geometry import (
    ObjectBPS, SceneHeightField, body_to_object_vectors, make_basis_points,
)
from chip.retarget.robot_motion import G1_DOF_NAMES


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------

def test_quat_order_roundtrip():
    q = np.array([[0.1, 0.2, 0.3, 0.9]])
    assert np.allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q)
    assert np.allclose(quat_xyzw_to_wxyz(q), [[0.9, 0.1, 0.2, 0.3]])


def test_yaw_from_quat_matches_scipy():
    for deg in [0, 45, 90, -120, 179]:
        q = R.from_euler("z", deg, degrees=True).as_quat()   # xyzw
        assert np.isclose(np.degrees(yaw_from_quat_xyzw(q)), deg, atol=1e-6)


def test_rot_z_rotates_x_to_y():
    Rm = rot_z(np.array(np.pi / 2))
    assert np.allclose(Rm @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_joint_reorder_by_name():
    # 模拟 IsaacLab 的交错顺序
    isaac = ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint"] + \
            [n for n in G1_DOF_NAMES if n not in ("left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint")]
    r = JointReorder(G1_DOF_NAMES, isaac)
    x = np.arange(29, dtype=float)
    y = r(x)
    assert y[0] == 0 and y[1] == G1_DOF_NAMES.index("right_hip_pitch_joint") and y[2] == 12
    assert np.allclose(r.inverse(y), x)
    # 批量维度也行
    assert r(np.tile(x, (5, 1))).shape == (5, 29)


def test_joint_reorder_rejects_mismatch():
    with pytest.raises(ValueError):
        JointReorder(G1_DOF_NAMES, list(G1_DOF_NAMES[:-1]) + ["bogus_joint"])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def make_scene():
    """地面 10×10 m 在 z=0，一张 1×1 m、高 0.8 m 的"桌子"放在 x∈[2,3], y∈[-0.5,0.5]。"""
    floor = trimesh.creation.box(extents=[10, 10, 0.02])
    floor.apply_translation([0, 0, -0.01])
    table = trimesh.creation.box(extents=[1, 1, 0.8])
    table.apply_translation([2.5, 0, 0.4])
    return trimesh.util.concatenate([floor, table])


def test_heightfield_sees_table():
    hf = SceneHeightField.from_mesh(make_scene(), res=0.1, pad=0.5)
    assert np.isclose(hf.sample(np.array([0.0, 0.0])), 0.0, atol=1e-3)
    assert np.isclose(hf.sample(np.array([2.5, 0.0])), 0.8, atol=1e-3)


def test_local_heightmap_is_heading_aligned():
    hf = SceneHeightField.from_mesh(make_scene(), res=0.05, pad=0.5)
    base = np.array([[1.9, 0.0, 0.9]])   # 站在桌子前 0.1 m，根高 0.9
    # 面朝 +x：桌子在前方 → 网格第 0 维靠后的行应为 0.8 − 0.9 = −0.1，身后为 0 − 0.9 = −0.9
    hm = hf.local_heightmap(base, np.array([0.0]), size=16, extent_m=1.6)
    assert hm.shape == (1, 16, 16)
    assert np.isclose(hm[0, -1, 8], -0.1, atol=0.05)
    assert np.isclose(hm[0, 0, 8], -0.9, atol=0.05)
    # 转身面朝 −x：桌子跑到身后
    hm2 = hf.local_heightmap(base, np.array([np.pi]), size=16, extent_m=1.6)
    assert np.isclose(hm2[0, 0, 8], -0.1, atol=0.05)
    assert np.isclose(hm2[0, -1, 8], -0.9, atol=0.05)


def test_heightfield_save_load(tmp_path):
    hf = SceneHeightField.from_mesh(make_scene(), res=0.2, pad=0.5)
    hf.save(tmp_path / "hf.npz")
    hf2 = SceneHeightField.load(tmp_path / "hf.npz")
    assert np.allclose(hf2.heights, hf.heights) and hf2.res == hf.res


def test_basis_points_fixed_and_in_ball():
    b1, b2 = make_basis_points(512, seed=0), make_basis_points(512, seed=0)
    assert b1.shape == (512, 3) and np.allclose(b1, b2)
    assert np.all(np.linalg.norm(b1, axis=1) <= 1.0 + 1e-6)


def test_bps_invariant_to_rigid_motion():
    """物体和锚点一起做刚体变换，BPS 编码不变。"""
    box = trimesh.creation.box(extents=[0.4, 0.3, 0.5])
    bps = ObjectBPS.from_mesh(box, make_basis_points(256, radius=1.5), n_surface=2000)
    pose0 = np.eye(4); pose0[:3, 3] = [0.6, 0.0, 0.25]
    e0 = bps.encode(pose0, anchor_pos=np.zeros(3), anchor_yaw=0.0)

    yaw = 1.1
    Rz = rot_z(np.array(yaw)); t = np.array([3.0, -2.0, 0.0])
    pose1 = np.eye(4); pose1[:3, :3] = Rz @ pose0[:3, :3]; pose1[:3, 3] = Rz @ pose0[:3, 3] + t
    e1 = bps.encode(pose1, anchor_pos=t, anchor_yaw=yaw)
    assert e0.shape == (256,)
    assert np.allclose(e0, e1, atol=1e-4)


def test_body_to_object_vectors_points_at_surface():
    box = trimesh.creation.box(extents=[1, 1, 1])     # 中心在原点，面在 ±0.5
    pose = np.eye(4); pose[:3, 3] = [2.0, 0.0, 0.5]
    body = np.array([[0.0, 0.0, 0.5]])                # 在物体 −x 侧 2 m 处
    v = body_to_object_vectors(body, box, pose)
    assert v.shape == (1, 3)
    assert np.allclose(v[0], [1.5, 0.0, 0.0], atol=1e-6)


def test_joint_reorder_fill_missing():
    from chip.envs.conversions import JointReorder
    import numpy as np, pytest
    src = ["a", "b", "c"]
    dst = ["c", "hand_x", "a", "b"]
    with pytest.raises(ValueError):
        JointReorder(src, dst)
    r = JointReorder(src, dst, fill_missing=0.0)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = r(x)
    assert y.shape == (2, 4)
    assert np.allclose(y, [[3, 0, 1, 2], [6, 0, 4, 5]])
    assert np.allclose(r.inverse(y), x)
    # src 多出来的关节必须报错
    with pytest.raises(ValueError):
        JointReorder(src + ["zzz"], dst, fill_missing=0.0)
