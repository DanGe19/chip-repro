"""inspect_robot_motion 的离屏渲染辅助：四元数转换 + 用 MjSpec 把物体加进模型。"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from chip.retarget.inspect_robot_motion import _mat_to_quat_wxyz, build_scene_model, _set_mocap
from chip.retarget.robot_motion import ObjectTrack

ROBOT_XML = """
<mujoco><worldbody>
  <body name="pelvis"><freejoint/><geom type="sphere" size="0.05"/>
    <body name="leg"><joint name="j0" type="hinge" axis="0 1 0"/><geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3"/></body>
  </body>
</worldbody></mujoco>
"""

CUBE_OBJ = """v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""


def test_mat_to_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        R = np.zeros(9); mujoco.mju_quat2Mat(R, q)
        q2 = _mat_to_quat_wxyz(R.reshape(3, 3))
        assert np.allclose(q2, q, atol=1e-6) or np.allclose(q2, -q, atol=1e-6)


def test_build_scene_model_adds_mocap_object(tmp_path):
    xml = tmp_path / "robot.xml"; xml.write_text(ROBOT_XML)
    root = tmp_path / "ds"; (root / "objs").mkdir(parents=True)
    (root / "objs" / "box.obj").write_text(CUBE_OBJ)
    T = 5
    pose = np.tile(np.eye(4), (T, 1, 1)); pose[:, 0, 3] = np.arange(T) * 0.1; pose[:, 2, 3] = 0.5
    track = ObjectTrack(name="box", mesh_path="objs/box.obj", pose=pose.astype(np.float32),
                        scale=np.array([0.2], np.float32), active=np.ones(T, bool))
    meta = {"source": "fake", "object_tracks": [track], "scene_mesh_path": None, "world_scale": 1.0, "world_offset": np.zeros(3)}
    model, tracks = build_scene_model(xml, meta, {"fake": root}, with_scene=False)
    assert len(tracks) == 1 and model.nmocap == 1
    assert model.nq == 7 + 1                      # mocap 不占 qpos
    data = mujoco.MjData(model)
    _set_mocap(model, data, tracks, 3)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tracks[0][0])
    assert np.allclose(data.xpos[bid], [0.3, 0.0, 0.5], atol=1e-6)


def test_missing_mesh_is_skipped(tmp_path):
    xml = tmp_path / "robot.xml"; xml.write_text(ROBOT_XML)
    track = ObjectTrack(name="ghost", mesh_path="nope.obj", pose=np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
                        scale=np.ones(1, np.float32), active=np.ones(2, bool))
    meta = {"source": "fake", "object_tracks": [track], "scene_mesh_path": None, "world_scale": 1.0, "world_offset": np.zeros(3)}
    model, tracks = build_scene_model(xml, meta, {"fake": tmp_path}, with_scene=False)
    assert tracks == [] and model.nmocap == 0
