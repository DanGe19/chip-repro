import math

import pytest
import torch

from chip.envs.rewards import (
    RewardCfg, Term, imitation_reward, kinematic_termination, load_reward_cfg, quat_angle, quat_rotate, yaw_quat,
)


def _yaw(yaw):
    return torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def make_state(N=4, B=3, yaw=0.0, shift=(0.0, 0.0, 0.0), with_vel=True, with_obj=True):
    q = _yaw(yaw).expand(N, 4).clone()
    anchor = torch.tensor([0.0, 0.0, 0.75]).expand(N, 3) + torch.tensor(shift)
    local = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.1, 0.3], [-0.1, -0.1, 0.5]]).expand(N, B, 3)
    body = anchor[:, None, :] + quat_rotate(q[:, None, :], local)
    s = {"anchor_pos": anchor.clone(), "anchor_rot": q, "body_pos": body,
         "body_rot": q[:, None, :].expand(N, B, 4).clone()}
    if with_vel:
        s["body_lin_vel"] = quat_rotate(q[:, None, :], torch.tensor([0.5, 0.0, 0.0]).expand(N, B, 3))
        s["body_ang_vel"] = torch.zeros(N, B, 3)
    if with_obj:
        s["object_pos"] = anchor + quat_rotate(q, torch.tensor([0.4, 0.0, 0.2]).expand(N, 3))
        s["object_rot"] = q.clone()
        s["object_active"] = torch.ones(N, dtype=torch.bool)
    return s


@pytest.fixture
def cfg():
    return load_reward_cfg("configs/stage1_imitation.yaml")


def test_config_loads_paper_terms(cfg):
    assert set(cfg.terms) == {"anchor_pos", "anchor_rot", "body_pos", "body_rot", "body_lin_vel", "body_ang_vel",
                              "object_rel_pos", "object_rot"}
    assert cfg.penalties["joint_limit"] == -10.0 and cfg.dt == pytest.approx(0.02)


def test_perfect_tracking_gives_max_reward(cfg):
    s = make_state()
    total, terms = imitation_reward(s, s, cfg)
    max_total = sum(t.weight for t in cfg.terms.values()) * cfg.dt
    assert torch.allclose(total, torch.full((4,), max_total))
    for k, v in terms.items():
        assert torch.allclose(v, torch.full((4,), cfg.terms[k].weight))


def test_relative_body_terms_ignore_global_offset(cfg):
    """整体平移+转身但姿态相同：锚点项掉分，相对身体项满分。"""
    ref = make_state()
    cur = make_state(yaw=0.6, shift=(0.3, -0.2, 0.0))
    _, terms = imitation_reward(cur, ref, cfg)
    assert terms["anchor_pos"].mean() < 0.5 * cfg.terms["anchor_pos"].weight + 1e-6
    assert terms["anchor_rot"].mean() < cfg.terms["anchor_rot"].weight
    for k in ("body_pos", "body_rot", "body_lin_vel", "body_ang_vel", "object_rel_pos"):
        assert torch.allclose(terms[k], torch.full((4,), cfg.terms[k].weight), atol=1e-5), k


def test_rbf_shape_and_sigma(cfg):
    ref = make_state()
    cur = make_state()
    cur["anchor_pos"][:, 0] += cfg.terms["anchor_pos"].sigma   # 误差正好 = σ → exp(-1)
    _, terms = imitation_reward(cur, ref, cfg)
    assert torch.allclose(terms["anchor_pos"], torch.full((4,), cfg.terms["anchor_pos"].weight * math.exp(-1)), atol=1e-5)


def test_object_terms_masked_when_inactive(cfg):
    ref = make_state()
    cur = make_state()
    ref["object_active"] = torch.tensor([True, False, True, False])
    _, terms = imitation_reward(cur, ref, cfg)
    assert terms["object_rel_pos"].tolist() == pytest.approx([0.5, 0.0, 0.5, 0.0])


def test_penalties(cfg):
    s = make_state()
    a = torch.zeros(4, 29)
    a_prev = torch.zeros(4, 29)
    a_prev[:, 0] = 0.5
    jp = torch.zeros(4, 29)
    jp[:, 3] = 2.0          # 超过上限 1.0 整整 1 rad
    lim = torch.stack([-torch.ones(29), torch.ones(29)], dim=1)
    contact = torch.tensor([0, 1, 0, 1])
    _, terms = imitation_reward(s, s, cfg, action=a, prev_action=a_prev, joint_pos=jp, joint_limits=lim,
                                undesired_contact=contact)
    assert torch.allclose(terms["action_rate"], torch.full((4,), -0.1 * 0.25))
    assert torch.allclose(terms["joint_limit"], torch.full((4,), -10.0))
    assert terms["undesired_contact"].tolist() == pytest.approx([0.0, -0.1, 0.0, -0.1])


def test_termination():
    ref = make_state()
    cur = make_state()
    cur["anchor_pos"][1, 0] += 0.8          # 跑偏
    cur["anchor_pos"][2, 2] = 0.2           # 摔倒
    done = kinematic_termination(cur, ref, pos_thr=0.5, height_thr=0.3)
    assert done.tolist() == [False, True, True, False]


def test_missing_optional_keys_are_skipped():
    cfg = RewardCfg(terms={"anchor_pos": Term(1.0, 0.3), "body_rot": Term(1.0, 0.4)})
    s = make_state(with_vel=False, with_obj=False)
    del s["body_rot"]
    total, terms = imitation_reward(s, s, cfg)
    assert set(terms) == {"anchor_pos"}


def test_yaw_quat_and_angle():
    q = _yaw(1.0)
    assert torch.allclose(yaw_quat(q[None])[0], q)
    assert quat_angle(_yaw(0.0)[None], _yaw(1.0)[None]).item() == pytest.approx(1.0, abs=1e-5)
