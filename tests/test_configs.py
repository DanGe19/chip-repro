"""确认 configs/ 里的 yaml 能加载，且关键数值和论文一致。不需要 GPU。"""
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def load(name):
    with open(CONFIGS / name) as f:
        return yaml.safe_load(f)


def test_stage1_matches_paper_table5_and_8():
    c = load("stage1_imitation.yaml")
    assert c["sim"]["physics_hz"] == 200 and c["sim"]["control_hz"] == 50
    assert c["ppo"]["minibatch_size"] == 24576
    assert c["ppo"]["rollout_horizon"] == 24
    assert c["reward"]["body_pos"] == {"weight": 1.0, "sigma": 0.30}
    assert c["reward"]["penalties"]["joint_limit"] == -10.0


def test_stage2_matches_paper_table6():
    c = load("stage2_prior.yaml")
    assert c["vqvae"]["codebook"]["size"] == 1024
    assert c["vqvae"]["codebook"]["dim"] == 64
    assert c["vqvae"]["commitment_beta"] == 0.25
    assert c["prior"]["backbone"]["num_layers"] == 4
    assert c["prior"]["modality_mask_prob"] == 0.3


def test_stage3_matches_paper_table7_and_9():
    c = load("stage3_posttrain.yaml")
    assert c["prior_regularization"]["kl_coef"] == 0.01
    assert c["prior_regularization"]["cfg_weight"] == 1.5
    assert c["inference"]["sampling_temperature"] == 0.05
    nav = c["tasks"]["scene_navigation"]["reward_weights"]
    assert nav["success"] == 50.0 and nav["failure"] == -50.0
    assert c["tasks"]["object_relocation"]["reward_weights"]["success"] == 20.0
