import torch

from chip.prior.vqvae import EMACodebook, SkillVQVAE, VQVAEConfig


def small_cfg(**kw):
    return VQVAEConfig(obs_dim=20, prop_dim=12, action_dim=6, codebook_size=32, code_dim=8,
                       encoder_hidden=[64, 64], decoder_hidden=[64, 64], **kw)


def test_shapes_and_token_range():
    torch.manual_seed(0)
    m = SkillVQVAE(small_cfg())
    obs, prop, act = torch.randn(50, 20), torch.randn(50, 12), torch.randn(50, 6)
    out = m(obs, prop)
    assert out["mean"].shape == (50, 6) and out["idx"].shape == (50,) and out["z_q"].shape == (50, 8)
    assert out["idx"].min() >= 0 and out["idx"].max() < 32
    tok = m.tokenize(obs)
    assert torch.equal(tok, m.tokenize(obs))                 # 确定性
    assert m.decode_token(prop, tok).shape == (50, 6)


def test_straight_through_gradient_reaches_encoder():
    torch.manual_seed(0)
    m = SkillVQVAE(small_cfg())
    obs, prop, act = torch.randn(32, 20), torch.randn(32, 12), torch.randn(32, 6)
    loss, _ = m.loss(obs, prop, act)
    loss.backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in m.encoder.parameters() if p.grad is not None)
    assert enc_grad > 0
    # 码本是 buffer，不该有梯度
    assert not any(n.startswith("codebook") for n, p in m.named_parameters())


def test_ema_codebook_moves_toward_data_and_tracks_usage():
    torch.manual_seed(0)
    cb = EMACodebook(size=4, dim=2, decay=0.5, dead_threshold=0.0)
    cb.train()
    # 两团数据，码本应该收敛到团中心附近
    a = torch.tensor([[5.0, 5.0]]).expand(64, 2) + 0.01 * torch.randn(64, 2)
    b = torch.tensor([[-5.0, -5.0]]).expand(64, 2) + 0.01 * torch.randn(64, 2)
    for _ in range(30):
        cb(torch.cat([a, b]))
    _, idx = cb(torch.cat([a, b]))
    assert len(idx.unique()) >= 2
    used = cb.embed[idx.unique()]
    assert all(min((u - torch.tensor([5.0, 5.0])).norm(), (u - torch.tensor([-5.0, -5.0])).norm()) < 0.5 for u in used)
    u = cb.usage()
    assert 1.0 < u["perplexity"] <= 4.0


def test_overfits_tiny_dataset():
    """能把一个小数据集的动作拟合下来，MSE 明显下降（证明三块接对了）。"""
    torch.manual_seed(0)
    m = SkillVQVAE(small_cfg())
    obs = torch.randn(256, 20)
    prop = torch.randn(256, 12)
    act = torch.tanh(obs[:, :6] * 0.7 + prop[:, :6] * 0.3)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    _, first = m.loss(obs, prop, act)
    for _ in range(300):
        loss, info = m.loss(obs, prop, act)
        opt.zero_grad(); loss.backward(); opt.step()
    assert info["action_mse"] < 0.3 * first["action_mse"]
    assert info["active_codes"] >= 2
