import torch

from chip.prior.context_prior import ContextPrior, PriorConfig


def small():
    return PriorConfig(prop_dim=12, history=5, context_dims={"height": 16, "goal_human": 3}, codebook_size=32,
                       d_model=32, num_layers=2, num_heads=4, ffn_dim=64, tokenizer_hidden=32)


def test_forward_shapes_and_missing_modalities():
    torch.manual_seed(0)
    m = ContextPrior(small()).eval()
    N = 6
    ph = torch.randn(N, 5, 12)
    ctx = {"height": torch.randn(N, 16), "goal_human": torch.randn(N, 3)}
    av = {"height": torch.tensor([1, 1, 1, 0, 0, 0]).bool(), "goal_human": torch.ones(N).bool()}
    logits = m(ph, ctx, av)
    assert logits.shape == (N, 32)
    # 完全不给模态也能跑（纯本体先验）
    assert m(ph).shape == (N, 32)
    # 被 mask 掉的模态不影响输出：把 height 的值改掉，avail=0 的那几行 logits 应不变
    ctx2 = {"height": torch.randn(N, 16), "goal_human": ctx["goal_human"]}
    l2 = m(ph, ctx2, av)
    assert torch.allclose(logits[3:], l2[3:], atol=1e-5)
    assert not torch.allclose(logits[:3], l2[:3], atol=1e-3)


def test_modality_dropout_only_in_training():
    torch.manual_seed(0)
    cfg = small(); cfg.modality_mask_prob = 1.0           # 训练时一定丢掉
    m = ContextPrior(cfg)
    ph = torch.randn(4, 5, 12); ctx = {"height": torch.randn(4, 16)}
    m.train(); a = m(ph, ctx)
    m.eval(); b = m(ph, ctx); c = m(ph)                   # eval 不丢；c 是无模态
    assert torch.allclose(a, c, atol=1e-5)                # 全丢 == 无模态
    assert not torch.allclose(a, b, atol=1e-3)


def test_loss_sample_guidance():
    torch.manual_seed(0)
    m = ContextPrior(small())
    ph = torch.randn(8, 5, 12); ctx = {"height": torch.randn(8, 16)}; av = {"height": torch.ones(8).bool()}
    tok = torch.randint(0, 32, (8,))
    loss, info = m.loss(ph, ctx, av, tok)
    assert loss.ndim == 0 and 0 <= info["acc"] <= 1
    loss.backward()
    m.eval()
    s = m.sample(ph, ctx, av, num_samples=7)
    assert s.shape == (8, 7) and s.min() >= 0 and s.max() < 32
    g = m.guided_logits(ph, ctx, av, guidance_scale=1.5)
    assert g.shape == (8, 32)
    # w=1 时等于有条件 logits
    g1 = m.guided_logits(ph, ctx, av, guidance_scale=1.0)
    assert torch.allclose(g1, m(ph, ctx, av), atol=1e-5)


def test_can_fit_simple_rule():
    """token = 本体最后一帧第 0 维的符号 → 两类；应能学到接近 100%。"""
    torch.manual_seed(0)
    cfg = small(); cfg.modality_mask_prob = 0.0
    m = ContextPrior(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    ph = torch.randn(512, 5, 12)
    tok = (ph[:, -1, 0] > 0).long()
    m.train()
    for _ in range(150):
        loss, info = m.loss(ph, {}, {}, tok)
        opt.zero_grad(); loss.backward(); opt.step()
    assert info["acc"] > 0.95
