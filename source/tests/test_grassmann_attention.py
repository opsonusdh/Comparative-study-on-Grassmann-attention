import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from loop_transformer import LoopConfig, LoopTransformer, load_checkpoint
from loop_transformer.attention import GrassmannFlowAttention


def tiny_cfg(**overrides):
    values = dict(
        vocab_size=32,
        dim=32,
        n_layers=2,
        n_heads=4,
        head_dim=8,
        ffn_hidden_dim=64,
        rope_dim=8,
        max_loops=2,
        min_loops=2,
        loop_sampling=False,
        attention_type="grassmann",
        grassmann_rank=8,
        grassmann_offsets=(1, 2, 4),
        groups=4,
        group_dim=8,
        moe_num_shared_experts=1,
        moe_num_routed_experts=2,
        moe_top_k=1,
        moe_expert_hidden_dim=32,
        activation_dictionary="moa7",
        activation_gate_mode="sigmoid",
        activation_top_k=2,
        loop_refinement_weight=0.0,
        loop_task_weight=0.0,
    )
    values.update(overrides)
    return LoopConfig(**values)


def test_grassmann_attention_shapes_gradients_and_no_pair_boundary():
    torch.manual_seed(1)
    module = GrassmannFlowAttention(dim=16, rank=6, offsets=(1, 2, 4))
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.equal(y[:, 0], torch.zeros_like(y[:, 0]))
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.reduce.weight.grad is not None
    assert module.plucker_proj.weight.grad is not None
    assert module.gate.weight.grad is not None


def test_grassmann_attention_is_strictly_causal_with_previous_offsets():
    torch.manual_seed(2)
    module = GrassmannFlowAttention(dim=16, rank=6, offsets=(1, 2, 4)).eval()
    x = torch.randn(1, 12, 16)
    changed = x.clone()
    changed[:, 8:, :] = torch.randn_like(changed[:, 8:, :]) * 20
    with torch.no_grad():
        original = module(x)
        perturbed = module(changed)
    assert torch.allclose(original[:, :8], perturbed[:, :8], atol=1e-6, rtol=1e-6)


def test_grassmann_attention_respects_explicit_pair_mask():
    torch.manual_seed(3)
    module = GrassmannFlowAttention(dim=12, rank=4, offsets=(1, 2)).eval()
    x = torch.randn(1, 6, 12)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.bool))
    mask[4, 3] = False
    mask[4, 2] = False
    with torch.no_grad():
        y = module(x, attention_mask=mask)
    assert torch.equal(y[:, 4], torch.zeros_like(y[:, 4]))


def test_full_loop_transformer_with_grassmann_attention_is_causal():
    torch.manual_seed(4)
    model = LoopTransformer(tiny_cfg()).eval()
    x = torch.randint(0, 32, (1, 12))
    changed = x.clone()
    changed[:, 9] = (changed[:, 9] + 1) % 32
    with torch.no_grad():
        logits_a = model(x, max_loops=2)[0][-1]
        logits_b = model(changed, max_loops=2)[0][-1]
    assert torch.allclose(logits_a[:, :9], logits_b[:, :9], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(logits_a[:, 9:], logits_b[:, 9:], atol=1e-7, rtol=1e-7)


def test_config_rejects_noncausal_or_duplicate_offsets():
    try:
        tiny_cfg(grassmann_offsets=(1, 0, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("zero offset must be rejected")
    try:
        tiny_cfg(grassmann_offsets=(1, 2, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate offsets must be rejected")


def test_default_deepseek_config_ignores_unused_grassmann_rank():
    # Baseline configs with a hidden width below Grassmann's default rank must
    # remain valid because the Grassmann settings are inactive on this path.
    cfg = LoopConfig(
        vocab_size=16, dim=16, n_layers=1, n_heads=2, head_dim=8,
        ffn_hidden_dim=32, rope_dim=8, groups=2,
    )
    assert cfg.attention_type == "deepseek"


def test_grassmann_attention_low_precision_is_finite():
    for dtype in (torch.float16, torch.bfloat16):
        module = GrassmannFlowAttention(dim=16, rank=6, offsets=(1, 2, 4)).to(dtype=dtype)
        x = torch.randn(2, 8, 16, dtype=dtype, requires_grad=True)
        y = module(x)
        y.float().square().mean().backward()
        assert torch.isfinite(y).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()


def test_grassmann_smooth_normalization_is_finite_for_repeated_vectors():
    torch.manual_seed(11)
    module = GrassmannFlowAttention(dim=16, rank=8, offsets=(1, 2, 4))
    one = torch.randn(2, 1, 16)
    x = one.expand(2, 12, 16).clone().requires_grad_(True)
    y = module(x)
    loss = y.float().square().mean()
    loss.backward()
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.reduce.weight.grad is not None
    assert torch.isfinite(module.reduce.weight.grad).all()
    # With the smooth epsilon, exact collinearity has a finite derivative.
    assert float(x.grad.abs().max()) < 1e4


def test_grassmann_model_adds_a_learned_absolute_token_position_table():
    torch.manual_seed(12)
    model = LoopTransformer(tiny_cfg(grassmann_positional_max_len=24)).eval()
    assert model.grassmann_positional_embedding is not None
    token_ids = torch.full((1, 8), 5, dtype=torch.long)
    embedded = model._embed_tokens(token_ids)
    assert embedded.shape == (1, 8, model.cfg.dim)
    assert not torch.allclose(embedded[:, 0], embedded[:, 1])

    embedded.sum().backward()
    grad = model.grassmann_positional_embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad[:8].abs().sum() > 0


def test_grassmann_absolute_position_table_enforces_max_length():
    model = LoopTransformer(tiny_cfg(grassmann_positional_max_len=7))
    with pytest.raises(ValueError, match="grassmann_positional_max_len"):
        model._embed_tokens(torch.ones((1, 8), dtype=torch.long))


def test_legacy_grassmann_checkpoint_loads_with_new_position_table(tmp_path):
    model = LoopTransformer(tiny_cfg())
    state = dict(model.state_dict())
    state.pop("grassmann_positional_embedding.weight")
    legacy_config = dataclasses.asdict(model.cfg)
    legacy_config.pop("grassmann_positional_max_len")
    path = tmp_path / "legacy_grassmann.pt"
    torch.save({
        "config": legacy_config,
        "model_state_dict": state,
        "step": 10,
        "extra": {},
    }, path)

    with pytest.warns(RuntimeWarning, match="predates learned absolute token positions"):
        restored = load_checkpoint(path, device="cpu")
    assert restored["model"].grassmann_positional_embedding is not None
    assert restored["model"].grassmann_positional_embedding.weight.shape == (1024, 32)


def test_resume_rejects_explicit_attention_architecture_switch():
    from train import _validate_resume_architecture_overrides

    model = LoopTransformer(tiny_cfg())
    args = SimpleNamespace(
        attention_type="deepseek",
        grassmann_rank=32,
        grassmann_offsets=[1, 2, 4],
        grassmann_positional_max_len=1024,
        _explicit_cli_overrides=frozenset({"attention_type"}),
    )
    with pytest.raises(ValueError, match="Cannot change attention-type while resuming"):
        _validate_resume_architecture_overrides(args, {"model": model})


def test_resume_without_attention_override_inherits_checkpoint_architecture():
    from train import _validate_resume_architecture_overrides

    model = LoopTransformer(tiny_cfg(grassmann_rank=8, grassmann_positional_max_len=128))
    args = SimpleNamespace(
        attention_type="deepseek",
        grassmann_rank=32,
        grassmann_offsets=[1, 2, 4],
        grassmann_positional_max_len=1024,
        _explicit_cli_overrides=frozenset(),
    )
    _validate_resume_architecture_overrides(args, {"model": model})
    assert args.attention_type == "grassmann"
    assert args.grassmann_rank == 8
    assert args.grassmann_offsets == (1, 2, 4)
    assert args.grassmann_positional_max_len == 128


def test_optimizer_resume_rejects_wrong_shaped_adam_moments():
    from train import _load_optimizer_state_compatible

    model = LoopTransformer(tiny_cfg())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randint(0, 32, (2, 8))
    loss, _ = model.compute_loss(batch, max_loops=2)
    loss.backward()
    optimizer.step()

    saved = copy.deepcopy(optimizer.state_dict())
    first_state = next(iter(saved["state"].values()))
    first_state["exp_avg"] = torch.zeros(3)
    fresh_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert not _load_optimizer_state_compatible(fresh_optimizer, saved, model)
    assert len(fresh_optimizer.state) == 0


def test_loop_controller_gradients_remain_finite_for_repeated_token_batches():
    torch.manual_seed(13)
    model = LoopTransformer(
        tiny_cfg(loop_controller=True, residual_loop_embedding=False)
    ).train()
    # Force the strict repeated-token/collinear case rather than relying on
    # random positional offsets to separate otherwise identical token states.
    with torch.no_grad():
        model.grassmann_positional_embedding.weight.zero_()
    batch = torch.full((2, 16), 7, dtype=torch.long)
    logits, _ = model(batch, max_loops=2)
    loss = F.cross_entropy(
        logits[-1][:, :-1].reshape(-1, model.cfg.vocab_size),
        batch[:, 1:].reshape(-1),
    )
    loss.backward()
    grad = model.loop_controller_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.norm()) < 100.0


def test_legacy_grassmann_optimizer_migrates_and_new_position_state_starts_fresh():
    from train import _load_optimizer_state_compatible

    model = LoopTransformer(tiny_cfg())
    legacy_parameters = [
        parameter for name, parameter in model.named_parameters()
        if name != "grassmann_positional_embedding.weight"
    ]
    legacy_optimizer = torch.optim.AdamW(legacy_parameters, lr=1e-3)
    batch = torch.randint(0, 32, (2, 8))
    loss, _ = model.compute_loss(batch, max_loops=2)
    loss.backward()
    legacy_optimizer.step()

    new_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert _load_optimizer_state_compatible(
        new_optimizer, copy.deepcopy(legacy_optimizer.state_dict()), model
    )
    assert len(new_optimizer.state) == len(legacy_optimizer.state)
    assert model.grassmann_positional_embedding.weight not in new_optimizer.state


def test_full_model_gradient_has_no_future_token_path_with_absolute_positions():
    torch.manual_seed(14)
    model = LoopTransformer(tiny_cfg()).eval()
    token_ids = torch.randint(0, 32, (1, 10))
    embedded = model._embed_tokens(token_ids).detach().requires_grad_(True)
    state = embedded
    for loop_idx in range(model.cfg.max_loops):
        state = model._one_loop(state, loop_idx=loop_idx)
    logits = model.lm_head(model.final_norm(state))
    # A scalar at position 5 must not depend on input embeddings at 6..T-1.
    scalar = logits[0, 5, 3]
    grad = torch.autograd.grad(scalar, embedded)[0]
    assert torch.count_nonzero(grad[:, 6:, :]).item() == 0
    assert float(grad[:, :6, :].abs().sum()) > 0.0
