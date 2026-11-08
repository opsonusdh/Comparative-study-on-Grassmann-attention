import torch

from loop_transformer import LoopConfig, LoopTransformer
from loop_transformer.attention import (
    CSAAttention,
    HCAAttention,
    SlidingWindowAttention,
    deepseek_loop_split_lags,
    deepseek_split_block_mask,
)
from train import build_parser, config_from_args


def tiny_cfg(**overrides):
    values = dict(
        vocab_size=32,
        dim=16,
        n_layers=4,
        n_heads=2,
        head_dim=8,
        ffn_hidden_dim=32,
        rope_dim=8,
        max_loops=4,
        min_loops=4,
        attention_type="deepseek-split",
        deepseek_split_stride=4,
        sw_window=3,
        csa_m=2,
        csa_top_k=4,
        hca_m_prime=2,
        groups=2,
        group_dim=8,
        moe_num_shared_experts=1,
        moe_num_routed_experts=2,
        moe_top_k=1,
        moe_expert_hidden_dim=16,
        activation_dictionary="moa7",
        activation_gate_mode="sigmoid",
        activation_top_k=2,
        loop_refinement_weight=0.0,
        loop_task_weight=0.0,
    )
    values.update(overrides)
    return LoopConfig(**values)


def _small_csa(split_stride=4):
    module = CSAAttention(
        dim=16,
        n_heads=2,
        head_dim=8,
        m=2,
        top_k=4,
        window_size=3,
        index_heads=2,
        index_dim=4,
        groups=2,
        group_dim=8,
        rope_dim=8,
        max_loops=4,
        split_stride=split_stride,
    )
    with torch.no_grad():
        module.grouped_proj.proj2.weight.normal_(mean=0.0, std=0.02)
    return module


def _small_hca(split_stride=4):
    module = HCAAttention(
        dim=16,
        n_heads=2,
        head_dim=8,
        m_prime=2,
        window_size=3,
        groups=2,
        group_dim=8,
        rope_dim=8,
        max_loops=4,
        split_stride=split_stride,
    )
    with torch.no_grad():
        module.grouped_proj.proj2.weight.normal_(mean=0.0, std=0.02)
    return module


def test_split_lags_follow_formula_and_cycle_every_four_loops():
    assert deepseek_loop_split_lags(17, 0, 4) == (1, 4, 8, 12, 16)
    assert deepseek_loop_split_lags(17, 1, 4) == (1, 5, 9, 13)
    assert deepseek_loop_split_lags(17, 2, 4) == (1, 2, 6, 10, 14)
    assert deepseek_loop_split_lags(17, 3, 4) == (1, 3, 7, 11, 15)
    assert deepseek_loop_split_lags(17, 4, 4) == (1, 4, 8, 12, 16)


def test_compressed_block_assignment_covers_visible_memory_once_plus_shared_newest():
    visible = torch.tril(torch.ones(12, 12, dtype=torch.bool))
    assignments = [deepseek_split_block_mask(visible, loop_idx, stride=4) for loop_idx in range(4)]
    union = torch.stack(assignments).any(dim=0)
    common = torch.stack(assignments).all(dim=0)

    assert torch.equal(union, visible)
    for query in range(visible.size(0)):
        # Only the newest causally visible compressed block is shared by all loops.
        expected_common = torch.zeros(visible.size(1), dtype=torch.bool)
        expected_common[query] = True
        assert torch.equal(common[query], expected_common)


def test_compressed_block_assignment_is_balanced_for_compression_stride_four():
    # A raw-token modulo split would put every m=4 compressed block into one
    # phase. This rank-based split correctly distributes compressed slots.
    visible = torch.ones(1, 32, dtype=torch.bool)
    counts = [int(deepseek_split_block_mask(visible, loop_idx, 4).sum()) for loop_idx in range(4)]
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == 32 + 3  # 32 ranks plus the shared newest block in other loops


def test_deepseek_split_preserves_original_swa_csa_hca_layer_routing():
    model = LoopTransformer(tiny_cfg()).eval()
    assert isinstance(model.blocks[0].attn, SlidingWindowAttention)
    assert isinstance(model.blocks[1].attn, SlidingWindowAttention)
    assert isinstance(model.blocks[2].attn, CSAAttention)
    assert isinstance(model.blocks[3].attn, HCAAttention)
    assert model.blocks[2].attn.split_stride == 4
    assert model.blocks[3].attn.split_stride == 4


def test_deepseek_and_deepseek_split_have_matching_parameter_structure():
    torch.manual_seed(101)
    baseline = LoopTransformer(tiny_cfg(attention_type="deepseek"))
    torch.manual_seed(101)
    split = LoopTransformer(tiny_cfg(attention_type="deepseek-split"))
    baseline_state = baseline.state_dict()
    split_state = split.state_dict()
    assert baseline_state.keys() == split_state.keys()
    for key in baseline_state:
        assert baseline_state[key].shape == split_state[key].shape, key


def test_csa_split_forward_backward_and_auxiliary_loss_are_finite():
    torch.manual_seed(17)
    module = _small_csa().train()
    x = torch.randn(2, 19, 16, requires_grad=True)
    outputs = []
    for loop_idx in range(4):
        y = module(x, loop_idx=loop_idx)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        outputs.append(y)
    assert len(module._aux_losses) == 4
    assert all(torch.isfinite(loss) for loss in module._aux_losses)
    (sum(y.square().mean() for y in outputs) + sum(module._aux_losses)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.index_down.weight.grad is not None


def test_hca_split_forward_backward_is_finite_for_all_loop_phases():
    torch.manual_seed(18)
    module = _small_hca().train()
    x = torch.randn(2, 19, 16, requires_grad=True)
    outputs = [module(x, loop_idx=i) for i in range(4)]
    assert all(y.shape == x.shape and torch.isfinite(y).all() for y in outputs)
    sum(y.square().mean() for y in outputs).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_hca_old_compressed_block_contributes_only_on_its_assigned_loop():
    torch.manual_seed(71)
    module = _small_hca().eval()
    x = torch.randn(1, 22, 16, requires_grad=True)
    block_gradients = []
    for loop_idx in range(4):
        y = module(x, loop_idx=loop_idx)
        grad = torch.autograd.grad(y[0, 20].sum(), x, retain_graph=True)[0]
        # For m_prime=2 at query 20, positions 16-17 form compressed-block age 2.
        # Age 2 belongs only to loop 2; the local window (18-20) is separate.
        block_gradients.append(float(grad[0, 16:18].abs().sum()))
    assert block_gradients[2] > 0.0
    assert block_gradients[0] == 0.0
    assert block_gradients[1] == 0.0
    assert block_gradients[3] == 0.0


def test_csa_and_hca_keep_the_entire_raw_local_window_on_every_loop():
    torch.manual_seed(19)
    x = torch.randn(1, 18, 16, requires_grad=True)
    for module in (_small_csa().eval(), _small_hca().eval()):
        for loop_idx in range(4):
            y = module(x, loop_idx=loop_idx)
            grad = torch.autograd.grad(y[0, 10].sum(), x, retain_graph=True)[0]
            # At query 10 with local window 3, raw positions 8, 9 and 10
            # remain directly available regardless of the loop's long-range split.
            assert all(grad[0, pos].abs().sum().item() > 0 for pos in (8, 9, 10))


def test_deepseek_split_model_runs_all_loops_and_is_causal():
    torch.manual_seed(20)
    model = LoopTransformer(tiny_cfg()).eval()
    tokens = torch.randint(0, 32, (1, 40))
    changed = tokens.clone()
    changed[:, 25:] = torch.randint(0, 32, changed[:, 25:].shape)
    with torch.no_grad():
        logits_a = model(tokens, max_loops=4)[0][-1]
        logits_b = model(changed, max_loops=4)[0][-1]
    assert logits_a.shape == (1, 40, 32)
    assert torch.allclose(logits_a[:, :25], logits_b[:, :25], atol=1e-5, rtol=1e-5)


def test_deepseek_split_cli_builds_expected_config():
    args = build_parser().parse_args([
        "--attention-type", "deepseek-split",
        "--deepseek-split-stride", "4",
        "--max-loops", "4",
        "--min-loops", "4",
    ])
    cfg = config_from_args(args)
    assert cfg.attention_type == "deepseek-split"
    assert cfg.deepseek_split_stride == 4


def test_deepseek_split_rejects_invalid_stride():
    try:
        tiny_cfg(deepseek_split_stride=1)
    except ValueError as exc:
        assert "deepseek_split_stride must be >= 2" in str(exc)
    else:
        raise AssertionError("invalid split stride was accepted")


def test_deepseek_split_checkpoint_round_trip_preserves_stride(tmp_path):
    from loop_transformer.checkpointing import load_checkpoint, save_checkpoint

    model = LoopTransformer(tiny_cfg(deepseek_split_stride=4))
    path = tmp_path / "deepseek_split.pt"
    save_checkpoint(path, model, step=11)
    restored = load_checkpoint(path, device="cpu")
    assert restored["model"].cfg.attention_type == "deepseek-split"
    assert restored["model"].cfg.deepseek_split_stride == 4
    assert restored["model"].num_parameters() == model.num_parameters()
    assert restored["step"] == 11
