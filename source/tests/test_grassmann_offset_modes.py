import shlex
from pathlib import Path

import torch

from loop_transformer import LoopConfig, LoopTransformer
from loop_transformer.attention import (
    GrassmannFlowAttention,
    adaptive_progressive_grassmann_offsets,
    progressive_grassmann_offsets,
    real_grassmann_offsets,
)
from loop_transformer.visualizer import LoopTransformerVisualizer
from train import build_parser, config_from_args


def tiny_cfg(**overrides):
    values = dict(
        vocab_size=32,
        dim=16,
        n_layers=2,
        n_heads=2,
        head_dim=8,
        ffn_hidden_dim=32,
        rope_dim=8,
        max_loops=4,
        min_loops=4,
        loop_sampling=False,
        attention_type="grassmann-new",
        grassmann_rank=4,
        grassmann_offsets=(1, 2, 4, 8, 12, 16),
        grassmann_offset_count=4,
        grassmann_offset_spacing_base=2,
        grassmann_long_range_stride=32,
        grassmann_positional_embedding=False,
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


def test_progressive_offsets_match_loop_dependent_arithmetic_schedule():
    assert progressive_grassmann_offsets(0, 4, 2) == (1, 2, 4, 6)
    assert progressive_grassmann_offsets(1, 4, 2) == (1, 3, 6, 9)
    assert progressive_grassmann_offsets(2, 4, 2) == (1, 4, 8, 12)
    assert progressive_grassmann_offsets(3, 4, 2) == (1, 5, 10, 15)


def test_offset_count_five_and_ten_are_configurable():
    assert progressive_grassmann_offsets(0, 5, 2) == (1, 2, 4, 6, 8)
    assert progressive_grassmann_offsets(3, 5, 2) == (1, 5, 10, 15, 20)
    assert progressive_grassmann_offsets(3, 10, 2) == (
        1, 5, 10, 15, 20, 25, 30, 35, 40, 45
    )


def test_adaptive_progressive_offsets_use_approved_n_dependent_stride_formula():
    # T=301 => C=300. R=4 and base=2.
    # N=4: Lmax=15, S=max(16, ceil(300/4))=75.
    assert adaptive_progressive_grassmann_offsets(0, 301, 4, 2, 4) == (
        1, 2, 4, 6, 75, 150, 225, 300
    )
    assert adaptive_progressive_grassmann_offsets(1, 301, 4, 2, 4) == (
        1, 3, 6, 9, 75, 150, 225, 300
    )
    assert adaptive_progressive_grassmann_offsets(2, 301, 4, 2, 4) == (
        1, 4, 8, 12, 75, 150, 225, 300
    )
    assert adaptive_progressive_grassmann_offsets(3, 301, 4, 2, 4) == (
        1, 5, 10, 15, 75, 150, 225, 300
    )

    # Larger N produces more local resolution and a denser global schedule.
    assert adaptive_progressive_grassmann_offsets(0, 301, 5, 2, 4) == (
        1, 2, 4, 6, 8, 60, 120, 180, 240, 300
    )
    assert adaptive_progressive_grassmann_offsets(0, 301, 10, 2, 4) == (
        1, 2, 4, 6, 8, 10, 12, 14, 16, 18,
        46, 92, 138, 184, 230, 276,
    )


def test_adaptive_progressive_offsets_handle_short_and_extreme_contexts():
    # With short context, S is forced just beyond the largest local offset.
    assert adaptive_progressive_grassmann_offsets(3, 31, 4, 2, 4) == (
        1, 5, 10, 15, 16
    )
    # Large N is bounded by available history and creates no invalid lags.
    offsets = adaptive_progressive_grassmann_offsets(3, 31, 1000, 2, 4)
    assert offsets == (1, 5, 10, 15, 20, 25, 30)
    assert all(0 < lag < 31 for lag in offsets)
    # N=1 is valid: mandatory lag 1 plus the single far-context global lag.
    assert adaptive_progressive_grassmann_offsets(0, 301, 1, 2, 4) == (1, 300)
    assert adaptive_progressive_grassmann_offsets(0, 1, 4, 2, 4) == ()


def test_adaptive_progressive_offsets_reject_invalid_recurrent_indices():
    import pytest
    with pytest.raises(ValueError, match="loop_idx"):
        adaptive_progressive_grassmann_offsets(4, 301, 4, 2, 4)
    with pytest.raises(ValueError, match="recurrent_loops"):
        adaptive_progressive_grassmann_offsets(0, 301, 4, 2, 0)


def test_real_grassmann_adds_stride_32_offsets_to_actual_context():
    offsets = real_grassmann_offsets(
        seq_len=301,
        local_offsets=(1, 2, 4, 8, 12, 16),
        long_range_stride=32,
    )
    assert offsets == (
        1, 2, 4, 8, 12, 16,
        32, 64, 96, 128, 160, 192, 224, 256, 288,
    )
    assert max(offsets) == 288  # next stride (320) exceeds the 301-position context


def test_fixed_grassmann_retains_the_legacy_offset_16():
    module = GrassmannFlowAttention(
        dim=16, rank=4, mode="grassmann", offsets=(1, 2, 4, 8, 12, 16)
    )
    assert module.offsets_for(seq_len=301) == (1, 2, 4, 8, 12, 16)


def test_real_grassmann_stride16_contains_all_sixteen_step_lookbacks():
    offsets = real_grassmann_offsets(
        seq_len=301,
        local_offsets=(1, 2, 4, 8, 12, 16),
        long_range_stride=16,
    )
    assert offsets == (
        1, 2, 4, 8, 12, 16, 32, 48, 64, 80, 96, 112,
        128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288,
    )
    assert all(v % 16 == 0 for v in offsets if v >= 16)


def test_grassmann_new_module_selects_offsets_by_recurrent_loop():
    module = GrassmannFlowAttention(
        dim=16, rank=4, mode="grassmann-new", offset_count=4, offset_spacing_base=2, max_loops=4
    )
    assert module.offsets_for(seq_len=301, loop_idx=0) == (1, 2, 4, 6, 75, 150, 225, 300)
    assert module.offsets_for(seq_len=301, loop_idx=1) == (1, 3, 6, 9, 75, 150, 225, 300)
    assert module.offsets_for(seq_len=301, loop_idx=2) == (1, 4, 8, 12, 75, 150, 225, 300)
    assert module.offsets_for(seq_len=301, loop_idx=3) == (1, 5, 10, 15, 75, 150, 225, 300)


def test_grassmann_new_caps_generated_offsets_to_actual_sequence_length():
    module = GrassmannFlowAttention(
        dim=16, rank=4, mode="grassmann-new", offset_count=1000, offset_spacing_base=2, max_loops=4
    )
    offsets = module.offsets_for(seq_len=31, loop_idx=3)
    assert offsets == (1, 5, 10, 15, 20, 25, 30)
    assert all(0 < delta < 31 for delta in offsets)


def test_real_grassmann_directly_uses_a_32_token_history_pair_and_remains_causal():
    torch.manual_seed(31)
    module = GrassmannFlowAttention(
        dim=16,
        rank=4,
        mode="real-grassmann",
        offsets=(1, 2, 4, 8, 12, 16),
        long_range_stride=32,
    ).eval()
    x = torch.randn(1, 64, 16, requires_grad=True)
    y = module(x, loop_idx=0)

    # Output position 40 has an explicit lag-32 pair with input position 8.
    grad = torch.autograd.grad(y[0, 40, 0], x)[0]
    assert grad[0, 8].abs().sum().item() > 0.0

    # No output position may depend on future inputs.
    assert torch.count_nonzero(grad[0, 41:]).item() == 0


def test_real_grassmann_stride16_has_a_live_long_range_gradient_and_is_causal():
    torch.manual_seed(310)
    module = GrassmannFlowAttention(
        dim=16, rank=4, mode="real-grassmann",
        offsets=(1, 2, 4, 8, 12, 16), long_range_stride=16,
    ).eval()
    x = torch.randn(1, 64, 16, requires_grad=True)
    y = module(x, loop_idx=0)
    grad = torch.autograd.grad(y[0, 40, 0], x)[0]
    assert grad[0, 24].abs().sum().item() > 0.0
    assert torch.count_nonzero(grad[0, 41:]).item() == 0


def test_grassmann_new_uses_long_range_adaptive_pairs_and_remains_causal():
    torch.manual_seed(311)
    module = GrassmannFlowAttention(
        dim=16, rank=4, mode="grassmann-new", offset_count=4,
        offset_spacing_base=2, max_loops=4,
    ).eval()
    x = torch.randn(1, 64, 16, requires_grad=True)
    y = module(x, loop_idx=0)

    # For T=64, N=4, R=4: S=max(16, ceil(63/4))=16.
    # Position 40 therefore has a direct lag-16 pair from position 24.
    grad = torch.autograd.grad(y[0, 40, 0], x)[0]
    assert grad[0, 24].abs().sum().item() > 0.0
    assert torch.count_nonzero(grad[0, 41:]).item() == 0


def test_grassmann_new_full_model_uses_loop_idx_and_is_causal():
    torch.manual_seed(32)
    model = LoopTransformer(tiny_cfg()).eval()
    tokens = torch.randint(0, 32, (1, 48))
    changed = tokens.clone()
    changed[:, 30:] = torch.randint(0, 32, changed[:, 30:].shape)
    with torch.no_grad():
        logits_a = model(tokens, max_loops=4)[0][-1]
        logits_b = model(changed, max_loops=4)[0][-1]
    assert torch.allclose(logits_a[:, :30], logits_b[:, :30], atol=1e-5, rtol=1e-5)


def test_all_grassmann_modes_have_same_parameter_structure_without_positions():
    cfg_kwargs = dict(grassmann_positional_embedding=False)
    counts = []
    keys = []
    for mode in ("grassmann", "grassmann-new", "real-grassmann"):
        torch.manual_seed(33)
        model = LoopTransformer(tiny_cfg(attention_type=mode, **cfg_kwargs))
        counts.append(model.num_parameters())
        keys.append({k: tuple(v.shape) for k, v in model.state_dict().items()})
    assert counts[0] == counts[1] == counts[2]
    assert keys[0] == keys[1] == keys[2]


def test_cli_exposes_all_three_grassmann_modes_and_hyperparameters():
    for mode in ("grassmann", "grassmann-new", "real-grassmann"):
        args = build_parser().parse_args([
            "--attention-type", mode,
            "--grassmann-offset-count", "4",
            "--grassmann-offset-spacing-base", "2",
            "--grassmann-long-range-stride", "32",
            "--no-grassmann-positional-embedding",
        ])
        cfg = config_from_args(args)
        assert cfg.attention_type == mode
        assert cfg.grassmann_offset_count == 4
        assert cfg.grassmann_offset_spacing_base == 2
        assert cfg.grassmann_long_range_stride == 32
        assert cfg.grassmann_positional_embedding is False


def test_real_grassmann_cli_default_is_sixteen_token_stride():
    args = build_parser().parse_args(["--attention-type", "real-grassmann"])
    cfg = config_from_args(args)
    assert cfg.grassmann_long_range_stride == 16


def test_published_kaggle_commands_select_the_intended_offset_modes():
    root = Path(__file__).resolve().parents[1]

    def parse_command(filename):
        contents = (root / filename).read_text(encoding="utf-8")
        assert "train.py" in contents
        arguments = shlex.split(contents.split("train.py", 1)[1].replace("\\\n", " "))
        return build_parser().parse_args(arguments)

    args_a = parse_command("GRASSMANN_ABLATION_A_NO_ABSOLUTE_POSITIONS.txt")
    args_b = parse_command("GRASSMANN_ABLATION_B_LEARNED_POSITIONS.txt")
    cfg_a, cfg_b = config_from_args(args_a), config_from_args(args_b)
    assert cfg_a.attention_type == cfg_b.attention_type == "real-grassmann"
    assert cfg_a.grassmann_long_range_stride == cfg_b.grassmann_long_range_stride == 16
    assert cfg_a.grassmann_offsets == cfg_b.grassmann_offsets == (1, 2, 4, 8, 12, 16)
    assert cfg_a.grassmann_positional_embedding is False
    assert cfg_b.grassmann_positional_embedding is True
    assert "_stride16" in str(args_a.checkpoint_dir)
    assert "_stride16" in str(args_b.checkpoint_dir)
    assert args_a.checkpoint_dir != args_b.checkpoint_dir

    args_n4 = parse_command("KAGGLE_GRASSMANN_NEW_N4_ADAPTIVE_COMMAND.txt")
    args_n5 = parse_command("KAGGLE_GRASSMANN_NEW_N5_COMMAND.txt")
    assert config_from_args(args_n4).attention_type == "grassmann-new"
    assert config_from_args(args_n4).grassmann_offset_count == 4
    assert config_from_args(args_n5).attention_type == "grassmann-new"
    assert config_from_args(args_n5).grassmann_offset_count == 5


def test_config_rejects_invalid_grassmann_offset_hyperparameters():
    for field, value in (
        ("grassmann_offset_count", 0),
        ("grassmann_offset_spacing_base", 0),
        ("grassmann_long_range_stride", 0),
    ):
        try:
            tiny_cfg(**{field: value})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid {field}={value} was accepted")


def test_visualizer_names_new_and_real_grassmann_modes(tmp_path):
    for mode, expected_label, expected_fragment in (
        ("grassmann-new", "Grassmann-New", "adaptive sparse-global causal Plücker mixing"),
        ("real-grassmann", "Real Grassmann", "stride-spaced long-range Plücker mixing"),
    ):
        model = LoopTransformer(tiny_cfg(attention_type=mode))
        viz = LoopTransformerVisualizer(model, tmp_path / mode)
        assert expected_label in viz.viz_label
        assert expected_fragment in viz.viz_description


def test_real_grassmann_uses_only_stride_multiples_through_context():
    module = GrassmannFlowAttention(
        dim=16,
        rank=4,
        mode="real-grassmann",
        offsets=(1, 2, 4, 8, 12, 16),
        long_range_stride=32,
    ).eval()
    offsets = module.offsets_for(seq_len=301)
    assert offsets[-1] == 288
    assert 300 not in offsets
    assert all(v % 32 == 0 for v in offsets if v > 16)


def test_new_modes_checkpoint_round_trip(tmp_path):
    from loop_transformer.checkpointing import load_checkpoint, save_checkpoint

    for mode in ("grassmann-new", "real-grassmann"):
        model = LoopTransformer(tiny_cfg(attention_type=mode))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        path = tmp_path / f"{mode}.pt"
        save_checkpoint(path, model, optimizer=optimizer, step=7)
        restored = load_checkpoint(path, device="cpu")
        assert restored["model"].cfg.attention_type == mode
        if mode == "grassmann-new":
            assert restored["model"].cfg.grassmann_offset_scheme == "adaptive_sparse_global_v1"
        assert restored["step"] == 7
        assert restored["model"].num_parameters() == model.num_parameters()


def test_pre_adaptive_grassmann_new_checkpoint_keeps_legacy_schedule_and_cannot_resume(tmp_path):
    from argparse import Namespace
    import pytest
    from loop_transformer.checkpointing import load_checkpoint, save_checkpoint
    from train import _validate_resume_architecture_overrides

    model = LoopTransformer(tiny_cfg(attention_type="grassmann-new"))
    path = tmp_path / "legacy_local_only.pt"
    save_checkpoint(path, model, step=12)

    # Simulate a checkpoint written before grassmann_offset_scheme existed.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"].pop("grassmann_offset_scheme")
    torch.save(payload, path)

    restored = load_checkpoint(path, device="cpu")
    restored_model = restored["model"]
    attention = restored_model.blocks[0].attn
    assert restored_model.cfg.grassmann_offset_scheme == "legacy_local_only_v0"
    assert attention.offsets_for(seq_len=301, loop_idx=3) == (1, 5, 10, 15)

    # The trainer must not silently mix the old local-only run with new offsets.
    with pytest.raises(ValueError, match="older local-only offset schedule"):
        _validate_resume_architecture_overrides(
            Namespace(), {"model": restored_model}
        )


def test_resume_guard_rejects_changing_progressive_offset_count():
    from argparse import Namespace
    import pytest
    from train import _validate_resume_architecture_overrides

    model = LoopTransformer(tiny_cfg(attention_type="grassmann-new"))
    args = Namespace(
        attention_type="grassmann-new",
        grassmann_rank=4,
        grassmann_offsets=(1, 2, 4, 8, 12, 16),
        grassmann_positional_embedding=False,
        grassmann_offset_count=5,
        grassmann_offset_spacing_base=2,
        grassmann_long_range_stride=32,
        _explicit_cli_overrides={"grassmann_offset_count"},
    )
    with pytest.raises(ValueError, match="grassmann-offset-count"):
        _validate_resume_architecture_overrides(
            args,
            {"model": model},
        )


def test_resume_guard_rejects_changing_progressive_recurrent_depth():
    from argparse import Namespace
    import pytest
    from train import _validate_resume_architecture_overrides

    model = LoopTransformer(tiny_cfg(attention_type="grassmann-new", max_loops=4))
    args = Namespace(
        attention_type="grassmann-new",
        grassmann_rank=4,
        grassmann_offsets=(1, 2, 4, 8, 12, 16),
        grassmann_positional_embedding=False,
        grassmann_offset_count=4,
        grassmann_offset_spacing_base=2,
        grassmann_long_range_stride=16,
        max_loops=5,
        _explicit_cli_overrides={"max_loops"},
    )
    with pytest.raises(ValueError, match="max-loops"):
        _validate_resume_architecture_overrides(args, {"model": model})
