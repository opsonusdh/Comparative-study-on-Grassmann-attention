import torch

from loop_transformer import LoopConfig, LoopTransformer
from loop_transformer.visualizer import LoopTransformerVisualizer
from train import build_parser, config_from_args


def _tiny_config(**overrides):
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
        grassmann_positional_max_len=16,
        grassmann_positional_seed=2009,
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


def _shared_named_parameters(model):
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if name != "grassmann_positional_embedding.weight"
    }


def test_position_table_does_not_shift_shared_model_initialization():
    torch.manual_seed(2009)
    without_positions = LoopTransformer(
        _tiny_config(grassmann_positional_embedding=False)
    )

    torch.manual_seed(2009)
    with_positions = LoopTransformer(
        _tiny_config(grassmann_positional_embedding=True)
    )

    assert without_positions.grassmann_positional_embedding is None
    assert with_positions.grassmann_positional_embedding is not None
    left = _shared_named_parameters(without_positions)
    right = _shared_named_parameters(with_positions)
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name



def test_position_table_adds_only_its_expected_parameters():
    torch.manual_seed(2009)
    without_positions = LoopTransformer(
        _tiny_config(grassmann_positional_embedding=False)
    )
    torch.manual_seed(2009)
    with_positions = LoopTransformer(
        _tiny_config(grassmann_positional_embedding=True)
    )
    difference = with_positions.num_parameters() - without_positions.num_parameters()
    assert difference == 16 * 32

def test_position_table_is_deterministic_and_rng_isolated():
    torch.manual_seed(100)
    model_a = LoopTransformer(_tiny_config(grassmann_positional_seed=77))
    torch.manual_seed(100)
    model_b = LoopTransformer(_tiny_config(grassmann_positional_seed=77))
    assert torch.equal(
        model_a.grassmann_positional_embedding.weight,
        model_b.grassmann_positional_embedding.weight,
    )


def test_cli_exposes_position_ablation_seed_and_visualization_label():
    args = build_parser().parse_args([
        "--attention-type", "grassmann",
        "--no-grassmann-positional-embedding",
        "--grassmann-positional-max-len", "301",
        "--grassmann-positional-seed", "2009",
        "--viz-label", "Grassmann | No Absolute Positions",
    ])
    cfg = config_from_args(args)
    assert cfg.grassmann_positional_embedding is False
    assert cfg.grassmann_positional_max_len == 301
    assert cfg.grassmann_positional_seed == 2009
    assert args.viz_label == "Grassmann | No Absolute Positions"


def test_visualizer_describes_both_grassmann_modes(tmp_path):
    torch.manual_seed(3)
    no_pos_model = LoopTransformer(_tiny_config(grassmann_positional_embedding=False))
    no_pos_viz = LoopTransformerVisualizer(no_pos_model, tmp_path / "no_pos")
    assert "no absolute positions" in no_pos_viz.viz_label
    assert "positions disabled" in no_pos_viz.viz_description

    torch.manual_seed(3)
    pos_model = LoopTransformer(_tiny_config(grassmann_positional_embedding=True))
    pos_viz = LoopTransformerVisualizer(pos_model, tmp_path / "with_pos")
    assert "learned absolute positions" in pos_viz.viz_label
    assert "max length 16" in pos_viz.viz_description


def test_custom_visualization_label_overrides_default(tmp_path):
    torch.manual_seed(4)
    model = LoopTransformer(_tiny_config(grassmann_positional_embedding=False))
    viz = LoopTransformerVisualizer(
        model,
        tmp_path / "custom",
        viz_label="Grassmann Ablation A",
    )
    assert viz.viz_label == "Grassmann Ablation A"
    assert "causal multi-scale Plücker mixing" in viz.viz_description


def test_real_grassmann_stride16_position_ablation_keeps_shared_init_aligned():
    torch.manual_seed(2009)
    without_positions = LoopTransformer(
        _tiny_config(
            attention_type="real-grassmann",
            grassmann_long_range_stride=16,
            grassmann_positional_embedding=False,
        )
    )
    torch.manual_seed(2009)
    with_positions = LoopTransformer(
        _tiny_config(
            attention_type="real-grassmann",
            grassmann_long_range_stride=16,
            grassmann_positional_embedding=True,
        )
    )
    left = _shared_named_parameters(without_positions)
    right = _shared_named_parameters(with_positions)
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name
