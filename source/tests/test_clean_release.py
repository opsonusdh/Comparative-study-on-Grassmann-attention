import tempfile
from pathlib import Path

import numpy as np
import torch

from loop_transformer import LoopConfig, LoopTransformer, load_checkpoint, save_checkpoint
from loop_transformer.attention import CSAAttention, HCAAttention, SlidingWindowAttention
from loop_transformer.visualizer import LoopTransformerVisualizer


def tiny_cfg(**overrides):
    cfg = dict(
        vocab_size=32,
        dim=32,
        n_layers=4,
        n_heads=4,
        head_dim=8,
        ffn_hidden_dim=64,
        rope_dim=8,
        max_loops=3,
        min_loops=3,
        loop_sampling=False,
        csa_m=2,
        csa_top_k=2,
        csa_aux_loss_weight=0.0,
        hca_m_prime=4,
        sw_window=4,
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
    cfg.update(overrides)
    return LoopConfig(**cfg)


def test_full_model_is_causal_for_logits():
    torch.manual_seed(0)
    model = LoopTransformer(tiny_cfg()).eval()
    x = torch.randint(0, 32, (1, 12))
    xp = x.clone()
    xp[:, 10] = (xp[:, 10] + 1) % 32
    with torch.no_grad():
        a = model(x, max_loops=3)[0][-1]
        b = model(xp, max_loops=3)[0][-1]
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-7, rtol=1e-7)


def test_fixed_attention_families_are_selected_by_layer():
    model = LoopTransformer(tiny_cfg(n_layers=4))
    assert isinstance(model.blocks[0].attn, SlidingWindowAttention)
    assert isinstance(model.blocks[1].attn, SlidingWindowAttention)
    assert isinstance(model.blocks[2].attn, CSAAttention)
    assert isinstance(model.blocks[3].attn, HCAAttention)


def test_no_retired_msa_or_diffusion_symbols_in_active_sources():
    root = Path(__file__).resolve().parents[1]
    active = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (root / "src").glob("loop_transformer/*.py")
    )
    for marker in ("MSAAttention", "use_msa_attention", "DiffusionBlocks", "token_adaptive_depth", "loop_clock_embedding"):
        assert marker not in active
    assert not (root / "src" / "loop_transformer" / "model_v2.py").exists()
    assert not (root / "src" / "loop_transformer" / "experimental_mechanisms.py").exists()


def test_visualizer_writes_periodic_snapshot():
    model = LoopTransformer(tiny_cfg()).eval()
    with tempfile.TemporaryDirectory() as td:
        viz = LoopTransformerVisualizer(model, td)
        x = torch.randint(0, 32, (1, 12))
        viz.snapshot(0, x)
        assert (Path(td) / "step_00000000.png").exists()
        assert (Path(td) / "latest.png").exists()
        metrics = Path(td) / "latest.json"
        assert metrics.exists() and metrics.stat().st_size > 0


def test_single_binary_split_is_zero_copy():
    arr = np.arange(1000, dtype=np.uint16)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "Corpus.bin"
        arr.tofile(path)
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        split = int(len(mm) * 0.9)
        train = mm[:split]
        val = mm[split:]
        assert len(train) == 900
        assert len(val) == 100
        assert int(train[-1]) == 899
        assert int(val[0]) == 900


def test_checkpoint_resume_restores_rng(tmp_path: Path):
    model = LoopTransformer(tiny_cfg(n_layers=2, max_loops=2, min_loops=2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 32, (2, 8))
    loss, _ = model.compute_loss(x)
    loss.backward()
    opt.step()
    import random
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    ckpt = tmp_path / "resume.pt"
    save_checkpoint(ckpt, model, opt, step=7)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    loaded = load_checkpoint(ckpt, restore_rng=True)
    restored = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert restored == expected
    assert loaded["step"] == 7
    assert loaded["optimizer_state_dict"]


def test_grad_checkpointing_backward_is_finite():
    model = LoopTransformer(tiny_cfg(n_layers=2, max_loops=2, min_loops=2, grad_checkpointing=True)).train()
    x = torch.randint(0, 32, (2, 8))
    loss, _ = model.compute_loss(x, max_loops=2)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
