from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from loop_transformer.visualizer import LoopTransformerVisualizer


class _ToyLoopModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(dim=4, loop_controller_bound=0.2)
        self._capture_layer_states = False
        self._last_loop_states = []
        self._last_loop_register_states = []
        self._last_loop_controller_stats = []

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, tokens: torch.Tensor):
        batch, seq_len = tokens.shape
        logits_a = torch.zeros(batch, seq_len, 3, device=tokens.device)
        logits_b = torch.zeros(batch, seq_len, 3, device=tokens.device)
        # Deliberately poor probe-batch logits. The injected full-validation
        # metrics must replace these values in the per-loop comparison chart.
        logits_a[..., 1] = 1.0
        logits_b[..., 2] = 1.0
        state = torch.ones(batch, seq_len, 4, device=tokens.device)
        self._last_loop_states = [state, state * 1.5]
        self._last_loop_register_states = []
        self._last_loop_controller_stats = []
        return [logits_a, logits_b], [torch.ones(batch), torch.ones(batch)]


def test_snapshot_uses_full_validation_metrics_for_per_loop_plot(tmp_path: Path) -> None:
    model = _ToyLoopModel()
    visualizer = LoopTransformerVisualizer(model, tmp_path)
    probe = torch.zeros(2, 5, dtype=torch.long)

    metrics = visualizer.snapshot(
        100,
        probe,
        validation_metrics={
            "val_loop_ce": [2.75, 2.25],
            "val_loop_accuracy_pct": [31.5, 42.0],
        },
    )

    assert metrics["loop_metric_source"] == "full_validation"
    assert metrics["loop_ce"] == [2.75, 2.25]
    assert metrics["loop_accuracy"] == [0.315, 0.42]
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "step_00000100.png").exists()


def test_snapshot_falls_back_to_probe_metrics_without_validation(tmp_path: Path) -> None:
    model = _ToyLoopModel()
    visualizer = LoopTransformerVisualizer(model, tmp_path)
    probe = torch.zeros(2, 5, dtype=torch.long)

    metrics = visualizer.snapshot(100, probe)

    assert metrics["loop_metric_source"] == "probe_batch"
    assert len(metrics["loop_ce"]) == 2
    assert len(metrics["loop_accuracy"]) == 2


def test_snapshot_rejects_loop_metric_count_mismatch(tmp_path: Path) -> None:
    model = _ToyLoopModel()
    visualizer = LoopTransformerVisualizer(model, tmp_path)
    probe = torch.zeros(2, 5, dtype=torch.long)

    try:
        visualizer.snapshot(
            100,
            probe,
            validation_metrics={
                "val_loop_ce": [2.0],
                "val_loop_accuracy_pct": [50.0],
            },
        )
    except ValueError as exc:
        assert "loop count" in str(exc)
    else:
        raise AssertionError("Expected validation loop-count mismatch to fail")


def test_resume_visualizer_continues_history_and_restores_original_probe(tmp_path: Path) -> None:
    model = _ToyLoopModel()
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        '\n'.join([
            '{"event":"train","step":1,"loss":4.0,"lr":0.0001,"grad_norm":1.0}',
            '{"event":"val","step":100,"loss":3.0,"accuracy":0.2}',
            '{"event":"train","step":600,"loss":2.0,"lr":0.00002,"grad_norm":0.5}',
            '{"event":"train","step":700,"loss":1.9,"lr":0.00001,"grad_norm":0.4}',
        ]) + '\n',
        encoding="utf-8",
    )
    original_probe = torch.tensor([[0, 1, 2, 1, 0]], dtype=torch.long)
    torch.save(original_probe, tmp_path / "probe.pt")

    fallback_probe = torch.zeros(1, 5, dtype=torch.long)
    visualizer = LoopTransformerVisualizer(
        model,
        tmp_path,
        probe_batch=fallback_probe,
        resume_step=600,
    )

    assert [row["step"] for row in visualizer.history if row["event"] == "train"] == [1, 600]
    assert torch.equal(visualizer.probe_batch, original_probe)
    persisted = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert max(row["step"] for row in persisted) == 600

    # A new snapshot must use the restored probe and append to the old series.
    visualizer.record_training(601, 1.8, 1e-5, 0.3)
    visualizer.snapshot(700, None)
    assert torch.equal(torch.load(tmp_path / "probe.pt", map_location="cpu", weights_only=True), original_probe)
    assert (tmp_path / "step_00000700.png").exists()


def test_resume_visualizer_rejects_missing_history_boundary(tmp_path: Path) -> None:
    model = _ToyLoopModel()
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        '{"event":"train","step":500,"loss":2.0,"lr":0.00002,"grad_norm":0.5}\n',
        encoding="utf-8",
    )
    torch.save(torch.zeros(1, 5, dtype=torch.long), tmp_path / "probe.pt")

    try:
        LoopTransformerVisualizer(model, tmp_path, resume_step=600)
    except RuntimeError as exc:
        assert "checkpoint boundary step 600" in str(exc)
    else:
        raise AssertionError("Expected mismatched visualization history to fail closed")
