"""Lightweight mechanistic visualizer used directly by the canonical trainer."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .config import is_grassmann_attention


class LoopTransformerVisualizer:
    """Save reproducible training snapshots without changing training RNG."""

    def __init__(
        self,
        model,
        output_dir: Path,
        probe_batch: Optional[torch.Tensor] = None,
        viz_label: Optional[str] = None,
        resume_step: Optional[int] = None,
    ):
        self.model = model
        self.viz_label = (str(viz_label).strip() if viz_label else "") or self._default_label(model)
        self.viz_description = self._describe_architecture(model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.probe_path = self.output_dir / "probe.pt"
        self.history_path = self.output_dir / "history.jsonl"
        self.latest_metrics_path = self.output_dir / "latest.json"
        self.history: list[dict[str, Any]] = []
        self.resume_step = int(resume_step) if resume_step is not None else None
        self._load_history(max_step=self.resume_step)

        # A resumed run must append to the existing graph, not quietly create a
        # new history at the next snapshot. Require the saved boundary record so
        # a wrong --viz-dir is detected before any optimizer step is executed.
        self.history_train_steps = [
            int(row["step"])
            for row in self.history
            if row.get("event") == "train" and isinstance(row.get("step"), int)
        ]
        if self.resume_step is not None:
            if not self.history_path.is_file():
                raise RuntimeError(
                    f"Visualizer resume history is missing: {self.history_path}. "
                    "Point --viz-dir at the restored experiment visualizations directory; "
                    "refusing to start a disconnected graph."
                )
            if self.resume_step not in self.history_train_steps:
                latest = max(self.history_train_steps, default=None)
                raise RuntimeError(
                    f"Visualizer history at {self.history_path} does not contain the "
                    f"checkpoint boundary step {self.resume_step} (latest saved train step={latest}). "
                    "Refusing to start a disconnected or mismatched graph."
                )

        self.probe_batch: Optional[torch.Tensor] = None
        if self.probe_path.is_file():
            try:
                try:
                    saved_probe = torch.load(self.probe_path, map_location="cpu", weights_only=True)
                except TypeError:  # compatibility with older supported PyTorch builds
                    saved_probe = torch.load(self.probe_path, map_location="cpu")
                if not torch.is_tensor(saved_probe) or saved_probe.ndim != 2:
                    raise ValueError("saved probe must be a rank-2 token tensor")
                self.probe_batch = saved_probe.detach().cpu().long()
            except Exception as exc:
                if self.resume_step is not None:
                    raise RuntimeError(
                        f"Could not restore the visualizer probe {self.probe_path}: {exc}"
                    ) from exc
        if self.probe_batch is None:
            if self.resume_step is not None:
                raise RuntimeError(
                    f"Visualizer probe is missing or invalid: {self.probe_path}. "
                    "Refusing to replace the original diagnostic batch during resume."
                )
            if probe_batch is not None:
                self.set_probe(probe_batch)

    @staticmethod
    def _default_label(model) -> str:
        cfg = model.cfg
        mode = getattr(cfg, "attention_type", "deepseek")
        if is_grassmann_attention(mode):
            position_mode = (
                "learned absolute positions"
                if getattr(cfg, "grassmann_positional_embedding", True)
                else "no absolute positions"
            )
            if mode == "grassmann-new":
                if getattr(cfg, "grassmann_offset_scheme", "legacy_local_only_v0") == "legacy_local_only_v0":
                    return f"Grassmann-New • Legacy Local-Only Offsets • {position_mode}"
                return f"Grassmann-New • Adaptive Local + Sparse Global Offsets • {position_mode}"
            if mode == "real-grassmann":
                return f"Real Grassmann • Sparse Long-Range Context • {position_mode}"
            return f"Grassmann Attention • Fixed Local Offsets • {position_mode}"
        if mode == "deepseek-split":
            return f"DeepSeek SWA / CSA / HCA Loop Split • stride {cfg.deepseek_split_stride}"
        return "DeepSeek Attention • SWA / CSA / HCA"

    @staticmethod
    def _describe_architecture(model) -> str:
        cfg = model.cfg
        mode = getattr(cfg, "attention_type", "deepseek")
        if is_grassmann_attention(mode):
            local_offsets = ", ".join(str(int(x)) for x in cfg.grassmann_offsets)
            is_legacy_local_only = (
                getattr(cfg, "grassmann_offset_scheme", "legacy_local_only_v0")
                == "legacy_local_only_v0"
            )
            if mode == "grassmann-new" and is_legacy_local_only:
                schedule = []
                for loop_idx in range(cfg.max_loops):
                    spacing = cfg.grassmann_offset_spacing_base + loop_idx
                    offsets = tuple(sorted({1, *(
                        k * spacing for k in range(1, cfg.grassmann_offset_count)
                    )}))
                    schedule.append(f"L{loop_idx}=[" + ",".join(map(str, offsets)) + "]")
                mechanism = (
                    "legacy progressive local-only causal Plücker mixing • "
                    f"local count={cfg.grassmann_offset_count} • "
                    + "; ".join(schedule)
                )
            elif mode == "grassmann-new":
                schedule = []
                for loop_idx in range(cfg.max_loops):
                    spacing = cfg.grassmann_offset_spacing_base + loop_idx
                    offsets = tuple(sorted({1, *(
                        k * spacing for k in range(1, cfg.grassmann_offset_count)
                    )}))
                    schedule.append(f"L{loop_idx}=[" + ",".join(map(str, offsets)) + "]")
                max_local = (cfg.grassmann_offset_count - 1) * (
                    cfg.grassmann_offset_spacing_base + cfg.max_loops - 1
                )
                mechanism = (
                    "progressive local + adaptive sparse-global causal Plücker mixing • "
                    f"local count={cfg.grassmann_offset_count} • "
                    f"adaptive global stride=max({max_local + 1}, ceil((T-1)/N)) • "
                    + "; ".join(schedule)
                )
            elif mode == "real-grassmann":
                mechanism = (
                    "causal local + stride-spaced long-range Plücker mixing • "
                    f"local offsets=[{local_offsets}] • "
                    f"long-range stride={cfg.grassmann_long_range_stride} "
                    "(extends to actual sequence length)"
                )
            else:
                mechanism = f"fixed causal multi-scale Plücker mixing • offsets=[{local_offsets}]"

            if getattr(cfg, "grassmann_positional_embedding", True):
                position = (
                    "learned absolute token positions "
                    f"(max length {getattr(cfg, 'grassmann_positional_max_len', 1024)})"
                )
            else:
                position = "absolute token positions disabled (ablation)"
            return f"Attention mechanism: {mechanism} • rank={cfg.grassmann_rank} • {position}"
        if mode == "deepseek-split":
            stride = int(cfg.deepseek_split_stride)
            schedule = []
            for loop_idx in range(cfg.max_loops):
                phase = loop_idx % stride
                ages = {1}
                for age in range(1, 65):
                    if age % stride == phase:
                        ages.add(age)
                schedule.append(
                    f"L{loop_idx}=[" + ",".join(map(str, sorted(ages))) + ",…]"
                )
            return (
                "Attention mechanism: SWA + CSA + HCA preserved; "
                "full SWA/raw local windows with loop-partitioned compressed long-range "
                f"CSA/HCA memory • stride={stride} • " + "; ".join(schedule)
            )
        return "Attention mechanism: original SWA / CSA / HCA"

    def _load_history(self, max_step: Optional[int] = None) -> None:
        if not self.history_path.exists():
            return
        original_lines = self.history_path.read_text(encoding="utf-8").splitlines()
        kept_lines: list[str] = []
        for line in original_lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = record.get("step")
            if max_step is not None and isinstance(step, int) and step > max_step:
                continue
            self.history.append(record)
            kept_lines.append(json.dumps(record, sort_keys=True))

        # If the visualization log got ahead of the last durable checkpoint,
        # trim those uncheckpointed points. Otherwise the new graph would show
        # stale future points before the resumed run reaches them again.
        if max_step is not None and len(kept_lines) != len(original_lines):
            tmp_path = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
            tmp_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
            tmp_path.replace(self.history_path)

    def _append(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def record_training(self, step: int, loss: float, lr: float, grad_norm: float | None) -> None:
        self._append({
            "event": "train",
            "step": int(step),
            "loss": float(loss),
            "lr": float(lr),
            "grad_norm": None if grad_norm is None else float(grad_norm),
        })

    def record_validation(self, step: int, loss: float, accuracy: float) -> None:
        self._append({
            "event": "val",
            "step": int(step),
            "loss": float(loss),
            "accuracy": float(accuracy),
        })

    def set_probe(self, batch: torch.Tensor) -> None:
        self.probe_batch = batch.detach().cpu().clone()
        torch.save(self.probe_batch, self.probe_path)

    @staticmethod
    def _safe_float(x: Any) -> float:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        return float(x)

    def snapshot(
        self,
        step: int,
        batch: Optional[torch.Tensor] = None,
        *,
        validation_metrics: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if batch is not None:
            self.set_probe(batch)
        if self.probe_batch is None:
            raise RuntimeError("Visualizer has no probe batch")

        probe = self.probe_batch
        device = next(self.model.parameters()).device
        probe = probe.to(device)
        was_training = self.model.training
        self.model.eval()
        self.model._capture_layer_states = True
        try:
            with torch.no_grad():
                loop_logits, lambdas = self.model(probe)
                states = [x.detach() for x in getattr(self.model, "_last_loop_states", [])]
                registers = [x.detach() for x in getattr(self.model, "_last_loop_register_states", [])]
                controller_stats = list(getattr(self.model, "_last_loop_controller_stats", []))
                targets = probe[:, 1:]
                loop_loss: list[float] = []
                loop_acc: list[float] = []
                for logits in loop_logits:
                    pred = logits[:, :-1].argmax(dim=-1)
                    loop_acc.append(float((pred == targets).float().mean().cpu()))
                    ce = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                        targets.reshape(-1),
                    )
                    loop_loss.append(float(ce.cpu()))

                # The probe batch is intentionally small and is used for
                # hidden-state/controller diagnostics. It is not a sound basis
                # for comparing per-loop language-model quality. When the
                # trainer has just evaluated the fixed validation set, replace
                # the probe-only CE/accuracy with the evaluator's token-weighted
                # full-validation aggregates. This reuses already-computed
                # logits and adds no validation forward pass.
                loop_metric_source = "probe_batch"
                if validation_metrics is not None:
                    val_loop_ce = validation_metrics.get("val_loop_ce")
                    val_loop_acc_pct = validation_metrics.get("val_loop_accuracy_pct")
                    if val_loop_ce is None or val_loop_acc_pct is None:
                        raise ValueError(
                            "validation_metrics must contain val_loop_ce and "
                            "val_loop_accuracy_pct"
                        )
                    if len(val_loop_ce) != len(loop_logits) or len(val_loop_acc_pct) != len(loop_logits):
                        raise ValueError(
                            "Full-validation per-loop metric count does not match "
                            f"the model loop count: logits={len(loop_logits)}, "
                            f"CE={len(val_loop_ce)}, accuracy={len(val_loop_acc_pct)}"
                        )
                    loop_loss = [float(value) for value in val_loop_ce]
                    loop_acc = [float(value) / 100.0 for value in val_loop_acc_pct]
                    loop_metric_source = "full_validation"
        finally:
            self.model._capture_layer_states = False
            if was_training:
                self.model.train()

        state_norms = [float(x.float().norm(dim=-1).mean().cpu()) for x in states]
        state_cosines = [float("nan")]
        for a, b in zip(states[:-1], states[1:]):
            av, bv = a.float(), b.float()
            state_cosines.append(float(F.cosine_similarity(av, bv, dim=-1).mean().cpu()))
        register_norms = [float(x.float().norm(dim=-1).mean().cpu()) for x in registers]
        growth = state_norms[-1] / max(state_norms[0], 1e-12) if state_norms else float("nan")

        routing = self.model.get_routing_debug() if hasattr(self.model, "get_routing_debug") else {}
        metrics = {
            "step": int(step),
            "params": int(self.model.num_parameters()),
            "loop_ce": loop_loss,
            "loop_accuracy": loop_acc,
            "loop_metric_source": loop_metric_source,
            "state_norms": state_norms,
            "state_cosine_prev": state_cosines,
            "register_norms": register_norms,
            "state_growth_ratio": growth,
            "controller": controller_stats,
            "routing": routing,
            "viz_label": self.viz_label,
            "architecture_description": self.viz_description,
            "health": "WARNING_STATE_GROWTH" if math.isfinite(growth) and growth > 100.0 else "OK",
        }
        self.latest_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._render(step, metrics)
        return metrics

    def _render(self, step: int, metrics: dict[str, Any]) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()

        train = [r for r in self.history if r.get("event") == "train"]
        val = [r for r in self.history if r.get("event") == "val"]

        ax = axes[0]
        if train:
            ax.plot([r["step"] for r in train], [r["loss"] for r in train], label="train objective")
        if val:
            ax.plot([r["step"] for r in val], [r["loss"] for r in val], marker="o", label="val CE")
        ax.set_title("Loss")
        ax.set_xlabel("Step")
        ax.grid(alpha=0.2)
        if train or val:
            ax.legend()

        ax = axes[1]
        if val:
            ax.plot([r["step"] for r in val], [100.0 * r["accuracy"] for r in val], marker="o")
        ax.set_title("Validation accuracy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Percent")
        ax.grid(alpha=0.2)

        ax = axes[2]
        loops = list(range(1, len(metrics["loop_ce"]) + 1))
        ax.plot(loops, metrics["loop_ce"], marker="o", label="CE")
        ax2 = ax.twinx()
        ax2.plot(loops, [100.0 * x for x in metrics["loop_accuracy"]], marker="s", label="accuracy")
        metric_source = metrics.get("loop_metric_source", "probe_batch")
        source_label = "full validation" if metric_source == "full_validation" else "probe batch"
        ax.set_title(f"Per-loop performance • {source_label}")
        ax.set_xlabel("Loop")
        ax.set_ylabel("CE")
        ax2.set_ylabel("Accuracy %")
        ax.grid(alpha=0.2)

        ax = axes[3]
        if metrics["state_norms"]:
            ax.plot(loops, metrics["state_norms"], marker="o")
        ax.set_title(f"State norm growth • {metrics['health']}")
        ax.set_xlabel("Loop")
        ax.set_ylabel("Mean ||state||")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.grid(alpha=0.2)

        ax = axes[4]
        if metrics["controller"]:
            sx = list(range(1, len(metrics["controller"]) + 1))
            scale = [row["scale_norm"] if row else 0.0 for row in metrics["controller"]]
            bias = [row["bias_norm"] if row else 0.0 for row in metrics["controller"]]
            ax.plot(sx, scale, marker="o", label="scale norm")
            ax.plot(sx, bias, marker="s", label="bias norm")
            bound = float(getattr(self.model.cfg, "loop_controller_bound", 0.0))
            ax.axhline(bound * math.sqrt(getattr(self.model.cfg, "dim", 1)), linestyle="--", label="scale bound norm")
            ax.legend()
        ax.set_title("Loop controller")
        ax.set_xlabel("Loop")
        ax.grid(alpha=0.2)

        ax = axes[5]
        if metrics["register_norms"]:
            rx = list(range(1, len(metrics["register_norms"]) + 1))
            ax.plot(rx, metrics["register_norms"], marker="o")
        cos = metrics["state_cosine_prev"]
        if len(cos) > 1:
            ax2 = ax.twinx()
            ax2.plot(list(range(1, len(cos) + 1)), cos, marker="s", label="state cosine")
            ax2.set_ylabel("Cosine")
        ax.set_title("Registers / state direction")
        ax.set_xlabel("Loop")
        ax.set_ylabel("Register norm")
        ax.grid(alpha=0.2)

        fig.suptitle(
            f"{self.viz_label} • step {step} • params={metrics['params']:,} • "
            f"state growth={metrics['state_growth_ratio']:.3g}×",
            fontsize=15,
            y=0.985,
        )
        fig.text(
            0.5,
            0.955,
            self.viz_description,
            ha="center",
            va="top",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(self.output_dir / f"step_{step:08d}.png", dpi=140, bbox_inches="tight")
        fig.savefig(self.output_dir / "latest.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
