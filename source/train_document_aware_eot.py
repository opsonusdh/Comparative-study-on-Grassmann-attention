#!/usr/bin/env python3
"""Canonical LoopLLM recurrent trainer.

Single training entrypoint for the production recurrent package.

Key properties:
- reads pre-tokenized .bin corpora through numpy.memmap;
- supports either explicit train/validation binaries or one binary with a
  chronological validation split;
- exposes every active LoopConfig field through CLI arguments;
- uses the project's production training recipe: AdamW, warmup + cosine LR,
  gradient accumulation/clipping, optional AMP, deterministic evaluation,
  checkpoint/resume with RNG restoration;
- optionally saves mechanistic visualizations every N optimizer steps;
- with --source Kaggle, restores and asynchronously synchronizes the live loop_llm tree to Google Drive.

No experimental attention variants, diffusion path, or alternate trainer is
kept in this entrypoint.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_transformer import LoopConfig, LoopTransformer, LoopTransformerVisualizer
from loop_transformer.checkpointing import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# CLI parsing helpers
# ---------------------------------------------------------------------------


def _optional_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def _optional_float(value: str) -> Optional[float]:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def _source_type(value: str) -> str:
    """Normalize execution source names so `Kaggle`, `kaggle`, etc. are equivalent."""
    normalized = str(value).strip().lower()
    if normalized not in {"local", "kaggle"}:
        raise argparse.ArgumentTypeError("source must be one of: local, kaggle")
    return normalized


def _add_bool(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(
        f"--{name}",
        dest=dest,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def build_parser() -> argparse.ArgumentParser:
    d = LoopConfig()
    p = argparse.ArgumentParser(
        description="Train the production recurrent LoopLLM model from .bin token data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--source", type=_source_type, default="local",
                    help="Runtime source backend. Use `Kaggle` to restore/sync /kaggle/working/loop_llm with Google Drive.")

    data = p.add_argument_group("data")
    data.add_argument("--data", type=Path, default=None,
                      help="One binary corpus; uses --val-fraction for a chronological split.")
    data.add_argument("--train-data", type=Path, default=None,
                      help="Explicit training .bin file.")
    data.add_argument("--val-data", type=Path, default=None,
                      help="Explicit validation .bin file.")
    data.add_argument("--data-dtype", choices=["uint8", "uint16", "uint32"], default="uint16")
    data.add_argument("--val-fraction", type=float, default=0.10,
                      help="Validation fraction when --data is used.")
    _add_bool(
        data,
        "document-aware-sampling",
        False,
        "Sample training windows entirely within EOT-delimited documents. "
        "Default is the unchanged flat random-window sampler.",
    )
    data.add_argument(
        "--eot-token-id",
        type=int,
        default=50256,
        help="Token ID used as the document EOT/EOS marker (the current p50k corpus uses 50256).",
    )
    data.add_argument(
        "--document-index",
        type=Path,
        default=None,
        help=(
            "Path to a cached EOT document-window index (.npz). When omitted and "
            "--document-aware-sampling is enabled, a derived index is created next to the checkpoint output."
        ),
    )

    model = p.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, default=d.vocab_size)
    model.add_argument("--dim", type=int, default=d.dim)
    model.add_argument("--n-layers", type=int, default=d.n_layers)
    model.add_argument("--n-heads", type=int, default=d.n_heads)
    model.add_argument("--head-dim", type=int, default=d.head_dim)
    model.add_argument("--ffn-hidden-dim", type=int, default=d.ffn_hidden_dim)
    model.add_argument("--rope-dim", type=int, default=d.rope_dim)
    model.add_argument("--embed-dim", type=_optional_int, default=d.embed_dim)
    model.add_argument("--embed-dim-out", type=_optional_int, default=d.embed_dim_out)
    model.add_argument("--max-loops", type=int, default=d.max_loops)
    model.add_argument("--min-loops", type=int, default=d.min_loops)

    objective = p.add_argument_group("recurrent objective")
    _add_bool(objective, "loop-sampling", d.loop_sampling, "Sample loop count during training.")
    objective.add_argument("--loop-sampling-lambda", type=_optional_float, default=d.loop_sampling_lambda)
    objective.add_argument("--beta-entropy", type=float, default=d.beta_entropy)
    objective.add_argument("--loop-supervision-weight", type=float, default=d.loop_supervision_weight)
    objective.add_argument("--joint-exit-loss-weight", type=float, default=d.joint_exit_loss_weight)
    objective.add_argument("--loop-monotonic-weight", type=float, default=d.loop_monotonic_weight)
    objective.add_argument("--loop-monotonic-margin", type=float, default=d.loop_monotonic_margin)
    objective.add_argument("--loop-refinement-weight", type=float, default=d.loop_refinement_weight)
    objective.add_argument("--loop-refinement-margin", type=float, default=d.loop_refinement_margin)
    objective.add_argument("--loop-task-weight", type=float, default=d.loop_task_weight)
    objective.add_argument("--loop-task-mode", choices=["none", "horizon"], default=d.loop_task_mode)
    objective.add_argument(
        "--late-loop-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra weight on final-loop next-token CE. The model's normal "
            "recurrent objective remains active; this term directly reinforces "
            "the final recurrent refinement without a second forward pass."
        ),
    )
    objective.add_argument("--shortcut-consistency-weight", type=float, default=d.shortcut_consistency_weight)
    objective.add_argument("--shortcut-consistency-temperature", type=float, default=d.shortcut_consistency_temperature)

    recurrent = p.add_argument_group("recurrent mechanisms")
    _add_bool(recurrent, "residual-loop-embedding", d.residual_loop_embedding, "Inject a learned loop identity embedding.")
    _add_bool(recurrent, "loop-positional-embedding", d.loop_positional_embedding, "Inject learned loop/time positional embeddings.")
    recurrent.add_argument("--loop-positional-max-len", type=int, default=d.loop_positional_max_len)
    recurrent.add_argument("--loop-positional-scale", type=float, default=d.loop_positional_scale)
    _add_bool(recurrent, "causal-register-bank", d.causal_register_bank, "Enable causal register memory across loops.")
    recurrent.add_argument("--causal-registers", type=int, default=d.causal_registers)
    recurrent.add_argument("--causal-register-update-scale", type=float, default=d.causal_register_update_scale)
    _add_bool(recurrent, "loop-transition-matrices", d.loop_transition_matrices, "Enable learned transitions between loops.")
    recurrent.add_argument("--loop-transition-init-scale", type=float, default=d.loop_transition_init_scale)
    _add_bool(recurrent, "loop-controller", d.loop_controller, "Enable the bounded recurrent controller.")
    recurrent.add_argument("--loop-controller-bound", type=float, default=d.loop_controller_bound,
                           help="Absolute elementwise controller bound after tanh.")

    attention = p.add_argument_group("attention")
    attention.add_argument("--sw-window", type=int, default=d.sw_window)
    attention.add_argument("--csa-m", type=int, default=d.csa_m)
    attention.add_argument("--csa-top-k", type=int, default=d.csa_top_k)
    attention.add_argument("--csa-aux-loss-weight", type=float, default=d.csa_aux_loss_weight)
    attention.add_argument("--hca-m-prime", type=int, default=d.hca_m_prime)
    attention.add_argument("--groups", type=int, default=d.groups)
    attention.add_argument("--group-dim", type=int, default=d.group_dim)

    moe = p.add_argument_group("MoE / activation routing")
    moe.add_argument("--moe-num-shared-experts", type=int, default=d.moe_num_shared_experts)
    moe.add_argument("--moe-num-routed-experts", type=int, default=d.moe_num_routed_experts)
    moe.add_argument("--moe-top-k", type=int, default=d.moe_top_k)
    moe.add_argument("--moe-expert-hidden-dim", type=_optional_int, default=d.moe_expert_hidden_dim)
    moe.add_argument("--activation-dictionary", choices=["legacy4", "moa7"], default=d.activation_dictionary)
    moe.add_argument("--activation-gate-mode", choices=["softmax", "sigmoid"], default=d.activation_gate_mode)
    moe.add_argument("--activation-top-k", type=int, default=d.activation_top_k)
    moe.add_argument("--activation-min-probability", type=float, default=d.activation_min_probability)
    moe.add_argument("--activation-balance-weight", type=float, default=d.activation_balance_weight)
    moe.add_argument("--activation-balance-tolerance", type=float, default=d.activation_balance_tolerance)
    moe.add_argument("--activation-bias-update-speed", type=float, default=d.activation_bias_update_speed)
    moe.add_argument("--moe-aux-loss-weight", type=float, default=d.moe_aux_loss_weight)
    moe.add_argument("--moe-bias-update-speed", type=float, default=d.moe_bias_update_speed)
    _add_bool(moe, "expert-local-activation-routing", d.expert_local_activation_routing,
              "Enable expert-local activation routers. Usually keep disabled for MoA7-Lean.")

    performance = p.add_argument_group("memory / performance")
    _add_bool(performance, "tie-embeddings", d.tie_embeddings, "Tie LM head weights to token embeddings.")
    performance.add_argument("--dropout", type=float, default=d.dropout)
    _add_bool(performance, "grad-checkpointing", d.grad_checkpointing, "Use activation checkpointing inside recurrent blocks.")
    performance.add_argument("--exit-gate-loop-embed-dim", type=int, default=d.exit_gate_loop_embed_dim)

    train = p.add_argument_group("training")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--seq-len", type=int, default=32)
    train.add_argument("--max-steps", type=int, default=800)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument(
        "--force-lr",
        type=float,
        default=None,
        help=(
            "Override the checkpoint LR and restart the cosine schedule from this "
            "LR at the resume step. The forced schedule has no warmup and decays "
            "to min_lr_ratio * LR by max_steps."
        ),
    )
    train.add_argument("--min-lr-ratio", type=float, default=0.10)
    train.add_argument("--beta1", type=float, default=0.9)
    train.add_argument("--beta2", type=float, default=0.95)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--grad-accum-steps", type=int, default=1)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--eval-interval", type=int, default=50)
    train.add_argument("--eval-iters", type=int, default=32)
    train.add_argument("--eval-seed", type=int, default=2027)
    train.add_argument("--checkpoint-interval", type=int, default=100)
    train.add_argument(
        "--drive-save-interval",
        type=int,
        default=0,
        help=(
            "When --source Kaggle is active, update the single remote latest.pt and the "
            "matching visualizations/step_XXXXXXXX.png every N optimizer steps. The "
            "Drive-save boundary also writes the full local resumable latest.pt so the "
            "remote weight is always resumable. Network upload runs in a separate OS "
            "process. No remote per-step checkpoint copies are created. 0 disables."
        ),
    )
    train.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "run")
    train.add_argument("--resume", type=Path, default=None,
                       help="Resume exactly from this canonical checkpoint.")
    train.add_argument("--log-interval", type=int, default=10)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--device", default=None,
                       help="cuda, cpu, or an explicit torch device. Defaults to CUDA when available.")
    train.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    train.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    _add_bool(train, "tf32", True, "Allow TF32 matrix math on CUDA when supported.")
    _add_bool(train, "amp", False, "Use autocast for the selected dtype on CUDA/CPU where supported.")

    viz = p.add_argument_group("visualization")
    viz.add_argument("--viz-dir", type=Path, default=ROOT / "visualizations" / "run")
    viz.add_argument("--viz-interval", type=int, default=100,
                     help="Save one mechanistic PNG/JSON snapshot every N optimizer steps; 0 disables snapshots.")
    viz.add_argument("--viz-probe-size", type=int, default=2)

    meta = p.add_argument_group("run metadata")
    meta.add_argument("--run-name", default=None)
    meta.add_argument("--notes", default="")

    return p


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse arguments and remember which CLI options were explicitly supplied.

    Resume loading can then restore omitted options from the checkpoint while
    honoring explicit runtime overrides such as ``--batch-size 4``.
    """
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    actions_by_option: dict[str, argparse.Action] = {}
    for action in parser._actions:
        for option in action.option_strings:
            actions_by_option[option] = action

    explicit: set[str] = set()
    for token in raw_argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option.startswith("--no-"):
            option = "--" + option[5:]
        action = actions_by_option.get(option)
        if action is not None:
            explicit.add(action.dest)

    args = parser.parse_args(raw_argv)
    setattr(args, "_explicit_cli_overrides", frozenset(explicit))
    return args


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
}


def open_bin(path: Path, dtype_name: str) -> np.memmap:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Binary corpus not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Binary corpus is empty: {path}")
    itemsize = np.dtype(DTYPE_MAP[dtype_name]).itemsize
    if path.stat().st_size % itemsize != 0:
        raise ValueError(
            f"{path} is {path.stat().st_size} bytes, not divisible by {itemsize}-byte {dtype_name} items."
        )
    return np.memmap(path, dtype=DTYPE_MAP[dtype_name], mode="r")


def validate_vocab_range(arr: np.memmap, vocab_size: int, name: str) -> None:
    # Full-array min/max is sequential and cheap for a memmap; fail early before
    # constructing a GPU model if a tokenizer/vocab mismatch is present.
    mn = int(arr.min())
    mx = int(arr.max())
    if mn < 0 or mx >= vocab_size:
        raise ValueError(
            f"{name} contains token ids outside [0, {vocab_size - 1}]: min={mn}, max={mx}."
        )


def prepare_datasets(args: argparse.Namespace) -> tuple[np.memmap, np.memmap]:
    if args.data is not None and (args.train_data is not None or args.val_data is not None):
        raise ValueError("Use either --data or --train-data/--val-data, not both.")

    if args.data is not None:
        all_data = open_bin(args.data, args.data_dtype)
        if not 0.0 < args.val_fraction < 1.0:
            raise ValueError("--val-fraction must be strictly between 0 and 1.")
        split = int(len(all_data) * (1.0 - args.val_fraction))
        if split <= args.seq_len or len(all_data) - split <= args.seq_len:
            raise ValueError("Chronological train/validation split leaves too few tokens for --seq-len.")
        train_data = all_data[:split]
        val_data = all_data[split:]
    else:
        if args.train_data is None or args.val_data is None:
            raise ValueError("Provide --data, or both --train-data and --val-data.")
        train_data = open_bin(args.train_data, args.data_dtype)
        val_data = open_bin(args.val_data, args.data_dtype)

    if len(train_data) <= args.seq_len:
        raise ValueError(f"Training corpus length {len(train_data)} must exceed seq_len={args.seq_len}.")
    if len(val_data) <= args.seq_len:
        raise ValueError(f"Validation corpus length {len(val_data)} must exceed seq_len={args.seq_len}.")

    validate_vocab_range(train_data, args.vocab_size, "train corpus")
    validate_vocab_range(val_data, args.vocab_size, "validation corpus")
    return train_data, val_data


@dataclasses.dataclass(frozen=True)
class DocumentWindowIndex:
    """Compact EOT-delimited document index for uniform window sampling.

    ``starts[i]`` is the first token of segment i (the token immediately after
    the previous EOT, except segment 0), while ``cumulative[i]`` is the
    cumulative number of valid seq_len+1 starts through segment i.
    ``required_tokens`` records the indexed window size for cache validation.
    """

    starts: np.ndarray
    cumulative: np.ndarray
    required_tokens: int

    @property
    def total_windows(self) -> int:
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def sample_starts(self, batch_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.total_windows < 1:
            raise ValueError("Document index contains no valid windows.")
        rng = rng or np.random.default_rng()
        draws = rng.integers(0, self.total_windows, size=batch_size, dtype=np.int64)
        seg_idx = np.searchsorted(self.cumulative, draws, side="right")
        previous = np.zeros_like(draws)
        nonzero = seg_idx > 0
        previous[nonzero] = self.cumulative[seg_idx[nonzero] - 1]
        return self.starts[seg_idx] + (draws - previous)


def _default_document_index_path(data_path: Path, seq_len: int, eot_token_id: int) -> Path:
    return data_path.with_name(
        f"{data_path.name}.docidx.seq{int(seq_len)}.eot{int(eot_token_id)}.npz"
    )


def _save_document_index(path: Path, index: DocumentWindowIndex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        starts=index.starts.astype(np.int64, copy=False),
        cumulative=index.cumulative.astype(np.int64, copy=False),
        required_tokens=np.asarray(index.required_tokens, dtype=np.int64),
    )
    tmp.replace(path)


def _load_document_index(path: Path, *, required_tokens: int, data_len: int) -> DocumentWindowIndex | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            starts = np.asarray(z["starts"], dtype=np.int64)
            cumulative = np.asarray(z["cumulative"], dtype=np.int64)
            saved_required = int(z["required_tokens"])
    except Exception:
        return None
    if saved_required != required_tokens:
        return None
    if starts.ndim != 1 or cumulative.ndim != 1 or starts.shape != cumulative.shape:
        return None
    if len(starts) == 0 or int(starts[0]) != 0:
        return None
    if np.any(starts < 0) or np.any(starts >= data_len):
        return None
    if np.any(np.diff(starts) <= 0):
        return None
    if np.any(np.diff(cumulative) <= 0) or int(cumulative[-1]) <= 0:
        return None
    if int(cumulative[-1]) > data_len:
        return None
    return DocumentWindowIndex(starts=starts, cumulative=cumulative, required_tokens=saved_required)


def build_document_window_index(
    data: np.memmap,
    seq_len: int,
    eot_token_id: int,
    *,
    cache_path: Path | None = None,
    chunk_tokens: int = 8_000_000,
) -> DocumentWindowIndex:
    """Build/cache an EOT-safe index without loading the corpus into RAM.

    A sampled window has ``seq_len+1`` tokens and contains no EOT token.
    This matches the existing EOT-safe experiment index: EOT-delimited
    documents shorter than the required window are skipped, and windows whose
    final token would be EOT are excluded. The index stores only one start and
    one cumulative count per EOT-delimited document, not one entry per window.
    """
    required_tokens = int(seq_len) + 1
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if eot_token_id < 0:
        raise ValueError("eot-token-id must be >= 0")
    if cache_path is not None:
        cached = _load_document_index(
            cache_path, required_tokens=required_tokens, data_len=len(data)
        )
        if cached is not None:
            return cached

    starts: list[int] = [0]
    for offset in range(0, len(data), chunk_tokens):
        stop = min(len(data), offset + chunk_tokens)
        chunk = np.asarray(data[offset:stop])
        eots = np.flatnonzero(chunk == eot_token_id)
        for rel in eots:
            next_start = offset + int(rel) + 1
            if next_start < len(data):
                starts.append(next_start)

    # Remove duplicates defensively in case a chunk boundary ever produces one.
    all_starts = np.asarray(starts, dtype=np.int64)
    if len(all_starts) > 1:
        all_starts = np.unique(all_starts)

    ends = np.empty_like(all_starts)
    if len(all_starts) > 1:
        ends[:-1] = all_starts[1:] - 1
    ends[-1] = len(data) - 1
    # Compute each segment's valid-window count before filtering short segments.
    # Otherwise an eligible segment immediately before a skipped short segment
    # would incorrectly absorb that skipped region into its count.
    counts = ends - all_starts + 1 - required_tokens
    eligible = counts >= 1
    starts_array = all_starts[eligible]
    counts = counts[eligible]
    if len(starts_array) == 0:
        raise ValueError(
            f"No EOT-delimited document contains at least seq_len+1={required_tokens} tokens. "
            f"Check --eot-token-id={eot_token_id} and the corpus."
        )

    cumulative = np.cumsum(counts.astype(np.int64, copy=False), dtype=np.int64)
    index = DocumentWindowIndex(
        starts=starts_array,
        cumulative=cumulative,
        required_tokens=required_tokens,
    )
    if cache_path is not None:
        _save_document_index(cache_path, index)
    return index


def make_document_aware_batch(
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    index: DocumentWindowIndex,
) -> torch.Tensor:
    """Uniformly sample valid starts from all EOT-safe windows."""
    if index.required_tokens != seq_len + 1:
        raise ValueError("Document index does not match the active seq_len.")
    starts = index.sample_starts(batch_size)
    window = seq_len + 1
    batch = np.stack(
        [np.asarray(data[int(s): int(s) + window], dtype=np.int64) for s in starts]
    )
    return torch.from_numpy(batch).to(device=device, dtype=torch.long)


def make_batch(
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    max_start = len(data) - (seq_len + 1)
    if max_start < 1:
        raise ValueError(f"Corpus too short for seq_len={seq_len}.")
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    batch = np.stack([np.asarray(data[int(s): int(s) + seq_len + 1], dtype=np.int64) for s in starts])
    return torch.from_numpy(batch).to(device=device, dtype=torch.long)


def make_fixed_eval_batches(
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    eval_iters: int,
    seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    max_start = len(data) - (seq_len + 1)
    if max_start < 1:
        raise ValueError(f"Validation corpus too short for seq_len={seq_len}.")
    rng = np.random.default_rng(seed)
    batches: list[torch.Tensor] = []
    for _ in range(eval_iters):
        starts = rng.integers(0, max_start + 1, size=batch_size)
        batch = np.stack([np.asarray(data[int(s): int(s) + seq_len + 1], dtype=np.int64) for s in starts])
        batches.append(torch.from_numpy(batch).to(device=device, dtype=torch.long))
    return batches


# ---------------------------------------------------------------------------
# Model / optimizer
# ---------------------------------------------------------------------------


def config_from_args(a: argparse.Namespace) -> LoopConfig:
    return LoopConfig(
        vocab_size=a.vocab_size,
        dim=a.dim,
        n_layers=a.n_layers,
        n_heads=a.n_heads,
        head_dim=a.head_dim,
        ffn_hidden_dim=a.ffn_hidden_dim,
        rope_dim=a.rope_dim,
        embed_dim=a.embed_dim,
        embed_dim_out=a.embed_dim_out,
        max_loops=a.max_loops,
        min_loops=a.min_loops,
        loop_sampling=a.loop_sampling,
        loop_sampling_lambda=a.loop_sampling_lambda,
        beta_entropy=a.beta_entropy,
        loop_supervision_weight=a.loop_supervision_weight,
        joint_exit_loss_weight=a.joint_exit_loss_weight,
        loop_monotonic_weight=a.loop_monotonic_weight,
        loop_monotonic_margin=a.loop_monotonic_margin,
        loop_refinement_weight=a.loop_refinement_weight,
        loop_refinement_margin=a.loop_refinement_margin,
        loop_task_weight=a.loop_task_weight,
        loop_task_mode=a.loop_task_mode,
        shortcut_consistency_weight=a.shortcut_consistency_weight,
        shortcut_consistency_temperature=a.shortcut_consistency_temperature,
        residual_loop_embedding=a.residual_loop_embedding,
        loop_positional_embedding=a.loop_positional_embedding,
        loop_positional_max_len=a.loop_positional_max_len,
        loop_positional_scale=a.loop_positional_scale,
        causal_register_bank=a.causal_register_bank,
        causal_registers=a.causal_registers,
        causal_register_update_scale=a.causal_register_update_scale,
        loop_transition_matrices=a.loop_transition_matrices,
        loop_transition_init_scale=a.loop_transition_init_scale,
        loop_controller=a.loop_controller,
        loop_controller_bound=a.loop_controller_bound,
        sw_window=a.sw_window,
        csa_m=a.csa_m,
        csa_top_k=a.csa_top_k,
        csa_aux_loss_weight=a.csa_aux_loss_weight,
        hca_m_prime=a.hca_m_prime,
        groups=a.groups,
        group_dim=a.group_dim,
        moe_num_shared_experts=a.moe_num_shared_experts,
        moe_num_routed_experts=a.moe_num_routed_experts,
        moe_top_k=a.moe_top_k,
        moe_expert_hidden_dim=a.moe_expert_hidden_dim,
        activation_dictionary=a.activation_dictionary,
        activation_gate_mode=a.activation_gate_mode,
        activation_top_k=a.activation_top_k,
        activation_min_probability=a.activation_min_probability,
        activation_balance_weight=a.activation_balance_weight,
        activation_balance_tolerance=a.activation_balance_tolerance,
        activation_bias_update_speed=a.activation_bias_update_speed,
        moe_aux_loss_weight=a.moe_aux_loss_weight,
        moe_bias_update_speed=a.moe_bias_update_speed,
        expert_local_activation_routing=a.expert_local_activation_routing,
        tie_embeddings=a.tie_embeddings,
        dropout=a.dropout,
        grad_checkpointing=a.grad_checkpointing,
        exit_gate_loop_embed_dim=a.exit_gate_loop_embed_dim,
    )


def make_optimizer(model: LoopTransformer, args: argparse.Namespace) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    no_decay_markers = (
        "loop_scale", "loop_bias", "loop_positional", "loop_transition",
        "causal_register", "loop_embedding", "norm", "tok_emb",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or any(marker in name.lower() for marker in no_decay_markers):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )


def scheduled_lr(
    step: int,
    args: argparse.Namespace,
    *,
    force_start_step: Optional[int] = None,
    force_peak_lr: Optional[float] = None,
) -> float:
    """Return the LR for an optimizer step.

    Normal training uses the original warmup + cosine schedule anchored at
    step 0. When ``force_start_step``/``force_peak_lr`` are supplied, a fresh
    cosine schedule is started at the first step after ``force_start_step``:
    the first resumed optimizer step uses exactly ``force_peak_lr`` and the
    schedule decays to ``min_lr_ratio * force_peak_lr`` at ``max_steps``.
    """
    if force_start_step is not None and force_peak_lr is not None:
        remaining = args.max_steps - force_start_step
        if remaining <= 1:
            return float(force_peak_lr)

        # The first optimizer step after the resume point is progress=0, so the
        # forced LR is applied exactly once before the cosine begins to decay.
        progress = (step - force_start_step - 1) / (remaining - 1)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(
            force_peak_lr
            * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)
        )

    if args.warmup_steps > 0 and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    if args.max_steps <= args.warmup_steps:
        return args.lr
    progress = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_checkpoint_extra(
    args: argparse.Namespace,
    *,
    run_name: str,
    best_val: float,
    last_eval: dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer],
    lr: float,
    forced_lr_start_step: Optional[int],
    forced_lr_peak: Optional[float],
    scaler: torch.amp.GradScaler,
    interrupted: bool = False,
) -> dict[str, Any]:
    """Construct checkpoint metadata with durable LR continuation state."""
    effective_lr = _checkpoint_lr(optimizer, lr)
    if forced_lr_start_step is not None and forced_lr_peak is not None:
        lr_state = {
            "mode": "forced_restart",
            "start_step": int(forced_lr_start_step),
            "peak_lr": float(forced_lr_peak),
            "current_lr": float(effective_lr),
            "min_lr_ratio": float(args.min_lr_ratio),
            "max_steps": int(args.max_steps),
        }
    else:
        lr_state = {
            "mode": "normal",
            "current_lr": float(effective_lr),
            "base_lr": float(args.lr),
            "min_lr_ratio": float(args.min_lr_ratio),
            "max_steps": int(args.max_steps),
        }

    return {
        "run_name": run_name,
        "notes": args.notes,
        "interrupted": bool(interrupted),
        "best_val_ce": best_val,
        "last_eval": last_eval,
        "lr_at_checkpoint": float(effective_lr),
        "lr_state": lr_state,
        # Keep this for backward compatibility with the previous package.
        "lr_schedule": (
            {
                "mode": "forced_restart",
                "start_step": int(forced_lr_start_step),
                "peak_lr": float(forced_lr_peak),
            }
            if forced_lr_start_step is not None and forced_lr_peak is not None
            else {"mode": "normal"}
        ),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else {},
        "args": _args_to_checkpoint_dict(args),
    }


# ---------------------------------------------------------------------------
# Evaluation / checkpoint bookkeeping
# ---------------------------------------------------------------------------


def evaluate(model: LoopTransformer, batches: list[torch.Tensor]) -> dict[str, Any]:
    model.eval()
    total_obj = 0.0
    total_ce = 0.0
    total_correct = 0
    total_tokens = 0
    loop_ce_sum: Optional[np.ndarray] = None
    loop_correct: Optional[np.ndarray] = None
    with torch.no_grad():
        for batch in batches:
            objective, step_losses = model.compute_loss(batch, max_loops=model.cfg.max_loops)
            logits_list, _ = model(batch, max_loops=model.cfg.max_loops)
            targets = batch[:, 1:]
            loop_ce = []
            loop_acc = []
            for logits in logits_list:
                flat_logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
                flat_targets = targets.reshape(-1)
                loop_ce.append(float(F.cross_entropy(flat_logits, flat_targets).item()))
                loop_acc.append(int((logits[:, :-1].argmax(-1) == targets).sum().item()))
            final_ce = loop_ce[-1]
            final_correct = loop_acc[-1]
            n_tokens = int(targets.numel())
            total_obj += float(objective.item())
            total_ce += final_ce
            total_correct += final_correct
            total_tokens += n_tokens
            ce_arr = np.asarray(loop_ce, dtype=np.float64)
            corr_arr = np.asarray(loop_acc, dtype=np.int64)
            loop_ce_sum = ce_arr if loop_ce_sum is None else loop_ce_sum + ce_arr
            loop_correct = corr_arr if loop_correct is None else loop_correct + corr_arr

    denom = max(1, len(batches))
    loop_ce_mean = (loop_ce_sum / denom).tolist() if loop_ce_sum is not None else []
    loop_acc = (100.0 * loop_correct / max(1, total_tokens)).tolist() if loop_correct is not None else []
    return {
        "val_objective": total_obj / denom,
        "val_ce": total_ce / denom,
        "val_accuracy": total_correct / max(1, total_tokens),
        "val_accuracy_pct": 100.0 * total_correct / max(1, total_tokens),
        "val_loop_ce": loop_ce_mean,
        "val_loop_accuracy_pct": loop_acc,
    }


def _args_to_checkpoint_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize public CLI arguments while excluding internal parser metadata."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }


def save_run_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    cfg: LoopConfig,
    train_data: np.memmap,
    val_data: np.memmap,
    model: LoopTransformer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": dataclasses.asdict(cfg),
        "training": _args_to_checkpoint_dict(args),
        "dataset": {
            "train_tokens": int(len(train_data)),
            "val_tokens": int(len(val_data)),
            "train_bytes": int(len(train_data) * train_data.dtype.itemsize),
            "val_bytes": int(len(val_data) * val_data.dtype.itemsize),
        },
        "parameters": int(model.num_parameters()),
        "effective_parameters": int(model.num_parameters(effective=True)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")



# ---------------------------------------------------------------------------
# Kaggle execution / Google Drive persistence
# ---------------------------------------------------------------------------

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
KAGGLE_STATE_ROOT = KAGGLE_WORKING_ROOT / "loop_llm"


def _find_unique_kaggle_input_file(filename: str) -> Path:
    """Find one exact-named input file anywhere under /kaggle/input."""
    if not KAGGLE_INPUT_ROOT.exists():
        raise RuntimeError("--source Kaggle was requested, but /kaggle/input does not exist.")
    matches = sorted(
        p for p in KAGGLE_INPUT_ROOT.rglob("*")
        if p.is_file() and p.name == filename
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {KAGGLE_INPUT_ROOT}. "
            "Attach the Kaggle Dataset containing the training binaries."
        )
    if len(matches) > 1:
        rendered = "\n".join(f"  {p}" for p in matches)
        raise RuntimeError(f"Found multiple {filename!r} files under /kaggle/input:\n{rendered}")
    return matches[0].resolve()


def _remap_kaggle_path(value: Optional[Path]) -> Optional[Path]:
    """Translate legacy Colab Drive paths to /kaggle/working/loop_llm."""
    if value is None:
        return None
    path = Path(value)
    raw = path.as_posix()
    colab_root = "/content/drive/MyDrive/loop_llm"
    kaggle_root = KAGGLE_STATE_ROOT.as_posix()
    prefix = colab_root + "/"
    if raw == colab_root:
        return KAGGLE_STATE_ROOT
    if raw.startswith(prefix):
        return Path(kaggle_root + "/" + raw[len(prefix):])
    try:
        if path.resolve() == (ROOT / "checkpoints" / "run").resolve():
            return KAGGLE_STATE_ROOT / "checkpoints" / "run"
        if path.resolve() == (ROOT / "visualizations" / "run").resolve():
            return KAGGLE_STATE_ROOT / "visualizations" / "run"
    except OSError:
        pass
    return path


def _configure_kaggle_paths(args: argparse.Namespace, *, validate_resume: bool = True) -> None:
    """Resolve Kaggle inputs and remap checkpoint/viz/resume paths."""
    if args.source != "kaggle":
        return
    train_input = _find_unique_kaggle_input_file("train.bin")
    val_input = _find_unique_kaggle_input_file("train.val.bin")
    for name in ("data", "train_data", "val_data", "checkpoint_dir", "viz_dir", "resume"):
        setattr(args, name, _remap_kaggle_path(getattr(args, name)))
    if args.data is not None and not args.data.exists():
        args.data = None
    if args.train_data is None or not args.train_data.exists():
        args.train_data = train_input
    if args.val_data is None or not args.val_data.exists():
        args.val_data = val_input
    if args.checkpoint_dir is None:
        args.checkpoint_dir = KAGGLE_STATE_ROOT / "checkpoints" / "run"
    if args.viz_dir is None:
        args.viz_dir = KAGGLE_STATE_ROOT / "visualizations" / "run"
    if validate_resume and args.resume is not None and not args.resume.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found after Kaggle path remapping: {args.resume}"
        )


def _kaggle_startup_status_path() -> Path:
    return Path("/kaggle/working/.loopllm_kaggle_startup_status.json")


def _write_kaggle_startup_status(path: Path, *, ok: bool, error: str = "") -> None:
    payload = {"ok": bool(ok), "error": str(error) if error else ""}
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _wait_for_kaggle_startup_status(path: Path, timeout_seconds: float = 900.0) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "ok" in payload:
                return payload
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError(
        f"Timed out waiting {timeout_seconds:.0f}s for rank 0 Kaggle Drive initialization "
        f"status at {path}."
    )


def _start_kaggle_drive_sync() -> tuple[Any, Any]:
    """Restore Drive state and start a completely out-of-band uploader process."""
    try:
        from drive_sync import DriveSyncError, DriveSyncWorker, KaggleDriveSync
    except ImportError as exc:
        raise RuntimeError("The Kaggle-enabled source package is missing drive_sync.py.") from exc
    try:
        syncer = KaggleDriveSync(
            KAGGLE_STATE_ROOT,
            root_folder_id=os.environ.get("LOOPLLM_GDRIVE_FOLDER_ID"),
            secret_name=os.environ.get("LOOPLLM_GDRIVE_SECRET_NAME", "GDRIVE_CREDENTIALS_JSON"),
        )
        syncer.restore()
        worker = DriveSyncWorker(syncer)
        worker.start()
        return syncer, worker
    except DriveSyncError as exc:
        raise RuntimeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------



RESUME_OUTPUT_OVERRIDES = {
    "source",
    "resume",
    "checkpoint_dir",
    "viz_dir",
    "run_name",
    "notes",
    "device",
    "max_steps",
    "force_lr",
    "drive_save_interval",
    # This is deliberately user-overridable so an existing checkpoint can be
    # forked into a late-loop-supervision ablation without changing the model.
    "late_loop_loss_weight",
}


def _restore_args_from_checkpoint(
    args: argparse.Namespace,
    resume_ckpt: dict[str, Any],
) -> None:
    """Restore the exact training/data configuration stored in a checkpoint.

    A resume must continue the same optimization trajectory. In particular,
    the AdamW hyperparameters and LR schedule must come from the checkpoint,
    not from fresh CLI defaults. Output/runtime options remain user-overridable.
    """
    saved = dict(resume_ckpt.get("extra", {}).get("args", {}))
    if not saved:
        return

    path_fields = {
        "data",
        "train_data",
        "val_data",
        "checkpoint_dir",
        "viz_dir",
        "resume",
        "document_index",
    }

    explicit = set(getattr(args, "_explicit_cli_overrides", ()))
    for key, value in saved.items():
        if key in RESUME_OUTPUT_OVERRIDES or key in explicit or not hasattr(args, key):
            continue
        if key in path_fields and value is not None:
            value = Path(value)
        setattr(args, key, value)


def _checkpoint_lr(optimizer: Optional[torch.optim.Optimizer], fallback: float) -> float:
    """Read the optimizer LR after state restoration."""
    if optimizer is not None and optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", fallback))
    return float(fallback)


def choose_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _runtime_device(requested: Optional[str], distributed: bool, local_rank: int) -> torch.device:
    """Resolve the actual process-local device, honoring torchrun's local rank."""
    device = choose_device(requested)
    if distributed:
        if device.type != "cuda":
            raise RuntimeError("Distributed multi-GPU training requires --device cuda.")
        return torch.device("cuda", local_rank)
    return device


def configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)


def autocast_context(device: torch.device, args: argparse.Namespace):
    if not args.amp or args.dtype == "float32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Distributed training
# ---------------------------------------------------------------------------


def _dist_world_size() -> int:
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError as exc:
        raise RuntimeError("WORLD_SIZE must be an integer when provided by torchrun.") from exc


def _init_distributed() -> tuple[bool, int, int, int]:
    """Initialize single-node NCCL DDP when launched by torchrun.

    Returns ``(enabled, rank, world_size, local_rank)``. The public
    ``--batch-size`` remains the global optimizer-step batch size; each rank
    receives its proportional local share.
    """
    world_size = _dist_world_size()
    if world_size <= 1:
        return False, 0, 1, 0

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU distributed training requires CUDA.")

    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(
            "WORLD_SIZE>1 but torchrun rank environment is missing. "
            "Launch with: torchrun --standalone --nproc-per-node=2 train.py ..."
        ) from exc

    gpu_count = torch.cuda.device_count()
    if local_rank < 0 or local_rank >= gpu_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {gpu_count} visible CUDA devices."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device("cuda", local_rank),
    )
    return True, rank, world_size, local_rank


def _dist_barrier(enabled: bool) -> None:
    if enabled:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _dist_shutdown(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _local_batch_size(global_batch: int, rank: int, world_size: int) -> int:
    """Split a global batch across ranks, allowing a final uneven partition."""
    if global_batch < world_size:
        raise ValueError(
            f"Global batch size {global_batch} is smaller than world size {world_size}. "
            "Use a batch size >= the number of GPUs."
        )
    base, remainder = divmod(global_batch, world_size)
    return base + (1 if rank < remainder else 0)


class _DistributedLossWrapper(nn.Module):
    """Expose LoopTransformer.compute_loss through DDP's forward hook."""

    def __init__(self, model: LoopTransformer) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        batch: torch.Tensor,
        *,
        max_loops: int,
        late_loop_weight: float,
    ) -> tuple[torch.Tensor, list[float]]:
        return self.model.compute_loss(
            batch,
            max_loops=max_loops,
            late_loop_weight=late_loop_weight,
        )


def train(args: argparse.Namespace) -> None:
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.batch_size < 1 or args.seq_len < 1 or args.grad_accum_steps < 1:
        raise ValueError("batch-size, seq-len, and grad-accum-steps must all be >= 1")
    if args.eval_interval < 1 or args.eval_iters < 1:
        raise ValueError("eval-interval and eval-iters must be >= 1")
    if args.checkpoint_interval < 1 or args.log_interval < 1:
        raise ValueError("checkpoint-interval and log-interval must be >= 1")
    if args.viz_interval < 0:
        raise ValueError("viz-interval must be >= 0")
    if args.drive_save_interval < 0:
        raise ValueError("drive-save-interval must be >= 0")

    # torchrun initializes DDP before any CUDA model allocation.
    distributed, rank, world_size, local_rank = _init_distributed()
    is_main = rank == 0

    # --source Kaggle is the only switch needed for the Kaggle runtime.
    # Only rank 0 touches Google Drive; all ranks share the same /kaggle/working
    # filesystem, so restore once and synchronize before any rank loads --resume.
    kaggle_syncer = None
    kaggle_worker = None
    if args.source == "kaggle":
        startup_status = _kaggle_startup_status_path()
        startup_error: Optional[str] = None
        if is_main:
            startup_status.unlink(missing_ok=True)
            try:
                kaggle_syncer, kaggle_worker = _start_kaggle_drive_sync()
                _write_kaggle_startup_status(startup_status, ok=True)
            except Exception as exc:
                startup_error = str(exc)
                _write_kaggle_startup_status(
                    startup_status,
                    ok=False,
                    error=startup_error,
                )

        # Do not use NCCL while rank 0 is doing Drive network I/O. Rank 1 waits
        # on a shared filesystem status file instead, then both ranks enter one
        # normal barrier after the startup decision is known. This prevents a
        # slow/failing Drive restore from turning into an NCCL timeout or a
        # stranded rank.
        status = _wait_for_kaggle_startup_status(startup_status)
        if not bool(status.get("ok", False)):
            _dist_shutdown(distributed)
            detail = str(status.get("error", "unknown Kaggle Drive initialization error"))
            if is_main and startup_error:
                raise RuntimeError(detail)
            raise RuntimeError(
                "Kaggle Drive initialization failed on rank 0. "
                f"Rank 0 reported: {detail}"
            )

        _dist_barrier(distributed)
        _configure_kaggle_paths(args)

    if not distributed and is_main and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(
            f"[Multi-GPU] {torch.cuda.device_count()} CUDA GPUs are visible, but this process was not "
            "launched with torchrun; only GPU 0 will be used. For both GPUs, launch "
            "with: torchrun --standalone --nproc-per-node=2 train.py ...",
            flush=True,
        )

    # DDP needs independent stochastic streams on each GPU. For a distributed
    # resume we intentionally do not restore the single-rank RNG snapshot from
    # the old checkpoint, because that would make all ranks sample identically.
    seed_everything(args.seed + rank if distributed else args.seed)
    device = _runtime_device(args.device, distributed, local_rank)
    configure_runtime(args, device)

    resume_ckpt = None
    forced_lr_start_step: Optional[int] = None
    forced_lr_peak: Optional[float] = None
    last_checkpoint_step = 0

    if args.resume is not None:
        # Load the checkpoint before constructing datasets so that seq_len, data
        # paths, batch size, and all other exact training settings are restored
        # from the saved run rather than silently falling back to CLI defaults.
        resume_ckpt = load_checkpoint(
            args.resume,
            device=device,
            optimizer=None,
            restore_rng=not distributed,
        )
        _restore_args_from_checkpoint(args, resume_ckpt)
        # A checkpoint may contain absolute Colab paths from the original run.
        # Normalize those paths again after restoring checkpoint metadata.
        if args.source == "kaggle":
            _configure_kaggle_paths(args)

        # max_steps is the one training option deliberately left overridable so
        # a user can extend a stopped run. With the parser default untouched,
        # retain the original checkpoint horizon.
        saved_args = dict(resume_ckpt.get("extra", {}).get("args", {}))
        parser_default_max_steps = 800
        if args.max_steps == parser_default_max_steps and saved_args.get("max_steps") is not None:
            args.max_steps = int(saved_args["max_steps"])

        # If the output directory was not intentionally changed, continue writing
        # checkpoints/metrics beside the resumed checkpoint.
        parser_default_ckpt_dir = ROOT / "checkpoints" / "run"
        if args.checkpoint_dir == parser_default_ckpt_dir:
            args.checkpoint_dir = Path(args.resume).resolve().parent
        parser_default_viz_dir = ROOT / "visualizations" / "run"
        if args.viz_dir == parser_default_viz_dir:
            args.viz_dir = args.checkpoint_dir / "visualizations"

        # Reconfigure runtime in case dtype/matmul/AMP settings were restored.
        device = _runtime_device(args.device, distributed, local_rank)
        configure_runtime(args, device)

        model = resume_ckpt["model"].to(device)
        cfg = model.cfg
        start_step = int(resume_ckpt.get("step", 0))
        last_checkpoint_step = start_step
        if start_step >= args.max_steps:
            raise ValueError(f"Checkpoint is already at step {start_step}, max_steps={args.max_steps}.")

        # A forced LR explicitly starts a new cosine schedule at the resume
        # point. Otherwise, preserve an already-active forced schedule from a
        # previous checkpoint so a plain resume can never jump back to the old
        # base-LR trajectory. ``lr_state`` is canonical; ``lr_schedule`` is
        # retained as a compatibility reader for checkpoints from the previous
        # build.
        resume_extra = dict(resume_ckpt.get("extra", {}))
        saved_lr_state = dict(resume_extra.get("lr_state", {}))
        saved_lr_schedule = dict(resume_extra.get("lr_schedule", {}))
        if args.force_lr is not None:
            if args.force_lr <= 0.0:
                raise ValueError("--force-lr must be > 0")
            forced_lr_start_step = start_step
            forced_lr_peak = float(args.force_lr)
            args.lr = forced_lr_peak
        else:
            schedule_state = saved_lr_state or saved_lr_schedule
            if schedule_state.get("mode") == "forced_restart":
                if schedule_state.get("start_step") is None or schedule_state.get("peak_lr") is None:
                    raise RuntimeError(
                        "Checkpoint contains an incomplete forced-LR schedule; "
                        "refusing to reconstruct the LR from CLI defaults."
                    )
                forced_lr_start_step = int(schedule_state["start_step"])
                forced_lr_peak = float(schedule_state["peak_lr"])
                if forced_lr_peak <= 0.0:
                    raise RuntimeError(
                        f"Checkpoint contains invalid forced peak LR: {forced_lr_peak!r}"
                    )
                args.lr = forced_lr_peak

        print(f"Resumed model config from checkpoint at step {start_step}.", flush=True)
        saved_batch_size = saved_args.get("batch_size")
        if saved_batch_size is not None and int(saved_batch_size) != int(args.batch_size):
            print(
                f"Resume override: batch_size checkpoint={int(saved_batch_size)} -> CLI={int(args.batch_size)}.",
                flush=True,
            )
        if forced_lr_start_step is not None and forced_lr_peak is not None:
            print(
                f"Forced LR schedule: peak={forced_lr_peak:.17g} "
                f"restarted at step {forced_lr_start_step}; no warmup.",
                flush=True,
            )
    else:
        cfg = config_from_args(args)
        model = LoopTransformer(cfg).to(device)
        start_step = 0
        last_checkpoint_step = 0

        # ``--force-lr`` is also valid on a fresh run: it simply selects the
        # cosine peak without altering the model architecture. A resumed run is
        # where this is primarily useful.
        if args.force_lr is not None:
            if args.force_lr <= 0.0:
                raise ValueError("--force-lr must be > 0")
            forced_lr_start_step = 0
            forced_lr_peak = float(args.force_lr)
            args.lr = forced_lr_peak

    train_data, val_data = prepare_datasets(args)

    document_index: DocumentWindowIndex | None = None
    if args.document_aware_sampling:
        if args.eot_token_id >= args.vocab_size:
            raise ValueError(
                f"--eot-token-id={args.eot_token_id} must be < vocab_size={args.vocab_size}."
            )
        index_path = args.document_index
        if index_path is None:
            train_path = args.train_data if args.train_data is not None else args.data
            if train_path is None:
                raise ValueError("Cannot derive a document-index path without an explicit training data path.")
            index_path = _default_document_index_path(Path(train_path), args.seq_len, args.eot_token_id)
        document_index = build_document_window_index(
            train_data,
            args.seq_len,
            args.eot_token_id,
            cache_path=index_path,
        )
        if is_main:
            print(
                f"Document-aware sampling: ENABLED | EOT={args.eot_token_id} | "
                f"segments={len(document_index.starts):,} | "
                f"valid_windows={document_index.total_windows:,} | index={index_path}",
                flush=True,
            )
    else:
        if is_main:
            print("Document-aware sampling: disabled | using flat random windows.", flush=True)

    if model.num_parameters() <= 0:
        raise RuntimeError("Model has no trainable parameters.")

    if distributed and (cfg.activation_bias_update_speed > 0.0 or cfg.moe_bias_update_speed > 0.0):
        raise RuntimeError(
            "Distributed training currently requires activation-bias and MoE-bias "
            "update speeds to be zero; those non-gradient updates are rank-local "
            "and would otherwise diverge replicas. Your current production recipe "
            "already uses zero for both."
        )

    local_batch_size = _local_batch_size(args.batch_size, rank, world_size)

    optimizer = make_optimizer(model, args)
    if resume_ckpt is not None and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=(args.amp and args.dtype == "float16" and device.type == "cuda"),
    )
    if resume_ckpt is not None:
        scaler_state = resume_ckpt.get("extra", {}).get("scaler_state_dict")
        if scaler_state and scaler.is_enabled():
            scaler.load_state_dict(scaler_state)

    train_model: nn.Module = model
    if distributed:
        # ``find_unused_parameters=True`` is required because sparse MoE routing
        # can leave some expert parameters unused on an individual microbatch.
        train_model = DDP(
            _DistributedLossWrapper(model),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            # The production model buffers are deterministic/static here
            # (RoPE inv_freq and zero-rate routing biases are identical on all
            # ranks). Avoid an unnecessary per-forward NCCL broadcast.
            broadcast_buffers=False,
        )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.checkpoint_dir.name
    metrics_path = args.checkpoint_dir / "metrics.jsonl"
    if is_main:
        save_run_metadata(args.checkpoint_dir, args, cfg, train_data, val_data, model)
    _dist_barrier(distributed)

    if is_main:
        eval_batches = make_fixed_eval_batches(
            val_data,
            args.batch_size,
            args.seq_len,
            args.eval_iters,
            args.eval_seed,
            device,
        )
        probe_batch = make_fixed_eval_batches(
            val_data,
            max(1, min(args.viz_probe_size, args.batch_size)),
            args.seq_len,
            1,
            args.eval_seed + 1,
            device,
        )[0]
    else:
        eval_batches = []
        probe_batch = None

    visualizer = None
    if is_main and args.viz_interval > 0:
        visualizer = LoopTransformerVisualizer(model, args.viz_dir, probe_batch=probe_batch.cpu())
        visualizer.set_probe(probe_batch.cpu())

    state = {"step": start_step, "stop": False}

    def _signal_handler(signum: int, _frame: Any) -> None:
        state["stop"] = True
        print(f"Received signal {signum}; saving a final checkpoint after the current micro-step.", flush=True)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    if is_main:
        print(f"Run: {run_name}")
        print(
            f"Training: global_batch_size={args.batch_size} | per_gpu_batch={local_batch_size} | "
            f"seq_len={args.seq_len} | grad_accum_steps={args.grad_accum_steps} | "
            f"dtype={args.dtype} | AMP={args.amp} | late_loop_loss_weight={args.late_loop_loss_weight}"
        )
        if distributed:
            local_batches = [
                _local_batch_size(args.batch_size, r, world_size)
                for r in range(world_size)
            ]
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            print(
                f"Distributed: NCCL DDP | GPUs={world_size} | local_batches={local_batches} | "
                f"devices={gpu_names}"
            )
        else:
            print(f"Device: {device}")
        print(f"Parameters: {model.num_parameters():,} physical / {model.num_parameters(effective=True):,} effective")
        print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
        print(f"Architecture: {json.dumps(dataclasses.asdict(cfg), sort_keys=True)}")

    best_val = float("inf")
    last_eval: dict[str, Any] = {}
    if resume_ckpt is not None:
        resume_extra = dict(resume_ckpt.get("extra", {}))
        best_val = float(resume_extra.get("best_val_ce", float("inf")))
        last_eval = dict(resume_extra.get("last_eval", {}))

        # The optimizer state already contains AdamW's internal step counters
        # and its last applied LR. Validate what we restored before continuing.
        saved_lr = resume_extra.get("lr_at_checkpoint")
        restored_lr = _checkpoint_lr(optimizer, args.lr)
        if forced_lr_start_step is not None and forced_lr_peak is not None:
            # The checkpoint necessarily contains the old LR, which is exactly
            # what --force-lr is meant to replace. Do not treat that as a
            # checkpoint corruption/mismatch. The first upcoming optimizer step
            # will apply the forced peak LR.
            print(
                f"Checkpoint optimizer LR was {restored_lr:.17g}; "
                f"overriding with new cosine peak {forced_lr_peak:.17g}.",
                flush=True,
            )
        elif saved_lr is not None and not math.isclose(restored_lr, float(saved_lr), rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f"Checkpoint LR mismatch: optimizer={restored_lr:.17g}, saved={float(saved_lr):.17g}."
            )
        else:
            print(f"Resumed optimizer at lr={restored_lr:.17g} | best_val={best_val:.6f}", flush=True)

    last_log_time = time.time()

    try:
        for step in range(start_step + 1, args.max_steps + 1):
            state["step"] = step
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            last_step_losses = None

            for micro in range(args.grad_accum_steps):
                if args.document_aware_sampling:
                    if document_index is None:
                        raise RuntimeError("Document-aware sampling was enabled but its index was not initialized.")
                    batch = make_document_aware_batch(
                        train_data, local_batch_size, args.seq_len, device, document_index
                    )
                else:
                    batch = make_batch(train_data, local_batch_size, args.seq_len, device)
                with autocast_context(device, args):
                    # Keep the production recurrent objective intact and add a
                    # small direct term on the final loop. The extra term reuses
                    # the same forward graph and final-loop CE already computed
                    # inside compute_loss(), so it does not require a second
                    # recurrent forward or a second set of vocabulary logits.
                    loss, step_losses = train_model(
                        batch,
                        max_loops=cfg.max_loops,
                        late_loop_weight=args.late_loop_loss_weight,
                    )
                    if distributed:
                        # DDP averages gradients across ranks. Weight each rank's
                        # local mean so uneven global batches (e.g. 3 -> 2+1) still
                        # produce the exact global-batch mean gradient.
                        rank_loss_scale = (world_size * local_batch_size) / float(args.batch_size)
                    else:
                        rank_loss_scale = 1.0
                    loss_for_backward = (loss * rank_loss_scale) / args.grad_accum_steps
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}, micro-step {micro}: {loss.item()}")
                if scaler.is_enabled():
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()
                total_loss += float(loss.item())
                last_step_losses = step_losses

            # Training loss is a diagnostic only. In DDP we intentionally log the
            # rank-0 local mean rather than inserting any user collective into the
            # training stream. DDP's reducer exclusively owns NCCL collectives.
            train_loss = total_loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm}")

            lr = scheduled_lr(
                step,
                args,
                force_start_step=forced_lr_start_step,
                force_peak_lr=forced_lr_peak,
            )
            set_lr(optimizer, lr)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            model.update_routing_biases()

            now = time.time()
            if is_main and (step % args.log_interval == 0 or step == 1):
                elapsed = now - last_log_time
                last_log_time = now
                tok_per_s = (args.batch_size * args.seq_len * args.grad_accum_steps * args.log_interval) / max(elapsed, 1e-6) if step % args.log_interval == 0 else 0.0
                print(
                    f"step {step:6d}/{args.max_steps} | train={train_loss:.6f} | "
                    f"lr={lr:.6g} | grad={float(grad_norm):.4g} | tok/s={tok_per_s:.1f}",
                    flush=True,
                )

            if is_main and visualizer is not None:
                visualizer.record_training(step, train_loss, lr, float(grad_norm))

            do_eval = (step % args.eval_interval == 0) or (step == args.max_steps)
            if do_eval:
                if distributed:
                    _dist_barrier(distributed)
                if is_main:
                    last_eval = evaluate(model, eval_batches)
                    visualizer and visualizer.record_validation(
                        step,
                        float(last_eval["val_ce"]),
                        float(last_eval["val_accuracy"]),
                    )
                    elapsed_total = time.time() - last_log_time
                    row = {
                        "step": step,
                        "train_loss": train_loss,
                        "lr": lr,
                        "grad_norm": float(grad_norm),
                        **last_eval,
                        "elapsed_since_last_log": elapsed_total,
                    }
                    append_metrics(metrics_path, row)
                    best_val = min(best_val, float(last_eval["val_ce"]))
                    print(
                        f"  val CE={last_eval['val_ce']:.6f} | val acc={last_eval['val_accuracy_pct']:.3f}% | "
                        f"loop CE={json.dumps(last_eval['val_loop_ce'])}",
                        flush=True,
                    )
                if distributed:
                    _dist_barrier(distributed)

            if is_main and visualizer is not None and args.viz_interval > 0 and (step % args.viz_interval == 0 or step == args.max_steps):
                metrics = visualizer.snapshot(step, probe_batch)
                print(
                    f"  viz: {metrics['health']} | state growth={metrics['state_growth_ratio']:.4g}x",
                    flush=True,
                )

            checkpoint_due = (step % args.checkpoint_interval == 0 or step == args.max_steps)
            drive_due = (
                kaggle_syncer is not None
                and kaggle_worker is not None
                and args.drive_save_interval > 0
                and step % args.drive_save_interval == 0
            )

            # A Drive-save boundary is also a full local checkpoint boundary. This
            # ensures Drive/latest.pt is always a resumable checkpoint, while the
            # actual Google Drive transfer remains completely out-of-band.
            if checkpoint_due or drive_due:
                if distributed:
                    _dist_barrier(distributed)
                if is_main:
                    extra = build_checkpoint_extra(
                        args,
                        run_name=run_name,
                        best_val=best_val,
                        last_eval=last_eval,
                        optimizer=optimizer,
                        lr=lr,
                        forced_lr_start_step=forced_lr_start_step,
                        forced_lr_peak=forced_lr_peak,
                        scaler=scaler,
                    )
                    save_checkpoint(
                        args.checkpoint_dir / "latest.pt",
                        model,
                        optimizer,
                        step=step,
                        extra=extra,
                    )
                    last_checkpoint_step = step
                    if checkpoint_due:
                        print(f"  checkpoint: {step}", flush=True)
                    elif drive_due:
                        print(f"  Drive checkpoint prepared: {step}", flush=True)
                if distributed:
                    _dist_barrier(distributed)

            if drive_due:
                # Guarantee a matching step image for every Drive save when the
                # visualizer is enabled. This image is generated from the exact model
                # state as the checkpoint above, then frozen by hard-link for upload.
                if is_main and visualizer is not None:
                    expected_image = args.viz_dir / f"step_{step:08d}.png"
                    if not expected_image.exists():
                        metrics = visualizer.snapshot(step, probe_batch)
                        print(
                            f"  viz: {metrics['health']} | state growth={metrics['state_growth_ratio']:.4g}x",
                            flush=True,
                        )

                if is_main:
                    ticket = kaggle_syncer.create_snapshot_ticket(
                        step=step,
                        checkpoint_path=args.checkpoint_dir / "latest.pt",
                    )
                    kaggle_worker.request_snapshot(ticket, step=step)

            if state["stop"]:
                if distributed:
                    _dist_barrier(distributed)
                if is_main:
                    ckpt_path = args.checkpoint_dir / f"latest.pt"
                    save_checkpoint(
                        ckpt_path,
                        model,
                        optimizer,
                        step=step,
                        extra=build_checkpoint_extra(
                            args,
                            run_name=run_name,
                            best_val=best_val,
                            last_eval=last_eval,
                            optimizer=optimizer,
                            lr=lr,
                            forced_lr_start_step=forced_lr_start_step,
                            forced_lr_peak=forced_lr_peak,
                            scaler=scaler,
                            interrupted=True,
                        ),
                    )
                    print(f"Interrupted safely at step {step}: {ckpt_path}", flush=True)
                if distributed:
                    _dist_barrier(distributed)
                return
    except KeyboardInterrupt:
        step = int(state["step"])
        if distributed:
            _dist_barrier(distributed)
        if is_main:
            ckpt_path = args.checkpoint_dir / f"latest.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                step=step,
                extra=build_checkpoint_extra(
                    args,
                    run_name=run_name,
                    best_val=best_val,
                    last_eval=last_eval,
                    optimizer=optimizer,
                    lr=args.lr,
                    forced_lr_start_step=forced_lr_start_step,
                    forced_lr_peak=forced_lr_peak,
                    scaler=scaler,
                    interrupted=True,
                ),
            )
            print(f"Interrupted safely at step {step}: {ckpt_path}", flush=True)
        if distributed:
            _dist_barrier(distributed)
        return
    finally:
        if is_main and kaggle_worker is not None:
            # One final best-effort sync. Drive failures are logged by the worker
            # and never replace the original training exception.
            kaggle_worker.close(flush=True)
        _dist_shutdown(distributed)

    if is_main:
        print("Training complete.", flush=True)
        if last_eval:
            print(json.dumps(last_eval, indent=2), flush=True)


if __name__ == "__main__":
    train(parse_args())
