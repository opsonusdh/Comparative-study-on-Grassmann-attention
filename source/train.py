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
- with --source Kaggle, resolves the Kaggle dataset/work paths; Drive upload is manual.

No experimental attention variants, diffusion path, or alternate trainer is
kept in this entrypoint.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import dataclasses
import datetime
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
                    help="Runtime source backend. Use `Kaggle` to resolve /kaggle/input and /kaggle/working/loop_llm paths; Drive upload is manual.")

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
    attention.add_argument(
        "--attention-type", choices=["deepseek", "deepseek-split", "grassmann", "grassmann-new", "real-grassmann"],
        default=d.attention_type,
        help="Select baseline DeepSeek SWA/CSA/HCA, loop-split sparse DeepSeek attention, or a Grassmann attention variant.",
    )
    attention.add_argument("--grassmann-rank", type=int, default=d.grassmann_rank)
    _add_bool(
        attention,
        "grassmann-positional-embedding",
        d.grassmann_positional_embedding,
        "Use Grassmann's learned absolute token-position table; disable for the no-position ablation.",
    )
    attention.add_argument(
        "--grassmann-positional-max-len", type=int, default=d.grassmann_positional_max_len,
        help="Maximum input token length supported by Grassmann's learned absolute position table.",
    )
    attention.add_argument(
        "--grassmann-positional-seed", type=_optional_int, default=d.grassmann_positional_seed,
        help="Dedicated seed for the position table; does not alter shared model initialization.",
    )
    attention.add_argument(
        "--grassmann-offsets", type=int, nargs="+", default=list(d.grassmann_offsets),
        help="Positive causal look-back offsets, e.g. 1 2 4 8 12 16.",
    )
    attention.add_argument(
        "--grassmann-offset-count", type=int, default=d.grassmann_offset_count,
        help=("For grassmann-new: number of local offsets per recurrent loop, including lag 1. "
              "An adaptive sparse-global tail is added automatically."),
    )
    attention.add_argument(
        "--grassmann-offset-spacing-base", type=int, default=d.grassmann_offset_spacing_base,
        help=("For grassmann-new: loop-0 local spacing; loop n uses spacing base+n. "
              "Global spacing is computed from N, max_loops, and runtime context length."),
    )
    attention.add_argument(
        "--grassmann-long-range-stride", type=int, default=d.grassmann_long_range_stride,
        help=("For real-grassmann: append offsets stride, 2*stride, ... up to "
              "the actual sequence length. Default 16: append every 16th look-back through the context."),
    )
    attention.add_argument("--deepseek-split-stride", type=int, default=d.deepseek_split_stride,
                           help=("For deepseek-split: partition period for compressed long-range "
                                 "CSA/HCA blocks; SWA and raw local windows stay complete (default 4)."))
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
            "Local manual-save boundary every N optimizer steps. No automatic Google Drive "
            "upload is performed; use the locally written latest.pt for manual backup. 0 disables."
        ),
    )
    train.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "run")
    train.add_argument("--resume", type=Path, default=None,
                       help="Resume exactly from this canonical checkpoint.")
    train.add_argument(
        "--require-resume",
        action="store_true",
        help="Abort unless --resume loads a complete checkpoint; prevents accidental fresh training.",
    )
    train.add_argument("--log-interval", type=int, default=10)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--device", default=None,
                       help="tpu, cuda, cpu, or an explicit torch device. Defaults to CUDA when available.")
    train.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    train.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high")
    _add_bool(train, "tf32", True, "Allow TF32 matrix math on CUDA when supported.")
    _add_bool(train, "amp", False, "Use autocast for the selected dtype on CUDA/CPU where supported.")

    viz = p.add_argument_group("visualization")
    viz.add_argument("--viz-dir", type=Path, default=ROOT / "visualizations" / "run")
    viz.add_argument("--viz-interval", type=int, default=100,
                     help="Save one mechanistic PNG/JSON snapshot every N optimizer steps; 0 disables snapshots.")
    viz.add_argument("--viz-probe-size", type=int, default=2)
    viz.add_argument(
        "--viz-label", default=None,
        help=(
            "Custom architecture/experiment label shown at the top of visualization plots. "
            "When omitted, a descriptive label is generated from the model configuration."
        ),
    )

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
    if args.require_resume and args.resume is None:
        parser.error("--require-resume was supplied, but --resume is missing; refusing to start a fresh run.")
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


def eot_cross_document_target_mask(batch: torch.Tensor, eot_token_id: int) -> torch.Tensor:
    """Return a next-token target mask that excludes only targets after EOT.

    For input ``[..., t0, t1, ...]``, the target for position ``i`` is token
    ``i + 1``. A target is masked only when its preceding input token is EOT,
    preserving all other within-document next-token targets.
    """
    if batch.ndim != 2:
        raise ValueError(f"batch must be rank-2 [batch, seq], got shape {tuple(batch.shape)}")
    if batch.size(1) < 2:
        return torch.empty((batch.size(0), 0), dtype=torch.bool, device=batch.device)
    return batch[:, :-1].ne(eot_token_id)


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
        attention_type=a.attention_type,
        deepseek_split_stride=a.deepseek_split_stride,
        grassmann_rank=a.grassmann_rank,
        grassmann_offsets=tuple(a.grassmann_offsets),
        grassmann_offset_count=a.grassmann_offset_count,
        grassmann_offset_spacing_base=a.grassmann_offset_spacing_base,
        grassmann_long_range_stride=a.grassmann_long_range_stride,
        grassmann_positional_embedding=a.grassmann_positional_embedding,
        grassmann_positional_max_len=a.grassmann_positional_max_len,
        grassmann_positional_seed=a.grassmann_positional_seed,
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
        "causal_register", "loop_embedding", "grassmann_positional", "norm", "tok_emb",
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
        # foreach=True may allocate a second parameter-sized set of temporary
        # tensors during AdamW.step(). On 15-GiB GPUs this can trigger OOM.
        # The scalar/tensor-at-a-time path is slower but has lower peak VRAM.
        foreach=False,
        fused=False,
    )


def _load_optimizer_state_compatible(
    optimizer: torch.optim.Optimizer,
    saved_state: dict[str, Any],
    model: LoopTransformer,
) -> bool:
    """Restore optimizer state across small parameter-group topology changes.

    The checkpoint optimizer does not contain parameter names, only optimizer
    parameter IDs.  Therefore a topology migration is safe only when the old
    and current parameter sequences can be matched unambiguously from the
    Adam state tensor shapes, preserving group order.  If they cannot be
    matched uniquely, return ``False`` so the caller can initialize a fresh
    optimizer rather than silently assigning a moment tensor to the wrong
    parameter.
    """
    saved_groups = list(saved_state.get("param_groups", []))
    current_param_objects = [list(group["params"]) for group in optimizer.param_groups]
    current_state = optimizer.state_dict()
    current_groups = list(current_state.get("param_groups", []))

    direct_shapes_match = len(saved_groups) == len(current_groups)
    if direct_shapes_match:
        for saved_group, current_params in zip(saved_groups, current_param_objects):
            saved_ids = list(saved_group.get("params", []))
            if len(saved_ids) != len(current_params):
                direct_shapes_match = False
                break
            saved_states_for_group = saved_state.get("state", {})
            for saved_id, param in zip(saved_ids, current_params):
                state = saved_states_for_group.get(saved_id, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = state.get(key) if isinstance(state, dict) else None
                    if torch.is_tensor(value) and tuple(value.shape) != tuple(param.shape):
                        direct_shapes_match = False
                        break
                if not direct_shapes_match:
                    break
            if not direct_shapes_match:
                break

    if direct_shapes_match:
        try:
            optimizer.load_state_dict(saved_state)
            return True
        except ValueError as exc:
            original_error = exc
    else:
        original_error = ValueError(
            "Saved Adam moment shapes/parameter counts do not match the current model."
        )

    current_state = optimizer.state_dict()
    current_groups = list(current_state.get("param_groups", []))

    if len(saved_groups) != len(current_groups):
        print(
            "Optimizer compatibility restore skipped: checkpoint/current optimizer "
            f"group counts differ ({len(saved_groups)} vs {len(current_groups)}). "
            "A fresh optimizer will be used.",
            flush=True,
        )
        return False

    current_param_groups = [
        list(group["params"]) for group in current_state.get("param_groups", [])
    ]
    current_param_objects = [
        list(group["params"]) for group in optimizer.param_groups
    ]
    saved_states = dict(saved_state.get("state", {}))

    def _state_shape(state: Any) -> tuple[int, ...] | None:
        if not isinstance(state, dict):
            return None
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value):
                return tuple(value.shape)
        return None

    def _map_group(
        saved_ids: list[Any],
        current_ids: list[Any],
        current_params: list[torch.nn.Parameter],
    ) -> list[tuple[Any, Any]] | None:
        """Find one unique order-preserving saved-id -> current-id mapping."""
        if len(saved_ids) > len(current_ids):
            return None

        extra = len(current_ids) - len(saved_ids)
        if extra == 0:
            candidates = [tuple()]
        elif extra == 1:
            candidates = [(i,) for i in range(len(current_ids))]
        else:
            # This trainer only needs tiny compatibility gaps. Refuse a large
            # ambiguous migration instead of spending unbounded time guessing.
            if extra > 8:
                return None
            candidates = []
            import itertools
            candidates = itertools.combinations(range(len(current_ids)), extra)

        valid: list[list[tuple[Any, Any]]] = []
        for skipped in candidates:
            skipped_set = set(skipped)
            pairs: list[tuple[Any, Any]] = []
            saved_index = 0
            ok = True
            for current_index, current_id in enumerate(current_ids):
                if current_index in skipped_set:
                    continue
                saved_id = saved_ids[saved_index]
                state = saved_states.get(saved_id)
                shape = _state_shape(state)
                if shape is not None and shape != tuple(current_params[current_index].shape):
                    ok = False
                    break
                pairs.append((saved_id, current_id))
                saved_index += 1
            if ok and saved_index == len(saved_ids):
                valid.append(pairs)
                if len(valid) > 1:
                    # Do not choose arbitrarily among equally plausible
                    # parameter alignments. That would corrupt Adam moments.
                    return None
        return valid[0] if valid else None

    all_pairs: list[tuple[Any, Any]] = []
    for group_index, (saved_group, current_group, current_params) in enumerate(
        zip(saved_groups, current_state["param_groups"], current_param_objects)
    ):
        saved_ids = list(saved_group.get("params", []))
        current_ids = list(current_group.get("params", []))
        mapped = _map_group(saved_ids, current_ids, current_params)
        if mapped is None:
            print(
                "Optimizer compatibility restore skipped: group "
                f"{group_index} could not be matched uniquely from Adam state shapes "
                "without risking a corrupted optimizer state. A fresh optimizer will be used. ",
                flush=True,
            )
            return False
        all_pairs.extend(mapped)

    remapped = copy.deepcopy(current_state)
    remapped_states: dict[Any, Any] = {}
    current_param_by_id = {
        current_id: param
        for ids, params in zip(current_param_groups, current_param_objects)
        for current_id, param in zip(ids, params)
    }

    for saved_id, current_id in all_pairs:
        state = saved_states.get(saved_id)
        if state is None:
            continue
        param = current_param_by_id[current_id]
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state.get(key)
            if tensor is not None and torch.is_tensor(tensor):
                expected = tuple(param.shape)
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        "Optimizer state shape mismatch during safe migration: "
                        f"{key} has shape {tuple(tensor.shape)}, expected {expected}."
                    ) from original_error
        remapped_states[current_id] = state

    remapped["state"] = remapped_states
    try:
        optimizer.load_state_dict(remapped)
    except ValueError as exc:
        raise ValueError(
            "Optimizer compatibility migration produced an invalid optimizer state."
        ) from exc

    matched = len(all_pairs)
    saved_total = sum(len(group.get("params", [])) for group in saved_groups)
    current_total = sum(len(group.get("params", [])) for group in current_state["param_groups"])
    print(
        "Optimizer compatibility restore: migrated "
        f"{matched}/{saved_total} checkpoint parameter states into the current "
        f"{current_total}-parameter optimizer; unmatched new parameters start with fresh Adam state.",
        flush=True,
    )
    return True


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


def _load_kaggle_drive_sync_class():
    """Load KaggleDriveSync from the trainer tree without starting any worker."""
    try:
        from drive_sync import DriveSyncError, KaggleDriveSync
        return DriveSyncError, KaggleDriveSync
    except ImportError:
        # A PJRT worker (or a notebook launched from another cwd) may not retain
        # the trainer directory on sys.path. Resolve the helper explicitly.
        import importlib.util

        candidates = [ROOT / "drive_sync.py"]
        if KAGGLE_INPUT_ROOT.exists():
            candidates.extend(
                p for p in KAGGLE_INPUT_ROOT.rglob("drive_sync.py") if p.is_file()
            )

        helper_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if helper_path is None:
            raise RuntimeError(
                "Kaggle mode requires drive_sync.py to restore the remote loop_llm tree. "
                "Place drive_sync.py beside train.py or include it in the attached Kaggle dataset."
            )

        spec = importlib.util.spec_from_file_location("loopllm_drive_sync", helper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load Kaggle Drive helper: {helper_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.DriveSyncError, module.KaggleDriveSync


def _restore_kaggle_drive_once() -> None:
    """Restore the remote loop_llm tree locally; never start an upload worker."""
    print(
        f"[Kaggle Drive] Remote restore requested; target={KAGGLE_STATE_ROOT}. "
        "This runs before model construction, so GPU utilization remains idle during restore.",
        flush=True,
    )
    DriveSyncError, KaggleDriveSync = _load_kaggle_drive_sync_class()
    try:
        syncer = KaggleDriveSync(
            KAGGLE_STATE_ROOT,
            remote_folder_name=os.environ.get("LOOPLLM_GDRIVE_FOLDER_NAME", "loop_llm"),
            secret_name=os.environ.get("LOOPLLM_GDRIVE_SECRET_NAME", "GDRIVE_CREDENTIALS_JSON"),
        )
        syncer.restore()
    except DriveSyncError as exc:
        raise RuntimeError(str(exc)) from exc
    print("[Kaggle Drive] Remote restore completed.", flush=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------



RESUME_OUTPUT_OVERRIDES = {
    "source",
    "resume",
    "require_resume",
    "checkpoint_dir",
    "viz_dir",
    "run_name",
    "viz_label",
    "notes",
    "device",
    "max_steps",
    "force_lr",
    "drive_save_interval",
    # This is deliberately user-overridable so an existing checkpoint can be
    # forked into a late-loop-supervision ablation without changing the model.
    "late_loop_loss_weight",
}


def _validate_resume_checkpoint(
    path: Path,
    resume_ckpt: dict[str, Any],
) -> int:
    """Fail closed unless a checkpoint contains a complete resumable state.

    A model-only checkpoint is a warm start, not a training resume.  Accepting
    one here used to let the run restart at step 1 with fresh Adam moments and
    a fresh LR warm-up while printing a misleading resume message.
    """
    if not isinstance(resume_ckpt, dict):
        raise TypeError(f"Resume checkpoint must contain a dictionary: {path}")

    step = resume_ckpt.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise RuntimeError(
            f"Refusing to resume from {path}: checkpoint has invalid completed step {step!r}. "
            "A training resume requires a checkpoint saved after at least one optimizer step."
        )

    optimizer_state = resume_ckpt.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise RuntimeError(
            f"Refusing to resume from {path}: optimizer_state_dict is missing. "
            "This file can provide model weights, but cannot continue the original optimizer trajectory."
        )
    if not isinstance(optimizer_state.get("param_groups"), list) or not optimizer_state["param_groups"]:
        raise RuntimeError(
            f"Refusing to resume from {path}: optimizer_state_dict has no parameter groups."
        )
    if not isinstance(optimizer_state.get("state"), dict) or not optimizer_state["state"]:
        raise RuntimeError(
            f"Refusing to resume from {path}: optimizer_state_dict contains no AdamW moment state. "
            "This is not a complete training checkpoint."
        )

    extra = resume_ckpt.get("extra")
    if not isinstance(extra, dict):
        raise RuntimeError(f"Refusing to resume from {path}: checkpoint extra metadata is missing.")
    saved_args = extra.get("args")
    if not isinstance(saved_args, dict):
        raise RuntimeError(
            f"Refusing to resume from {path}: saved training arguments are missing; "
            "the original LR schedule cannot be reconstructed safely."
        )

    required_args = ("lr", "warmup_steps", "min_lr_ratio", "max_steps")
    missing_args = [name for name in required_args if name not in saved_args]
    if missing_args:
        raise RuntimeError(
            f"Refusing to resume from {path}: saved training arguments are missing "
            f"{missing_args}; refusing to guess the LR schedule from parser defaults."
        )

    schedule = extra.get("lr_state") or extra.get("lr_schedule")
    if not isinstance(schedule, dict) or schedule.get("mode") not in {"normal", "forced_restart"}:
        raise RuntimeError(
            f"Refusing to resume from {path}: checkpoint has no recognized LR schedule state."
        )

    try:
        base_lr = float(saved_args["lr"])
        warmup_steps = int(saved_args["warmup_steps"])
        min_lr_ratio = float(saved_args["min_lr_ratio"])
        saved_max_steps = int(saved_args["max_steps"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Refusing to resume from {path}: invalid saved LR schedule arguments.") from exc
    if not math.isfinite(base_lr) or base_lr <= 0.0:
        raise RuntimeError(f"Refusing to resume from {path}: invalid saved base LR {base_lr!r}.")
    if warmup_steps < 0 or saved_max_steps < 1 or not (0.0 < min_lr_ratio <= 1.0):
        raise RuntimeError(
            f"Refusing to resume from {path}: invalid saved LR schedule "
            f"(warmup_steps={warmup_steps}, min_lr_ratio={min_lr_ratio}, max_steps={saved_max_steps})."
        )
    if step >= saved_max_steps:
        # A completed checkpoint is valid, but there is no next step under its
        # original schedule. Extending it requires an explicit --force-lr run.
        pass

    if schedule["mode"] == "normal":
        schedule_lr = schedule.get("base_lr", base_lr)
        schedule_min = schedule.get("min_lr_ratio", min_lr_ratio)
        schedule_steps = schedule.get("max_steps", saved_max_steps)
        if not math.isclose(float(schedule_lr), base_lr, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f"Refusing to resume from {path}: LR metadata disagrees "
                f"(lr_state.base_lr={schedule_lr}, args.lr={base_lr})."
            )
        if not math.isclose(float(schedule_min), min_lr_ratio, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(f"Refusing to resume from {path}: LR minimum metadata disagrees.")
        if int(schedule_steps) != saved_max_steps:
            raise RuntimeError(f"Refusing to resume from {path}: LR schedule horizon metadata disagrees.")
    else:
        forced_start = schedule.get("start_step")
        forced_peak = schedule.get("peak_lr")
        if forced_start is None or forced_peak is None:
            raise RuntimeError(f"Refusing to resume from {path}: forced LR schedule metadata is incomplete.")
        if int(forced_start) < 0 or int(forced_start) > step or not math.isfinite(float(forced_peak)) or float(forced_peak) <= 0.0:
            raise RuntimeError(f"Refusing to resume from {path}: forced LR schedule metadata is invalid.")

    saved_lr = extra.get("lr_at_checkpoint")
    if saved_lr is None or not math.isfinite(float(saved_lr)) or float(saved_lr) <= 0.0:
        raise RuntimeError(
            f"Refusing to resume from {path}: checkpoint LR is missing or invalid; "
            "cannot verify the optimizer and scheduler are aligned."
        )

    return step


def _validate_resume_schedule_overrides(
    args: argparse.Namespace,
    resume_ckpt: dict[str, Any],
) -> None:
    """Prevent accidental schedule changes on an ordinary resume.

    --force-lr is the explicit opt-in for starting a new cosine schedule from
    the checkpoint step. Without it, LR, warm-up, minimum ratio, and schedule
    horizon must match the saved experiment.
    """
    saved_args = dict(resume_ckpt.get("extra", {}).get("args", {}))
    explicit = set(getattr(args, "_explicit_cli_overrides", ()))
    force_lr = getattr(args, "force_lr", None)

    schedule_fields = ("lr", "warmup_steps", "min_lr_ratio", "max_steps")
    for name in schedule_fields:
        if name not in saved_args or not hasattr(args, name):
            continue
        saved_value = saved_args[name]
        current_value = getattr(args, name)
        if name == "lr" and force_lr is not None:
            continue
        if name in {"warmup_steps", "min_lr_ratio", "max_steps"} and force_lr is not None:
            # A forced restart deliberately uses a new remaining-step cosine
            # schedule; warmup is not used in this mode.
            continue
        if name not in explicit:
            continue
        try:
            equal = math.isclose(float(current_value), float(saved_value), rel_tol=0.0, abs_tol=1e-15)
        except (TypeError, ValueError):
            equal = current_value == saved_value
        if not equal:
            raise ValueError(
                f"Cannot change --{name.replace('_', '-')} during an ordinary resume: "
                f"checkpoint={saved_value!r}, CLI={current_value!r}. "
                "Omit the override to continue the saved schedule, or use --force-lr "
                "to intentionally start a new cosine schedule."
            )


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


def _validate_resume_architecture_overrides(
    args: argparse.Namespace,
    resume_ckpt: dict[str, Any],
) -> None:
    """Reject explicit attention-architecture changes during checkpoint resume.

    The model architecture is reconstructed from the checkpoint. Previously a
    conflicting ``--attention-type`` could be recorded in run metadata while
    the loaded model silently kept its original attention. Structural
    Grassmann options are checked too, because rank/offset/table-size changes
    cannot be applied to an already-constructed checkpoint model.
    """
    model = resume_ckpt.get("model")
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise RuntimeError("Resume checkpoint has no reconstructed model configuration.")

    fields = [
        "attention_type",
        "grassmann_rank",
        "grassmann_positional_embedding",
    ]
    checkpoint_attention_type = getattr(cfg, "attention_type", "deepseek")
    if checkpoint_attention_type == "grassmann-new":
        checkpoint_scheme = getattr(cfg, "grassmann_offset_scheme", "legacy_local_only_v0")
        if checkpoint_scheme != "adaptive_sparse_global_v1":
            raise ValueError(
                "This grassmann-new checkpoint uses the older local-only offset schedule. "
                "It cannot be resumed as the adaptive sparse-global experiment because "
                "that would change the attention geometry mid-run. Load it for inspection, "
                "or start a fresh adaptive run in a new checkpoint directory."
            )
    if checkpoint_attention_type in {"grassmann", "real-grassmann"}:
        fields.append("grassmann_offsets")
    if checkpoint_attention_type == "grassmann-new":
        fields.extend((
            "grassmann_offset_count",
            "grassmann_offset_spacing_base",
            "max_loops",
            "grassmann_offset_scheme",
        ))
    if checkpoint_attention_type == "real-grassmann":
        fields.append("grassmann_long_range_stride")
    if checkpoint_attention_type == "deepseek-split":
        fields.extend(("deepseek_split_stride", "max_loops"))
    # Older Grassmann checkpoints predate the positional table/config fields.
    # Infer their mode from the serialized model rather than rejecting or
    # misidentifying those checkpoints during resume.
    if hasattr(cfg, "grassmann_positional_embedding"):
        checkpoint_has_positions = bool(cfg.grassmann_positional_embedding)
    else:
        checkpoint_has_positions = (
            getattr(model, "grassmann_positional_embedding", None) is not None
        )
    if checkpoint_has_positions:
        fields.append("grassmann_positional_max_len")
    explicit = set(getattr(args, "_explicit_cli_overrides", ()))
    for name in fields:
        if name == "grassmann_positional_embedding":
            checkpoint_value = checkpoint_has_positions
        else:
            checkpoint_value = getattr(cfg, name, getattr(args, name, None))
        requested_value = getattr(args, name, checkpoint_value)
        if name == "grassmann_offsets":
            requested_value = tuple(int(v) for v in requested_value)
            checkpoint_value = tuple(int(v) for v in checkpoint_value)
        if name in explicit and requested_value != checkpoint_value:
            raise ValueError(
                f"Cannot change {name.replace('_', '-')} while resuming: "
                f"checkpoint={checkpoint_value!r}, CLI={requested_value!r}. "
                "Start a fresh run with a new checkpoint directory to change "
                "the attention architecture."
            )
        # Keep metadata and later checkpoint arguments aligned with the model
        # actually loaded, including older checkpoints whose saved CLI args may
        # be incomplete or inconsistent with their serialized model config.
        setattr(args, name, checkpoint_value)
        # Backfill newly introduced config fields for checkpoints serialized
        # before the positional-table ablation was added.
        if not hasattr(cfg, name):
            setattr(cfg, name, checkpoint_value)


def _checkpoint_lr(optimizer: Optional[torch.optim.Optimizer], fallback: float) -> float:
    """Read the optimizer LR after state restoration."""
    if optimizer is not None and optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", fallback))
    return float(fallback)


def choose_device(requested: Optional[str]) -> torch.device:
    if requested:
        requested_name = str(requested).strip().lower()
        if requested_name == "tpu":
            try:
                import torch_xla
            except ImportError as exc:
                raise RuntimeError(
                    "--device tpu requires PyTorch/XLA (torch_xla). "
                    "Install a torch_xla build matching your PyTorch/TPU runtime."
                ) from exc

            # PyTorch/XLA exposes the TPU through the normal XLA device type.
            # Prefer the current top-level API, with the legacy API as a fallback
            # for older torch_xla releases.
            if hasattr(torch_xla, "device"):
                return torch_xla.device()
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _runtime_device(requested: Optional[str], distributed: bool, local_rank: int) -> torch.device:
    """Resolve the actual process-local device."""
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


def _init_distributed(
    requested_device: Optional[str] = None,
    *,
    timeout_seconds: int = 120,
) -> tuple[bool, int, int, int]:
    """Initialize single-node NCCL DDP when launched by torchrun.

    Returns ``(enabled, rank, world_size, local_rank)``. The public
    ``--batch-size`` remains the global optimizer-step batch size; each rank
    receives its proportional local share. TPU mode intentionally uses the
    single-process XLA path in this entrypoint rather than CUDA DDP.

    ``timeout_seconds`` controls the initial process-group timeout. Kaggle
    startup passes a longer value because rank 0 performs a potentially slow
    Google Drive restore before the first collective; normal training uses the
    shorter default timeout. The Kaggle path resets the timeout after restore.

    The process group may already have been initialized by the Kaggle startup
    path so that only rank 0 restores the shared Drive tree. In that case,
    reuse it rather than trying to initialize NCCL a second time.
    """
    if requested_device and str(requested_device).strip().lower() == "tpu":
        return False, 0, 1, 0

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        except ValueError as exc:
            raise RuntimeError("LOCAL_RANK must be an integer when provided by torchrun.") from exc
        return True, rank, world_size, local_rank

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

    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError(
            f"timeout_seconds must be a positive integer, got {timeout_seconds!r}."
        )

    # Keep normal DDP failures bounded. Kaggle startup explicitly requests a
    # longer timeout because rank 0 restores Drive while other ranks wait at a
    # startup broadcast; the timeout is reset to 120 seconds after that broadcast.
    print(
        f"[DDP rank {rank}/{world_size}] Initializing NCCL on cuda:{local_rank}; "
        f"visible_gpus={gpu_count}, torch={torch.__version__}, "
        f"process_group_timeout={timeout_seconds}s.",
        flush=True,
    )
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_seconds),
            device_id=torch.device("cuda", local_rank),
        )
    except Exception as exc:
        raise RuntimeError(
            f"NCCL process-group initialization failed on rank {rank}/{world_size} "
            f"(cuda:{local_rank}). This is a distributed startup failure, not a model-training stall. "
            "Confirm both GPUs are visible with `torch.cuda.device_count()` and launch with "
            "`torchrun --standalone --nproc-per-node=2 train.py ...`. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print(f"[DDP rank {rank}/{world_size}] NCCL initialized.", flush=True)
    return True, rank, world_size, local_rank


def _set_distributed_timeout(seconds: int, *, reason: str) -> bool:
    """Change the initialized process-group timeout when supported by PyTorch.

    PyTorch exposes ``dist.set_timeout`` in newer releases. Older supported
    builds (including the 2.10 line) expose the equivalent
    ``distributed_c10d._set_pg_timeout`` helper. If neither is available, the
    caller keeps the longer startup timeout and emits a warning rather than
    failing a healthy run.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if not isinstance(seconds, int) or seconds <= 0:
        raise ValueError(f"seconds must be a positive integer, got {seconds!r}.")

    setter = getattr(dist, "set_timeout", None)
    setter_name = "torch.distributed.set_timeout"
    if not callable(setter):
        try:
            import torch.distributed.distributed_c10d as c10d
            setter = getattr(c10d, "_set_pg_timeout", None)
            setter_name = "torch.distributed.distributed_c10d._set_pg_timeout"
        except Exception:
            setter = None

    if not callable(setter):
        print(
            f"[DDP] WARNING: this PyTorch build does not expose a process-group "
            f"timeout setter; retaining the current timeout after {reason}.",
            flush=True,
        )
        return False

    try:
        setter(datetime.timedelta(seconds=seconds), group=dist.group.WORLD)
    except Exception as exc:
        print(
            f"[DDP] WARNING: could not set process-group timeout to {seconds}s "
            f"using {setter_name}: {type(exc).__name__}: {exc}. "
            "The current timeout remains in effect.",
            flush=True,
        )
        return False

    print(
        f"[DDP] Process-group timeout set to {seconds}s ({reason}).",
        flush=True,
    )
    return True


def _dist_barrier(enabled: bool) -> None:
    if enabled:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _dist_shutdown(enabled: bool) -> None:
    if enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _tpu_barrier(enabled: bool, tag: str) -> None:
    """Synchronize TPU worker processes through the XLA runtime."""
    if not enabled:
        return
    import torch_xla.core.xla_model as xm
    xm.rendezvous(f"loopllm_{tag}")


def _configure_xla_checkpointing() -> None:
    """Use PyTorch/XLA's checkpoint implementation on TPU workers.

    The model's source imports ``torch.utils.checkpoint.checkpoint`` directly
    and requests ``use_reentrant=False``. That is not a supported XLA path in
    the current torch_xla checkpoint implementation. Do not emulate XLA as a
    ``torch.xla`` device module; route checkpoint calls to the XLA-aware
    implementation instead and force its supported reentrant mode.
    """
    import importlib
    from torch_xla.utils import checkpoint as xla_checkpoint

    model_module = importlib.import_module("loop_transformer.model")

    def _checkpoint(function, *args, use_reentrant=False, **kwargs):
        # torch_xla.utils.checkpoint currently supports only its XLA-aware
        # reentrant implementation. Preserve RNG state unless the caller
        # explicitly disables it. The model's existing call-site passes only
        # ``use_reentrant=False``, so no model.py change is required.
        return xla_checkpoint.checkpoint(
            function, *args, use_reentrant=True, **kwargs
        )

    model_module.grad_checkpoint = _checkpoint


def _distributed_barrier(cuda_enabled: bool, tpu_enabled: bool, tag: str) -> None:
    if cuda_enabled:
        _dist_barrier(True)
    elif tpu_enabled:
        _tpu_barrier(True, tag)


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


def train(
    args: argparse.Namespace,
    *,
    tpu_worker: bool = False,
    tpu_index: Optional[int] = None,
) -> None:
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
    if args.viz_probe_size < 1:
        raise ValueError("viz-probe-size must be >= 1")
    if args.drive_save_interval < 0:
        raise ValueError("drive-save-interval must be >= 0")

    tpu_distributed = bool(tpu_worker or (args.device is not None and str(args.device).strip().lower() == "tpu"))
    if tpu_distributed:
        if not tpu_worker:
            raise RuntimeError("TPU workers must be launched through torch_xla.launch().")
        _configure_xla_checkpointing()
        import torch_xla.runtime as xr
        rank = int(xr.global_ordinal())
        world_size = int(xr.world_size())
        local_rank = rank
        distributed = False
    else:
        distributed, rank, world_size, local_rank = _init_distributed(args.device)
    is_main = rank == 0
    if distributed and args.batch_size % world_size != 0:
        raise ValueError(
            f"For balanced two-GPU training, --batch-size ({args.batch_size}) must be "
            f"divisible by world_size ({world_size}). Uneven per-rank batches create "
            "different activation peaks and can leave one GPU waiting at DDP collectives. "
            f"Use --batch-size {((args.batch_size + world_size - 1) // world_size) * world_size}."
        )

    # --source Kaggle only resolves paths here. The remote loop_llm tree is
    # restored once by _run_from_cli() before any TPU workers are spawned.
    # Workers then consume the shared /kaggle/working tree and never touch Drive.
    if args.source == "kaggle":
        _configure_kaggle_paths(args)

    if not distributed and not tpu_distributed and is_main and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(
            f"[Multi-GPU] {torch.cuda.device_count()} CUDA GPUs are visible, but this process was not "
            "launched with torchrun; only GPU 0 will be used. For both GPUs, launch "
            "with: torchrun --standalone --nproc-per-node=2 train.py ...",
            flush=True,
        )

    # DDP needs independent stochastic streams on each GPU. For a distributed
    # resume we intentionally do not restore the single-rank RNG snapshot from
    # the old checkpoint, because that would make all ranks sample identically.
    seed_everything(args.seed + rank if (distributed or tpu_distributed) else args.seed)
    print(
        f"[Startup rank {rank}] Resolving runtime device and checkpoint state...",
        flush=True,
    )
    device = _runtime_device(args.device, distributed, local_rank)
    configure_runtime(args, device)
    print(f"[Startup rank {rank}] Runtime device={device}.", flush=True)

    resume_ckpt = None
    forced_lr_start_step: Optional[int] = None
    forced_lr_peak: Optional[float] = None
    last_checkpoint_step = 0

    if args.resume is not None:
        # Load the checkpoint before constructing datasets so that seq_len, data
        # paths, batch size, and all other exact training settings are restored
        # from the saved run rather than silently falling back to CLI defaults.
        # Load TPU resumes through CPU first. This lets a CUDA/CPU checkpoint
        # be remapped without requiring the XLA serializer to interpret the
        # source device tag during deserialization; the restored model is then
        # moved to XLA immediately below.
        checkpoint_device = torch.device("cpu") if device.type == "xla" else device
        resume_ckpt = load_checkpoint(
            args.resume,
            device=checkpoint_device,
            optimizer=None,
            restore_rng=not (distributed or tpu_distributed),
        )
        validated_checkpoint_step = _validate_resume_checkpoint(args.resume, resume_ckpt)
        _restore_args_from_checkpoint(args, resume_ckpt)
        _validate_resume_schedule_overrides(args, resume_ckpt)
        _validate_resume_architecture_overrides(args, resume_ckpt)
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
        start_step = validated_checkpoint_step
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

        print(
            f"[Resume] Accepted checkpoint {args.resume}: completed_step={start_step}; "
            f"next_optimizer_step={start_step + 1}; saved_max_steps={saved_args.get('max_steps')}; "
            f"LR mode={((resume_extra.get('lr_state') or resume_extra.get('lr_schedule') or {}).get('mode', 'unknown'))}.",
            flush=True,
        )
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
        if getattr(args, "require_resume", False):
            raise RuntimeError("--require-resume is active, but no resume checkpoint was provided.")
        if is_main:
            print("[Resume] No --resume checkpoint supplied; this is a fresh training run.", flush=True)
        cfg = config_from_args(args)
        print(f"[Startup rank {rank}] Building model on {device}...", flush=True)
        model = LoopTransformer(cfg).to(device)
        print(f"[Startup rank {rank}] Model created and moved to {device}.", flush=True)
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

    print(f"[Startup rank {rank}] Opening and validating training/validation binaries...", flush=True)
    train_data, val_data = prepare_datasets(args)
    print(f"[Startup rank {rank}] Dataset validation complete.", flush=True)

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

    if (distributed or tpu_distributed) and (cfg.activation_bias_update_speed > 0.0 or cfg.moe_bias_update_speed > 0.0):
        raise RuntimeError(
            "Distributed training currently requires activation-bias and MoE-bias "
            "update speeds to be zero; those non-gradient updates are rank-local "
            "and would otherwise diverge replicas. Your current production recipe "
            "already uses zero for both."
        )

    local_batch_size = _local_batch_size(args.batch_size, rank, world_size)

    optimizer = make_optimizer(model, args)
    if resume_ckpt is not None:
        optimizer_state_restored = _load_optimizer_state_compatible(
            optimizer,
            resume_ckpt["optimizer_state_dict"],
            model,
        )
        if not optimizer_state_restored:
            raise RuntimeError(
                "Resume aborted: the checkpoint AdamW state could not be mapped safely to this model. "
                "No training step was run. Use the matching source revision/checkpoint or start a new run."
            )
        restored_lr = _checkpoint_lr(optimizer, args.lr)
        expected_lr = float(resume_ckpt["extra"]["lr_at_checkpoint"])
        if not math.isclose(restored_lr, expected_lr, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                "Resume aborted: restored optimizer LR does not match checkpoint metadata "
                f"(optimizer={restored_lr:.17g}, metadata={expected_lr:.17g})."
            )
        print(
            f"[Resume] Full training state restored: completed_step={start_step}, "
            f"next_step={start_step + 1}, max_steps={args.max_steps}, "
            f"optimizer_lr={restored_lr:.17g}.",
            flush=True,
        )

    if device.type == "xla":
        # TPU training normally uses bf16 without gradient scaling. Preserve the
        # existing --amp flag for bf16 autocast, and use the XLA scaler only when
        # the user explicitly selects fp16 AMP.
        if args.amp and args.dtype == "float16":
            try:
                from torch_xla.amp import GradScaler as XlaGradScaler
            except ImportError as exc:
                raise RuntimeError(
                    "TPU fp16 AMP requires torch_xla.amp.GradScaler from your PyTorch/XLA installation."
                ) from exc
            scaler = XlaGradScaler()
        else:
            scaler = torch.amp.GradScaler(device="cuda", enabled=False)
    else:
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
            # Reuse gradient storage as DDP buckets instead of keeping a
            # separate full-size gradient buffer plus communication buckets.
            gradient_as_bucket_view=True,
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
    _distributed_barrier(distributed, tpu_distributed, "generic")

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
            args.viz_probe_size,
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
        visualizer = LoopTransformerVisualizer(
            model,
            args.viz_dir,
            probe_batch=probe_batch.cpu(),
            viz_label=args.viz_label,
            resume_step=start_step if resume_ckpt is not None else None,
        )
        print(
            f"[Visualizer] directory={args.viz_dir}; "
            f"history_records={len(visualizer.history)}; "
            f"last_train_step={max(visualizer.history_train_steps, default=0)}; "
            f"probe={'restored' if resume_ckpt is not None else 'initialized'}. ",
            flush=True,
        )

    state = {"step": start_step, "stop": False}

    def _signal_handler(signum: int, _frame: Any) -> None:
        state["stop"] = True
        print(f"Received signal {signum}; saving a final checkpoint after the current micro-step.", flush=True)

    # PyTorch/XLA PJRT invokes TPU workers from a non-main worker thread.
    # Python only permits signal.signal() from the main thread, so install
    # process signal handlers only for ordinary CPU/CUDA execution. PJRT owns
    # TPU worker lifecycle/signals at the launcher level.
    if not tpu_distributed:
        signal.signal(signal.SIGINT, _signal_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _signal_handler)

    if is_main:
        print(f"Run: {run_name}")
        batch_label = "per_core_batch" if tpu_distributed else "per_gpu_batch"
        print(
            f"Training: global_batch_size={args.batch_size} | "
            f"{batch_label}={local_batch_size} | "
            f"seq_len={args.seq_len} | grad_accum_steps={args.grad_accum_steps} | "
            f"dtype={args.dtype} | AMP={args.amp} | late_loop_loss_weight={args.late_loop_loss_weight}"
        )
        if tpu_distributed:
            local_batches = [
                _local_batch_size(args.batch_size, r, world_size)
                for r in range(world_size)
            ]
            print(
                f"Distributed: PyTorch/XLA multi-core | TPU cores={world_size} | "
                f"local_batches={local_batches} | worker={rank}",
                flush=True,
            )
        elif distributed:
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
            tpu_loss_terms: list[torch.Tensor] = []
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
                # DDP should synchronize gradients only on the final
                # accumulation micro-step. Earlier micro-steps retain local
                # gradients and avoid repeated bucket communication.
                sync_context = (
                    train_model.no_sync()
                    if distributed and micro < args.grad_accum_steps - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    with autocast_context(device, args):
                        if distributed:
                            loss, step_losses = train_model(
                                batch,
                                max_loops=cfg.max_loops,
                                late_loop_weight=args.late_loop_loss_weight,
                            )
                        else:
                            loss, step_losses = model.compute_loss(
                                batch,
                                max_loops=cfg.max_loops,
                                late_loop_weight=args.late_loop_loss_weight,
                            )
                        if distributed or tpu_distributed:
                            # Equal per-rank batches make this scale exactly 1.0.
                            rank_loss_scale = (world_size * local_batch_size) / float(args.batch_size)
                        else:
                            rank_loss_scale = 1.0
                        loss_for_backward = (loss * rank_loss_scale) / args.grad_accum_steps
                    if not tpu_distributed and not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at step {step}, micro-step {micro}: {loss.item()}"
                        )
                    if scaler.is_enabled():
                        scaler.scale(loss_for_backward).backward()
                    else:
                        loss_for_backward.backward()
                if tpu_distributed:
                    # Keep XLA scalars lazy. Accessing loss.item() before the
                    # optimizer step forces an early graph execution/compilation.
                    tpu_loss_terms.append(loss.detach())
                else:
                    total_loss += float(loss.item())
                last_step_losses = step_losses

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            if device.type == "xla":
                # Reduce gradients through XLA's optimizer-aware path before
                # clipping. This is the supported multi-core primitive and avoids
                # maintaining a second hand-written all-reduce path.
                import torch_xla.core.xla_model as xm
                xm.reduce_gradients(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            lr = scheduled_lr(
                step,
                args,
                force_start_step=forced_lr_start_step,
                force_peak_lr=forced_lr_peak,
            )
            set_lr(optimizer, lr)
            if device.type == "xla":
                import torch_xla.core.xla_model as xm
                if scaler.is_enabled():
                    # Gradients were already averaged by xm.reduce_gradients().
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            elif scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            model.update_routing_biases()
            if device.type == "xla":
                # Finish the complete lazy TPU step exactly once: reduced/clipped
                # gradients, optimizer update, and routing-bias update all belong
                # to this step boundary. Only after this marker do we materialize
                # scalar diagnostics.
                import torch_xla.core.xla_model as xm
                if not tpu_loss_terms:
                    raise RuntimeError("TPU training produced no loss terms for this step.")
                total_loss_tensor = torch.stack(tpu_loss_terms).sum()
                xm.mark_step()
                train_loss = float(total_loss_tensor.cpu().item()) / args.grad_accum_steps
                if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {step}: {grad_norm}"
                    )
            else:
                train_loss = total_loss / args.grad_accum_steps
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    raise FloatingPointError(
                        f"Non-finite gradient norm at step {step}: {grad_norm}"
                    )

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

            if device.type == "cuda" and (step % args.log_interval == 0 or step == 1):
                torch.cuda.synchronize(device)
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                mem_free, mem_total = torch.cuda.mem_get_info(device)
                print(
                    f"[VRAM rank={rank} gpu={local_rank}] step={step} "
                    f"allocated={mem_alloc:.2f}GiB reserved={mem_reserved:.2f}GiB "
                    f"peak={mem_peak:.2f}GiB free={mem_free / (1024 ** 3):.2f}/"
                    f"{mem_total / (1024 ** 3):.2f}GiB",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats(device)

            if is_main and visualizer is not None:
                visualizer.record_training(step, train_loss, lr, float(grad_norm))

            do_eval = (step % args.eval_interval == 0) or (step == args.max_steps)
            if do_eval:
                if distributed or tpu_distributed:
                    _distributed_barrier(distributed, tpu_distributed, f"eval_pre_{step}")
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
                if distributed or tpu_distributed:
                    _distributed_barrier(distributed, tpu_distributed, f"eval_post_{step}")

            if is_main and visualizer is not None and args.viz_interval > 0 and (step % args.viz_interval == 0 or step == args.max_steps):
                metrics = visualizer.snapshot(
                    step,
                    None,  # preserve the original probe.pt loaded at resume
                    validation_metrics=last_eval if do_eval and last_eval else None,
                )
                print(
                    f"  viz: {metrics['health']} | state growth={metrics['state_growth_ratio']:.4g}x",
                    flush=True,
                )

            checkpoint_due = (step % args.checkpoint_interval == 0 or step == args.max_steps)
            # Keep the existing flag available as a local manual-save boundary.
            manual_save_due = (
                args.drive_save_interval > 0
                and step % args.drive_save_interval == 0
            )

            if checkpoint_due or manual_save_due:
                if distributed or tpu_distributed:
                    _distributed_barrier(distributed, tpu_distributed, f"checkpoint_pre_{step}")
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
                    elif manual_save_due:
                        print(f"  manual checkpoint prepared: {step}", flush=True)
                if distributed or tpu_distributed:
                    _distributed_barrier(distributed, tpu_distributed, f"checkpoint_post_{step}")

            if state["stop"]:
                if distributed or tpu_distributed:
                    _distributed_barrier(distributed, tpu_distributed, f"interrupt_pre_{step}")
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
                if distributed or tpu_distributed:
                    _distributed_barrier(distributed, tpu_distributed, f"interrupt_post_{step}")
                return
    except KeyboardInterrupt:
        step = int(state["step"])
        if distributed or tpu_distributed:
            _distributed_barrier(distributed, tpu_distributed, f"keyboard_pre_{step}")
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
        if distributed or tpu_distributed:
            _distributed_barrier(distributed, tpu_distributed, f"keyboard_post_{step}")
        return
    finally:
        _dist_shutdown(distributed)

    if is_main:
        print("Training complete.", flush=True)
        if last_eval:
            print(json.dumps(last_eval, indent=2), flush=True)


def _tpu_worker(index: int, args: argparse.Namespace) -> None:
    # PyTorch/XLA assigns exactly one TPU device to each launched worker.
    # Device/runtime discovery must happen inside this worker.
    train(args, tpu_worker=True, tpu_index=index)


def _run_from_cli() -> None:
    args = parse_args()
    requested_device = str(args.device).strip().lower() if args.device is not None else None
    print(
        f"[Startup] source={args.source} device={requested_device or 'auto'} "
        f"world_size={os.environ.get('WORLD_SIZE', '1')} pid={os.getpid()}.",
        flush=True,
    )

    # Preserve the intended Kaggle workflow: every Kaggle launch restores the
    # complete remote loop_llm tree before training. Under torchrun, initialize
    # DDP first and let rank 0 restore once; then broadcast success/failure so
    # other ranks cannot race the staging directory or load stale checkpoints.
    if args.source == "kaggle":
        requested_world_size = _dist_world_size()
        if requested_device == "tpu" and requested_world_size > 1:
            raise RuntimeError(
                "Do not launch --device tpu through torchrun. This entrypoint restores "
                "Kaggle state once before torch_xla.launch(); invoke train.py directly for TPU."
            )

        if requested_world_size > 1:
            print(
                f"[Startup] DDP startup: rank 0 restores the complete Google Drive "
                f"loop_llm tree to {KAGGLE_STATE_ROOT}; other ranks wait for the "
                "restore-status broadcast.",
                flush=True,
            )
        else:
            print(
                f"[Startup] Restoring the complete Google Drive loop_llm tree to {KAGGLE_STATE_ROOT}...",
                flush=True,
            )
        if requested_world_size > 1:
            try:
                restore_timeout_seconds = int(
                    os.environ.get("LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS", "1800")
                )
            except ValueError as exc:
                raise RuntimeError(
                    "LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS must be an integer."
                ) from exc
            if restore_timeout_seconds < 120:
                raise ValueError(
                    "LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS must be at least 120 seconds."
                )

            distributed, rank, world_size, local_rank = _init_distributed(
                args.device,
                timeout_seconds=restore_timeout_seconds,
            )
            if not distributed:
                raise RuntimeError(
                    "Kaggle Drive restore with WORLD_SIZE>1 requires CUDA DDP. "
                    "Use --device cuda with torchrun, or launch a single process."
                )

            restore_status: list[Optional[str]] = [None]
            if rank == 0:
                try:
                    _restore_kaggle_drive_once()
                except Exception as exc:
                    restore_status[0] = f"{type(exc).__name__}: {exc}"

            try:
                # Rank 1 waits here while rank 0 restores the full Drive tree.
                # A default 120-second NCCL timeout is too short for large
                # restores (e.g. 167 files took >120 seconds). Use the extended
                # startup timeout for this one collective, then restore the
                # normal bounded timeout before model construction/training.
                print(
                    f"[Kaggle Drive] Waiting for rank-0 restore status; "
                    f"startup timeout={restore_timeout_seconds}s.",
                    flush=True,
                )
                dist.broadcast_object_list(
                    restore_status,
                    src=0,
                    device=torch.device("cuda", local_rank),
                )
            except Exception:
                _dist_shutdown(distributed)
                raise

            # Reset on every rank after the startup broadcast. If the installed
            # PyTorch lacks a timeout setter, the longer timeout remains in
            # effect; this is safer than reusing the 120-second timeout for Drive.
            _set_distributed_timeout(
                120,
                reason="Google Drive restore complete; normal DDP timeout restored",
            )

            if restore_status[0] is not None:
                _dist_shutdown(distributed)
                raise RuntimeError(
                    "Kaggle Drive restore failed on rank 0; all workers stopped before training: "
                    + str(restore_status[0])
                )
            if rank == 0:
                print(
                    "[Kaggle Drive] Restore completed once on rank 0; all workers may continue.",
                    flush=True,
                )
        else:
            # Single-process CUDA/CPU and the TPU launcher parent use this path.
            _restore_kaggle_drive_once()

        print("[Startup] Google Drive restoration succeeded; resolving Kaggle paths...", flush=True)
        _configure_kaggle_paths(args)

    if requested_device == "tpu":
        # Kaggle's TPU notebook environment exports cluster-style TPU variables
        # (notably TPU_PROCESS_ADDRESSES=local). Remove those stale cluster hints
        # before importing/initializing XLA.
        os.environ.pop("TPU_PROCESS_ADDRESSES", None)
        os.environ.pop("CLOUD_TPU_TASK_ID", None)
        os.environ["PJRT_DEVICE"] = "TPU"
        try:
            import torch_xla
        except ImportError as exc:
            raise RuntimeError(
                "--device tpu requires PyTorch/XLA (torch_xla). "
                "Install a torch_xla build matching your PyTorch/TPU runtime."
            ) from exc
        # launch() discovers the runtime device count automatically, so on a
        # TPU v5e-8 it creates one worker per TPU core (8 workers).
        torch_xla.launch(_tpu_worker, args=(args,))
    else:
        train(args)


if __name__ == "__main__":
    _run_from_cli()
