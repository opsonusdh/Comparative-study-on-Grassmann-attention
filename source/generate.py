#!/usr/bin/env python3
"""Sample text from a LoopLLM checkpoint.

This script is intentionally inference-only. It loads a .pt checkpoint safely on CPU first, then moves the model to the selected backend,
reconstructs the exact LoopConfig stored in that checkpoint, encodes the prompt
with the project's p50k_base tokenizer, and autoregressively samples tokens.

Common controls:
  --checkpoint      Path to the .pt checkpoint
  --prompt          Starting text/token string
  --max-new-tokens  Number of tokens to generate
  --temperature     Sampling temperature
  --top-k           Top-k filtering (0 disables)
  --top-p           Nucleus sampling (1 disables)
  --seed            Reproducible sampling seed
  --loops           Fixed loop count; default is the trained max loop count
  --device          auto/cuda/cpu/xpu/dml
  --greedy          Greedy argmax decoding
  --repetition-penalty  Optional repetition penalty

It uses the checkpoint's stored architecture/configuration. No optimizer state
is loaded and no gradients are created.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Project import path
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def find_src_dir() -> Path:
    """Find the repository's src/ directory in common script placements."""
    candidates = [
        SCRIPT_DIR / "src",
        SCRIPT_DIR.parent / "src",
        Path.cwd() / "src",
        Path.cwd() / "loopllm_clean_rfc" / "src",
    ]
    for candidate in candidates:
        if (candidate / "loop_transformer").is_dir():
            return candidate
    raise RuntimeError(
        "Could not find the LoopLLM src/ directory. Put this script in the project root "
        "or scripts/ directory, or run it from the project root."
    )


SRC = find_src_dir()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_transformer.checkpointing import load_checkpoint, resolve_device  # noqa: E402


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.manual_seed_all(seed)
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply the standard token-wise repetition penalty to the last-step logits."""
    if penalty == 1.0 or penalty <= 0.0:
        return logits

    out = logits.clone()
    seen = token_ids.unique()
    values = out[:, seen]
    out[:, seen] = torch.where(values < 0, values * penalty, values / penalty)
    return out


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    cutoff = values[:, [-1]]
    return torch.where(
        logits < cutoff,
        torch.full_like(logits, float("-inf")),
        logits,
    )


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("--top-p must be in (0, 1].")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    remove = cumulative > top_p
    # Keep the first token that crosses the threshold.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    filtered_sorted = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)
    return filtered


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    greedy: bool,
    repetition_penalty: float,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    logits = apply_repetition_penalty(logits, token_ids, repetition_penalty)

    if greedy:
        next_id = logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        confidence = float(probs.gather(-1, next_id).item())
        return next_id, confidence

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("--temperature must be finite and > 0 unless --greedy is used.")

    scaled = logits / temperature
    scaled = top_k_filter(scaled, top_k)
    scaled = top_p_filter(scaled, top_p)

    probs = F.softmax(scaled, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
        raise RuntimeError("Sampling filter removed all probability mass.")

    next_id = torch.multinomial(probs, num_samples=1)
    confidence = float(probs.gather(-1, next_id).item())
    return next_id, confidence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate text from a LoopLLM .pt checkpoint with configurable sampling."
    )
    p.add_argument("--checkpoint", required=True, help="Path to the LoopLLM .pt checkpoint.")
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument("--prompt", help="Starting text/token string.")
    start.add_argument(
        "--start-token-id",
        type=int,
        help="Start from exactly one tokenizer token id instead of text.",
    )
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Fixed number of recurrent loops. Default: checkpoint max_loops.",
    )
    p.add_argument(
        "--extrapolate-loops",
        action="store_true",
        help=(
            "Allow loops beyond the checkpoint's trained ceiling for zero-shot "
            "recurrent extrapolation. Loop-conditioned parameters are held at "
            "their final trained value; the shared recurrent core and causal "
            "register bank continue iterating. This is an evaluation-only mode."
        ),
    )
    p.add_argument(
        "--adaptive-exit",
        action="store_true",
        help="Use the model's built-in adaptive/Q-exit path instead of fixed loops.",
    )
    p.add_argument(
        "--exit-threshold",
        type=float,
        default=0.8,
        help="Q-exit CDF threshold when --adaptive-exit is enabled.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu", "dml", "directml", "privateuseone", "privateuseone:0"),
        default="auto",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Use argmax decoding; temperature/top-k/top-p are ignored.",
    )
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help=">1 discourages previously generated tokens; 1 disables it.",
    )
    p.add_argument(
        "--show-token-info",
        action="store_true",
        help="Print every generated token id, probability, and loop count.",
    )
    p.add_argument(
        "--show-final-only",
        action="store_true",
        help="Suppress streaming and print only the final generated text.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if args.repetition_penalty <= 0.0:
        raise ValueError("--repetition-penalty must be > 0")
    if args.loops is not None and args.loops <= 0:
        raise ValueError("--loops must be > 0")
    if args.adaptive_exit and args.loops is not None:
        raise ValueError("Use either --adaptive-exit or --loops, not both.")
    if args.adaptive_exit and args.extrapolate_loops:
        raise ValueError("--extrapolate-loops is only valid with fixed --loops generation.")
    if not 0.0 < args.exit_threshold <= 1.0:
        raise ValueError("--exit-threshold must be in (0, 1].")
    if not args.greedy and args.temperature <= 0.0:
        raise ValueError("--temperature must be > 0 unless --greedy is used.")

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")

    # Load through the project's checkpoint loader so the exact stored
    # LoopConfig is reconstructed from the .pt file.
    ckpt = load_checkpoint(checkpoint, device="cpu")
    model = ckpt["model"].to(device)
    model.eval()

    configured_loops = model.cfg.max_loops
    run_loops = args.loops if args.loops is not None else configured_loops
    if args.adaptive_exit:
        run_loops = configured_loops
    if run_loops > configured_loops and not args.extrapolate_loops:
        raise ValueError(
            f"Requested {run_loops} loops, but checkpoint was trained/configured for "
            f"{configured_loops}. Re-run with --extrapolate-loops for explicit "
            "zero-shot recurrent extrapolation."
        )
    if args.extrapolate_loops and run_loops <= configured_loops:
        raise ValueError(
            "--extrapolate-loops is only meaningful when --loops exceeds the "
            "checkpoint's configured loop count."
        )
    if args.extrapolate_loops and model.cfg.loop_transition_matrices:
        raise ValueError(
            "This checkpoint uses explicit loop-transition matrices, so a loop "
            "beyond the trained ceiling has no trained transition to apply."
        )

    if args.seed is not None:
        seed_everything(args.seed)
        print(f"Seed: {args.seed}")

    try:
        import tiktoken  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required for p50k_base tokenization. Install it with: pip install tiktoken"
        ) from exc

    tokenizer = tiktoken.get_encoding("p50k_base")
    eos_token_id = tokenizer.eot_token
    if args.start_token_id is not None:
        if not 0 <= args.start_token_id < model.cfg.vocab_size:
            raise ValueError(
                f"--start-token-id must be in [0, {model.cfg.vocab_size - 1}]"
            )
        prompt_ids = [args.start_token_id]
        prompt_text = tokenizer.decode(prompt_ids)
    else:
        prompt_ids = tokenizer.encode(args.prompt)
        if not prompt_ids:
            raise ValueError("Prompt produced zero tokens.")
        prompt_text = args.prompt

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated_ids: list[int] = []
    loop_history: list[int] = []
    confidence_history: list[float] = []
    stopped_on_eos = False

    print(f"Checkpoint step: {ckpt.get('step', 'unknown')}")
    print(f"Parameters: {model.num_parameters():,}")
    print(f"Configured loops: {configured_loops}")
    print(f"Generation loops: {'adaptive' if args.adaptive_exit else run_loops}")
    if args.extrapolate_loops:
        print(
            "Loop extrapolation: ENABLED | loops beyond the trained ceiling reuse "
            "the final trained loop-conditioned parameters while the shared "
            "recurrent core and causal register state continue."
        )
    print(f"Prompt tokens: {len(prompt_ids)}")
    print(f"Prompt: {prompt_text!r}")
    if args.adaptive_exit:
        print(f"Exit threshold: {args.exit_threshold:.3f}")
    print("\n--- GENERATED TEXT ---")

    # The model's forward() returns logits for every loop. Under no_grad this is
    # inference-only, and the final selected loop is used for token sampling.
    with torch.no_grad():
        for step in range(args.max_new_tokens):
            if args.adaptive_exit:
                chosen, chosen_loop, _ = model.forward_adaptive_exit(
                    idx, exit_threshold=args.exit_threshold
                )
                chosen_logits = chosen[:, -1, :]
            else:
                loop_logits, _ = model(
                    idx,
                    max_loops=run_loops,
                    allow_loop_extrapolation=args.extrapolate_loops,
                )
                chosen_logits = loop_logits[-1][:, -1, :]
                chosen_loop = run_loops

            next_id, confidence = sample_next_token(
                chosen_logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                greedy=args.greedy,
                repetition_penalty=args.repetition_penalty,
                token_ids=idx,
            )

            idx = torch.cat([idx, next_id], dim=1)
            token_id = int(next_id.item())
            token_text = tokenizer.decode([token_id])
            loop_history.append(chosen_loop)
            confidence_history.append(confidence)

            if args.show_token_info:
                print(
                    f"\n[{step + 1:04d}] token_id={token_id:5d} "
                    f"prob={confidence:.6f} loops={chosen_loop} "
                    f"token={token_text!r}"
                )

            # p50k_base uses token 50256 for <|endoftext|>.  It is a document
            # boundary and therefore terminates this sample; do not present
            # tokens after it as if they were model continuation text.
            if token_id == eos_token_id:
                stopped_on_eos = True
                break

            generated_ids.append(token_id)

            if not args.show_final_only and not args.show_token_info:
                print(token_text, end="", flush=True)

    final_text = tokenizer.decode(generated_ids)

    if args.show_final_only or args.show_token_info:
        print("\n" + final_text, end="")
    print("\n")

    mean_conf = sum(confidence_history) / max(1, len(confidence_history))
    print("--- SUMMARY ---")
    print(f"Generated tokens: {len(generated_ids)}")
    print(f"Stopped on EOS: {'yes' if stopped_on_eos else 'no'}")
    print(f"Mean token probability after filtering: {mean_conf:.6f}")
    if args.adaptive_exit:
        mean_loops = sum(loop_history) / max(1, len(loop_history))
        print(f"Mean loops used: {mean_loops:.3f} / {configured_loops}")
    else:
        print(f"Loop count per token: {run_loops}")
    print(f"Sampling: {'greedy' if args.greedy else f'temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}'}")
    print(f"Repetition penalty: {args.repetition_penalty}")


if __name__ == "__main__":
    main()
