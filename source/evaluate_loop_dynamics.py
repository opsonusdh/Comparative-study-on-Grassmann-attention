#!/usr/bin/env python3
"""
evaluate_loop_dynamics.py

Evaluate recurrent-loop dynamics in LoopLLM.

Metrics per loop:
  - final-token CE
  - mean probability assigned to the true next token
  - mean predictive entropy
  - top-1 token at the final prediction position
  - top-5 overlap with the previous loop
  - hidden-state norm at the final prediction position
  - state delta ||h_t - h_(t-1)||
  - normalized state delta ||h_t-h_(t-1)|| / ||h_(t-1)||
  - causal-register norm at the final position
  - causal-register cosine similarity to the previous loop
  - KL(P_t || P_(t-1)) at the final prediction position

Usage for an open-ended context:
  python evaluate_loop_dynamics.py ^
    --checkpoint D:\\latest.pt ^
    --context "if (condition) {\\n" ^
    --max-loops 16 ^
    --extrapolate-loops ^
    --device cpu

Usage for true next-token metrics:
  The final token of --eval-text is treated as the ground-truth target;
  all preceding tokens are the context.

  python evaluate_loop_dynamics.py ^
    --checkpoint D:\\latest.pt ^
    --eval-text "The cat sat on the mat." ^
    --max-loops 16 ^
    --extrapolate-loops ^
    --device cpu ^
    --output loop_dynamics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from loop_transformer.checkpointing import load_checkpoint, resolve_device


def get_tokenizer():
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError(
            "tiktoken is required. Install it with: pip install tiktoken"
        ) from exc
    return tiktoken.get_encoding("p50k_base")


def decode_token(enc, token_id: int) -> str:
    try:
        raw = enc.decode_single_token_bytes(int(token_id))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        try:
            return enc.decode([int(token_id)])
        except Exception:
            return f"<token:{int(token_id)}>"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_inputs(
    enc,
    *,
    context: Optional[str],
    eval_text: Optional[str],
) -> tuple[torch.Tensor, Optional[torch.Tensor], str]:
    if (context is None) == (eval_text is None):
        raise ValueError("Provide exactly one of --context or --eval-text.")

    if eval_text is not None:
        ids = enc.encode(eval_text, allowed_special={"<|endoftext|>"})
        if len(ids) < 2:
            raise ValueError("--eval-text must tokenize to at least two tokens.")
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long)
        targets = torch.tensor([ids[1:]], dtype=torch.long)
        return input_ids, targets, eval_text

    ids = enc.encode(context, allowed_special={"<|endoftext|>"})
    if not ids:
        raise ValueError("--context tokenized to zero tokens.")
    input_ids = torch.tensor([ids], dtype=torch.long)
    return input_ids, None, context


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def true_token_metrics(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor],
) -> tuple[float, float]:
    """
    logits: [1, T, V]
    targets: [1, T], or None

    Returns:
      final-token CE at the final prediction position
      mean probability assigned to the true next token
    """
    if targets is None:
        return float("nan"), float("nan")

    flat_logits = logits.float().reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    probs = F.softmax(flat_logits, dim=-1)
    true_prob = probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)

    return float(ce[-1].item()), float(true_prob.mean().item())


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.reshape(a.size(0), -1).float()
    bv = b.reshape(b.size(0), -1).float()
    return float(F.cosine_similarity(av, bv, dim=-1, eps=1e-8).mean().item())


def evaluate(
    model,
    enc,
    input_ids: torch.Tensor,
    targets: Optional[torch.Tensor],
    max_loops: int,
    *,
    allow_loop_extrapolation: bool,
) -> list[dict]:
    device = next(model.parameters()).device
    idx = input_ids.to(device)
    tgt = targets.to(device) if targets is not None else None

    model.eval()
    with torch.no_grad():
        loop_logits, _ = model(
            idx,
            max_loops=max_loops,
            allow_loop_extrapolation=allow_loop_extrapolation,
        )

    states = list(getattr(model, "_last_loop_states", []))
    registers = list(getattr(model, "_last_loop_register_states", []))

    if len(states) != max_loops:
        raise RuntimeError(
            f"Expected {max_loops} saved hidden states, got {len(states)}."
        )

    rows: list[dict] = []
    prev_top5: Optional[torch.Tensor] = None
    prev_state: Optional[torch.Tensor] = None
    prev_register: Optional[torch.Tensor] = None
    prev_final_log_probs: Optional[torch.Tensor] = None

    for i in range(max_loops):
        logits = loop_logits[i].float()

        # Mean entropy across all positions in the supplied context.
        entropy = float(entropy_from_logits(logits).mean().item())

        # Final position distribution.
        final_logits = logits[:, -1, :]
        final_probs = F.softmax(final_logits, dim=-1)
        topk = min(5, final_probs.size(-1))
        top5 = torch.topk(final_probs, k=topk, dim=-1).indices[0]
        top1_id = int(top5[0].item())
        top1_text = decode_token(enc, top1_id)

        final_ce, mean_true_prob = true_token_metrics(logits, tgt)

        # Final-position hidden state.
        state = states[i][:, -1, :].float()
        hidden_norm = float(state.norm(dim=-1).mean().item())

        if prev_state is None:
            state_delta = float("nan")
            relative_state_delta = float("nan")
        else:
            delta = (state - prev_state).norm(dim=-1)
            prev_norm = prev_state.norm(dim=-1).clamp_min(1e-12)
            state_delta = float(delta.mean().item())
            relative_state_delta = float((delta / prev_norm).mean().item())

        # KL divergence at the final prediction position:
        # KL(P_t || P_(t-1)), in nats.
        current_final_log_probs = F.log_softmax(final_logits, dim=-1)
        if prev_final_log_probs is None:
            kl_prev = float("nan")
        else:
            current_probs = current_final_log_probs.exp()
            kl_prev = float(
                (
                    current_probs
                    * (current_final_log_probs - prev_final_log_probs)
                )
                .sum(dim=-1)
                .mean()
                .item()
            )

        # Final-position register state.
        if registers:
            register = registers[i][:, -1, :, :].float()

            register_norm = float(
                register.reshape(register.size(0), -1)
                .norm(dim=-1)
                .mean()
                .item()
            )

            if prev_register is None:
                register_cos = float("nan")
            else:
                register_cos = cosine_flat(register, prev_register)
        else:
            register = None
            register_norm = float("nan")
            register_cos = float("nan")

        if prev_top5 is None:
            overlap = float("nan")
        else:
            a = set(int(x) for x in prev_top5.tolist())
            b = set(int(x) for x in top5.tolist())
            overlap = len(a & b) / float(topk)

        rows.append(
            {
                "loop": i + 1,
                "final_token_ce": final_ce,
                "mean_token_probability": mean_true_prob,
                "entropy_nats": entropy,
                "top1_token_id": top1_id,
                "top1_token": top1_text,
                "top5_overlap_prev": overlap,
                "hidden_state_norm": hidden_norm,
                "state_delta_l2": state_delta,
                "relative_state_delta": relative_state_delta,
                "register_norm": register_norm,
                "register_cosine_prev": register_cos,
                "kl_prev_nats": kl_prev,
            }
        )

        prev_top5 = top5.detach().cpu()
        prev_state = state.detach()
        prev_register = register.detach() if register is not None else None
        prev_final_log_probs = current_final_log_probs.detach()

    return rows


def print_table(rows: list[dict]) -> None:
    headers = [
        "loop",
        "final-CE",
        "mean-true-P",
        "entropy",
        "top-1",
        "top5-overlap",
        "||h||",
        "||Δh||",
        "rel-Δh",
        "||R||",
        "cos(Rprev)",
        "KL(Pt||Pprev)",
    ]

    rendered: list[list[str]] = []
    for r in rows:
        rendered.append(
            [
                str(r["loop"]),
                f'{r["final_token_ce"]:.6f}'
                if math.isfinite(r["final_token_ce"])
                else "nan",
                f'{r["mean_token_probability"]:.6f}'
                if math.isfinite(r["mean_token_probability"])
                else "nan",
                f'{r["entropy_nats"]:.6f}',
                f'{r["top1_token_id"]}:{r["top1_token"]!r}',
                f'{r["top5_overlap_prev"]:.3f}'
                if math.isfinite(r["top5_overlap_prev"])
                else "nan",
                f'{r["hidden_state_norm"]:.6f}',
                f'{r["state_delta_l2"]:.6f}'
                if math.isfinite(r["state_delta_l2"])
                else "nan",
                f'{r["relative_state_delta"]:.6f}'
                if math.isfinite(r["relative_state_delta"])
                else "nan",
                f'{r["register_norm"]:.6f}'
                if math.isfinite(r["register_norm"])
                else "nan",
                f'{r["register_cosine_prev"]:.6f}'
                if math.isfinite(r["register_cosine_prev"])
                else "nan",
                f'{r["kl_prev_nats"]:.6f}'
                if math.isfinite(r["kl_prev_nats"])
                else "nan",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rendered:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    print()
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rendered:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    print()


def save_results(rows: list[dict], output: Optional[str]) -> None:
    if not output:
        return

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved: {path}")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure LoopLLM recurrent-loop dynamics."
    )

    p.add_argument("--checkpoint", required=True)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--context",
        help="Open-ended context. CE/true-token probability will be NaN.",
    )
    src.add_argument(
        "--eval-text",
        help=(
            "Known text. The final token is the target; all earlier tokens "
            "are context."
        ),
    )

    p.add_argument(
        "--max-loops",
        type=int,
        default=16,
        help="Evaluate loops 1..N.",
    )

    p.add_argument(
        "--extrapolate-loops",
        action="store_true",
        help="Permit N > configured/trained loop count.",
    )

    p.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "xpu",
            "dml",
            "directml",
            "privateuseone",
            "privateuseone:0",
        ],
    )

    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--output",
        help="CSV or JSON path for results.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_loops < 1:
        raise ValueError("--max-loops must be >= 1")

    seed_everything(args.seed)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    ckpt = load_checkpoint(
        args.checkpoint,
        device=device,
        restore_rng=False,
    )

    model = ckpt["model"]
    cfg = model.cfg

    print(f"Checkpoint step: {ckpt.get('step', 'unknown')}")
    print(f"Parameters: {model.num_parameters()}")
    print(f"Configured loops: {cfg.max_loops}")
    print(f"Evaluation loops: {args.max_loops}")

    if args.max_loops > cfg.max_loops and not args.extrapolate_loops:
        raise ValueError(
            f"Requested {args.max_loops} loops, but the checkpoint is "
            f"configured for {cfg.max_loops}. Add --extrapolate-loops."
        )

    if args.max_loops > cfg.max_loops:
        print(
            "Loop extrapolation: ENABLED | loops beyond the trained ceiling "
            "reuse terminal trained loop-conditioned parameters while the "
            "shared recurrent core and causal registers continue."
        )

    enc = get_tokenizer()

    input_ids, targets, source_text = build_inputs(
        enc,
        context=args.context,
        eval_text=args.eval_text,
    )

    print(f"Input tokens: {input_ids.size(1)}")
    print(f"Input text: {source_text!r}")

    rows = evaluate(
        model,
        enc,
        input_ids,
        targets,
        args.max_loops,
        allow_loop_extrapolation=args.extrapolate_loops,
    )

    print_table(rows)
    save_results(rows, args.output)


if __name__ == "__main__":
    main()
