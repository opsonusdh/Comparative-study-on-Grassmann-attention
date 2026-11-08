from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

# Allow running directly from scripts/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_transformer.checkpointing import load_checkpoint


def resolve_device(spec: str) -> torch.device:
    """Resolve cpu/cuda/xpu/dml/auto without assuming CUDA is present."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        try:
            import torch_directml
            return torch_directml.device()
        except (ImportError, RuntimeError):
            return torch.device("cpu")

    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if spec == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU requested, but torch.xpu.is_available() is False.")
        return torch.device("xpu")

    if spec == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError(
                "DML requested, but torch-directml is not installed. "
                "Install it with: pip install torch-directml"
            ) from exc
        return torch_directml.device()

    return torch.device("cpu")


def load_model_to_device(path: Path, device: torch.device, label: str):
    """Load checkpoint on CPU first, then move the model to the target backend."""
    print(f"Loading {label} on CPU: {path}")
    ckpt = load_checkpoint(path, device="cpu")
    model = ckpt["model"].to(device)
    model.eval()
    return ckpt, model


def tensor_digest(tensor: torch.Tensor) -> str:
    """Stable digest for checkpoint-integrity diagnostics."""
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def overlay_trained_gate(base_model: torch.nn.Module, gate_model: torch.nn.Module) -> None:
    """Copy all exit_gate.* tensors from Stage-II into the matching Stage-I model."""
    base_state = base_model.state_dict()
    gate_state = gate_model.state_dict()

    gate_keys = [
        key for key in gate_state
        if key.startswith("exit_gate.")
    ]
    if not gate_keys:
        raise RuntimeError("Gate checkpoint contains no exit_gate parameters.")

    base_non_gate_digests = {
        key: tensor_digest(value)
        for key, value in base_state.items()
        if not key.startswith("exit_gate.")
    }

    copied = []
    with torch.no_grad():
        for key in gate_keys:
            if key not in base_state:
                raise RuntimeError(f"Gate tensor {key!r} is missing from the base model.")
            if base_state[key].shape != gate_state[key].shape:
                raise RuntimeError(
                    f"Gate tensor shape mismatch for {key}: "
                    f"base={tuple(base_state[key].shape)} "
                    f"gate={tuple(gate_state[key].shape)}"
                )
            base_state[key].copy_(gate_state[key].to(device=base_state[key].device))
            copied.append(key)

    # Verify that the overlay did not accidentally touch non-gate tensors.
    after_state = base_model.state_dict()
    changed_non_gate = [
        key for key, digest in base_non_gate_digests.items()
        if tensor_digest(after_state[key]) != digest
    ]
    if changed_non_gate:
        raise RuntimeError(
            "Gate overlay changed non-gate tensors: " + ", ".join(changed_non_gate[:10])
        )

    print(
        "Gate overlay: copied "
        f"{len(copied)} gate tensors; non-gate tensors unchanged."
    )


def _print_routing_debug(model, step: int) -> None:
    """Print routing diagnostics exposed by the model, if available."""
    if not hasattr(model, "get_routing_debug"):
        return
    info = model.get_routing_debug()
    if not info:
        return

    print(f"[DEBUG] routing diagnostics at generation step {step}")

    act = info.get("activation")
    if act:
        print(f"[DEBUG] activation top-k: {act.get('top_k')}")
        probs = act.get("mean_probs")
        usage = act.get("hard_usage")
        entropy = act.get("mean_entropy")
        if probs is not None:
            for i, p in enumerate(probs):
                print(f"  A{i}: mean_prob={p:.6f}")
        if usage is not None:
            for i, p in enumerate(usage):
                print(f"  A{i}: hard_usage={p:.6f}")
        if entropy is not None:
            print(f"  activation_entropy={entropy:.6f}")

    moe = info.get("moe")
    if moe:
        print(f"[DEBUG] MoE top-k: {moe.get('top_k')}")
        probs = moe.get("mean_probs")
        usage = moe.get("hard_usage")
        entropy = moe.get("mean_entropy")
        if probs is not None:
            for i, p in enumerate(probs):
                print(f"  E{i}: mean_prob={p:.6f}")
        if usage is not None:
            for i, p in enumerate(usage):
                print(f"  E{i}: hard_usage={p:.6f}")
        if entropy is not None:
            print(f"  moe_entropy={entropy:.6f}")


def _clear_routing_debug(model) -> None:
    if hasattr(model, "clear_routing_debug"):
        model.clear_routing_debug()


def select_loop(
    loop_logits: list[torch.Tensor],
    lambdas: list[torch.Tensor],
    threshold: float,
) -> tuple[torch.Tensor, int, float, list[float]]:
    """Q-exit selection matching the model's current generation semantics."""
    chosen = loop_logits[-1]
    chosen_loop = len(loop_logits)
    survival = 1.0
    cdf = 0.0
    lambda_values: list[float] = []

    for t, (lgts, lam) in enumerate(zip(loop_logits, lambdas)):
        lam_value = float(lam.mean().item())
        lambda_values.append(lam_value)
        p_t = lam_value * survival
        cdf += p_t

        if t < len(loop_logits) - 1:
            survival *= 1.0 - lam_value

        if cdf >= threshold:
            chosen = lgts
            chosen_loop = t + 1
            break

    return chosen, chosen_loop, cdf, lambda_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate text from LoopTransformer, optionally overlaying a trained "
            "Stage-II exit gate. Supports CPU/CUDA/XPU/DirectML."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Stage-I/base checkpoint")
    parser.add_argument(
        "--gate-checkpoint",
        default=None,
        help=(
            "Stage-II exit-gate checkpoint (for example exit_gate/best.pt). "
            "Only exit_gate.* tensors (projection + loop embedding) are copied into the base model."
        ),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Fixed loop count. Omit when using the trained exit gate.",
    )
    parser.add_argument(
        "--exit-threshold",
        type=float,
        default=0.8,
        help="Q-exit CDF threshold used by the trained gate.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "xpu", "dml"),
        default="auto",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed torch sampling. Omit for nondeterministic sampling where supported.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print top predictions, lambdas, and routing diagnostics at every step.",
    )
    parser.add_argument(
        "--show-loop",
        action="store_true",
        help="Print the loop selected for every generated token.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print the final generated text once instead of streaming tokens.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import tiktoken  # type: ignore

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if not (0.0 < args.exit_threshold <= 1.0):
        raise ValueError("--exit-threshold must be in (0, 1]")
    if args.loops is not None and args.loops <= 0:
        raise ValueError("--loops must be positive")
    if args.loops is None and not args.gate_checkpoint:
        print(
            "[WARNING] Adaptive generation requested without --gate-checkpoint. "
            "The gate stored in --checkpoint will be used as-is."
        )

    device = resolve_device(args.device)
    print(f"Device: {device}")

    base_path = Path(args.checkpoint)
    if not base_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {base_path}")

    # Load base model on CPU first. This is especially important for DML.
    base_ckpt, model = load_model_to_device(
        base_path,
        device,
        "Stage-I/base checkpoint",
    )

    gate_ckpt = None
    if args.gate_checkpoint:
        gate_path = Path(args.gate_checkpoint)
        if not gate_path.exists():
            raise FileNotFoundError(f"Gate checkpoint not found: {gate_path}")

        gate_ckpt, gate_model = load_model_to_device(
            gate_path,
            device,
            "Stage-II gate checkpoint",
        )

        if gate_ckpt.get("extra", {}).get("stage") != "stage2_exit_gate":
            raise ValueError(
                "--gate-checkpoint is not marked as stage2_exit_gate. "
                "Refusing to overlay an arbitrary checkpoint."
            )

        if model.cfg != gate_model.cfg:
            raise ValueError(
                "Base and gate checkpoints have different model configs. "
                "They must be architecture-compatible."
            )

        overlay_trained_gate(model, gate_model)

        del gate_model
        gate_ckpt = gate_ckpt
        print(f"Gate checkpoint step: {gate_ckpt.get('step', 'unknown')}")

    model.eval()

    print(f"Base checkpoint step: {base_ckpt.get('step', 'unknown')}")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Configured max loops: {model.cfg.max_loops}")
    print(f"Adaptive gate: {'ON' if args.loops is None else 'OFF'}")
    if args.loops is None:
        print(f"Exit threshold: {args.exit_threshold:.3f}")

    activation_top_k = getattr(model.cfg, "activation_top_k", None)
    num_activations = getattr(model.cfg, "num_activations", 4)
    moe_top_k = getattr(model.cfg, "moe_top_k", None)
    num_experts = getattr(model.cfg, "num_experts", None)
    print(
        f"Activation routing: top-k={activation_top_k}, "
        f"num_activations={num_activations}"
    )
    if moe_top_k is not None:
        print(f"MoE routing: top-k={moe_top_k}, num_experts={num_experts}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            try:
                torch.xpu.manual_seed_all(args.seed)
            except AttributeError:
                pass
        print(f"Sampling seed: {args.seed}")

    tokenizer = tiktoken.get_encoding("p50k_base")
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt produced zero tokens.")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"Prompt tokens: {idx.shape[1]}")
    print("Prompt IDs:", idx[0].tolist())
    print("Decoded:", tokenizer.decode(idx[0].tolist()))
    print("\n--- GENERATION ---\n")

    generated_ids: list[int] = []
    selected_loops: list[int] = []
    selected_cdfs: list[float] = []

    with torch.no_grad():
        for step in range(args.max_new_tokens):
            _clear_routing_debug(model)

            run_loops = args.loops if args.loops is not None else model.cfg.max_loops
            loop_logits, lambdas = model.forward(idx, max_loops=run_loops)

            if args.loops is not None:
                chosen = loop_logits[-1]
                chosen_loop = len(loop_logits)
                cdf = 1.0
                lambda_values = [float(x.mean().item()) for x in lambdas]
            else:
                chosen, chosen_loop, cdf, lambda_values = select_loop(
                    loop_logits,
                    lambdas,
                    args.exit_threshold,
                )

            selected_loops.append(chosen_loop)
            selected_cdfs.append(cdf)

            logits = chosen[:, -1, :]
            scaled_logits = logits / max(args.temperature, 1e-6)

            if args.debug:
                probs = F.softmax(scaled_logits, dim=-1)
                debug_k = min(20, probs.size(-1))
                top_probs, top_ids = torch.topk(probs, debug_k, dim=-1)

                print(
                    f"\n[DEBUG] generation step {step + 1}/{args.max_new_tokens}"
                )
                print(f"[DEBUG] sequence length: {idx.shape[1]}")
                print(f"[DEBUG] loops used: {chosen_loop}")
                print(f"[DEBUG] Q-exit CDF: {cdf:.6f}")
                print(
                    "[DEBUG] lambdas: "
                    + ", ".join(
                        f"L{i + 1}={value:.6f}"
                        for i, value in enumerate(lambda_values)
                    )
                )
                _print_routing_debug(model, step + 1)
                print("[DEBUG] top predictions:")
                for rank, (token_id, prob) in enumerate(
                    zip(top_ids[0].tolist(), top_probs[0].tolist()),
                    start=1,
                ):
                    token_text = tokenizer.decode([token_id])
                    print(
                        f"  {rank:2d}. id={token_id:5d} "
                        f"prob={prob:.6f} token={token_text!r}"
                    )

            if args.top_k > 0:
                k = min(args.top_k, scaled_logits.size(-1))
                values, _ = torch.topk(scaled_logits, k, dim=-1)
                threshold = values[:, [-1]]
                scaled_logits = torch.where(
                    scaled_logits < threshold,
                    torch.full_like(scaled_logits, float("-inf")),
                    scaled_logits,
                )

            probs = F.softmax(scaled_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            token_id = int(next_id.item())
            generated_ids.append(token_id)
            token_text = tokenizer.decode([token_id])

            if args.show_loop:
                print(
                    f"\n[loop={chosen_loop} cdf={cdf:.3f}] ",
                    end="",
                    flush=True,
                )

            if args.no_stream:
                continue

            print(token_text, end="", flush=True)

    if args.no_stream:
        print(tokenizer.decode(generated_ids), end="")

    print("\n")
    mean_depth = sum(selected_loops) / len(selected_loops)
    print("--- GENERATION SUMMARY ---")
    print(f"Tokens generated: {len(generated_ids)}")
    print(f"Mean loops used: {mean_depth:.3f} / {model.cfg.max_loops}")
    print(
        f"Estimated recurrent compute saved: "
        f"{100.0 * (1.0 - mean_depth / model.cfg.max_loops):.2f}%"
    )
    for loop_number in range(1, model.cfg.max_loops + 1):
        count = sum(1 for value in selected_loops if value == loop_number)
        rate = 100.0 * count / len(selected_loops)
        print(f"Loop {loop_number}: {count:4d} tokens ({rate:6.2f}%)")


if __name__ == "__main__":
    main()
