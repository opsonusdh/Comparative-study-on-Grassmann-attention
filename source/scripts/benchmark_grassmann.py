#!/usr/bin/env python3
"""Small controlled LM + multi-hop retrieval comparison for Grassmann attention.

This is a reproducible smoke/quality experiment, not a Wikitext benchmark.
Both variants share the same LoopLM backbone, training batches, optimizer,
steps, and evaluation set; only attention_type and its attention-specific
hyperparameters differ.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from loop_transformer import LoopConfig, LoopTransformer


N_NODES = 8
NODE_BASE = 7
HOP_BASE = NODE_BASE + N_NODES
VOCAB_SIZE = HOP_BASE + 4
SEQ_LEN = 1 + (4 * N_NODES) + 5 + 1
ANSWER_POSITION = SEQ_LEN - 3


def make_example(rng: np.random.Generator, hop: int) -> np.ndarray:
    """One random 8-node directed cycle and a query that follows `hop` edges."""
    cycle = rng.permutation(N_NODES).tolist()
    next_node = {cycle[i]: cycle[(i + 1) % N_NODES] for i in range(N_NODES)}
    start = int(rng.integers(0, N_NODES))
    target = start
    for _ in range(hop):
        target = next_node[target]

    edges = [(key, value) for key, value in next_node.items()]
    rng.shuffle(edges)
    tokens = [0]  # BOS
    for key, value in edges:
        tokens.extend((1, NODE_BASE + key, 2, NODE_BASE + value))  # MAP key TO value
    tokens.extend((3, NODE_BASE + start, HOP_BASE + hop - 1, 4, NODE_BASE + target, 5))
    if len(tokens) != SEQ_LEN:
        raise AssertionError(f"bad sample length {len(tokens)} != {SEQ_LEN}")
    return np.asarray(tokens, dtype=np.int64)


def build_model(attention_type: str, seed: int, grassmann_rank: int, offsets: tuple[int, ...]) -> LoopTransformer:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cfg = LoopConfig(
        vocab_size=VOCAB_SIZE,
        dim=32,
        n_layers=2,
        n_heads=4,
        head_dim=8,
        ffn_hidden_dim=64,
        rope_dim=8,
        max_loops=2,
        min_loops=2,
        loop_sampling=False,
        attention_type=attention_type,
        grassmann_rank=grassmann_rank,
        grassmann_offsets=offsets,
        sw_window=64,
        groups=4,
        group_dim=16,
        moe_num_shared_experts=1,
        moe_num_routed_experts=2,
        moe_top_k=1,
        moe_expert_hidden_dim=32,
        activation_dictionary="moa7",
        activation_gate_mode="sigmoid",
        activation_top_k=2,
        activation_min_probability=0.0,
        activation_balance_weight=0.0,
        moe_aux_loss_weight=0.0,
        loop_refinement_weight=0.0,
        loop_task_weight=0.0,
        dropout=0.0,
    )
    return LoopTransformer(cfg)


def evaluate(
    model: LoopTransformer,
    data: torch.Tensor,
    hops: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float | dict[str, float]]:
    model.eval()
    total_ce = 0.0
    total_correct = 0
    total_tokens = 0
    answer_correct = {h: 0 for h in range(1, 5)}
    answer_total = {h: 0 for h in range(1, 5)}
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size].to(device)
            hop_batch = hops[start : start + batch_size]
            logits = model(batch, max_loops=2)[0][-1]
            pred_logits = logits[:, :-1].contiguous()
            targets = batch[:, 1:].contiguous()
            losses = F.cross_entropy(
                pred_logits.view(-1, VOCAB_SIZE), targets.view(-1), reduction="sum"
            )
            total_ce += float(losses.item())
            predictions = pred_logits.argmax(dim=-1)
            total_correct += int((predictions == targets).sum().item())
            total_tokens += targets.numel()
            answers = predictions[:, ANSWER_POSITION]
            gold = batch[:, ANSWER_POSITION + 1]
            for i, hop in enumerate(hop_batch):
                h = int(hop)
                answer_total[h] += 1
                answer_correct[h] += int(answers[i].item() == gold[i].item())
    mean_ce = total_ce / total_tokens
    return {
        "val_ce": mean_ce,
        "val_perplexity": math.exp(min(mean_ce, 20.0)),
        "val_next_token_accuracy": total_correct / total_tokens,
        "multi_hop_answer_accuracy": {
            str(h): answer_correct[h] / max(1, answer_total[h]) for h in range(1, 5)
        },
        "multi_hop_answer_accuracy_overall": sum(answer_correct.values()) / max(1, sum(answer_total.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--grassmann-rank", type=int, default=8)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--train-examples", type=int, default=2048)
    parser.add_argument("--val-examples-per-hop", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("grassmann_benchmark_results.json"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data_rng = np.random.default_rng(args.seed)
    # Consume one common, fixed validation set for both variants.
    val_rows, val_hops_list = [], []
    for h in range(1, 5):
        for _ in range(args.val_examples_per_hop):
            val_rows.append(make_example(data_rng, h))
            val_hops_list.append(h)
    val_data = torch.from_numpy(np.stack(val_rows))
    val_hops = np.asarray(val_hops_list, dtype=np.int64)

    train_rows = []
    for _ in range(args.train_examples):
        h = int(data_rng.integers(1, 5))
        train_rows.append(make_example(data_rng, h))
    train_data = torch.from_numpy(np.stack(train_rows))

    # Both models receive bit-identical initial values for every parameter
    # outside their attention modules. This isolates the attention swap.
    reference = build_model("deepseek", args.seed, args.grassmann_rank, tuple(args.offsets))
    shared_init = {
        key: value.detach().clone()
        for key, value in reference.state_dict().items()
        if ".attn." not in key
    }
    del reference

    sampler = np.random.default_rng(args.seed + 17)
    order = sampler.integers(0, len(train_data), size=(args.steps, args.batch_size))
    batches = [train_data[idx] for idx in order]
    results = {}

    def run_fixed_batches(attention_type: str) -> dict:
        device = torch.device("cpu")
        model = build_model(attention_type, args.seed, args.grassmann_rank, tuple(args.offsets)).to(device)
        model_state = model.state_dict()
        with torch.no_grad():
            for key, value in shared_init.items():
                if key in model_state and model_state[key].shape == value.shape:
                    model_state[key].copy_(value)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        best_val = float("inf")
        best_metrics = None
        start = time.perf_counter()
        for step, fixed_batch in enumerate(batches, 1):
            model.train()
            batch = fixed_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.compute_loss(batch, max_loops=2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{attention_type} produced non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, val_data, val_hops, device)
                if metrics["val_ce"] < best_val:
                    best_val = float(metrics["val_ce"])
                    best_metrics = metrics
                print(
                    f"[{attention_type}] step {step:4d}/{args.steps} "
                    f"train_ce={float(loss.detach()):.4f} val_ce={metrics['val_ce']:.4f} "
                    f"val_acc={metrics['val_next_token_accuracy']:.4f} "
                    f"answer_acc={metrics['multi_hop_answer_accuracy_overall']:.4f}",
                    flush=True,
                )
        elapsed = time.perf_counter() - start
        final_metrics = evaluate(model, val_data, val_hops, device)
        active_attn_params = sum(p.numel() for block in model.blocks for p in block.attn.parameters())
        return {
            "attention_type": attention_type,
            "steps": args.steps,
            "seconds": elapsed,
            "total_model_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "gradient_active_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None
            ),
            "active_attention_parameters": active_attn_params,
            "best_validation": best_metrics,
            "final_validation": final_metrics,
        }
    results["deepseek"] = run_fixed_batches("deepseek")
    results["grassmann"] = run_fixed_batches("grassmann")

    payload = {
        "benchmark": "synthetic random-cycle multi-hop next-token LM",
        "sequence_length": SEQ_LEN,
        "vocabulary_size": VOCAB_SIZE,
        "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "validation_examples_per_hop": args.val_examples_per_hop,
        "answer_prediction_position": ANSWER_POSITION,
        "chance_answer_accuracy": 1.0 / N_NODES,
        "model_config": {
            "dim": 32, "layers": 2, "heads": 4, "head_dim": 8,
            "ffn_hidden_dim": 64, "loops": 2,
            "grassmann_rank": args.grassmann_rank,
            "grassmann_offsets": args.offsets,
        },
        "same_non_attention_initialization": True,
        "same_train_batches": True,
        "same_optimizer": "AdamW",
        "learning_rate": args.lr,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nRESULTS_JSON=" + str(args.output.resolve()), flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
