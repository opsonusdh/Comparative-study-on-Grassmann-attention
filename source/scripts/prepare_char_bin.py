#!/usr/bin/env python3
"""Pack a UTF-8 character corpus into LoopLLM's zero-copy binary format.

The character vocabulary is sorted deterministically. The output is a flat
uint16/uint32 array of token IDs suitable for the canonical train.py trainer.
A chronological validation split and a JSON vocabulary sidecar are emitted.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--dtype", choices=["uint16", "uint32"], default="uint16")
    args = ap.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    text = args.input.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = np.fromiter((stoi[ch] for ch in text), dtype=np.uint16 if args.dtype == "uint16" else np.uint32)
    cut = int(len(ids) * (1.0 - args.val_fraction))
    if cut <= 0 or cut >= len(ids):
        raise ValueError("corpus is too small for the requested validation split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ids[:cut].tofile(args.output)
    val_path = args.output.with_name(args.output.stem + ".val" + args.output.suffix)
    ids[cut:].tofile(val_path)
    vocab_path = args.output.with_name(args.output.stem + ".vocab.json")
    vocab_path.write_text(
        json.dumps({"stoi": stoi, "itos": {str(i): ch for ch, i in stoi.items()}}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"characters={len(chars):,} tokens={len(ids):,} train={cut:,} val={len(ids)-cut:,}")
    print(f"train_bin={args.output}")
    print(f"val_bin={val_path}")
    print(f"vocab_json={vocab_path}")


if __name__ == "__main__":
    main()
