# Safe checkpoint resumption

The trainer now treats `--resume` as a full training-state restore, not a
model-weight warm start.

## Required checkpoint contents

A resumable checkpoint must contain:

- a positive completed optimizer step (`step >= 1`);
- the model configuration and model state;
- a non-empty AdamW optimizer state, including parameter groups and moment
  tensors;
- saved training arguments (`lr`, `warmup_steps`, `min_lr_ratio`, and
  `max_steps`);
- LR schedule metadata and the LR recorded at checkpoint time.

If any of these are missing or inconsistent, startup stops before the first
training step. The trainer will not quietly replace AdamW with a new optimizer
or infer a schedule from parser defaults.

## Schedule behavior

A normal resume keeps the original global-step warm-up/cosine schedule. For
example, resuming a 5,000-step run saved at step 600 continues at step 601 and
uses the LR calculated for global step 601. It does not warm up again.

An explicit change to `--lr`, `--warmup-steps`, `--min-lr-ratio`, or
`--max-steps` is rejected during an ordinary resume when it conflicts with the
checkpoint. Use `--force-lr` only when intentionally starting a new cosine
schedule from the checkpoint step.

## Preventing an accidental fresh run

Add both flags to a resume launch:

```bash
--resume /path/to/checkpoints/latest.pt \
--require-resume
```

`--require-resume` makes the CLI reject a command that omitted `--resume`.
A successful resume prints both the accepted checkpoint's completed step and
the next optimizer step, followed by confirmation that AdamW state and its
saved LR were restored. A fresh run prints an explicit `[Resume] No --resume
checkpoint supplied` message.

## Visualizer continuity

For Kaggle resume runs, keep `--viz-dir` pointed at the restored experiment's
canonical `visualizations/` folder under `/kaggle/working/loop_llm`. The
visualizer loads `history.jsonl`, restores the saved `probe.pt`, and requires a
training-history record at the checkpoint boundary. History points later than
the checkpoint are discarded before appending, preventing uncheckpointed
future points from being plotted. New training records then continue at the
next optimizer step (for a step-600 checkpoint, step 601); snapshots are
written at the configured interval (the next periodic snapshot is step 700
when `--viz-interval 100`).
