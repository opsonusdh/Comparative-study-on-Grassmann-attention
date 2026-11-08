# Grassmann positional-embedding ablation (two matched 5,000-step runs)

Run the two command files **separately**, not concurrently, on the same Kaggle runtime/dataset.

- `GRASSMANN_ABLATION_A_NO_ABSOLUTE_POSITIONS.txt`: Grassmann mixer with no learned absolute token-position table.
- `GRASSMANN_ABLATION_B_LEARNED_POSITIONS.txt`: same Grassmann mixer with the paper's learned absolute token-position table.

## Controlled settings

Both commands use the same `--seed 2009`, `--grassmann-positional-seed 2009`, data files, batch size, sequence length, 8x512 architecture, MoE settings, four recurrent loops, rank 32, offsets, optimizer, LR schedule, evaluation batches, and 5,000-step budget. The only intended model-architecture difference is whether the Grassmann absolute-position table exists. Its initialization uses an isolated CPU RNG context, so it does not consume/shift the RNG stream used to initialize shared model parameters. A regression test checks exact equality of all shared initialized parameters between A and B.

The learned table has 301 rows because the training input tensors contain `seq_len + 1` tokens (300 + 1).

## Learning-rate schedule

These are fresh runs and intentionally use `--lr 3e-5 --warmup-steps 200`. Do **not** add `--force-lr`: in this trainer, `--force-lr` deliberately bypasses warmup and starts a forced cosine schedule at step zero.

## Output isolation

The runs write to separate checkpoint and visualization folders below `/kaggle/working/grassmann_ablation_runs/`, **outside** `/kaggle/working/loop_llm`. This is deliberate: each Kaggle invocation's Drive restore atomically replaces the entire `/kaggle/working/loop_llm` tree. Keeping experiment outputs outside that tree prevents launch B from deleting launch A's outputs. Neither command uses `--resume`, so neither can silently load the previous DeepSeek or Grassmann checkpoint. Run A and then B in the same Kaggle runtime session if you want both local output folders available together.

**Drive persistence note:** in this trainer, `--source Kaggle` restores from Drive, while `--drive-save-interval` only creates an additional local checkpoint boundary. It does not automatically upload the output folders to Google Drive. Before the Kaggle session ends, manually download or copy the two output folders somewhere persistent. Do not assume the restore flag also enables upload.

## Visualization

Each dashboard has a custom experiment title and an automatically generated subtitle naming the causal multi-scale Pluecker mixer, rank, offsets and positional mode. `--viz-probe-size 5` now really means five examples, even though the training global batch is four; the probe is evaluated on rank 0 without gradients. The headline validation metrics and per-loop language-model metrics continue to come from the full fixed validation batches. State/controller/register diagnostics use the five-example probe.

## Caveat

This is a controlled two-condition ablation, not a complete statistical study. For stronger evidence, repeat both conditions with additional matched seeds after these first 5,000-step runs. The already-trained DeepSeek run can remain the external reference; no new DeepSeek training is included.
