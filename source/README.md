# LoopLLM Recurrent: MoA7 + RFC Production Training Package

This package contains one canonical recurrent training entrypoint, `train.py`.
The model is intentionally configured entirely from command-line arguments. There
is no second RFC trainer and no architecture-specific JSON preset required for
training.

## Active architecture

The active model contains only mechanisms kept in the production research path:

- shared-weight recurrent Transformer loops;
- Sliding Window Attention, Compressed Sparse Attention (CSA), and Heavy Compressed Attention (HCA);
- MLP Attention Residuals (AttnRes), local to each physical pass;
- DeepSeekMoE-style shared + routed experts;
- MoA7 seven-activation token/state-dependent routing;
- causal register memory;
- loop controller with a bounded `tanh` safety envelope;
- optional loop positional embeddings and loop transitions;
- ExitGate / adaptive-exit support;
- recurrent refinement, supervision, monotonic, and task objectives.

Removed from the active package: MSA attention, token-adaptive-depth experiment,
loop-clock experiment, fixed basis-cycling experiment, legacy experimental memory
classes, alternate model-v2 file, and diffusion training code. Removed historical
material is not required to train the model.

## Grassmann attention modes

The canonical trainer exposes three causal Grassmann variants alongside the unchanged DeepSeek baseline:

- `--attention-type grassmann`: unchanged legacy local offsets, including lag 16: `[1, 2, 4, 8, 12, 16]`.
- `--attention-type grassmann-new --grassmann-offset-count 4`: progressive per-loop local offsets plus an adaptive sparse-global tail. The global stride is calculated from `N`, `max_loops`, and the runtime sequence length.
- `--attention-type real-grassmann`: fixed local offsets plus regularly spaced global look-backs. Its default stride is 16; set `--grassmann-long-range-stride 32` for stride 32.

All variants use previous-token pairs `(z_t, z_{t-Delta})`. Repeated 16-token sampling and the adaptive global tail are project extensions, not features specified in arXiv:2512.19428v1. See [GRASSMANN_OFFSET_MODES.md](GRASSMANN_OFFSET_MODES.md) for exact schedules, complexity implications, A/B experiment commands, and checkpoint safeguards.

## Data format

Training consumes pre-tokenized flat `.bin` files through `numpy.memmap`.
Use either:

1. `--train-data` plus `--val-data`, or
2. `--data` plus `--val-fraction` for a chronological split without copying the
   corpus.

The bundled TinyShakespeare data are `data/tinyshakespeare.bin` and
`data/tinyshakespeare.val.bin`, with `data/tinyshakespeare.vocab.json`.

## Exact RFC-style TinyShakespeare command

This is the corrected RFC configuration used for the production run. The main
change from the previously bundled research checkpoint is the bounded controller.

```bash
python train.py \
  --train-data data/tinyshakespeare.bin \
  --val-data data/tinyshakespeare.val.bin \
  --data-dtype uint16 \
  --vocab-size 65 \
  --dim 64 \
  --n-layers 3 \
  --n-heads 4 \
  --head-dim 16 \
  --ffn-hidden-dim 256 \
  --rope-dim 16 \
  --max-loops 4 \
  --min-loops 4 \
  --no-loop-sampling \
  --loop-refinement-weight 0.02 \
  --loop-refinement-margin 0.001 \
  --residual-loop-embedding \
  --no-loop-positional-embedding \
  --causal-register-bank \
  --causal-registers 4 \
  --causal-register-update-scale 0.10 \
  --no-loop-transition-matrices \
  --loop-controller \
  --loop-controller-bound 0.20 \
  --sw-window 32 \
  --csa-m 2 \
  --csa-top-k 4 \
  --csa-aux-loss-weight 0.0 \
  --hca-m-prime 8 \
  --groups 8 \
  --group-dim 1024 \
  --moe-num-shared-experts 1 \
  --moe-num-routed-experts 2 \
  --moe-top-k 1 \
  --moe-expert-hidden-dim 128 \
  --activation-dictionary moa7 \
  --activation-gate-mode sigmoid \
  --activation-top-k 7 \
  --activation-min-probability 0.0 \
  --activation-balance-weight 0.0 \
  --activation-bias-update-speed 0.0 \
  --moe-aux-loss-weight 0.0 \
  --moe-bias-update-speed 0.0 \
  --no-expert-local-activation-routing \
  --tie-embeddings \
  --dropout 0.0 \
  --no-grad-checkpointing \
  --batch-size 2 \
  --seq-len 32 \
  --max-steps 800 \
  --lr 0.002 \
  --min-lr-ratio 0.05 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight-decay 0.01 \
  --warmup-steps 100 \
  --grad-accum-steps 1 \
  --grad-clip 1.0 \
  --eval-interval 50 \
  --eval-iters 32 \
  --checkpoint-interval 100 \
  --seed 123 \
  --device cuda \
  --dtype float32 \
  --no-amp \
  --checkpoint-dir checkpoints/rfc_seed123 \
  --viz-dir visualizations/rfc_seed123 \
  --viz-interval 100 \
  --viz-probe-size 2
```

The cosine scheduler matches the former RFC trainer: after warmup it decays to
`5%` of the base learning rate at the final step.

## Colab

```bash
pip install -r requirements.txt
python train.py --help
```

For Google Drive output, point `--checkpoint-dir` and `--viz-dir` at a mounted
Drive directory. Training can be resumed from `latest.pt`:

```bash
python train.py \
  --resume /content/drive/MyDrive/loopllm/checkpoints/rfc_seed123/latest.pt \
  --max-steps 20000 \
  --checkpoint-dir /content/drive/MyDrive/loopllm/checkpoints/rfc_seed123 \
  --viz-dir /content/drive/MyDrive/loopllm/visualizations/rfc_seed123
```

On resume, model architecture is taken from the checkpoint so its parameter
shape cannot silently change halfway through an experiment.

## Visualization

Set `--viz-interval N` to save a complete image every N optimizer steps.
The output directory contains:

- `step_XXXXXXXX.png`: deterministic mechanistic dashboard;
- `latest.png`: most recent dashboard;
- `latest.json`: raw per-loop telemetry;
- `history.jsonl`: training/evaluation history;
- `probe.pt`: the fixed probe batch used for mechanistic snapshots.

The dashboard includes final and per-loop CE/accuracy, hidden-state norms and
cosines, controller norms, register norms, and routing summaries. State growth
above 100x is explicitly labelled as a health warning.

## Experimental Grassmann attention

The original attention stack remains the default. To replace only the attention
branch with the causal Grassmann/Pluecker mixer from the linked paper:

```bash
python train.py \
  --attention-type grassmann \
  --grassmann-rank 32 \
  --grassmann-offsets 1 2 4 8 12 16 \
  --grassmann-positional-max-len 1024
```

For autoregressive language modeling, this implementation pairs the current
projected token with previous positions (`z[t]` and `z[t-delta]`). The paper's
printed `z[t+delta]` pairing accesses future tokens despite being described as
causal, so it is intentionally corrected here. Grassmann mode also adds the
paper's learned absolute token-position table once at the input embeddings;
this is separate from the optional recurrent-loop positional embedding. The
default `deepseek` path,
outer residual, FFN/MoE, recurrent loop, optimizer, checkpointing, and Kaggle
Google Drive restoration/synchronization are otherwise preserved. See
`GRASSMANN_EXPERIMENT.md` for implementation details and the small synthetic
multi-hop comparison; `scripts/benchmark_grassmann.py` reproduces that test.
For a fresh two-GPU Kaggle run, use `KAGGLE_GRASSMANN_COMMAND.txt`; it selects
Grassmann explicitly, uses the learned token-position table, starts at 3e-5 LR,
and writes to a separate checkpoint directory rather than resuming DeepSeek.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The cleaned package's release tests cover causality, attention-family selection,
retired-code removal, `.bin` handling, checkpoint RNG restoration,
visualization, and gradient-checkpointed backward passes.

## Kaggle execution

Use `--source Kaggle` to run the same trainer on Kaggle without rewriting the
existing Colab-style command. The trainer discovers the attached `train.bin` and
`train.val.bin` under `/kaggle/input`, maps legacy `/content/drive/MyDrive/loop_llm`
paths into `/kaggle/working/loop_llm`, restores the full persistent Drive tree
before loading `latest.pt`, and starts one background synchronization worker.

The worker requests a synchronization every 1000 optimizer steps and mirrors
new/changed files plus intentional deletions. Checkpoints are atomically written
by the existing checkpointing code; large atomic files such as `.pt` checkpoints
are snapshotted with a hard link when possible so training can continue while the
background upload reads a stable inode. Append-style logs/JSONL files are copied
to a staging area before upload.

Kaggle authentication uses `kaggle_secrets.UserSecretsClient` with the secret
label `GDRIVE_CREDENTIALS_JSON`. See `KAGGLE_SETUP.md` and
`scripts/create_drive_credentials.py`.


## Grassmann positional ablation

For the matched 5,000-step comparison, use the two command files:

- `GRASSMANN_ABLATION_A_NO_ABSOLUTE_POSITIONS.txt`
- `GRASSMANN_ABLATION_B_LEARNED_POSITIONS.txt`

See `GRASSMANN_POSITION_ABLATION_README.md` for the controlled settings, output paths, positional-RNG isolation, visualization labels, and Kaggle restore/persistence notes. The CLI supports `--grassmann-positional-embedding` / `--no-grassmann-positional-embedding`, `--grassmann-positional-seed`, and `--viz-label`.


### Distributed Kaggle Drive-restore timeout

With `torchrun` and `--source Kaggle`, rank 0 restores the persistent Drive tree
while the other ranks wait for a startup status broadcast. Large trees can take
longer than the normal 120-second DDP collective timeout. The trainer now uses
a 30-minute timeout for this startup phase only, then resets the process group
to 120 seconds before model construction and training. Override the restore
window with `LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS` when needed (minimum
120 seconds). The restore still runs once on rank 0; NCCL's socket fallback
messages about unavailable optional plugins are informational when NCCL reports
`Init COMPLETE` on every rank.

## Strict checkpoint resume

For an exact continuation, pass `--resume /path/to/checkpoints/latest.pt`. The
trainer validates the saved global step, AdamW moment state, and LR schedule
metadata before running. It refuses to silently start a new optimizer or warm-up
when resume state is incomplete. Add `--require-resume` to make a missing
`--resume` argument a CLI error. Ordinary resume preserves the saved LR schedule;
use `--force-lr` only when deliberately starting a new cosine schedule. See
`RESUME_SAFETY.md` for details.

## DeepSeek loop-split SWA / CSA / HCA experiment

The experimental `--attention-type deepseek-split` mode preserves the original
DeepSeek attention family routing: layers 0-1 use Sliding Window Attention
(SWA), even layers from 2 onward use Compressed Sparse Attention (CSA), and
odd layers from 3 onward use Heavily Compressed Attention (HCA).

The split is applied inside the existing mechanisms, not by replacing them
with a fourth attention implementation. SWA still attends to every causally
valid token in its configured `--sw-window`. CSA and HCA also retain their
entire raw-token local windows on every recurrent pass. Only their compressed
long-range memory blocks are assigned to loops. For each query, compressed
blocks are ranked newest-to-oldest; rank 1 is shared, and older ranks follow
`{1} union {stride*n + (loop_idx mod stride)}`. CSA still runs its learned
indexer and top-k selection over its assigned compressed blocks; HCA attends
to all assigned compressed blocks. The four loop phases therefore cover the
complete compressed long-range memory without dropping the local detail path.

With `--deepseek-split-stride 4`, the compressed-block ranks are:

- loop 0: 1, 4, 8, 12, ...
- loop 1: 1, 5, 9, 13, ...
- loop 2: 1, 2, 6, 10, ...
- loop 3: 1, 3, 7, 11, ...
- loop 4: repeats loop 0

The rank is measured in compressed-block slots, not raw token offsets. This
keeps the split balanced even when CSA/HCA compression factors are multiples
of the loop stride. Use `--sw-window 200` to retain a full 200-token causal
local window in SWA and in the raw-local branches of CSA/HCA.

The mode adds no attention parameters and preserves the baseline parameter
shapes/state-dict keys; it changes only which compressed long-range candidates
are considered on each recurrent pass. Use a separate checkpoint directory
for a clean experiment. The original `deepseek`, all Grassmann modes, MoE/RFC,
loop controllers, training objectives, and data pipeline remain unchanged.
For the two-GPU Kaggle command, use `KAGGLE_DEEPSEEK_SPLIT_COMMAND.txt`.
