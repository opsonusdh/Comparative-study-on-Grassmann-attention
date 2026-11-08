# Release verification

## Source baseline

Built from the newly supplied `loopllm_moa7(1).zip` only. Previous generated
Kaggle/Drive packages were not used as source material.

The original non-generated source files were compared byte-for-byte. There are
no missing files and no unintended changes outside the intended integration
files:

- `train.py`
- `README.md`
- `VERSION.txt`
- `drive_sync.py` (new)
- `KAGGLE_SETUP.md` (new)
- `KAGGLE_COMMAND.txt` (new)
- `KAGGLE_REQUIREMENTS.txt` (new)
- `scripts/create_drive_credentials.py` (new)
- `tests/test_kaggle_drive_sync.py` (new)

Python bytecode caches are excluded from the release archive.

## Automated verification

- Python compilation: PASS
- Existing project tests + Drive-sync tests: **18/18 PASS**
- Exact user training command parsing with `--source Kaggle`: PASS
- Case-insensitive `Kaggle` source value: PASS
- `/content/drive/MyDrive/train.bin` -> Kaggle input discovery: PASS
- `/content/drive/MyDrive/train.val.bin` -> Kaggle input discovery: PASS
- `/content/drive/MyDrive/loop_llm/...` -> `/kaggle/working/loop_llm/...`: PASS
- Legacy checkpoint-embedded Colab data paths re-normalized after resume: PASS
- Full remote-tree restore behavior: PASS (fake Drive integration test)
- New/changed/deleted file synchronization: PASS (fake Drive integration test)
- Background worker and final-sync shutdown: PASS
- Upload failure does not falsely commit a file as synchronized: PASS by implementation/test coverage
- Drive sync cadence independent of local checkpoint cadence: PASS by implementation review

## Design checks

The trainer keeps the existing optimization/model recipe intact. `--source Kaggle`
is the only new CLI mode exposed to the training command.

The Kaggle backend restores the complete `loop_llm` working tree before a resume
checkpoint is loaded, then mirrors only changed/new files every 1000 optimizer
steps in one background worker. Atomic PyTorch checkpoint writes are preserved;
large atomic artifacts can be hard-linked into the upload staging area so an
in-flight upload does not read a subsequently replaced checkpoint inode. JSONL
and similar append-style files are copied to staging before upload.

## External verification limitation

A live Google Drive upload cannot be exercised in this build environment because
it does not have the user's Kaggle Secret or Google OAuth session. The Drive API
and PyDrive2 behaviors used by the implementation were checked against their
current documentation, and the Drive operations were exercised through a fake
Drive implementation in the automated tests.


## Current Grassmann ablation release verification

- Full project test suite: **48/48 PASS**.
- Python compilation: PASS.
- Both A/B Kaggle command files: shell syntax PASS and full CLI parse PASS.
- A/B argument parity: PASS; only the positional-table mode and run/output labels differ.
- Shared parameter initialization: bit-identical between A/B at the same global seed; the optional position table uses an isolated CPU RNG.
- One-step CPU end-to-end smoke for both modes: PASS, including checkpoint write, five-example probe tensor, latest metrics JSON, and visualization PNG.
- The two runs write outside the Kaggle Drive restore target to avoid the second restore deleting the first run's local outputs.
- A live Kaggle T4/NCCL production run and live Google Drive upload were not available in this build environment.


## Kaggle DDP Drive-restore timeout fix

- Root cause reproduced from user log: rank 1 waited in `broadcast_object_list`
  while rank 0 restored 167 Drive files; the process group had a 120-second
  timeout and aborted before rank 0 could broadcast completion. NCCL init itself
  completed successfully on both T4s.
- Kaggle DDP startup now initializes with a configurable 1800-second restore
  timeout (`LOOPLLM_DRIVE_RESTORE_TIMEOUT_SECONDS`) and resets the group to
  120 seconds immediately after the restore-status broadcast.
- On PyTorch versions exposing `torch.distributed.set_timeout`, that API is used;
  older builds fall back to `distributed_c10d._set_pg_timeout`. If neither is
  available, the longer timeout remains in effect and a warning is printed.
- Distributed timeout unit tests: PASS.
- Live Gloo process-group timeout change/reset: PASS.
- Two-rank delayed-broadcast smoke test: PASS.
- Full test suite at the DDP timeout release: 52/52 PASS.
- `train.py --help` and Python compilation: PASS.
- Live Kaggle NCCL + Google Drive restore has not been exercised in this
  environment because Kaggle GPUs and the user's Google credentials are not
  available here.


## Grassmann offset modes

- `grassmann` retains its previous fixed-offset path.
- `grassmann-new` resolves offsets from the recurrent loop index and configurable offset count/spacing base.
- `real-grassmann` resolves local offsets plus stride-spaced multiples strictly below the runtime sequence length.
- CPU tests cover schedules, CLI parsing, parameter-structure equality, long-range gradients, future-token causality, and visualization labels.
- New full test suite: **66/66 PASS**.
- `train.py --help`: all three Grassmann modes and their hyperparameters are present.
- Both generated Kaggle command files parse through the actual CLI and preserve the matched 5,000-step / batch-4 / seed-2009 experiment settings.
- One-step CPU forward/backward/AdamW update with gradient checkpointing passed for all three Grassmann modes.
- New/real mode checkpoint round-trips passed; resume guard rejects changing progressive offset count mid-run.
- The original `deepseek` and `grassmann` modes were instantiated with the same seed in the pre-change and updated trees: state-dict hashes, parameter counts, and tensor counts match exactly for both modes.
- Real-Grassmann's 301-position offset set is `[1,2,4,8,12,16,32,64,96,128,160,192,224,256,288]`; future-token gradient checks pass.
- Live Kaggle T4 throughput/VRAM and live Drive restoration were not available in this environment.

## Adaptive sparse-global offsets and corrected stride-16 A/B update

This section supersedes the earlier 66/66 offset-mode test count above.

- Full project test suite after the adaptive-offset update: **78/78 PASS**.
- Python compilation: PASS for `train.py`, all `src/loop_transformer/*.py`, and all test modules.
- `grassmann-new` exact schedule tests: PASS for N=4, N=5, and N=10 at T=301; tested boundary cases include T=1, short context, N=1, and N much larger than the context.
- Adaptive rule verified exactly: `C=T-1`; `L_max=(N-1)*(spacing_base+max_loops-1)`; `S=max(L_max+1,ceil(C/N))`; global lags are positive multiples of S not exceeding C.
- Long-range gradient and future-token causality tests: PASS for adaptive `grassmann-new` and fixed stride-16 `real-grassmann`.
- Legacy `grassmann` still returns precisely `[1,2,4,8,12,16]` with no additional multiples.
- Legacy-checkpoint compatibility: a simulated pre-adaptive `grassmann-new` checkpoint loads with its original local-only schedule; the trainer rejects resuming it under the new adaptive implementation instead of silently changing the attention geometry.
- New `grassmann-new` checkpoint round trips retain the `adaptive_sparse_global_v1` schedule marker. Resume validation includes N, spacing base, and `max_loops`.
- `real-grassmann` default stride is 16; explicit stride 32 remains supported. The corrected A/B run files explicitly set stride 16 and use new `_stride16` checkpoint/visualization directories.
- The N=4 and N=5 adaptive commands and the A/B stride-16 commands parse through the real argument parser. The explicit stride-32 command also parses. All five Kaggle command files pass `bash -n` after stripping the notebook `!` prefix.
- CPU full-trainer one-step smoke tests passed for `grassmann-new` N=4 and `real-grassmann` stride 16, both with and without gradient checkpointing. Each completed forward/backward, optimizer step, evaluation, and checkpoint write.
- Additional CPU smoke tests passed forward/backward, finite-gradient checks, AdamW update, and checkpoint round-trip for the legacy, adaptive, and fixed-stride modes.
- Model state-dict hashes and parameter counts for `deepseek` and legacy `grassmann` are identical between the pre-update source tree and this release under the same seed.

A live Kaggle two-T4/NCCL run, peak-VRAM measurement at the 301-position context, and live Google Drive restore/upload were not available in this environment. The fixed stride-16 A/B mode evaluates 23 unique look-back offsets at T=301, so it is more expensive than the six-offset historical configuration. Run a short GPU pilot and inspect peak VRAM/tokens-per-second before committing to the full 5,000-step A/B jobs. The N=4 adaptive mode has 8 offsets per recurrent pass at T=301 under the current four-loop configuration.

## Strict resume-state and LR-schedule guard

- A `--resume` checkpoint must contain a positive completed optimizer step, a
  non-empty AdamW state, saved schedule arguments, LR schedule metadata, and a
  valid checkpoint LR. Missing state now aborts before training.
- Unsafe optimizer-state migration now aborts instead of silently continuing
  with fresh AdamW moments.
- Ordinary resume rejects conflicting explicit LR, warm-up, minimum-LR-ratio,
  or schedule-horizon overrides. `--force-lr` remains the deliberate path for
  starting a new cosine schedule at the resume step.
- Added `--require-resume` to prevent an accidentally omitted `--resume` flag;
  it fails during CLI parsing rather than launching a fresh run.
- Startup now reports the checkpoint's completed step, next optimizer step,
  total horizon, and restored optimizer LR.
- Full project tests after this change: **87/87 PASS**.
- Python compilation and `train.py --help`: PASS.
- Live Kaggle T4 resume against the user's Google Drive checkpoint has not been
  run in this environment; the checkpoint contents are validated on Kaggle
  before any training step begins.


## 2026-11-05 — strict resume and visualizer continuity

- Resume guard: requires model/config, completed optimizer step, AdamW moment state, saved LR arguments, and LR metadata; rejects incomplete state rather than silently starting fresh.
- Visualizer resume: loads the existing `history.jsonl`, truncates any entries beyond the last durable checkpoint, restores the exact saved `probe.pt`, and requires a training record at the checkpoint boundary.
- Training resumes at `checkpoint_step + 1`; visualization snapshots continue on the configured interval (for a step-600 checkpoint and interval 100, the next periodic image is step 700).
- Kaggle DeepSeek resume command now writes checkpoint and visualization files into the same restored canonical tree under `/kaggle/working/loop_llm/deepseek_baseline_runs/DeepSeek_SWA_CSA_HCA/`.
- Added tests for history continuity, future-point truncation, saved-probe restoration, and missing-boundary rejection.
- Full project test suite: **89/89 PASS**.
