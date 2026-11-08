# Grassmann attention experiment

## Scope and causal correction

Baseline: `loopllm_moa7_drive_restore_fixed.zip`. The existing recurrent model, feed-forward/MoE blocks, optimizer, data handling, checkpointing, visualization, Kaggle/Google Drive restoration and synchronization paths were retained. The default remains `--attention-type deepseek`. The only model computation replaced by the experimental option is the attention branch.

Paper: [Attention Is Not What You Need: Grassmann Flows as an Attention-Free Alternative for Sequence Modeling](https://arxiv.org/html/2512.19428v1), §3.2. The paper describes pairs `(z_t, z_{t+Δ})` and calls them causal. That is a future-token dependency for left-to-right next-token modeling. This implementation corrects it to `(z_t, z_{t-Δ})` for every positive offset. It never reads a position later than `t`.

For each offset, the module computes the Plücker coordinates

`p_ij(t, Δ) = z_t,i * z_(t-Δ),j - z_t,j * z_(t-Δ),i`,

normalizes the Plücker vector, projects it back to model width, averages valid offsets, and uses the paper's gated fusion followed by LayerNorm. Since this LoopLM's residual is owned outside the attention module, the attention module returns `LayerNorm(mixed) - x` as a branch delta. No outer residual, FFN, recurrent loop, router, or training objective was reworked. Positions without a valid previous pair return a zero attention delta.

## Automated tests

- Python syntax compilation: PASS.
- Full project test suite: 29/29 PASS.
- Grassmann unit tests verify output shape, finite gradients (fp32/fp16/bf16), first-token zero boundary, explicit pair masking, and that changing future hidden states does not change earlier outputs.
- Full-model next-token causality test: PASS.
- Training CLI accepts `--attention-type grassmann`, `--grassmann-rank`, and `--grassmann-offsets`.

## Small language-model comparison

This is a controlled synthetic smoke/quality test, not a replication of the paper's WikiText-2 results. Each model had 32-dimensional hidden states, 2 physical layers, 4 heads in the baseline, 2 recurrent loops, FFN width 64, 19-token vocabulary, and sequence length 39. The Grassmann module used rank 8 and offsets `{1, 2, 4, 8}`. Both models used AdamW at `2e-3`, 100 optimizer steps, batch size 32, identical training batches, and bit-identical initialization for all non-attention parameters. The validation set contained 1,024 examples, balanced across 1-, 2-, 3-, and 4-hop queries. Two seeds (1234 and 1235) completed; a third attempted run did not complete and is excluded from the reported statistics.

The task encodes a random directed 8-node cycle as shuffled `MAP key TO value` records, then asks the model to follow 1–4 links from a query node. Answer chance is 12.5%. The answer accuracy is the exact token accuracy at the answer position; next-token accuracy and CE/PPL are over all shifted sequence positions.

| Metric (mean ± sample SD, 2 seeds) | Original attention | Grassmann attention |
|---|---:|---:|
| Validation CE ↓ | 1.1606 ± 0.0388 | 1.1310 ± 0.0119 |
| Validation perplexity ↓ | 3.193 ± 0.124 | 3.099 ± 0.037 |
| Next-token accuracy ↑ | 0.5595 ± 0.0008 | 0.5399 ± 0.0014 |
| Multi-hop answer accuracy ↑ | 0.1323 ± 0.0062 | 0.1284 ± 0.0007 |
| Training wall time (s) ↓, CPU | 11.31 ± 0.00 | 10.18 ± 0.01 |
| Active attention parameters | 8512 | 6672 |

### Answer accuracy by hop count

| Hops | Original attention | Grassmann attention | Chance |
|---:|---:|---:|---:|
| 1 | 0.1484 ± 0.0331 | 0.1055 ± 0.0166 | 0.1250 |
| 2 | 0.1309 ± 0.0249 | 0.1289 ± 0.0221 | 0.1250 |
| 3 | 0.1152 ± 0.0249 | 0.1328 ± 0.0000 | 0.1250 |
| 4 | 0.1348 ± 0.0083 | 0.1465 ± 0.0028 | 0.1250 |

## Interpretation and limitations

Grassmann produced slightly lower mean validation CE/perplexity in this tiny run, while the original attention had higher mean next-token accuracy. Multi-hop answer accuracy was essentially at chance for both (12.5% chance; 13.23% original vs 12.84% Grassmann), so this experiment does **not** establish multi-hop reasoning competence or a win for Grassmann. One hundred steps is a short undertraining regime, and the results are only two seeds on a synthetic task. The result is a functional/causal implementation check plus an initial comparison, not evidence that the method is superior.

The execution environment used CPU-only PyTorch (`torch 2.10.0+cpu`; no CUDA device available). Wall times are CPU measurements and must not be interpreted as T4 performance. A proper GPU throughput/peak-memory comparison and longer convergence study still need to be run on Kaggle.

## Running it

Select the new module in the existing trainer:

```bash
python train.py \
  --attention-type grassmann \
  --grassmann-rank 32 \
  --grassmann-offsets 1 2 4 8 12 16 \
  # retain your existing training arguments
```

Run the controlled small benchmark:

```bash
PYTHONPATH=src python scripts/benchmark_grassmann.py \
  --steps 100 --batch-size 32 --eval-every 25 \
  --train-examples 1024 --val-examples-per-hop 256
```

The default attention path remains the original `deepseek` implementation. The new option is opt-in.


## Post-review implementation fixes (2026-11-03)

A subsequent code review identified three issues and this release addresses them:

1. **Absolute token positions (§3.1):** Grassmann mode now owns a learned
   `grassmann_positional_embedding` table of shape
   `[grassmann_positional_max_len, dim]`, added to token embeddings exactly once
   before the recurrent stack. It is separate from `loop_positional_embedding`,
   which encodes recurrent depth. The DeepSeek default creates no such table and
   its parameter/state-dict structure is unchanged. Configure the limit with
   `--grassmann-positional-max-len` (default 1024).
2. **Stable Plücker normalization:** the hard norm clamp has been replaced with
   smooth normalization `p * rsqrt(sum(p*p) + 1e-4)`. This bounds the derivative
   when pairs are identical or nearly collinear, avoiding the `1/||p||` gradient
   singularity on repeated-token runs. Computation remains in fp32 before the
   projected feature is cast back to the model dtype.
3. **Resume architecture guard:** explicit changes to attention type, rank,
   offsets, or Grassmann position-table length are rejected when resuming a
   checkpoint. If the attention option is omitted, the checkpoint's architecture
   is selected and run metadata is aligned with it. Start a fresh checkpoint
   directory to switch from DeepSeek to Grassmann or vice versa.

Legacy Grassmann checkpoints from before the positional table are still loadable:
the old parameters are restored, the new position table is initialized afresh,
and optimizer moments for that new parameter start empty. Existing Adam moments
for matching parameters are migrated only when their shapes can be matched
safely; otherwise optimizer restoration is declined rather than misapplied.

The original two-seed benchmark tables above were generated **before** the
positional-table and smooth-normalization changes. They remain a record of that
older smoke experiment and must not be treated as results for the updated model.
A new matched benchmark should be run before drawing accuracy or throughput
conclusions about this revision.

### Updated verification

- Full test suite: 39/39 PASS.
- Exact repeated-vector Plücker backward: finite and bounded gradients.
- Full model with loop controller and repeated-token batches (including the
  exactly repeated-position stress case): finite, bounded controller gradients
  after the normalization fix.
- Full-model autograd check: a logit at position 5 has exactly zero gradient to
  token embeddings at positions 6 and later, with the absolute position table on.
- Learned absolute position table: receives gradients, distinguishes token
  positions, and enforces its configured maximum sequence length.
- Legacy Grassmann checkpoint loading: only the new position-table key may be
  absent; all other missing/unexpected state keys remain errors.
- Legacy optimizer migration: previous Adam moments restore; the new position
  table starts with fresh moments.
- Explicit DeepSeek/Grassmann resume mismatch: rejected before training.
- Default DeepSeek model: no Grassmann position table is instantiated.


## Matched positional-embedding ablation (current release)

This release adds two CLI-selectable Grassmann conditions:

- `--no-grassmann-positional-embedding`: causal Grassmann mixing without a learned absolute token-position table.
- `--grassmann-positional-embedding`: the paper's learned absolute token-position table (default).

The position table is constructed and initialized inside an isolated CPU RNG context. Its constructor and initialization do not advance the global model-initialization RNG. At the same global seed, all shared non-position parameters are therefore bit-identical between the two conditions. A dedicated `--grassmann-positional-seed` controls the table independently; when omitted it derives from the global torch seed.

Visualization accepts `--viz-label` for a custom experiment title. A generated subtitle describes causal multi-scale Pluecker mixing, rank, look-back offsets, and positional mode. `--viz-probe-size` now uses the requested number of examples exactly (it is no longer silently capped at the training batch size); the probe remains rank-0-only and no-grad. Per-loop validation CE/accuracy use full fixed validation aggregates when evaluation ran on the same step.

The matched run commands are `GRASSMANN_ABLATION_A_NO_ABSOLUTE_POSITIONS.txt` and `GRASSMANN_ABLATION_B_LEARNED_POSITIONS.txt`. They differ in positional-table enablement and run/output labels only; both are fresh 5,000-step runs using the same data, seed, LR schedule, batch and architecture settings. The scripts intentionally omit `--force-lr` so the 200-step warmup is active.

Current verification: **48/48 project tests pass**, Python compilation passes, both notebook command files pass shell syntax parsing and CLI argument parsing, and one-step CPU end-to-end smoke runs for both position modes successfully produce checkpoints, five-example probe files, metrics JSON, and visualization PNGs. A T4/NCCL production run was not available in this build environment.


## Kaggle ablation output safety

The A/B command files write output under `/kaggle/working/grassmann_ablation_runs/`, outside the Drive restore target `/kaggle/working/loop_llm`. `KaggleDriveSync.restore()` swaps/replaces the complete restore target at every invocation; placing A's outputs inside that directory would let invocation B erase them during startup. The command files avoid that data-loss hazard. The current `--drive-save-interval` argument produces local checkpoints; it does not itself upload them to Drive. Copy/download the run folders to persistent storage before the Kaggle runtime ends.


## Adaptive global offsets and corrected 16-step A/B reruns

A subsequent implementation review found two different omissions in the earlier experiment setup: `grassmann-new` produced only progressive local offsets, and the earlier A/B command files selected the legacy fixed-offset mode. The latter contained local lag 16 but did not repeat global look-backs at 16-token intervals.

The corrected `grassmann-new` implementation combines its loop-dependent local set with a sparse global set. For `N` local offsets, recurrent depth `R`, and actual runtime tensor length `T`, it uses `C=T-1`, `L_max=(N-1)*(b+R-1)`, `S=max(L_max+1, ceil(C/N))`, and global offsets `{j*S | j*S <= C}`. For T=301, R=4, b=2, N=4 gives S=75 and global offsets 75, 150, 225, 300; N=5 gives S=60; N=10 gives S=46. The first global lag is always beyond the largest configured local lag.

The refreshed A/B command files now use `real-grassmann` with `--grassmann-long-range-stride 16`, so their 301-position offset set is `[1,2,4,8,12,16,32,...,288]` (23 unique lags). Their outputs are redirected to new `*_stride16` folders; they do not overwrite existing historical checkpoints. The default real-Grassmann stride is also 16, while stride 32 remains selectable.

These long-range and adaptive schedules are project extensions, not features stated in arXiv:2512.19428v1. They must be labelled separately from the paper's fixed local-window examples. CPU tests cover exact schedule values, context boundaries, invalid recurrent indices, gradients through long-range pairs, causal future-token exclusion, CLI defaults, checkpoint resume guards, and the legacy fixed-offset path. A real Kaggle/T4 throughput and VRAM test is still required before starting the full 5,000-step A/B runs.


The checkpoint loader now versions the `grassmann-new` offset schedule. Checkpoints produced before this adaptive change are detected by the missing `grassmann_offset_scheme` field and reconstructed with their former local-only schedule for evaluation. Training refuses to resume these pre-adaptive checkpoints under the adaptive mode; this avoids a silent attention change during resumed optimization. New checkpoints store `adaptive_sparse_global_v1`, and resume validation checks the schedule marker and recurrent depth.
