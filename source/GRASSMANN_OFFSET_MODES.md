# Grassmann offset modes

This release keeps three distinct offset strategies. All use the causal pair `(z_t, z_{t-Delta})`; no mode reads a future token. The runtime sequence length is the actual model input length. With `--seq-len 300`, the shifted input tensor is 301 positions, so the largest valid lag is 300.

## 1. Legacy `grassmann`: fixed local offsets

This mode is preserved for historical reproducibility:

```bash
--attention-type grassmann \
--grassmann-offsets 1 2 4 8 12 16
```

Every recurrent pass uses exactly these six lags. **Lag 16 is included, but this mode does not append 32, 48, 64, and so on.** This distinction matters when reproducing older A/B experiments. Its parameter structure and offset behavior remain unchanged.

## 2. `grassmann-new`: progressive local offsets plus adaptive sparse-global offsets

Let `N` be `--grassmann-offset-count`, the number of **local** offsets per recurrent pass, including mandatory lag 1. Let `b` be `--grassmann-offset-spacing-base` and `R` be configured `--max-loops`. For zero-based recurrent pass `n`, the local schedule is:

```text
L_n = {1} union { k * (b+n) : k=1,...,N-1 }
```

The global schedule is computed once from the full recurrent stack and the runtime sequence length `T`:

```text
C     = T - 1
L_max = (N - 1) * (b + R - 1)
S     = max(L_max + 1, ceil(C / N))
G     = { j*S : j=1,...,floor(C/S) }
O_n   = sorted_unique(L_n union G)
```

Therefore the first global lag is always strictly greater than the largest configured local lag across all recurrent passes. Global lags are sparse multiples of `S`, not a dense sweep of the interval after the local offsets. Offsets beyond the current tensor's largest valid lag are omitted. If `N` is extremely large relative to the context, local offsets are capped to valid history and the global set can be empty. `N=1` remains valid: the local lag is 1 and the global tail contains the far-context lag when one exists.

For the current 301-position input (`T=301`, `C=300`), `b=2`, `R=4`:

| N | Local offsets by recurrent pass (n=0,1,2,3) | Adaptive S | Global offsets | Total offsets/pass |
|---:|---|---:|---|---:|
| 4 | `[1,2,4,6]`; `[1,3,6,9]`; `[1,4,8,12]`; `[1,5,10,15]` | 75 | `[75,150,225,300]` | 8 |
| 5 | `[1,2,4,6,8]`; `[1,3,6,9,12]`; `[1,4,8,12,16]`; `[1,5,10,15,20]` | 60 | `[60,120,180,240,300]` | 10 |
| 10 | ten local lags per pass, with loop spacing 2, 3, 4, 5 | 46 | `[46,92,138,184,230,276]` | 16 |

The N=4 low-memory command is `KAGGLE_GRASSMANN_NEW_N4_ADAPTIVE_COMMAND.txt`. The N=5 command is `KAGGLE_GRASSMANN_NEW_N5_COMMAND.txt`. Since both `T` and `N` affect `S`, the global stride adjusts automatically for shorter or longer context windows. The attention module processes offsets incrementally; it does not construct a dense `[B,T,num_offsets,plucker_dim]` tensor.

CLI example:

```bash
--attention-type grassmann-new \
--grassmann-offset-count 4 \
--grassmann-offset-spacing-base 2 \
--max-loops 4
```

For fixed `N`, the adaptive rule keeps the number of selected lags bounded by a quantity proportional to `N` as context length grows, so pair work remains linear in `T` for fixed recurrent depth and `N`.

## 3. `real-grassmann`: fixed-stride global look-backs

This mode combines the local list with all positive multiples of an explicit global stride that fit the current context. The default is now **16**; set 32 explicitly for the lower-density variant:

```bash
--attention-type real-grassmann \
--grassmann-offsets 1 2 4 8 12 16 \
--grassmann-long-range-stride 16
```

At `T=301`, stride 16 produces the unique lag set:

```text
[1, 2, 4, 8, 12, 16, 32, 48, 64, 80, 96, 112,
 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288]
```

The local lag 16 and the first global multiple 16 are deduplicated. This is 23 distinct offsets per pass. At larger contexts, fixed-stride pair count grows approximately as `O(T^2 / stride)`, so stride 16 is materially more expensive than stride 32. Measure peak VRAM and tokens/second on the target T4 before committing to a long run. The corrected A/B commands explicitly use stride 16 and new output directories:

- `GRASSMANN_ABLATION_A_NO_ABSOLUTE_POSITIONS.txt`
- `GRASSMANN_ABLATION_B_LEARNED_POSITIONS.txt`

These commands are fresh 5,000-step runs and no longer write into the historical A/B output folders. They differ only in the absolute-position setting and descriptive run labels; the position table uses an isolated RNG so shared model parameters remain aligned.

## 4. Paper-version clarification

The cited paper is Zhang Chong, *Attention Is Not What You Need: Grassmann Flows as an Attention-Free Alternative for Sequence Modeling*, arXiv:2512.19428v1. Its method examples use local offsets `{1, 2, 4, 8, 12, 16}`. It does **not** specify repeated stride-16 or stride-32 global offsets, nor the N-dependent adaptive schedule above. Those are project extensions and must be reported as such. The paper's `r=32` is the reduced Grassmann projection dimension, not an instruction to sample offsets every 32 positions.

Paper: https://arxiv.org/html/2512.19428v1

The paper's displayed `z_{t+Delta}` pairing is future-looking despite the causal description. This implementation intentionally uses `z_{t-Delta}` for left-to-right language modeling.

## 5. Checkpoints and resume safety

The selected attention mode and structural offset configuration are retained in the model configuration. Resuming `grassmann-new` validates `N`, local spacing base, `max_loops`, and the schedule marker, because each affects the resolved offset schedule. Resuming `real-grassmann` validates the fixed long-range stride. Changing these settings requires a fresh run and a new output directory. The legacy `grassmann` path remains fixed at the serialized local offset list.

Pre-adaptive `grassmann-new` checkpoints did not contain a schedule marker. They are loaded as `legacy_local_only_v0` so evaluation/generation preserves their original local-only behavior. The training entrypoint refuses to resume them under the new adaptive implementation; this avoids silently changing attention geometry partway through a run. Start a fresh run for adaptive global offsets.
