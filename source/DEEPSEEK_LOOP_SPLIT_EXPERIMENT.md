# DeepSeek Loop-Split SWA / CSA / HCA Experiment

## Goal

Test whether recurrent passes can divide compressed long-range context while
preserving the three existing DeepSeek attention mechanisms. This is an
experimental `deepseek-split` mode, not a replacement with a fourth SWA-like
attention module.

## Attention-family routing

The original physical-layer routing is preserved exactly:

| Physical layer | Attention family | `deepseek-split` behavior |
|---:|---|---|
| 0-1 | SWA | Unchanged. Every causally valid position in the configured sliding window is attended on every loop. |
| 2, 4, 6, ... | CSA | Learned indexer and top-k selection remain active, but operate only on that loop's eligible compressed long-range blocks. The raw-token local window remains complete. |
| 3, 5, 7, ... | HCA | Attends to all compressed long-range blocks assigned to the current loop. The raw-token local window remains complete. |

`--sw-window` defines the full raw-token local window used by SWA and by the
local-detail branches of CSA/HCA. For example, `--sw-window 200` means that
these branches retain all causally valid positions in the most recent 200-token
window. Those positions are **not** split among recurrent loops.

## Split schedule

Only the compressed long-range memory bank in CSA/HCA is split. For each query,
causally visible compressed blocks are ranked newest-to-oldest with one-based
ages. Let `l` be the zero-based recurrent loop and `s` the split stride. The
assigned ranks are:

```text
A_l = {1} union {s*n + (l mod s) : n >= 0, 1 <= s*n + (l mod s) <= visible_blocks}
```

For stride 4:

| Recurrent loop | Compressed-block ages |
|---:|---|
| 0 | 1, 4, 8, 12, 16, ... |
| 1 | 1, 5, 9, 13, 17, ... |
| 2 | 1, 2, 6, 10, 14, ... |
| 3 | 1, 3, 7, 11, 15, ... |
| 4 | repeats loop 0 |

Age 1 is shared by every loop. Every older compressed block belongs to exactly
one loop phase. The union across `s` phases therefore recovers the entire
causally visible compressed memory. The loop-1 formula produces `1, 5, 9, ...`;
lag/rank 2 belongs to loop 2, not loop 1.

The schedule is applied to compressed-block ranks rather than raw-token lag
modulo `s`. For example, if the CSA compression factor is 4 and we used raw
look-back modulo 4, each block boundary could land in the same modulo class,
putting nearly the entire bank into one pass. Ranking compressed slots avoids
that aliasing and keeps the partition balanced regardless of `csa_m` or
`hca_m_prime`.

## Compute path

- SWA is the original full causal sliding-window implementation.
- CSA compresses the context as before, gathers only the loop-assigned
  compressed blocks, scores only those candidates with its learned indexer,
  then applies top-k within that eligible set. Its full raw local window is
  concatenated into the same attention softmax as before.
- HCA compresses the context as before, gathers only the loop-assigned
  compressed blocks, and attends to all of them. Its full raw local window is
  also retained.
- Both compressed-memory gathers are compact `[B,T,K,D]` tensors, with
  `K` bounded by the number of blocks assigned to one phase. They do not first
  materialize a dense `[B,T,N,D]` expansion of the entire compressed bank.
- CSA's indexer auxiliary KL loss uses the exact same loop-eligible candidate
  set as CSA's top-k selection; its detach/no-grad gradient-isolation rules
  remain in place.

No attention parameters or buffers are added. The `deepseek` and
`deepseek-split` parameter state-dict keys and tensor shapes remain the same;
the difference is the candidate set used by CSA/HCA on each recurrent pass.
The baseline `deepseek` path is unchanged when `split_stride=None`.

## Run

```bash
python train.py \
  --attention-type deepseek-split \
  --deepseek-split-stride 4 \
  --sw-window 200 \
  --max-loops 4 \
  --min-loops 4 \
  ...
```

Use a new checkpoint/output directory for the experiment so baseline and split
metrics remain separate. Although parameter shapes are compatible, resume
validation intentionally rejects changing `attention_type` mid-run; do not
change an in-progress baseline run into the split experiment.

## Evaluation

Compare against `--attention-type deepseek` using the same tokenizer, data
split, model dimensions, optimizer, seed, effective batch size, training-token
budget, and evaluation procedure. Track validation CE, next-token accuracy,
tokens/sec, peak allocated VRAM, and scaling across multiple context lengths.
This is an architectural hypothesis, not an established improvement.
