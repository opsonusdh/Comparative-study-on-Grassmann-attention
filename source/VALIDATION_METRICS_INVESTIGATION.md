# Validation Metrics Investigation: DeepSeek vs Grassmann

## Findings from the step-500 charts

The headline validation metrics and the per-loop panel in the old visualizer were not measured on the same sample set.

- The Kaggle run uses `--batch-size 2 --eval-iters 20`, so the headline evaluator scores 20 fixed batches (40 sequences total, with the shown configuration).
- The visualizer uses `--viz-probe-size 5` and scores only one five-sequence probe batch for its per-loop CE/accuracy panel.
- `evaluate()` already computes `val_loop_ce` and `val_loop_accuracy_pct` over the full fixed validation set, but the old `visualizer.snapshot()` ignored those values and independently scored the probe batch.

Therefore, lower per-loop CE in the old Grassmann probe panel does not establish that Grassmann has better full-validation CE at every loop. It is a small-probe diagnostic, while the headline result is a broader validation aggregate. The updated visualizer uses the full-validation per-loop aggregates when a visualization coincides with evaluation; it clearly labels the fallback as `probe batch` when it does not.

## Which parameter count belongs to which architecture?

I instantiated both architectures from the source using the configuration printed in the user's startup log:

- vocabulary 50,281; dimension 512; 8 layers; 8 heads; FFN hidden size 2,048
- 4 recurrent loops; controller and causal register bank enabled
- 1 shared + 4 routed experts; top-k 2; expert hidden dimension 512
- Grassmann rank 32; offsets `[1, 2, 4, 8, 12, 16]`
- tied embeddings; positional table length 301 for Grassmann

The counts reproduce the chart titles exactly:

| Architecture | Parameters |
|---|---:|
| DeepSeek attention | 67,576,149 |
| Grassmann attention, position length 301 | 66,049,557 |
| Grassmann attention, position length 300 | 66,049,045 |

The 512-parameter difference between the last two rows is exactly one additional 512-dimensional positional vector. It is not a parameter leak during the forward pass.

At these matching settings, Grassmann has 1,526,592 fewer total parameters (about 2.26%). Its learned 301-position table adds 154,112 parameters, already included in its total.

## Correct reading of the step-500 comparison

Using the startup log's `attention_type=grassmann` and exact parameter-count reproduction to identify the runs:

- The 67,576,149-parameter chart is DeepSeek.
- The 66,049,557-parameter chart is Grassmann.

At step 500, the plotted full-validation endpoint is approximately:

| Metric | DeepSeek | Grassmann |
|---|---:|---:|
| Final-loop validation CE | 6.63 | 6.72 |
| Final-loop validation accuracy | 13.47% | 11.90% |

DeepSeek is ahead in the aggregate headline validation metrics at this point by approximately 0.09 CE and 1.57 percentage points accuracy. This is one run at 500 steps, not a multi-seed or statistically conclusive result.

The old per-loop CE/accuracy panels are probe-batch measurements and should not be used to reverse that conclusion. The updated charts will put per-loop CE/accuracy on the full validation set when the validation and visualization steps coincide.

## Other diagnostic interpretation

The controller panel reports the mean L2 norms of the *bounded outputs* after `0.2 * tanh(raw)`; it does not report raw controller parameter norms. With dimension 512, the theoretical maximum norm is approximately `0.2 * sqrt(512) = 4.525`. The Grassmann panel's values around 3.6-3.7 are high relative to that bound (roughly 80%), so saturation is worth monitoring, but this plot alone does not demonstrate exploding parameters or gradients.

The displayed state-growth ratios (3.03x and 4.07x) are over four recurrent loops. Both are far below the visualizer's existing warning threshold of 100x. They indicate stronger growth in the Grassmann run, not an immediate numerical failure.

## Initialization caveat for a strict A/B test

Reusing the same global random seed does not guarantee identical initialization of shared non-attention parameters. The Grassmann model creates a positional embedding before the transformer blocks, and its attention modules consume a different number of random values from DeepSeek attention. In a direct same-seed construction check, token embeddings and the loop-controller projection matched, but shared FFN expert weights and the causal-register base did not.

A stricter experiment should copy all same-name, same-shape non-attention parameters/buffers from one common initialized model into the other, while keeping the two attention modules and the Grassmann-only positional table separately initialized. Then use identical fixed evaluation batches and preferably multiple seeds.

## Code change in this update

The visualizer now accepts the `evaluate()` result. On steps where evaluation and visualization coincide, the per-loop CE and accuracy panels use `val_loop_ce` and `val_loop_accuracy_pct` from the same full validation set as the headline metrics. Hidden-state, register and controller diagnostics still use the fixed probe batch. When there is no same-step evaluation result, the chart explicitly labels its per-loop metrics as `probe batch`.

No model, optimizer, loss, checkpoint, data sampling, Google Drive or DDP behavior was changed by this metrics-only patch.


## Training-curve label correction

The former blue legend said `train CE`, but the value passed to `record_training()` is the complete `compute_loss()` objective: direct CE averaged across loops plus configured auxiliary terms (including the late-loop reinforcement and refinement penalty in this run). The orange validation CE is plain next-token CE from the final loop. They are not the same quantity. The chart legend now correctly says `train objective`.
