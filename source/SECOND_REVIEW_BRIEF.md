# Independent second-pass review brief

You are reviewing a patch to the LoopLLM MOA7/RFC Grassmann training project. Do not assume the implementation is correct because tests pass. Inspect the source and the two Kaggle command files independently.

## User's intended experiment

The user has already trained the DeepSeek/SWA/CSA/HCA baseline for a long run and does not want another DeepSeek training run. They want exactly two fresh, matched Grassmann experiments, each for 5,000 optimizer steps:

A. Grassmann causal multi-scale Pluecker mixing, with no learned absolute token-position table.
B. The same Grassmann mixer, with the paper's learned absolute token-position table.

The only intended model difference between A and B is the presence of that positional table. All shared model parameters must be initialized identically under the same global seed. The position table must use a dedicated/isolate RNG so constructing it does not advance the global model initialization RNG. The table length must be 301 because the trainer feeds `seq_len + 1` token IDs when `--seq-len 300`.

Both runs must keep the user's existing configuration, data, global batch 4, sequence length 300, FP32/no AMP, rank 32, offsets 1/2/4/8/12/16, four recurrent loops, MoA7, RFC, MoE and controller settings. The LR must be `3e-5` with 200 warmup steps. `--force-lr` must be absent because this code's forced schedule bypasses warmup. Runs must have distinct fresh checkpoint and visualization directories and must not use `--resume`.

## Visualization requirements

The plot title must clearly identify Grassmann and distinguish condition A from condition B. The subtitle must state that this is causal multi-scale Pluecker mixing, list rank and look-back offsets, and state whether learned absolute token positions are enabled. A user-supplied `--viz-label` must override the default title. The requested five-example probe must really contain five examples even when the training global batch is four. Probe diagnostics are no-grad and rank-0-only; validation CE/accuracy should continue to use the full fixed validation aggregates when available.

## Files to review

- `src/loop_transformer/model.py` (RNG isolation, position addition, causal behaviour)
- `src/loop_transformer/config.py` (new fields and validation)
- `train.py` (CLI, config mapping, resume validation, probe sizing, visualizer wiring, warmup scheduling)
- `src/loop_transformer/visualizer.py` (title/subtitle and metrics metadata)
- `tests/test_grassmann_ablation_modes.py` and the existing Grassmann/visualizer tests
- `GRASSMANN_ABLATION_A_NO_ABSOLUTE_POSITIONS.txt`
- `GRASSMANN_ABLATION_B_LEARNED_POSITIONS.txt`

## Questions to answer

1. Do A and B have bit-identical shared parameter initialization at the same seed? Is the position table reproducible and isolated from CPU/CUDA global RNG state?
2. Are there any other architecture, loss, optimizer, data, batch, or scheduler differences between the two commands?
3. Does `--viz-probe-size 5` really make a five-example diagnostic probe? Does it change training gradients or optimizer state?
4. Does the visualization label accurately explain the Grassmann computation without the old generic LoopLLM/MoA7-RFC title?
5. Can old DeepSeek and Grassmann checkpoints still load safely? Are attention-mode mismatches rejected on resume?
6. Are both command files valid in a Kaggle notebook cell, including shell continuations, argument names, output paths, and no accidental checkpoint resume?
7. What important test or assumption is missing? State any unresolved concerns.

Return a concise audit with files/line references, concrete findings, missing tests, and explicit clarifications. Do not rewrite the training architecture or silently change experimental settings.
