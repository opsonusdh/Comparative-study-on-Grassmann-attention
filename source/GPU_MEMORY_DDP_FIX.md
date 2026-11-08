# GPU memory / DDP stability fix

This patch addresses the actual training entry point used by the DeepSeek Split SWA/CSA/HCA experiment.

## Changes

1. AdamW is constructed with `foreach=False` and `fused=False`. The CUDA foreach implementation may allocate parameter-sized temporary tensor lists during `optimizer.step()`, creating a large transient VRAM spike. The non-foreach path trades some speed for lower peak memory.
2. DDP uses `gradient_as_bucket_view=True`, allowing gradient tensors to share storage with communication buckets instead of retaining a separate full gradient copy.
3. Gradient accumulation uses `DDP.no_sync()` on all but the final micro-step, and both forward and backward are inside the context. This avoids redundant all-reduces during accumulation.
4. Multi-GPU runs now require the global batch size to divide evenly across ranks. Previously, the remainder was assigned to rank 0, giving one GPU a larger activation footprint and creating exactly the sort of asymmetric allocation seen in the dashboard.
5. Each rank logs allocated, reserved, peak, and free CUDA memory at the training log interval.

## Existing command

The supplied Kaggle command already uses `--batch-size 2` for two GPUs, so each GPU receives one sample. Keep that for the first retry. `--grad-checkpointing` is already enabled. The command uses float32; if peak activation memory remains too high, use `--amp --dtype float16` on NVIDIA T4 GPUs, after validating numerical stability on a short run.

## Resume

Keep the same checkpoint directory to resume from the last completed checkpoint. The optimizer algorithm is still AdamW, so optimizer state remains compatible. The change to foreach/fused affects execution strategy, not the stored moment tensors.

## Interpreting the new logs

Each worker prints `[VRAM rank=... gpu=...]`. If one rank stops printing while the other reaches a collective, inspect the first traceback from the silent rank. A GPU showing allocated memory but 0% utilization can simply be holding CUDA allocations after its worker has failed or while the surviving worker waits at an NCCL collective; the dashboard alone cannot identify the original exception.
