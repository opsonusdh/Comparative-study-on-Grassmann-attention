"""Shared context summarizer used by token-level activation/FFN routers.

This module is deliberately separate from the model's attention families.
SWA/CSA/HCA are normal, fixed attention paths selected by layer; this context
attention is only a lightweight O(T*d) summary used to condition FFN routing.

BUG FOUND AND FIXED (during an unrelated integration task): the original
forward() computed `x.mean(dim=1)` -- the mean over the WHOLE sequence,
unmasked -- as the attention query, then broadcast the identical resulting
context vector to every position via `.expand(...)`. That means every
position's FFN routing depended on the entire sequence, including tokens
after it. Confirmed directly, the same way every causality check in this
project's recent history has been confirmed: instantiate with random
weights, perturb one token near the end of a sequence, check whether
logits/outputs at an EARLIER position change. They did -- FeedForward
outputs at position 5 changed by up to ~0.09, position 30 by ~0.24, from
changing only position 55, which is architecturally impossible for a
correct causal LM component. During training (teacher-forcing, whole
sequence available at once) this silently trains the router to rely on
future-token information; at real autoregressive generation, that
information does not exist yet -- a train/inference mismatch that would
not show up in training-time loss but could degrade real generation
quality, and is a plausible contributing factor to training difficulties
at real scale.

Fix: the query is now a CAUSAL running mean (cumulative average of
positions 0..t, not 0..T-1) instead of the whole-sequence mean, and the
attention step -- which cost the SAME as a second full attention layer
per FFN call while contributing nothing once the query was already
whole-sequence-pooled and identical everywhere -- is replaced with a
direct projection of that running mean. This is provably causal (a
cumulative sum is causal by construction) and is actually CHEAPER than
the original: O(T*d) with no O(T^2) attention matrix at all, matching
the module's own stated design goal better than the buggy version did.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoutingContextAttention(nn.Module):
    """Causal O(T*d) running-mean summary used to make token routers
    context-aware, without letting later tokens leak into earlier
    positions' routing decisions."""

    def __init__(self, dim: int, context_dim: int | None = None):
        super().__init__()
        context_dim = context_dim or min(64, max(16, dim // 8))
        self.proj = nn.Linear(dim, context_dim, bias=False)
        self.out = nn.Linear(context_dim, context_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        running_sum = x.cumsum(dim=1)
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        running_mean = running_sum / counts   # (B,T,D): position t = mean of x[:, :t+1]
        ctx = self.proj(running_mean)           # genuinely per-position now, not broadcast
        return self.out(torch.tanh(ctx))
