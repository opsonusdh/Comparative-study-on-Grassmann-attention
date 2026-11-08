"""MLP-only attention-dependent residuals.

This module deliberately does not participate in recurrent-loop state routing.
It only provides a learned residual over earlier MLP branch outputs inside the
current Transformer pass, and its history is reset every recurrent iteration.
"""
from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn
from .layers import RMSNorm


class MLPAttentionResidual(nn.Module):
    """Attention-weighted residual over prior MLP outputs in one loop.

    The residual gate is zero-initialized, so the module starts as an exact
    no-op and can only introduce cross-layer MLP residuals as training learns
    them. The learned query chooses among earlier MLP outputs; it never sees
    previous recurrent-loop states.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.key_norm = RMSNorm(dim)
        self.query = nn.Parameter(torch.zeros(dim))
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, history: Sequence[torch.Tensor]) -> torch.Tensor:
        if not history:
            raise ValueError("MLPAttentionResidual requires non-empty history")
        sources = torch.stack(list(history), dim=0)  # [N,B,T,D]
        keys = self.key_norm(sources)
        logits = torch.einsum("d,nbtd->nbt", self.query, keys)
        weights = logits.softmax(dim=0)
        mixed = torch.einsum("nbt,nbtd->btd", weights, sources)
        return mixed * self.gate
