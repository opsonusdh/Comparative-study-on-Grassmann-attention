"""Causal recurrent register memory used by LoopLLM."""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalRegisterBank(nn.Module):
    """Per-position causal register memory passed between recurrent loops.

    Every position writes to a small learned register bank using only the
    causal prefix up to that position. The resulting read is safe to feed
    into later recurrent loops because no future token contributes to it.
    """

    def __init__(self, dim: int, num_registers: int = 4, update_scale: float = 0.1):
        super().__init__()
        if num_registers < 1:
            raise ValueError("num_registers must be >= 1")
        if update_scale <= 0:
            raise ValueError("update_scale must be > 0")
        self.dim = dim
        self.num_registers = num_registers
        self.update_scale = float(update_scale)
        self.base = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.write = nn.Linear(dim, num_registers * dim, bias=False)
        self.router = nn.Linear(dim, num_registers)
        self.read = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        previous: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        prefix = x.cumsum(dim=1) / counts
        update = torch.tanh(self.write(prefix)).view(B, T, self.num_registers, D)
        slots = self.base.view(1, 1, self.num_registers, D) + self.update_scale * update
        if previous is not None:
            slots = slots + previous
        weights = torch.softmax(self.router(prefix), dim=-1).unsqueeze(-1)
        register_read = (slots * weights).sum(dim=2)
        register_read = self.read(register_read)
        return slots, register_read
