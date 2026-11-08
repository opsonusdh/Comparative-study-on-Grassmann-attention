"""Norms, positional encoding, and small tensor utilities shared across
attention implementations.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(t, t, device=device, dtype=torch.bool))


def topk_mask(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    k = min(k, scores.size(-1))
    return torch.topk(scores, k=k, dim=-1)


def safe_eps(dtype: torch.dtype) -> float:
    """Smallest safe positive epsilon for `dtype`, for clamping a value away
    from zero right before log() or division.

    A fixed Python float like 1e-8 silently rounds to EXACTLY 0 in
    float16, whose smallest positive normal is ~6.1e-5 (confirmed: `torch.
    tensor(1e-8, dtype=torch.float16).item()` is `0.0`). `some_fp16_zero_
    tensor.clamp_min(1e-8)` is then still exactly zero, so `.log()` of the
    "clamped" value is -inf, and any downstream product with the
    original (also ~0) term becomes `0 * -inf = NaN` under autocast/fp16.
    This is the same category of bug as the mask-value overflow already
    fixed via `torch.finfo(dtype).min` in attention.py's `_kl_mask_value`
    -- the epsilon side of dtype-aware sentinels, not the mask side.
    """
    return torch.finfo(dtype).tiny


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum-of-squares and rsqrt run in fp32 regardless of x's dtype,
        then the result is cast back to x's original dtype.

        x.pow(2) overflows float16 (max ~65504) for any |x| >~ 256 --
        entirely plausible activation magnitude, not a contrived edge
        case. Confirmed directly: a uniform fp16 input of magnitude 260
        gives x.pow(2) == inf, then rsqrt(inf + eps) == 0, so RMSNorm's
        output is silently EXACTLY 0 -- no NaN, no crash, no warning, just
        a normalization layer that quietly zeroes out whatever fed it
        (attention output, a query/key vector, a loop's residual delta...)
        every time an upstream value gets a bit large. Since RMSNorm sits
        at attn_norm/ffn_norm/final_norm/q_norm/k_norm/mem_norm/key_norm
        -- i.e. nearly every sub-layer boundary -- this is a plausible
        source of hard-to-trace instability specifically under fp16,
        never reproducible under fp32 (whose max is ~3.4e38, so x.pow(2)
        would need |x| >~ 1.8e19 to overflow -- not a realistic concern).
        Computing in fp32 internally is the standard fix (matches e.g.
        LLaMA's RMSNorm) and costs nothing extra when x is already fp32.
        """
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(rms + self.eps)
        return (x_normed * self.weight.float()).to(input_dtype)


class PartialRoPE(nn.Module):
    """Apply RoPE only to the last rope_dim dimensions of each head."""

    def __init__(self, head_dim: int, rope_dim: int = 64, base: int = 10_000):
        super().__init__()
        if rope_dim % 2 != 0 or rope_dim > head_dim:
            raise ValueError(
                f"rope_dim must be even and <= head_dim, got rope_dim={rope_dim}, "
                f"head_dim={head_dim}."
            )
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, T: int, device, dtype):
        t = torch.arange(T, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype=dtype, device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, H, D]"""
        if self.rope_dim == 0:
            return x
        B, T, H, D = x.shape
        x_main = x[..., : D - self.rope_dim]
        x_rope = x[..., D - self.rope_dim :].view(B, T, H, self.rope_dim // 2, 2)
        cos, sin = self._cos_sin(T, x.device, x.dtype)
        cos = cos[..., : self.rope_dim].view(1, T, 1, self.rope_dim // 2, 2)
        sin = sin[..., : self.rope_dim].view(1, T, 1, self.rope_dim // 2, 2)
        x1, x2 = x_rope[..., 0], x_rope[..., 1]
        rop = torch.stack(
            (x1 * cos[..., 0] - x2 * sin[..., 0],
             x1 * sin[..., 0] + x2 * cos[..., 0]),
            dim=-1,
        ).flatten(-2)

        # ``rope_dim == D`` is a valid configuration: all head dimensions are
        # rotary and there is no non-rotary remainder.  In that case the
        # slice above has shape ``[..., 0]`` for ``x_main``.  CPU/CUDA PyTorch
        # can concatenate a zero-width tensor, but DirectML's privateuseone
        # backend rejects this particular empty-tensor ``cat`` invocation with
        # ``RuntimeError: The parameter is incorrect``.  There is no reason to
        # concatenate an empty remainder anyway, so return the rotated tensor
        # directly.  This preserves the exact mathematical operation and avoids
        # relying on undefined/unsupported empty-tensor backend behavior.
        if x_main.shape[-1] == 0:
            return rop
        return torch.cat((x_main, rop), dim=-1)


def causal_windows(x: torch.Tensor, window: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For every position t, gather the causal window of up to `window` raw
    token vectors ending at t (positions [t-window+1, t]), left-padded
    with zeros where the window would reach before position 0.

    Used as the "recent raw detail" branch for CSA/HCA -- shared here so
    both get identical, correct causal behavior instead of duplicating it.

    Returns
    -------
    windows : [B, T, window, C]  -- raw token vectors per window slot
    valid   : [B, T, window]     -- True where the slot is a real (non-pad)
                                     token; slot (window-1) is always valid
                                     (it's position t itself), so every
                                     query always has >=1 valid candidate.
    """
    B, T, C = x.shape
    pad = window - 1
    x_pad = F.pad(x, (0, 0, pad, 0))          # [B, T+pad, C]
    windows = x_pad.unfold(1, window, 1)       # [B, T, C, window]
    windows = windows.permute(0, 1, 3, 2)      # [B, T, window, C]

    pos    = torch.arange(T, device=x.device)
    offset = torch.arange(window, device=x.device)
    src_pos = pos[:, None] + offset[None, :] - pad     # [T, window]
    valid = (src_pos >= 0)[None].expand(B, T, window)
    return windows, valid
