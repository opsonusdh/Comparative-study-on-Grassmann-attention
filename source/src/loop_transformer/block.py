"""TransformerBlock (attention + FFN, external residuals) and ExitGate
(Ouro LoopLM's per-loop halting predictor).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import (
    CSAAttention, GrassmannFlowAttention, HCAAttention, SlidingWindowAttention,
)
from .feedforward import FeedForward
from .layers import RMSNorm


class TransformerBlock(nn.Module):
    """
    Single transformer block: attention + FFN. The model owns the normal
    attention residual and the MLP-only attention-dependent residual.
    ``forward_attn`` and ``forward_ffn`` return branch deltas so the model
    can place those residuals explicitly and keep them local to one loop.

    Attention type by layer index:
        idx < 2       → SlidingWindowAttention  (warm-up layers)
        idx >= 2, even → CSAAttention            (compressed sparse)
        idx >= 2, odd  → HCAAttention            (heavily compressed)
    """

    def __init__(
        self,
        dim:            int,
        n_heads:        int,
        head_dim:       int,
        layer_idx:      int,
        ffn_hidden_dim: int,
        attention_type: str = "deepseek",
        grassmann_rank: int = 32,
        grassmann_offsets: tuple[int, ...] = (1, 2, 4, 8, 12, 16),
        grassmann_offset_count: int = 5,
        grassmann_offset_spacing_base: int = 2,
        grassmann_long_range_stride: int = 16,
        grassmann_offset_scheme: str = "adaptive_sparse_global_v1",
        csa_m:          int   = 4,
        csa_top_k:      int   = 64,
        hca_m_prime:    int   = 128,
        sw_window:      int   = 128,
        deepseek_split_stride: int = 4,
        groups:         int   = 8,
        group_dim:      int   = 1024,
        rope_dim:       int   = 64,
        moe_num_shared_experts: int = 1,
        moe_num_routed_experts: int = 4,
        moe_top_k: int = 1,
        moe_expert_hidden_dim: int | None = None,
        max_loops: int = 4,
        activation_balance_weight: float = 0.01,
        moe_aux_loss_weight: float = 0.01,
        activation_top_k: int = 4,
        activation_min_probability: float = 0.10,
        activation_balance_tolerance: float = 0.25,
        activation_bias_update_speed: float = 0.001,
        moe_bias_update_speed: float = 0.001,
        activation_dictionary: str = "legacy4",
        activation_gate_mode: str = "softmax",
        expert_local_activation_routing: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm  = RMSNorm(dim)
        # Standard Transformer residual-branch dropout. Because residuals
        # are owned by MLP-only attention residuals, dropout is applied to the raw sub-layer
        # delta immediately before the caller adds that delta.
        self.dropout = nn.Dropout(dropout)

        if attention_type in {"grassmann", "grassmann-new", "real-grassmann"}:
            self.attn = GrassmannFlowAttention(
                dim=dim, rank=grassmann_rank, offsets=grassmann_offsets,
                mode=attention_type,
                offset_count=grassmann_offset_count,
                offset_spacing_base=grassmann_offset_spacing_base,
                long_range_stride=grassmann_long_range_stride,
                max_loops=max_loops,
                offset_scheme=grassmann_offset_scheme,
            )
        elif attention_type in {"deepseek", "deepseek-split"}:
            # Both modes preserve the physical-layer family routing:
            # layers 0-1 use full SWA, even layers >=2 use CSA, and odd
            # layers >=3 use HCA. The split mode only partitions the
            # compressed long-range memory inside CSA/HCA; SWA is untouched.
            split_stride = (
                deepseek_split_stride if attention_type == "deepseek-split" else None
            )
            base_swa = SlidingWindowAttention(
                dim, n_heads, head_dim, window_size=sw_window,
                rope_dim=rope_dim, max_loops=max_loops,
            )
            base_csa = CSAAttention(
                dim=dim, n_heads=n_heads, head_dim=head_dim,
                m=csa_m, top_k=csa_top_k, window_size=sw_window,
                index_heads=min(32, n_heads), groups=groups, group_dim=group_dim,
                rope_dim=rope_dim, max_loops=max_loops,
                split_stride=split_stride,
            )
            base_hca = HCAAttention(
                dim=dim, n_heads=n_heads, head_dim=head_dim,
                m_prime=hca_m_prime, window_size=sw_window,
                groups=groups, group_dim=group_dim, rope_dim=rope_dim,
                max_loops=max_loops, split_stride=split_stride,
            )

            # Fixed attention family by physical layer. There is deliberately
            # no router between SWA/CSA/HCA.
            if layer_idx < 2:
                self.attn = base_swa
            elif layer_idx % 2 == 0:
                self.attn = base_csa
            else:
                self.attn = base_hca
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type!r}")

        self.ffn = FeedForward(
            dim, ffn_hidden_dim,
            num_shared_experts=moe_num_shared_experts,
            num_routed_experts=moe_num_routed_experts,
            moe_top_k=moe_top_k,
            expert_hidden_dim=moe_expert_hidden_dim,
            max_loops=max_loops,
            activation_balance_weight=activation_balance_weight,
            moe_aux_loss_weight=moe_aux_loss_weight,
            activation_top_k=activation_top_k,
            activation_min_probability=activation_min_probability,
            activation_balance_tolerance=activation_balance_tolerance,
            activation_bias_update_speed=activation_bias_update_speed,
            moe_bias_update_speed=moe_bias_update_speed,
            activation_dictionary=activation_dictionary,
            activation_gate_mode=activation_gate_mode,
            expert_local_activation_routing=expert_local_activation_routing,
        )

        # Per-loop FiLM conditioning on the (normed) sub-layer input, applied
        # to BOTH attention and FFN -- so the same shared weights can behave
        # differently at each loop depth, not just at each layer. Previously
        # attention had NO loop identity at all (forward_attn took no
        # loop_idx), while FFN only had it inside the activation/expert
        # ROUTERS (conditioning a routing decision, not what the FFN itself
        # receives). This is a different, complementary level: it shapes the
        # INPUT every sub-layer sees, before any routing or attention math
        # runs at all.
        #
        # Parametrized as (1 + delta) with delta and bias BOTH zero-init, not
        # scale starting at 1 directly: at delta=0 this is an exact identity
        # (safe to warm-start from an existing checkpoint with zero
        # disruption), and -- with weight decay active -- decay pulling delta
        # toward 0 REINFORCES the identity property instead of eroding it.
        # A directly-initialized-to-1 scale would instead get pulled toward
        # 0 by decay, silently shrinking the sub-layer's effective input
        # over training. (The existing activation_router/expert_router loop
        # FiLM in feedforward.py has this same erosion issue; not fixed here
        # to keep this change scoped to the attention-conditioning gap.)
        self.attn_loop_scale = nn.Parameter(torch.zeros(max_loops, dim))
        self.attn_loop_bias = nn.Parameter(torch.zeros(max_loops, dim))
        self.ffn_loop_scale = nn.Parameter(torch.zeros(max_loops, dim))
        self.ffn_loop_bias = nn.Parameter(torch.zeros(max_loops, dim))


    def _film(
        self,
        h: torch.Tensor,
        loop_idx: int,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        loop_idx = int(max(0, min(loop_idx, scale.size(0) - 1)))
        h = h * (1.0 + scale[loop_idx]) + bias[loop_idx]
        return h

    def forward_attn(
        self, x: torch.Tensor, loop_idx: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attention sub-layer delta: f_attn(x).  No residual."""
        h = self._film(
            self.attn_norm(x), loop_idx, self.attn_loop_scale, self.attn_loop_bias,
        )
        return self.dropout(self.attn(h, loop_idx=loop_idx, attention_mask=attention_mask))

    def forward_ffn(
        self, x: torch.Tensor, loop_idx: int = 0,
    ) -> torch.Tensor:
        """FFN sub-layer delta: f_ffn(x).  No residual."""
        h = self._film(
            self.ffn_norm(x), loop_idx, self.ffn_loop_scale, self.ffn_loop_bias,
        )
        return self.dropout(self.ffn(h, loop_idx=loop_idx))


class ExitGate(nn.Module):
    """Per-loop instantaneous exit probability with explicit loop identity."""

    def __init__(self, dim: int, loop_embed_dim: int = 16, max_loops: int = 4):
        super().__init__()
        self.dim = dim
        self.loop_embed_dim = loop_embed_dim
        self.max_loops = max_loops
        if loop_embed_dim > 0:
            self.loop_embedding = nn.Embedding(max_loops, loop_embed_dim)
            nn.init.normal_(self.loop_embedding.weight, mean=0.0, std=0.02)
            in_dim = dim + loop_embed_dim
        else:
            self.loop_embedding = None
            in_dim = dim
        self.proj = nn.Linear(in_dim, 1, bias=True)

    def forward(self, h: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        """h: [B,T,D] -> lambda_t [B]; loop_idx is 0-based."""
        # Use the final causal position so the loop-depth decision for
        # next-token prediction cannot depend on hidden states at positions
        # after the token being predicted.
        pooled = h[:, -1, :]
        if self.loop_embedding is not None:
            idx = max(0, min(int(loop_idx), self.max_loops - 1))
            loop_vec = self.loop_embedding.weight[idx].unsqueeze(0).expand(pooled.size(0), -1)
            pooled = torch.cat([pooled, loop_vec], dim=-1)
        return torch.sigmoid(self.proj(pooled)).squeeze(-1)



