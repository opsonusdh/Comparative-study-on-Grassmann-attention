"""LoopTransformer -- the unified model.

Three-paper architecture:

Paper 1 -- Memory   (DeepSeek):
    SWA / CSA / HCA mechanisms handle long-range context within
    the token sequence at each call to forward_attn().

Paper 2 -- Amnesia  (Kimi AttnRes, adapted):
    Attention-dependent residuals are used ONLY on the MLP/FFN branch
    inside a single Transformer pass. Each layer can learn a weighted
    residual over earlier MLP outputs from the current loop. The history
    is reset at every recurrent loop; recurrent loops themselves remain
    ordinary state-to-state applications of the shared Transformer stack.

Paper 3 -- Reasoning (Ouro LoopLM):
    The N-block stack is looped R times with SHARED weights.
    A per-loop exit gate models a discrete distribution over
    halting steps; training minimises an entropy-regularised
    expected loss (Ouro Eq 4) that prevents collapse to always
    running R loops.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .attention import CSAAttention
import math
from .mlp_attnres import MLPAttentionResidual
from .registers import CausalRegisterBank


from .block import ExitGate, TransformerBlock
from .config import LoopConfig, is_grassmann_attention
from .layers import RMSNorm, safe_eps


class LoopTransformer(nn.Module):
    def __init__(self, cfg: LoopConfig):
        super().__init__()
        self.cfg = cfg

        # ── Embedding factorization (optional) ──────────────────────
        # cfg.embed_dim / cfg.embed_dim_out let the input and output
        # embeddings each be smaller than cfg.dim, with a Linear
        # projection bridging the gap -- see LoopConfig's docstring for
        # the parameter-count math and why input/output are decoupled
        # rather than forced to match (Chung et al., "Rethinking
        # Embedding Coupling in Pre-trained Language Models": naive
        # symmetric ALBERT-style shrinkage of both sides specifically
        # hurts vocab-diverse/multilingual models). Both default to
        # cfg.dim (no factorization, no projection) when left unset --
        # existing configs are completely unaffected.
        embed_dim     = cfg.embed_dim if cfg.embed_dim is not None else cfg.dim
        embed_dim_out = cfg.embed_dim_out if cfg.embed_dim_out is not None else embed_dim

        self.tok_emb = nn.Embedding(cfg.vocab_size, embed_dim)
        # nn.Embedding's PyTorch default init is std=1 -- reasonable for a
        # standalone lookup table, but wrong once this matrix is also used
        # as a Linear projection weight (tie_embeddings=True, below):
        # summing `embed_dim` terms of std~1 each gives logits std~
        # sqrt(embed_dim), not O(1). Standard GPT-2/nanoGPT-style std=0.02
        # keeps logits well-scaled regardless of tying or factorization.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        self.embed_proj_in = (
            nn.Linear(embed_dim, cfg.dim, bias=False) if embed_dim != cfg.dim else None
        )

        # The Grassmann paper (§3.1) uses learned ABSOLUTE TOKEN positions
        # h_t = E(x_t) + P_t. This is intentionally distinct from
        # loop_positional_embedding below, which encodes recurrent-loop depth.
        # It is active only for the Grassmann architecture so the original
        # DeepSeek/SWA/CSA/HCA model and checkpoints remain unchanged.
        self.grassmann_positional_embedding = None
        if is_grassmann_attention(cfg.attention_type) and cfg.grassmann_positional_embedding:
            # Construct and initialize the optional table in an isolated RNG
            # context. nn.Embedding initializes itself in __init__, so merely
            # re-seeding an explicit init afterwards would still advance the
            # global RNG and shift every subsequently-created model parameter.
            # fork_rng restores the caller's CPU RNG state on exit, ensuring
            # the shared MoA7/RFC/MoE/controller weights are identical between
            # the position-on and position-off ablation at the same --seed.
            with torch.random.fork_rng(devices=[]):
                # Seed only the CPU default generator used by nn.Embedding's
                # constructor; torch.manual_seed() would also reset CUDA RNGs.
                position_seed = (
                    int(cfg.grassmann_positional_seed)
                    if cfg.grassmann_positional_seed is not None
                    else int(torch.initial_seed())
                )
                torch.random.default_generator.manual_seed(position_seed)
                position_embedding = nn.Embedding(
                    cfg.grassmann_positional_max_len, cfg.dim
                )
                nn.init.normal_(
                    position_embedding.weight, mean=0.0, std=0.01
                )
            self.grassmann_positional_embedding = position_embedding

        # N transformer blocks -- same weights used each loop (LoopLM)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.dim, n_heads=cfg.n_heads, head_dim=cfg.head_dim,
                layer_idx=i, ffn_hidden_dim=cfg.ffn_hidden_dim,
                attention_type=cfg.attention_type,
                grassmann_rank=cfg.grassmann_rank,
                grassmann_offsets=cfg.grassmann_offsets,
                grassmann_offset_count=cfg.grassmann_offset_count,
                grassmann_offset_spacing_base=cfg.grassmann_offset_spacing_base,
                grassmann_long_range_stride=cfg.grassmann_long_range_stride,
                grassmann_offset_scheme=cfg.grassmann_offset_scheme,
                csa_m=cfg.csa_m, csa_top_k=cfg.csa_top_k,
                hca_m_prime=cfg.hca_m_prime, sw_window=cfg.sw_window,
                deepseek_split_stride=cfg.deepseek_split_stride,
                groups=cfg.groups, group_dim=cfg.group_dim, rope_dim=cfg.rope_dim,
                moe_num_shared_experts=cfg.moe_num_shared_experts,
                moe_num_routed_experts=cfg.moe_num_routed_experts,
                moe_top_k=cfg.moe_top_k,
                moe_expert_hidden_dim=cfg.moe_expert_hidden_dim,
                max_loops=cfg.max_loops,
                activation_balance_weight=cfg.activation_balance_weight,
                moe_aux_loss_weight=cfg.moe_aux_loss_weight,
                activation_top_k=cfg.activation_top_k,
                activation_min_probability=cfg.activation_min_probability,
                activation_balance_tolerance=cfg.activation_balance_tolerance,
                activation_bias_update_speed=cfg.activation_bias_update_speed,
                moe_bias_update_speed=cfg.moe_bias_update_speed,
                activation_dictionary=cfg.activation_dictionary,
                activation_gate_mode=cfg.activation_gate_mode,
                expert_local_activation_routing=cfg.expert_local_activation_routing,
                dropout=cfg.dropout,
            )
            for i in range(cfg.n_layers)
        ])

        # Attention-dependent residuals are intentionally LOCAL to the MLP/FFN
        # path inside a single Transformer pass. They reset at every recurrent
        # loop, so no attention-residual connection crosses loop boundaries.
        self.mlp_attnres = nn.ModuleList(
            [MLPAttentionResidual(cfg.dim) for _ in range(cfg.n_layers)]
        )

        # Exit gate + LM head -- shared weights across loop steps (LoopLM §4.1)
        self.exit_gate  = ExitGate(cfg.dim, loop_embed_dim=cfg.exit_gate_loop_embed_dim, max_loops=cfg.max_loops)

        # Per-loop identity signal injected directly into the residual
        # stream, once per loop, before any of the max_loops blocks run --
        # see config.py's residual_loop_embedding/loop_controller docstrings
        # and LoopLLM_diagnosis_and_fix.md (RC3 + Tier 2.3) for the full
        # rationale. Both are additive/no-op at init so an existing
        # checkpoint that predates these flags can still be warm-started
        # from (loading with strict=False and these two left at init
        # values is a safe no-op until they've actually been trained).
        self.loop_embedding_residual = (
            nn.Embedding(cfg.max_loops, cfg.dim) if cfg.residual_loop_embedding else None
        )
        if self.loop_embedding_residual is not None:
            nn.init.normal_(self.loop_embedding_residual.weight, mean=0.0, std=0.02)  # matches tok_emb's init scale

        self.loop_positional_embedding = (
            nn.Parameter(torch.randn(cfg.max_loops, cfg.loop_positional_max_len, cfg.dim) * 0.01)
            if cfg.loop_positional_embedding else None
        )
        self.causal_register_bank = (
            CausalRegisterBank(
                cfg.dim,
                num_registers=cfg.causal_registers,
                update_scale=cfg.causal_register_update_scale,
            )
            if cfg.causal_register_bank else None
        )
        self.loop_transition_matrices = nn.ParameterList()
        if cfg.loop_transition_matrices:
            for _ in range(max(0, cfg.max_loops - 1)):
                w = torch.eye(cfg.dim)
                if cfg.loop_transition_init_scale > 0:
                    w = w + torch.randn(cfg.dim, cfg.dim) * cfg.loop_transition_init_scale
                self.loop_transition_matrices.append(nn.Parameter(w))

        self.loop_controller_proj = (
            nn.Linear(cfg.dim, 2 * cfg.dim) if cfg.loop_controller else None
        )
        if self.loop_controller_proj is not None:
            nn.init.zeros_(self.loop_controller_proj.weight)   # zero-init -> exact
            nn.init.zeros_(self.loop_controller_proj.bias)     # identity at init, same
                                                                # rationale as block.py's
                                                                # FiLM zero-init. Unlike
                                                                # the FiLM tables, this IS
                                                                # a real weight matrix and
                                                                # is intentionally left in
                                                                # the decayed optimizer
                                                                # group (name contains
                                                                # neither loop_scale,
                                                                # loop_bias, nor
                                                                # loop_embedding).

        self.loop_controller_bound = float(cfg.loop_controller_bound)
        self._last_loop_controller_stats = []

        self.final_norm = RMSNorm(cfg.dim)


        self.embed_proj_out = (
            nn.Linear(cfg.dim, embed_dim_out, bias=False) if embed_dim_out != cfg.dim else None
        )
        self.lm_head = nn.Linear(embed_dim_out, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # nn.Embedding.weight is [vocab, embed_dim]; nn.Linear(embed_dim_out,
            # vocab).weight is [vocab, embed_dim_out] -- only the same shape
            # (hence tie-able) when embed_dim_out == embed_dim, which
            # LoopConfig's validation already guarantees by the time we get
            # here whenever tie_embeddings=True (see config.py).
            self.lm_head.weight = self.tok_emb.weight

        # Set by compute_loss() after each call, for inspection/logging --
        # e.g. to confirm the CSA indexer is actually learning over the
        # course of training. Kept as a plain attribute rather than a third
        # return value from compute_loss() so the existing (total_loss,
        # step_losses) call signature stays exactly backward compatible.
        self.last_csa_aux_loss: Optional[torch.Tensor] = None
        self.last_activation_balance_loss: Optional[torch.Tensor] = None
        self.last_moe_aux_loss: Optional[torch.Tensor] = None
        self.last_activation_probs: Optional[torch.Tensor] = None
        self.last_moe_load: Optional[torch.Tensor] = None
        self.last_loop_supervision_loss: Optional[torch.Tensor] = None
        self.last_loop_refinement_loss: Optional[torch.Tensor] = None
        self.last_loop_task_loss: Optional[torch.Tensor] = None
        self.last_loop_monotonic_loss: Optional[torch.Tensor] = None
        self.last_joint_exit_loss: Optional[torch.Tensor] = None
        self.last_loop_improvements: Optional[torch.Tensor] = None
        self.last_loop_state_cosines: Optional[torch.Tensor] = None
        self.last_loop_transition_norms: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update_routing_biases(self) -> None:
        """Apply one non-gradient load-balancing update to FFN routers."""
        for block in self.blocks:
            updater = getattr(block.ffn, "_update_routing_biases", None)
            if updater is not None:
                updater()

    def num_parameters(self, effective: bool = False) -> int:
        """
        Physical parameter count (what's actually stored) by default.

        effective=True instead reports the parameter count an equivalent
        NON-weight-shared model would need to match this model's per-token
        compute depth: physical_params + (max_loops - 1) * shared_block_params,
        since self.blocks is reused max_loops times at runtime. This is the
        comparison papers like Ouro report ("a 1.4B LoopLM with 4 loops
        matches a 4B standard transformer") -- a compute/depth comparison,
        not a storage comparison.
        """
        total = sum(p.numel() for p in self.parameters())
        if not effective:
            return total
        shared = sum(p.numel() for p in self.blocks.parameters())
        return total + (self.cfg.max_loops - 1) * shared

    # ------------------------------------------------------------------
    # Internal: one full loop pass
    # ------------------------------------------------------------------

    def _run_layer(
        self,
        block_idx: int,
        loop_idx: int,
        state: torch.Tensor,
        mlp_history: tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one Transformer layer and return (new_state, ffn_delta)."""
        block = self.blocks[block_idx]
        attn_out = block.forward_attn(
            state, loop_idx=loop_idx, attention_mask=attention_mask,
        )
        state_after_attn = state + attn_out
        ffn_out = block.forward_ffn(
            state_after_attn, loop_idx=loop_idx,
        )
        if mlp_history:
            ffn_out = ffn_out + self.mlp_attnres[block_idx](list(mlp_history))
        return state_after_attn + ffn_out, ffn_out

    def _one_loop_impl(
        self, loop_idx: int, initial_state: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Run one Transformer pass with selective checkpointing.

        Attention-mixture layers are stateful (they accumulate lightweight
        routing diagnostics during a forward), and repeatedly checkpointing a
        stateful router can retain stale autograd references across shared
        recurrent iterations. We therefore checkpoint the ordinary layers
        individually, while executing mixture layers directly. This keeps most
        activation savings without compromising router correctness.
        """
        h = initial_state

        # For explicit zero-shot loop extrapolation, loop-indexed conditioning
        # has no learned entries beyond cfg.max_loops-1.  The only principled
        # way to continue the already-trained recurrent operator without
        # inventing new parameters is to hold the final trained conditioning
        # state fixed while continuing to apply the shared recurrent core.
        #
        # This is deliberately a *terminal hold*, not a claim that the model
        # learned a new loop identity for t > cfg.max_loops.  It isolates the
        # question we actually want to test: whether the learned state
        # transition (including the causal register bank) remains useful when
        # iterated beyond the training horizon.
        conditioning_loop_idx = min(int(loop_idx), self.cfg.max_loops - 1)

        if self.loop_positional_embedding is not None:
            if h.size(1) > self.cfg.loop_positional_max_len:
                raise ValueError(
                    f"sequence length {h.size(1)} exceeds loop_positional_max_len="
                    f"{self.cfg.loop_positional_max_len}"
                )
            h = h + self.cfg.loop_positional_scale * self.loop_positional_embedding[
                conditioning_loop_idx, :h.size(1)
            ].unsqueeze(0)
        if self.loop_embedding_residual is not None:
            idx = torch.full(
                (1,), conditioning_loop_idx, dtype=torch.long, device=h.device
            )
            h = h + self.loop_embedding_residual(idx).view(1, 1, -1)
        if self.loop_controller_proj is not None:
            # pure "what's been computed so far" signal -- input-dependent,
            # deliberately orthogonal to the loop_idx-only signal above.
            # Mean-pooling over sequence is a simplification (a per-token
            # controller is a reasonable future refinement, at the cost of
            # more compute); this is intentionally the smaller first step.
            counts = torch.arange(
                1,
                initial_state.size(1) + 1,
                device=initial_state.device,
                dtype=initial_state.dtype,
            ).view(1, -1, 1)
            pooled = initial_state.cumsum(dim=1) / counts
            ctrl_scale, ctrl_bias = self.loop_controller_proj(pooled).chunk(2, dim=-1)
            bound = self.loop_controller_bound
            ctrl_scale = bound * torch.tanh(ctrl_scale)
            ctrl_bias = bound * torch.tanh(ctrl_bias)
            self._last_loop_controller_stats.append({
                "scale_norm": float(ctrl_scale.detach().float().norm(dim=-1).mean().item()),
                "bias_norm": float(ctrl_bias.detach().float().norm(dim=-1).mean().item()),
            })
            h = h * (1.0 + ctrl_scale) + ctrl_bias
        else:
            self._last_loop_controller_stats.append(None)
        mlp_history: list[torch.Tensor] = []
        capture_layers = bool(getattr(self, "_capture_layer_states", False))
        captured_layers: list[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            checkpointable = self.cfg.grad_checkpointing and self.training
            if checkpointable:
                args = (h, *mlp_history)
                def layer_fn(*inputs, layer_idx=i, recurrent_idx=loop_idx,
                             mask=attention_mask):
                    state_in = inputs[0]
                    history = tuple(inputs[1:])
                    return self._run_layer(
                        layer_idx, recurrent_idx, state_in, history, mask
                    )
                h, ffn_out = grad_checkpoint(
                    layer_fn, *args, use_reentrant=False
                )
            else:
                h, ffn_out = self._run_layer(
                    i, loop_idx, h, tuple(mlp_history), attention_mask
                )
            mlp_history.append(ffn_out)
            if capture_layers:
                captured_layers.append(h.detach())

        if capture_layers:
            self._captured_layer_states = captured_layers

        return h

    def _one_loop(
        self, state: torch.Tensor, loop_idx: int = 0, *,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run exactly one shared recurrent Transformer pass."""
        return self._one_loop_impl(loop_idx, state, attention_mask)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """Embed token IDs and add Grassmann's learned absolute positions.

        The positional table is added once before the recurrent stack, as in
        paper §3.1. It is not re-added at each loop, which would conflate token
        position with recurrent depth and change the intended computation.
        """
        embedding = self.tok_emb(idx)
        if self.embed_proj_in is not None:
            embedding = self.embed_proj_in(embedding)

        position_embedding = getattr(self, "grassmann_positional_embedding", None)
        if position_embedding is not None:
            seq_len = idx.size(1)
            max_len = position_embedding.num_embeddings
            if seq_len > max_len:
                raise ValueError(
                    f"sequence length {seq_len} exceeds grassmann_positional_max_len={max_len}"
                )
            position_ids = torch.arange(seq_len, device=idx.device)
            embedding = embedding + position_embedding(position_ids).unsqueeze(0)
        return embedding

    def forward(
        self,
        idx:       torch.Tensor,
        max_loops: Optional[int] = None,
        *,
        allow_loop_extrapolation: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Full looped forward pass.

        Returns
        -------
        loop_logits : list of [B, T, V] tensors, one per loop step
        lambdas     : list of [B] tensors -- instantaneous exit probs λ_t
        """
        max_loops = max_loops if max_loops is not None else self.cfg.max_loops
        self._clear_ffn_aux_losses()
        state = self._embed_tokens(idx)

        if max_loops > self.cfg.max_loops:
            if not allow_loop_extrapolation:
                raise ValueError(
                    f"Requested {max_loops} loops, but checkpoint was trained/configured "
                    f"for {self.cfg.max_loops}. Pass allow_loop_extrapolation=True only "
                    "for explicit zero-shot recurrent-loop extrapolation."
                )
            if len(self.loop_transition_matrices) > 0:
                raise ValueError(
                    "Zero-shot loop extrapolation is not supported when explicit "
                    "loop-transition matrices are enabled, because those transitions "
                    "are not shared and there is no trained transition for loops "
                    f">={self.cfg.max_loops + 1}."
                )

        loop_logits: List[torch.Tensor] = []
        lambdas: List[torch.Tensor] = []
        self._last_loop_states = []
        self._last_loop_register_states = []
        self._last_loop_transition_deltas = []
        self._last_loop_controller_stats = []
        register_state = None

        for loop_idx in range(max_loops):
            state = self._one_loop(state, loop_idx=loop_idx)

            if self.causal_register_bank is not None:
                register_state, register_read = self.causal_register_bank(state, register_state)
                state = state + register_read
                self._last_loop_register_states.append(register_state.detach())

            if loop_idx < max_loops - 1 and self.loop_transition_matrices:
                before = state
                state = F.linear(state, self.loop_transition_matrices[loop_idx])
                self._last_loop_transition_deltas.append((state - before).detach())

            self._last_loop_states.append(state.detach())

            h_normed = self.final_norm(state)
            pre_logits = self.embed_proj_out(h_normed) if self.embed_proj_out is not None else h_normed
            loop_logits.append(self.lm_head(pre_logits))
            lambdas.append(self.exit_gate(h_normed, loop_idx=loop_idx))

        return loop_logits, lambdas

    @torch.no_grad()
    def forward_adaptive_exit(
        self,
        idx: torch.Tensor,
        *,
        exit_threshold: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run recurrent loops with genuine per-example early exit."""
        if not 0.0 < exit_threshold <= 1.0:
            raise ValueError("exit_threshold must be in (0, 1].")
        if idx.ndim != 2:
            raise ValueError(f"idx must be [B,T], got {tuple(idx.shape)}")

        B = idx.size(0)
        device = idx.device
        state = self._embed_tokens(idx)

        active = torch.arange(B, device=device)
        active_state = state
        selected_logits: list[Optional[torch.Tensor]] = [None] * B
        selected_loops = torch.full((B,), self.cfg.max_loops, device=device, dtype=torch.long)
        exit_cdf = torch.zeros(B, device=device, dtype=state.dtype)
        survival = torch.ones(B, device=device, dtype=state.dtype)

        register_state = None
        for loop_idx in range(self.cfg.max_loops):
            active_state = self._one_loop(active_state, loop_idx=loop_idx)
            if self.causal_register_bank is not None:
                register_state, register_read = self.causal_register_bank(active_state, register_state)
                active_state = active_state + register_read
            if loop_idx < self.cfg.max_loops - 1 and self.loop_transition_matrices:
                active_state = F.linear(active_state, self.loop_transition_matrices[loop_idx])
            h_normed = self.final_norm(active_state)
            pre_logits = self.embed_proj_out(h_normed) if self.embed_proj_out is not None else h_normed
            logits = self.lm_head(pre_logits)
            lam = self.exit_gate(h_normed, loop_idx=loop_idx).clamp(0.0, 1.0)

            if loop_idx < self.cfg.max_loops - 1:
                p_exit = survival[active] * lam
                new_survival = survival[active] * (1.0 - lam)
                threshold_hit = (exit_cdf[active] + p_exit) >= exit_threshold
            else:
                p_exit = survival[active]
                new_survival = torch.zeros_like(p_exit)
                threshold_hit = torch.ones_like(p_exit, dtype=torch.bool)

            cumulative = exit_cdf[active] + p_exit
            current_active = active
            if threshold_hit.any():
                done = current_active[threshold_hit]
                done_logits = logits[threshold_hit].detach()
                for row, original_idx in enumerate(done.tolist()):
                    selected_logits[original_idx] = done_logits[row]
                selected_loops[done] = loop_idx + 1
                exit_cdf[done] = cumulative[threshold_hit]

            if threshold_hit.all():
                break

            keep = ~threshold_hit
            active = current_active[keep]
            active_state = active_state[keep]
            if register_state is not None:
                register_state = register_state[keep]
            survival[active] = new_survival[keep]
            exit_cdf[active] = cumulative[keep]

        for i, value in enumerate(selected_logits):
            if value is None:
                raise RuntimeError(f"Example {i} failed to receive an exit result")

        return torch.stack([x for x in selected_logits if x is not None], dim=0), selected_loops, exit_cdf







    @torch.no_grad()

    # ------------------------------------------------------------------
    # Training objective  (Ouro §3.3, Eq 4)
    # ------------------------------------------------------------------

    def _sample_loop_count(self) -> int:
        """
        Sample a training-time loop count R ~ clip(Poisson(λ), min_loops,
        max_loops), used by compute_loss() when it isn't given an explicit
        max_loops. Only samples in training mode -- eval/inference always
        get the fixed cfg.max_loops ceiling (mirrors how grad_checkpointing
        similarly checks self.training elsewhere in this file), so
        validation loss stays a comparable, fixed-depth metric across runs
        rather than fluctuating with whatever depth happened to get drawn.

        Rationale: see LoopConfig.loop_sampling.
        """
        if not self.cfg.loop_sampling or not self.training:
            return self.cfg.max_loops
        lam = self.cfg.loop_sampling_lambda
        if lam is None:
            lam = self.cfg.max_loops / 2.0
        r = torch.poisson(torch.tensor(float(lam))).item()
        return int(max(self.cfg.min_loops, min(self.cfg.max_loops, round(r))))


    def _clear_csa_aux_losses(self) -> None:
        """Defensive reset before a forward pass: if something else (a
        direct model.forward() call, an interrupted previous compute_loss)
        left entries sitting in a CSAAttention instance's _aux_losses list,
        don't let them leak into this call's collection."""
        for module in self.modules():
            if isinstance(module, CSAAttention):
                module._aux_losses.clear()

    def _collect_csa_aux_loss(self) -> torch.Tensor:
        """
        Collect this call's indexer auxiliary KL-divergence loss (see
        CSAAttention._indexer_aux_loss) across every CSA layer, and clear
        each instance's list afterward.

        Filters to entries with requires_grad=True before combining: under
        gradient checkpointing, CSAAttention.forward() runs TWICE per loop
        (once in a throwaway no_grad dry-run internal to
        torch.utils.checkpoint, once again with grad enabled during
        backward recomputation) -- naively averaging both would silently
        double-count this loss whenever checkpointing is active, since the
        no_grad entry is a real (if unusable) tensor, not absent. Verified
        this interaction empirically while building it, not just reasoned
        through -- see tests/test_indexer_aux_loss.py.

        Returns a zero tensor (not None) when there's nothing to collect
        (e.g. no CSA layers at this depth, or eval mode), so callers can
        add it to the total loss unconditionally.
        """
        losses: List[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, CSAAttention):
                losses.extend(t for t in module._aux_losses if t.requires_grad)
                module._aux_losses.clear()
        if not losses:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(losses).mean()



    def _clear_ffn_aux_losses(self) -> None:
        for module in self.modules():
            if hasattr(module, "_clear_aux_losses"):
                module._clear_aux_losses()

    def _collect_ffn_aux_losses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        activation_losses = []
        moe_losses = []
        activation_probs = []
        moe_loads = []
        for module in self.modules():
            if hasattr(module, "collect_aux_loss"):
                a, m = module.collect_aux_loss()
                if a.requires_grad:
                    activation_losses.append(a)
                if m.requires_grad:
                    moe_losses.append(m)
            if hasattr(module, "last_activation_probs_mean"):
                p = module.last_activation_probs_mean
                if p is not None:
                    activation_probs.append(p)
            if hasattr(module, "last_expert_load"):
                l = module.last_expert_load
                if l is not None:
                    moe_loads.append(l)
        device = next(self.parameters()).device
        a = torch.stack(activation_losses).mean() if activation_losses else torch.zeros((), device=device)
        m = torch.stack(moe_losses).mean() if moe_losses else torch.zeros((), device=device)
        self.last_activation_balance_loss = a.detach()
        self.last_moe_aux_loss = m.detach()
        self.last_activation_probs = torch.stack(activation_probs).mean(dim=0) if activation_probs else None
        self.last_moe_load = torch.stack(moe_loads).mean(dim=0) if moe_loads else None
        return a, m

    def get_routing_debug(self) -> Dict[str, Dict[str, object]]:
        """
        Snapshot of routing diagnostics from the most recent forward() call,
        averaged across every FeedForward layer -- what scripts/generate.py's
        --debug flag prints per generation step.

        Deliberately independent of _collect_ffn_aux_losses()/compute_loss():
        this only READS each FeedForward's last_* properties (already
        populated as a side effect of forward() running _route_activations/
        _run_sparse_routed) and never clears anything, so it's safe to call
        after a plain forward() or generate() call -- no interaction with
        training's aux-loss bookkeeping.

        Returns {} if no FeedForward layer has run a forward pass yet
        (matches generate.py's `if not info: return`).
        """
        act_probs, act_hard, act_entropy = [], [], []
        moe_probs, moe_hard, moe_entropy = [], [], []

        for module in self.modules():
            if hasattr(module, "last_activation_probs_dense_mean"):
                p = module.last_activation_probs_dense_mean
                if p is not None:
                    act_probs.append(p)
            if hasattr(module, "last_activation_probs_mean"):
                p = module.last_activation_probs_mean
                if p is not None:
                    act_hard.append(p)
            if hasattr(module, "last_activation_entropy_mean"):
                e = module.last_activation_entropy_mean
                if e is not None:
                    act_entropy.append(e)
            if hasattr(module, "last_expert_probs_mean"):
                p = module.last_expert_probs_mean
                if p is not None:
                    moe_probs.append(p)
            if hasattr(module, "last_expert_load"):
                l = module.last_expert_load
                if l is not None:
                    moe_hard.append(l)
            if hasattr(module, "last_expert_entropy_mean"):
                e = module.last_expert_entropy_mean
                if e is not None:
                    moe_entropy.append(e)

        info: Dict[str, Dict[str, object]] = {}
        if act_probs or act_hard or act_entropy:
            info["activation"] = {
                "top_k": self.cfg.activation_top_k,
                "mean_probs": torch.stack(act_probs).mean(dim=0).tolist() if act_probs else None,
                "hard_usage": torch.stack(act_hard).mean(dim=0).tolist() if act_hard else None,
                "mean_entropy": torch.stack(act_entropy).mean().item() if act_entropy else None,
            }
        if moe_probs or moe_hard or moe_entropy:
            info["moe"] = {
                "top_k": self.cfg.moe_top_k,
                "mean_probs": torch.stack(moe_probs).mean(dim=0).tolist() if moe_probs else None,
                "hard_usage": torch.stack(moe_hard).mean(dim=0).tolist() if moe_hard else None,
                "mean_entropy": torch.stack(moe_entropy).mean().item() if moe_entropy else None,
            }
        return info

    def clear_routing_debug(self) -> None:
        """Reset get_routing_debug()'s per-layer snapshot fields so a stale
        forward pass's numbers can't leak into the next one -- e.g.
        scripts/generate.py calls this before each new generation step.
        Mirrors _clear_ffn_aux_losses()'s hasattr-dispatch pattern, but
        targets the debug-only fields, not the aux-loss lists -- keeps this
        fully independent of training's loss bookkeeping."""
        for module in self.modules():
            if hasattr(module, "_clear_routing_diagnostics"):
                module._clear_routing_diagnostics()


    def compute_loss(
        self,
        idx:       torch.Tensor,
        max_loops: Optional[int] = None,
        beta:      Optional[float] = None,
        teacher:             Optional["LoopTransformer"] = None,
        distill_alpha:       float = 0.5,
        distill_temperature: float = 2.0,
        late_loop_weight:     float = 0.0,
        target_mask:          Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Entropy-regularised expected loss (Ouro Eq 4), computed PER-EXAMPLE
        and then averaged over the batch -- p_φ(t|x) is explicitly a function
        of x (Ouro Eq 3), so different sequences in a batch can prefer
        different exit points; collapsing to a single batch-wide scalar
        before combining would throw that away.

            L = Σ_t  p_φ(t|x) · L^(t)(x)  −  β · H(p_φ(·|x))

        Exit distribution via survival model (Ouro Eq 3), per example:
            p̃_t(x) = λ_t(x) · ∏_{j<t}(1 − λ_j(x))
            p_T(x)  = ∏_{j<T}(1 − λ_j(x))         (remaining mass → last step)

        The entropy term prevents the gate from collapsing to always
        exiting at step T_max (the self-reinforcement failure mode
        described in Ouro §3.3).

        If max_loops isn't given explicitly and self.training is True and
        cfg.loop_sampling is on, the loop count for THIS call is sampled
        via _sample_loop_count() rather than fixed at cfg.max_loops --
        so step_losses' length below varies call to call in that regime;
        it always reflects however many loops actually ran this time.

        Also adds cfg.csa_aux_loss_weight * (CSA indexer's KL-divergence
        auxiliary loss), gradient-isolated so it only trains the indexer
        submodules -- see CSAAttention._indexer_aux_loss. Its value is
        stashed on self.last_csa_aux_loss afterward for inspection/logging
        (e.g. to confirm the indexer is actually learning over training),
        without changing this method's return signature.

        Late-loop reinforcement (optional): `late_loop_weight` adds a direct
        cross-entropy term on the final executed loop. This deliberately keeps
        the normal recurrent objective active while increasing the training
        pressure on the deepest recurrent state. It reuses the final-loop CE
        already computed below, so it introduces no second forward pass and no
        second copy of vocabulary logits.

        Knowledge distillation (optional): pass `teacher`, another
        LoopTransformer instance (any size/config, same vocab_size), to
        blend the standard cross-entropy loss with a temperature-scaled
        KL-divergence toward the teacher's output distribution (Hinton
        et al.'s classic soft-label distillation, still the standard
        production approach as of 2026 -- 5-30x inference cost reduction
        typically retaining ~95-97% of teacher quality is the commonly
        reported range, not a guarantee for any specific setup).

            L^(t)_combined = alpha * L^(t)_CE  +  (1-alpha) * T^2 * KL(teacher || student_t)

        then the SAME p_exit weighting and entropy regularization above
        apply to the combined per-step loss. The teacher runs once under
        torch.no_grad() at its OWN cfg.max_loops (its trained ceiling,
        which may differ from the student's), using its final loop's
        output as the fixed target distribution; the teacher's own
        training-mode flag is saved and restored around the call, so this
        doesn't have side effects on a teacher you're still training
        elsewhere.

        Worth knowing before reaching for a much larger teacher: Distillation
        Scaling Laws (Busbridge et al., 2025) found a real capacity-gap
        effect -- too large a teacher/student ratio can hurt transfer
        rather than help it, with optimal teacher scale tracking student
        scale roughly linearly rather than "bigger is strictly better."
        This implementation is classical (fixed-corpus) distillation, not
        the newer on-policy variant (student generates its own rollouts,
        teacher supervises those) that 2026 literature reports further
        gains from -- a reasonable next step, not implemented here.

        Returns
        -------
        total_loss  : scalar tensor (differentiable)
        step_losses : [T] tensor of per-loop LM losses, batch-averaged
                      (detached, for logging only); T = loops run this call.
                      When teacher is given, these are the COMBINED
                      (CE + KD) per-step losses, not CE alone.
        """
        if target_mask is not None:
            expected_shape = (idx.size(0), max(0, idx.size(1) - 1))
            if target_mask.shape != expected_shape:
                raise ValueError(
                    f"target_mask must be {expected_shape}, got {tuple(target_mask.shape)}"
                )
            target_mask = target_mask.to(device=idx.device, dtype=torch.bool)
            if not bool(target_mask.any()):
                raise ValueError("target_mask excludes every next-token target.")
        if teacher is not None:
            if teacher.cfg.vocab_size != self.cfg.vocab_size:
                raise ValueError(
                    f"teacher.cfg.vocab_size ({teacher.cfg.vocab_size}) != "
                    f"self.cfg.vocab_size ({self.cfg.vocab_size}) -- logit-level "
                    f"distillation requires a shared vocabulary/tokenizer."
                )
            if not (0.0 <= distill_alpha <= 1.0):
                raise ValueError(f"distill_alpha must be in [0, 1], got {distill_alpha}")
            if distill_temperature <= 0:
                raise ValueError(f"distill_temperature must be > 0, got {distill_temperature}")
        if late_loop_weight < 0:
            raise ValueError(f"late_loop_weight must be >= 0, got {late_loop_weight}")

        max_loops = max_loops if max_loops is not None else self._sample_loop_count()
        beta      = self.cfg.beta_entropy if beta is None else beta

        self._clear_csa_aux_losses()
        loop_logits, lambdas = self.forward(idx, max_loops)
        B = idx.size(0)

        # ── Per-example, per-loop cross-entropy L^(t)(x) ─────────────
        # reduction="none" + reshape keeps the batch dimension alive so
        # each example can be weighted by its own p_φ(t|x) below.
        step_losses_per_example = []   # will hold B-vectors, one per loop
        target_mask_f = (
            target_mask.to(dtype=loop_logits[0].dtype)
            if target_mask is not None
            else None
        )
        for logits in loop_logits:
            lgt = logits[:, :-1].contiguous()      # [B, T-1, V]
            tgt = idx[:, 1:].contiguous()          # [B, T-1]
            ce = F.cross_entropy(
                lgt.view(-1, lgt.size(-1)), tgt.view(-1), reduction="none",
            ).view(B, -1)
            if target_mask_f is None:
                ce = ce.mean(dim=1)
            else:
                counts = target_mask_f.sum(dim=1)
                if bool((counts == 0).any()):
                    raise ValueError("target_mask excludes every target in at least one example.")
                ce = (ce * target_mask_f).sum(dim=1) / counts
            step_losses_per_example.append(ce)
        step_losses_per_example = torch.stack(step_losses_per_example, dim=1)  # [B, T]
        primary_step_losses = step_losses_per_example

        # Distinct task per loop, without replacing the primary LM objective.
        # Loop t predicts t+1 tokens ahead (1-, 2-, 3-, 4-token horizons).
        # This makes the auxiliary objective genuinely depth-specific while
        # every loop still receives the normal next-token CE signal.
        loop_task_loss = torch.zeros((), device=idx.device, dtype=primary_step_losses.dtype)
        if self.cfg.loop_task_weight > 0 and self.cfg.loop_task_mode == "horizon":
            task_losses = []
            for t, logits in enumerate(loop_logits):
                horizon = t + 1
                if idx.size(1) <= horizon:
                    continue
                pred = logits[:, :-(horizon)].contiguous()
                tgt = idx[:, horizon:].contiguous()
                task_losses.append(F.cross_entropy(
                    pred.reshape(-1, pred.size(-1)), tgt.reshape(-1), reduction="mean"
                ))
            if task_losses:
                w = torch.arange(1, len(task_losses) + 1, device=idx.device, dtype=primary_step_losses.dtype)
                w = w / w.sum() * len(task_losses)
                loop_task_loss = torch.stack(task_losses).mul(w).mean()
        self.last_loop_task_loss = loop_task_loss.detach()

        if teacher is not None:
            teacher_was_training = teacher.training
            teacher.eval()
            with torch.no_grad():
                teacher_loop_logits, _ = teacher.forward(idx, max_loops=teacher.cfg.max_loops)
                teacher_logits = teacher_loop_logits[-1][:, :-1].detach()  # [B, T-1, V]
                teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)
            if teacher_was_training:
                teacher.train()

            kd_step_losses = []
            for logits in loop_logits:
                student_log_probs = F.log_softmax(
                    logits[:, :-1].contiguous() / distill_temperature, dim=-1,
                )  # [B, T-1, V]
                kd = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
                if target_mask_f is None:
                    kd = kd.mean(dim=1)
                else:
                    kd = (kd * target_mask_f).sum(dim=1) / target_mask_f.sum(dim=1)
                kd_step_losses.append(kd * (distill_temperature ** 2))  # [B]
            kd_step_losses = torch.stack(kd_step_losses, dim=1)  # [B, T]

            step_losses_per_example = (
                distill_alpha * step_losses_per_example
                + (1.0 - distill_alpha) * kd_step_losses
            )

        # ── Per-example exit distribution p_φ(t|x) via survival model ──
        p_exit = []
        survival = torch.ones_like(lambdas[0])     # [B]
        for t in range(max_loops):
            lam_t = lambdas[t]                     # [B]
            if t < max_loops - 1:
                p_t = lam_t * survival
                survival = survival * (1.0 - lam_t)
            else:
                p_t = survival                     # absorb remaining mass
            p_exit.append(p_t)
        p_exit = torch.stack(p_exit, dim=1)         # [B, T], rows sum to ~1

        # ── Objective, averaged over the batch ─────────────────────────
        direct_depth_loss = step_losses_per_example.mean()
        expected_loss = (p_exit * step_losses_per_example).sum(dim=1).mean()
        entropy       = -(p_exit * p_exit.clamp(min=safe_eps(p_exit.dtype)).log()).sum(dim=1).mean()
        joint_exit_loss = expected_loss - beta * entropy
        w_exit = self.cfg.joint_exit_loss_weight
        total_loss = (1.0 - min(w_exit, 1.0)) * direct_depth_loss + min(w_exit, 1.0) * joint_exit_loss
        self.last_joint_exit_loss = joint_exit_loss.detach()

        # Optional direct supervision for every executed loop. The weights
        # increase linearly with depth and are normalized to sum to one, so
        # this behaves as an auxiliary objective rather than changing the
        # loss scale merely because max_loops is larger. The default weight
        # is 0.0, preserving the original training objective exactly.
        if self.cfg.loop_supervision_weight > 0.0:
            n_loops = step_losses_per_example.size(1)
            loop_weights = torch.arange(
                1, n_loops + 1,
                device=step_losses_per_example.device,
                dtype=step_losses_per_example.dtype,
            )
            loop_weights = loop_weights / loop_weights.sum()
            loop_supervision_loss = (
                step_losses_per_example * loop_weights.unsqueeze(0)
            ).sum(dim=1).mean()
        else:
            loop_supervision_loss = torch.zeros(
                (),
                device=step_losses_per_example.device,
                dtype=step_losses_per_example.dtype,
            )
        self.last_loop_supervision_loss = loop_supervision_loss.detach()
        total_loss = total_loss + self.cfg.loop_supervision_weight * loop_supervision_loss

        csa_aux_loss = self._collect_csa_aux_loss()
        self.last_csa_aux_loss = csa_aux_loss.detach()
        total_loss = total_loss + self.cfg.csa_aux_loss_weight * csa_aux_loss


        activation_balance_loss, moe_aux_loss = self._collect_ffn_aux_losses()
        total_loss = total_loss + self.cfg.activation_balance_weight * activation_balance_loss
        total_loss = total_loss + self.cfg.moe_aux_loss_weight * moe_aux_loss

        # Per-example refinement: the next loop is rewarded only when it beats
        # the previous loop by at least a small margin. ReLU keeps this a soft
        # auxiliary signal instead of forcing impossible monotonicity on every
        # stochastic batch.
        if self.cfg.loop_refinement_weight > 0 and primary_step_losses.size(1) > 1:
            improvements = primary_step_losses[:, :-1] - primary_step_losses[:, 1:]
            refinement_loss = F.relu(self.cfg.loop_refinement_margin - improvements).mean()
        else:
            refinement_loss = torch.zeros((), device=idx.device, dtype=primary_step_losses.dtype)
        self.last_loop_refinement_loss = refinement_loss.detach()
        total_loss = total_loss + self.cfg.loop_refinement_weight * refinement_loss

        # Legacy batch-level monotonic penalty is kept separate and disabled
        # by default. It remains useful for controlled ablations.
        if self.cfg.loop_monotonic_weight > 0 and primary_step_losses.size(1) > 1:
            mean_losses = primary_step_losses.mean(dim=0)
            monotonic_loss = F.relu(
                mean_losses[1:] - mean_losses[:-1] + self.cfg.loop_monotonic_margin
            ).mean()
        else:
            monotonic_loss = torch.zeros((), device=idx.device, dtype=primary_step_losses.dtype)
        self.last_loop_monotonic_loss = monotonic_loss.detach()
        total_loss = total_loss + self.cfg.loop_monotonic_weight * monotonic_loss

        total_loss = total_loss + self.cfg.loop_task_weight * loop_task_loss

        # Explicitly reinforce the deepest recurrent exit without replacing the
        # existing objective. `primary_step_losses` contains the original CE
        # graph, so reusing its final column adds only a tiny scalar operation
        # and does not retain another set of [B, T, V] vocabulary logits.
        late_loop_ce = primary_step_losses[:, -1].mean()
        self.last_late_loop_loss = late_loop_ce.detach()
        total_loss = total_loss + late_loop_weight * late_loop_ce

        # Shortcut consistency: earlier checkpoints should remain useful
        # approximations of the final route. This is intentionally separate
        # from the refinement margin: it stabilizes variable-depth inference
        # without forcing the hidden states themselves to collapse.
        if self.cfg.shortcut_consistency_weight > 0 and len(loop_logits) > 1:
            teacher_logits = loop_logits[-1][:, :-1].detach()
            T = self.cfg.shortcut_consistency_temperature
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            sc_terms = []
            for lgts in loop_logits[:-1]:
                student_logp = F.log_softmax(lgts[:, :-1] / T, dim=-1)
                sc_terms.append(F.kl_div(student_logp, teacher_probs, reduction="batchmean") * (T * T))
            if sc_terms:
                total_loss = total_loss + self.cfg.shortcut_consistency_weight * torch.stack(sc_terms).mean()

        detached_losses = primary_step_losses.mean(dim=0).detach()
        if detached_losses.numel() > 1:
            self.last_loop_improvements = (detached_losses[:-1] - detached_losses[1:]).detach()
        else:
            self.last_loop_improvements = torch.empty(0, device=idx.device)

        # State-cosine diagnostic is intentionally non-differentiable.
        # It tells us whether recurrence is actually moving the latent state.
        if hasattr(self, "_last_loop_states") and len(self._last_loop_states) > 1:
            cos_terms = []
            for a, b in zip(self._last_loop_states[:-1], self._last_loop_states[1:]):
                aa = a.mean(dim=1)
                bb = b.mean(dim=1)
                cos_terms.append(torch.nn.functional.cosine_similarity(aa, bb, dim=-1).mean())
            self.last_loop_state_cosines = torch.stack(cos_terms).detach()
        else:
            self.last_loop_state_cosines = torch.empty(0, device=idx.device)
        return total_loss, detached_losses  # [T]

    # ------------------------------------------------------------------
    # Inference with adaptive early exit  (Ouro §3.2, Q-exit criterion)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          Optional[int] = None,
        n_loops:        Optional[int] = None,
        exit_threshold: float = 0.8,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        n_loops=None   → adaptive exit: stop at first loop where CDF ≥ threshold
                          (inherently bounded by cfg.max_loops -- the Q-exit
                          loop below can only stop EARLIER than run_loops,
                          never later, so this path never extrapolates on
                          its own)
        n_loops=k      → always run exactly k loops (fixed depth). If
                          k > cfg.max_loops, this explicitly extrapolates
                          past the trained depth -- see warning below.

        Depth-extrapolation warning: A Mechanistic Analysis of Looped
        Reasoning Language Models (arXiv:2604.11791) specifically found
        that Ouro-style looped models show "structural drift" rather than
        clean convergence when run past their trained loop count, unlike
        some other looped architectures that stayed stable under the same
        test. Our LoopLM design follows Ouro's recipe, so this risk isn't
        ruled out for this model either. We warn rather than block --
        n_loops > max_loops may be exactly what you're deliberately
        testing -- but treat results at that depth as unverified.
        """
        if n_loops is not None and n_loops > self.cfg.max_loops:
            warnings.warn(
                f"generate(n_loops={n_loops}) exceeds cfg.max_loops="
                f"{self.cfg.max_loops} (the trained ceiling). Ouro-style "
                f"looped models have specifically been found to show "
                f"structural drift rather than clean convergence when "
                f"extrapolated past their trained loop count (arXiv:"
                f"2604.11791) -- results at this depth are unverified for "
                f"this model. Proceeding anyway since this may be "
                f"deliberate.",
                stacklevel=2,
            )
        run_loops = n_loops if n_loops is not None else self.cfg.max_loops

        for _ in range(max_new_tokens):
            if n_loops is not None:
                loop_logits, _ = self.forward(idx, run_loops)
                chosen = loop_logits[-1]
            else:
                chosen, _, _ = self.forward_adaptive_exit(
                    idx,
                    exit_threshold=exit_threshold,
                )

            logits = chosen[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.where(logits < v[:, [-1]],
                                     torch.full_like(logits, float("-inf")), logits)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx
