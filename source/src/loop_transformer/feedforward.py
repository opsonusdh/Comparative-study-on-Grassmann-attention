"""DeepSeekMoE-style sparse FFN with shared experts and activation routing.

The original dense SwiGLU FFN is replaced by a compact DeepSeekMoE-inspired
Mixture-of-Experts block: one always-on shared expert plus top-k routed
fine-grained experts.  The activation dictionary is configurable. The
production-default path preserves the historical 4-way dictionary, while the
MoA7 preset enables the 7-way token/state-dependent dictionary used in the
RFC research runs.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import safe_eps
from .routing import RoutingContextAttention


class SwiGLUExpert(nn.Module):
    """A small gated FFN expert with a 7-way activation dictionary."""

    def __init__(self, dim: int, hidden_dim: int, num_activations: int = 4, create_local_router: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.w3.weight)
        self.num_activations = num_activations
        # Expert-local activation router: each expert can learn a distinct
        # token-dependent preference over the shared activation dictionary.
        # LEAN VARIANT: only allocated when actually used -- verified
        # ablation (moa7_nolocal_123 full_eval.json) shows this contributes
        # ~0.00036 nats, so paying ~448 params/expert for it is a bad trade.
        self.activation_router = nn.Linear(dim, num_activations, bias=False) if create_local_router else None

    @staticmethod
    def _activation_stack(x: torch.Tensor, num_activations: int) -> torch.Tensor:
        """Return the configured activation dictionary.

        MoA7 order is a fixed research contract: Identity, GELU, SiLU, ReLU,
        LeakyReLU, bounded-ReLU-squared, tanh. The legacy 4-way dictionary is
        retained exactly for backward compatibility with older checkpoints.
        """
        relu = F.relu(x)
        bounded_relu_sq = relu.square() / (1.0 + relu)
        if num_activations == 7:
            return torch.stack((
                x,
                F.gelu(x),
                F.silu(x),
                relu,
                F.leaky_relu(x, negative_slope=0.01),
                bounded_relu_sq,
                torch.tanh(x),
            ), dim=-1)
        if num_activations == 4:
            return torch.stack((
                F.silu(x),
                F.gelu(x),
                bounded_relu_sq,
                F.elu(x),
            ), dim=-1)
        raise ValueError(f"unsupported activation dictionary width: {num_activations}")

    def forward(
        self,
        x: torch.Tensor,
        activation_probs: torch.Tensor,
        global_activation_logits: torch.Tensor | None = None,
        use_local_activation_routing: bool = False,
        loop_idx: int = 0,
    ) -> torch.Tensor:
        gate = self.w1(x)
        value = self.w2(x)

        activations = self._activation_stack(gate, self.num_activations)

        if use_local_activation_routing:
            if global_activation_logits is None:
                raise ValueError("global_activation_logits required for local activation routing")
            # Global routing supplies the shared, loop-conditioned prior; each
            # expert adds an independent token-dependent correction before the
            # sigmoid gate. The sigmoid gates are intentionally independent
            # rather than simplex-normalized, so the learned gate can also
            # control total activation strength.
            local_logits = global_activation_logits + self.activation_router(x)
            local_probs = torch.sigmoid(local_logits)
        else:
            local_probs = activation_probs

        mixed_gate = (activations * local_probs.unsqueeze(-2)).sum(dim=-1)

        return self.w3(mixed_gate * value)


class FeedForward(nn.Module):
    """DeepSeekMoE-style shared + routed FFN with configurable activation routing.

    Compute is controlled by ``expert_hidden_dim`` and ``moe_top_k``.  The
    recommended small-model setting is one shared expert + one routed expert,
    each with half the original dense FFN width, keeping active FFN multiply
    width approximately equal to the old dense SwiGLU while increasing total
    parameter capacity through the inactive routed experts.

    ``activation_top_k`` controls how many activations can be active
    for each token (1..4). ``activation_balance_loss`` is KL(mean routing ||
    uniform), so a non-uniform batch-average activation distribution is explicitly
    penalized. This does *not* force every token to use all activations: individual
    tokens remain free to choose a sparse specialist set. The same design is used
    for the expert router via
    a standard top-k load-balance auxiliary loss.
    """

    NUM_ACTIVATIONS = 7

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        num_shared_experts: int = 1,
        num_routed_experts: int = 4,
        moe_top_k: int = 1,
        expert_hidden_dim: Optional[int] = None,
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
    ):
        super().__init__()
        activation_dictionary = str(activation_dictionary)
        activation_gate_mode = str(activation_gate_mode)
        if activation_dictionary not in {"legacy4", "moa7"}:
            raise ValueError(f"unsupported activation_dictionary: {activation_dictionary}")
        if activation_gate_mode not in {"sigmoid", "softmax"}:
            raise ValueError(f"unsupported activation_gate_mode: {activation_gate_mode}")
        self.NUM_ACTIVATIONS = 7 if activation_dictionary == "moa7" else 4
        if num_shared_experts < 1:
            raise ValueError("num_shared_experts must be >= 1")
        if num_routed_experts < 1:
            raise ValueError("num_routed_experts must be >= 1")
        if not 1 <= moe_top_k <= num_routed_experts:
            raise ValueError("moe_top_k must be in [1, num_routed_experts]")
        if max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if not 1 <= activation_top_k <= self.NUM_ACTIVATIONS:
            raise ValueError(
                f"activation_top_k must be in [1, {self.NUM_ACTIVATIONS}]"
            )

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.moe_top_k = moe_top_k
        self.max_loops = max_loops
        self.activation_balance_weight = activation_balance_weight
        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.activation_top_k = activation_top_k
        self.activation_min_probability = float(activation_min_probability)
        self.activation_balance_tolerance = float(activation_balance_tolerance)
        if not 0.0 <= self.activation_min_probability < 1.0 / self.NUM_ACTIVATIONS:
            raise ValueError("activation_min_probability must be in [0, 1/num_activations)")
        if self.activation_balance_tolerance < 0.0:
            raise ValueError("activation_balance_tolerance must be >= 0")
        self.activation_bias_update_speed = float(activation_bias_update_speed)
        self.moe_bias_update_speed = float(moe_bias_update_speed)
        self.activation_dictionary = activation_dictionary
        self.activation_gate_mode = activation_gate_mode
        self.expert_local_activation_routing = bool(expert_local_activation_routing)

        # Non-gradient loss-free-balancing biases. They are updated once per
        # optimizer step by LoopTransformer.update_routing_biases().
        self.register_buffer(
            "activation_bias",
            torch.zeros(self.NUM_ACTIVATIONS),
            persistent=True,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_routed_experts),
            persistent=True,
        )

        if expert_hidden_dim is None:
            # Preserve roughly the old dense active width when one shared
            # expert and top-1 routed expert are active simultaneously.
            active_experts = num_shared_experts + moe_top_k
            expert_hidden_dim = max(1, hidden_dim // active_experts)
        if expert_hidden_dim < 1:
            raise ValueError("expert_hidden_dim must be >= 1")
        self.expert_hidden_dim = expert_hidden_dim

        self.shared_experts = nn.ModuleList(
            SwiGLUExpert(dim, expert_hidden_dim, self.NUM_ACTIVATIONS, create_local_router=self.expert_local_activation_routing)
            for _ in range(num_shared_experts)
        )
        self.routed_experts = nn.ModuleList(
            SwiGLUExpert(dim, expert_hidden_dim, self.NUM_ACTIVATIONS, create_local_router=self.expert_local_activation_routing)
            for _ in range(num_routed_experts)
        )

        # DeepSeek-style token router for fine-grained routed experts.
        # FiLM-style per-loop conditioning (see activation_router below for
        # the full reasoning) -- MoE previously had NO loop-awareness at
        # all, unlike activation routing's (weaker) additive-only bias.
        self.expert_router = nn.Linear(dim, num_routed_experts, bias=False)
        self.expert_loop_scale = nn.Parameter(torch.ones(max_loops, num_routed_experts))
        self.expert_loop_bias = nn.Parameter(torch.zeros(max_loops, num_routed_experts))

        # Tiny activation router. FiLM-style per-loop conditioning: the
        # shared router's raw logits are scaled AND shifted per loop_idx,
        # not just shifted. Pure additive bias (raw + loop_bias[loop_idx])
        # only lets a loop shift every token's baseline preference by the
        # same constant vector, independent of the token itself -- it
        # can't express "loop 1 prefers activation A for tokens with
        # property P, but loop 3 prefers B for those same tokens", since
        # that needs the loop index to interact WITH the token's features,
        # not just offset the result after the fact. Scaling raw_logits *
        # loop_scale[loop_idx] first lets each loop amplify or dampen how
        # strongly the (shared) weight matrix's per-activation projection
        # of x drives that logit, before the shift is added -- cheap (just
        # 2*max_loops*NUM_ACTIVATIONS extra scalars, no new weight matrix
        # or embedding-size hyperparameter to pick) and still keeps the
        # underlying feature extraction (the w_i . x projections
        # themselves) shared and sample-efficient across loops, unlike
        # giving every loop a fully separate router weight matrix would.
        # loop_scale inits to 1 and loop_bias to 0, so at step 0 this is
        # exactly activation_router(x) with no loop conditioning at all --
        # matches this codebase's existing zero-init-starts-as-identity
        # convention (see mlp_attnres.py).
        self.activation_router = nn.Linear(dim, self.NUM_ACTIVATIONS, bias=False)
        self.loop_scale = nn.Parameter(torch.ones(max_loops, self.NUM_ACTIVATIONS))
        self.loop_bias = nn.Parameter(torch.zeros(max_loops, self.NUM_ACTIVATIONS))

        # Context-aware routing: a tiny attention pool lets routing decisions
        # depend on the surrounding sequence, not only the token's own vector.
        # This is deliberately separate from the main attention so the router
        # adds O(T*d) context work rather than another O(T^2) attention map.
        self.routing_context = RoutingContextAttention(dim)
        self.activation_context_proj = nn.Linear(dim + min(64, max(16, dim // 8)), self.NUM_ACTIVATIONS, bias=False)
        self.expert_context_proj = nn.Linear(dim + min(64, max(16, dim // 8)), num_routed_experts, bias=False)
        nn.init.zeros_(self.activation_context_proj.weight)
        nn.init.zeros_(self.expert_context_proj.weight)

        # Populated during forward for the training objective / debug logging.
        self._activation_aux_losses = []
        self._moe_aux_losses = []
        self._last_activation_probs_mean: Optional[torch.Tensor] = None
        self._last_activation_probs_dense_mean: Optional[torch.Tensor] = None
        self._last_activation_entropy_mean: Optional[torch.Tensor] = None
        self._last_expert_probs_mean: Optional[torch.Tensor] = None
        self._last_expert_load: Optional[torch.Tensor] = None
        self._last_expert_entropy_mean: Optional[torch.Tensor] = None

    @property
    def last_activation_probs_mean(self) -> Optional[torch.Tensor]:
        """Post-sparsify (hard, actually-dispatched) activation usage mean."""
        return self._last_activation_probs_mean

    @property
    def last_activation_probs_dense_mean(self) -> Optional[torch.Tensor]:
        """Pre-sparsify (dense, soft) activation routing mean -- what the
        balance loss trains against; see _route_activations."""
        return self._last_activation_probs_dense_mean

    @property
    def last_activation_entropy_mean(self) -> Optional[torch.Tensor]:
        """Mean per-token entropy of the dense activation distribution."""
        return self._last_activation_entropy_mean

    @property
    def last_expert_probs_mean(self) -> Optional[torch.Tensor]:
        return self._last_expert_probs_mean

    @property
    def last_expert_load(self) -> Optional[torch.Tensor]:
        return self._last_expert_load

    @property
    def last_expert_entropy_mean(self) -> Optional[torch.Tensor]:
        """Mean per-token entropy of the dense MoE router distribution."""
        return self._last_expert_entropy_mean

    def _clear_routing_diagnostics(self) -> None:
        """Reset get_routing_debug()'s snapshot fields. Deliberately separate
        from _clear_aux_losses(): that one clears the loss-accumulation
        lists compute_loss() depends on; this one only clears the debug
        snapshot generate.py reads, so the two never interact."""
        self._last_activation_probs_mean = None
        self._last_activation_probs_dense_mean = None
        self._last_activation_entropy_mean = None
        self._last_expert_probs_mean = None
        self._last_expert_load = None
        self._last_expert_entropy_mean = None

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        """Mean per-row entropy (nats) of a [..., K] probability distribution.
        Used only for debug diagnostics (get_routing_debug), not the loss."""
        eps = safe_eps(probs.dtype)
        row_entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
        return row_entropy.mean()

    def _activation_balance_loss(
        self,
        dense_probs: torch.Tensor,
        hard_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Keep population diversity while allowing token-level specialization.

        The dense router has a differentiable per-token floor. Its batch mean
        is allowed to move freely inside a tolerance band around uniform. If
        sparse top-k dispatch is enabled, we additionally discourage the *actual
        dispatched* batch mass of any activation from falling below the same
        floor. Thus an activation can be nearly absent for one token while still
        remaining alive across the population.
        """
        dense_q = dense_probs.mean(dim=0)
        uniform = 1.0 / dense_probs.size(-1)
        lower = max(self.activation_min_probability, uniform - self.activation_balance_tolerance)
        upper = min(1.0, uniform + self.activation_balance_tolerance)
        dense_below = F.relu(lower - dense_q)
        dense_above = F.relu(dense_q - upper)
        loss = (dense_below.square() + dense_above.square()).mean()

        if hard_probs is not None:
            hard_q = hard_probs.mean(dim=0)
            hard_below = F.relu(self.activation_min_probability - hard_q)
            loss = loss + hard_below.square().mean()
        return loss

    @staticmethod
    def _moe_load_balance_loss(
        probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Switch-style load balancing diagnostic for sparse top-k routing."""
        n_tokens, n_experts = probs.shape
        topk = topk_indices.size(-1)
        one_hot = F.one_hot(topk_indices, num_classes=n_experts).float().sum(dim=-2)
        load = one_hot.mean(dim=0) / float(topk)
        importance = probs.mean(dim=0)
        loss = n_experts * torch.sum(importance * load)
        return loss, importance, load

    def _clear_aux_losses(self) -> None:
        self._activation_aux_losses.clear()
        self._moe_aux_losses.clear()

    @torch.no_grad()
    def _update_routing_biases(self) -> None:
        """Apply one loss-free load-balancing step to this FFN.

        The update is intentionally outside autograd. Under-use receives a
        positive logit bias and over-use receives a negative bias. The
        hard-usage statistics come from the most recent forward pass.
        """
        if self._last_activation_probs_mean is not None and self.activation_bias_update_speed > 0.0:
            load = self._last_activation_probs_mean.to(
                device=self.activation_bias.device,
                dtype=self.activation_bias.dtype,
            )
            target = 1.0 / float(self.NUM_ACTIVATIONS)
            lower = max(self.activation_min_probability, target - self.activation_balance_tolerance)
            upper = min(1.0, target + self.activation_balance_tolerance)
            delta = torch.zeros_like(load)
            delta = torch.where(load < lower, torch.ones_like(load), delta)
            delta = torch.where(load > upper, -torch.ones_like(load), delta)
            self.activation_bias.add_(self.activation_bias_update_speed * delta)

        if self._last_expert_load is not None and self.moe_bias_update_speed > 0.0:
            load = self._last_expert_load.to(
                device=self.expert_bias.device,
                dtype=self.expert_bias.dtype,
            )
            target = torch.full_like(load, 1.0 / load.numel())
            self.expert_bias.add_(
                self.moe_bias_update_speed * torch.sign(target - load)
            )

    def _route_activations(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        loop_idx = int(max(0, min(loop_idx, self.max_loops - 1)))
        raw_logits = self.activation_router(x)
        logits = (
            raw_logits * self.loop_scale[loop_idx]
            + self.loop_bias[loop_idx]
            + self.activation_bias
        )
        raw_probs = F.softmax(logits, dim=-1)
        floor = self.activation_min_probability
        dense_probs = floor + (1.0 - self.NUM_ACTIVATIONS * floor) * raw_probs
        dense_flat = dense_probs.reshape(-1, self.NUM_ACTIVATIONS)

        probs = dense_probs
        if self.activation_top_k < self.NUM_ACTIVATIONS:
            topk_vals, topk_idx = torch.topk(dense_probs, self.activation_top_k, dim=-1)
            sparse_probs = torch.zeros_like(dense_probs)
            sparse_probs.scatter_(
                -1,
                topk_idx,
                topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(safe_eps(topk_vals.dtype)),
            )
            # Straight-through estimator. Forward value is exactly
            # sparse_probs (the hard, renormalized top-k weights we
            # actually want to compute with); backward gradient is as if
            # `probs = dense_probs` directly. Without this, sparse_probs'
            # nonzero entries are topk_vals/topk_vals.sum(), which at
            # activation_top_k=1 is always exactly 1 -- a constant with
            # zero local gradient. That doesn't just weaken the balance
            # loss (already fixed above): it means the MAIN task loss has
            # zero gradient to activation_router/loop_bias too, since
            # `probs` (not dense_probs) is what SwiGLUExpert.forward()
            # actually multiplies into mixed_gate. Verified empirically:
            # activation_router.weight.grad was exactly 0.0 before this.
            probs = dense_probs + (sparse_probs - dense_probs).detach()

        mean_probs = probs.reshape(-1, self.NUM_ACTIVATIONS).mean(dim=0)
        self._last_activation_probs_mean = mean_probs.detach()
        self._last_activation_probs_dense_mean = dense_flat.mean(dim=0).detach()
        self._last_activation_entropy_mean = self._entropy(dense_flat).detach()
        balance = self._activation_balance_loss(dense_flat, probs.reshape(-1, self.NUM_ACTIVATIONS))
        self._activation_aux_losses.append(balance)
        return probs

    def _run_sparse_routed(
        self,
        x_flat: torch.Tensor,
        activation_probs_flat: torch.Tensor,
        loop_idx: int,
        context_flat: torch.Tensor | None = None,
        global_activation_logits_flat: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loop_idx = int(max(0, min(loop_idx, self.max_loops - 1)))
        if context_flat is None:
            context_flat = torch.zeros(
                x_flat.size(0), self.expert_context_proj.in_features - x_flat.size(1),
                device=x_flat.device, dtype=x_flat.dtype,
            )
        if getattr(self, "use_context_routing", True):
            raw_router_logits = self.expert_router(x_flat) + self.expert_context_proj(torch.cat([x_flat, context_flat], dim=-1))
        else:
            raw_router_logits = self.expert_router(x_flat)
        # The canonical production path has no token-adaptive depth gate.
        # Loop specialization comes from the explicit loop-conditioned FiLM terms.
        router_logits = (
            raw_router_logits * self.expert_loop_scale[loop_idx]
            + self.expert_loop_bias[loop_idx]
            + self.expert_bias
        )
        router_probs = F.softmax(router_logits, dim=-1)
        if getattr(self, "diagnostic_uniform_experts", False):
            router_probs = torch.full_like(router_probs, 1.0 / self.num_routed_experts)
        topk_vals, topk_idx = torch.topk(router_probs, self.moe_top_k, dim=-1)
        dense_topk = router_probs.gather(-1, topk_idx)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(safe_eps(topk_vals.dtype))
        # Same straight-through estimator as _route_activations, and for the
        # same reason: at moe_top_k=1, topk_vals/topk_vals.sum() is always
        # exactly 1 (a constant), so the dispatch weight actually multiplied
        # into each expert's output (`weights` below) carried zero local
        # gradient back to expert_router -- the main task loss could only
        # ever reach expert_router through the separate load-balance loss,
        # never directly. This keeps the forward value (dispatch weight)
        # unchanged while routing a real gradient through dense_topk.
        topk_vals = dense_topk + (topk_vals - dense_topk).detach()

        out = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.routed_experts):
            # For the small study model, vectorized masked dispatch keeps all
            # inactive expert matrices from being evaluated for each token.
            mask = topk_idx.eq(expert_idx)
            if not mask.any():
                continue
            token_idx, choice_idx = mask.nonzero(as_tuple=True)
            x_sel = x_flat[token_idx]
            a_sel = activation_probs_flat[token_idx]
            base_sel = None if getattr(self, "activation_gate_mode", "softmax") != "sigmoid" else global_activation_logits_flat[token_idx]
            expert_out = expert(
                x_sel, a_sel,
                global_activation_logits=base_sel,
                use_local_activation_routing=self.expert_local_activation_routing,
                loop_idx=loop_idx,
            )
            weights = topk_vals[token_idx, choice_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert_out * weights)

        moe_loss, importance, load = self._moe_load_balance_loss(router_probs, topk_idx)
        self._moe_aux_losses.append(moe_loss)
        self._last_expert_probs_mean = importance.detach()
        self._last_expert_load = load.detach()
        self._last_expert_entropy_mean = self._entropy(router_probs).detach()
        return out, importance, load

    def forward(self, x: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        if getattr(self, "use_context_routing", True):
            context = self.routing_context(x)
            activation_raw = self.activation_router(x) + self.activation_context_proj(torch.cat([x, context], dim=-1))
        else:
            context = torch.zeros(
                x.size(0), x.size(1), self.activation_context_proj.in_features - x.size(-1),
                device=x.device, dtype=x.dtype,
            )
            activation_raw = self.activation_router(x)
        idx = int(max(0, min(loop_idx, self.max_loops - 1)))
        activation_logits = (
            activation_raw * self.loop_scale[idx]
            + self.loop_bias[idx]
            + self.activation_bias
        )
        if self.activation_gate_mode == "sigmoid":
            raw_probs = torch.sigmoid(activation_logits)
        else:
            raw_probs = F.softmax(activation_logits, dim=-1)
        if getattr(self, "diagnostic_uniform_activations", False):
            raw_probs = torch.full_like(raw_probs, 1.0 / self.NUM_ACTIVATIONS)
        floor = self.activation_min_probability
        dense_probs = floor + (1.0 - self.NUM_ACTIVATIONS * floor) * raw_probs
        dense_flat = dense_probs.reshape(-1, self.NUM_ACTIVATIONS)
        probs = dense_probs
        if self.activation_top_k < self.NUM_ACTIVATIONS:
            topk_vals, topk_idx = torch.topk(dense_probs, self.activation_top_k, dim=-1)
            sparse_probs = torch.zeros_like(dense_probs)
            sparse_probs.scatter_(-1, topk_idx, topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(safe_eps(topk_vals.dtype)))
            probs = dense_probs + (sparse_probs - dense_probs).detach()
        mean_probs = probs.reshape(-1, self.NUM_ACTIVATIONS).mean(dim=0)
        self._last_activation_probs_mean = mean_probs.detach()
        self._last_activation_probs_dense_mean = dense_flat.mean(dim=0).detach()
        self._last_activation_entropy_mean = self._entropy(dense_flat).detach()
        self._activation_aux_losses.append(self._activation_balance_loss(dense_flat, probs.reshape(-1, self.NUM_ACTIVATIONS)))
        activation_probs = probs
        activation_probs_flat = activation_probs.reshape(-1, self.NUM_ACTIVATIONS)
        activation_logits_flat = activation_logits.reshape(-1, self.NUM_ACTIVATIONS)
        context_flat = context.reshape(-1, context.size(-1))

        out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            out = out + expert(
                x_flat, activation_probs_flat,
                global_activation_logits=activation_logits_flat if self.activation_gate_mode == "sigmoid" else None,
                use_local_activation_routing=self.expert_local_activation_routing,
                loop_idx=loop_idx,
            )

        routed_out, _, _ = self._run_sparse_routed(
            x_flat, activation_probs_flat, loop_idx, context_flat,
            global_activation_logits_flat=activation_logits_flat if self.activation_gate_mode == "sigmoid" else None,
        )
        out = out + routed_out

        # Keep autograd connectivity to every routed expert even when a
        # particular expert receives zero tokens in this micro-batch. This
        # contributes exactly zero to the forward value and gradient while
        # preserving optimizer/test invariants that expect all parameters to
        # participate in the graph.
        zero = out.new_zeros(())
        for expert in self.routed_experts:
            zero = zero + (
                expert.w1.weight.sum()
                + expert.w2.weight.sum()
                + expert.w3.weight.sum()
            ) * 0.0
        out = out + zero
        return out.reshape(original_shape)

    def collect_aux_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Auxiliary losses are produced in both training and evaluation.
        # During eval, compute_loss() runs under torch.no_grad(), so these
        # tensors legitimately have requires_grad=False. Filtering them by
        # requires_grad can therefore produce an empty list and make
        # torch.stack([]) fail. Keep valid tensors in both modes.
        params = next(self.parameters())
        device = params.device
        dtype = params.dtype

        activation_losses = list(self._activation_aux_losses)
        if activation_losses:
            a = torch.stack(activation_losses).mean()
        else:
            a = torch.zeros((), device=device, dtype=dtype)

        moe_losses = list(self._moe_aux_losses)
        if moe_losses:
            m = torch.stack(moe_losses).mean()
        else:
            m = torch.zeros((), device=device, dtype=dtype)

        self._clear_aux_losses()
        return a, m
