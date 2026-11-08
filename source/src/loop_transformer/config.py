"""Configuration for the production recurrent LoopLLM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


GRASSMANN_ATTENTION_TYPES = frozenset({"grassmann", "grassmann-new", "real-grassmann"})


def is_grassmann_attention(attention_type: str) -> bool:
    """Return whether an attention type uses the Grassmann/Pluecker mixer."""
    return str(attention_type) in GRASSMANN_ATTENTION_TYPES


class LoopConfigError(ValueError):
    """Raised when a LoopConfig value is invalid."""


@dataclass
class LoopConfig:
    # Core model dimensions.
    vocab_size: int = 65
    dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    head_dim: int = 16
    ffn_hidden_dim: int = 256
    rope_dim: int = 16
    embed_dim: Optional[int] = None
    embed_dim_out: Optional[int] = None

    # Recurrent training/objective.
    max_loops: int = 4
    min_loops: int = 4
    loop_sampling: bool = False
    loop_sampling_lambda: Optional[float] = None
    beta_entropy: float = 0.0
    loop_supervision_weight: float = 0.0
    joint_exit_loss_weight: float = 0.0
    loop_monotonic_weight: float = 0.0
    loop_monotonic_margin: float = 0.01
    loop_refinement_weight: float = 0.02
    loop_refinement_margin: float = 0.001
    loop_task_weight: float = 0.0
    loop_task_mode: str = "none"
    shortcut_consistency_weight: float = 0.0
    shortcut_consistency_temperature: float = 2.0

    # Recurrent specialization mechanisms that have actually been tested.
    residual_loop_embedding: bool = True
    loop_positional_embedding: bool = False
    loop_positional_max_len: int = 1024
    loop_positional_scale: float = 0.10
    causal_register_bank: bool = False
    causal_registers: int = 4
    causal_register_update_scale: float = 0.10
    loop_transition_matrices: bool = False
    loop_transition_init_scale: float = 0.02
    loop_controller: bool = False
    # Controller output is bounded elementwise by +/- this value before the
    # recurrent FiLM operation h <- h*(1+scale) + bias. 0.2 was validated
    # against the previously observed multi-loop state explosion.
    loop_controller_bound: float = 0.20

    # Attention implementation. "deepseek" preserves the original fixed
    # SWA/CSA/HCA layer routing; "grassmann" replaces only those attention
    # branches with causal multi-scale Pluecker mixing.
    attention_type: str = "deepseek"
    grassmann_rank: int = 32
    grassmann_offsets: tuple[int, ...] = (1, 2, 4, 8, 12, 16)
    # grassmann-new: number of local offsets per pass (including mandatory lag 1).
    # Loop n uses spacing base+n, plus an adaptive sparse-global tail.
    grassmann_offset_count: int = 5
    grassmann_offset_spacing_base: int = 2
    # Versioned schedule marker. Old grassmann-new checkpoints without this
    # field are loaded as legacy local-only for faithful inspection.
    grassmann_offset_scheme: str = "adaptive_sparse_global_v1"
    # real-grassmann: append every positive multiple of this stride up to the
    # actual input sequence length to the fixed local offset set.
    grassmann_long_range_stride: int = 16
    # Learned absolute token positions from Grassmann paper §3.1. This is
    # separate from loop_positional_embedding, which encodes recurrent depth.
    # Set grassmann_positional_embedding=False for the no-absolute-position ablation.
    grassmann_positional_embedding: bool = True
    grassmann_positional_max_len: int = 1024
    # Dedicated seed for the optional position table. Its initialization runs
    # inside a forked RNG context and cannot perturb shared model initialization.
    grassmann_positional_seed: Optional[int] = None

    # DeepSeek-style attention families. Layer routing is fixed:
    # layers 0-1 = SWA, layer >=2 even = CSA, layer >=3 odd = HCA.
    # deepseek-split: preserve SWA/CSA/HCA routing; partition only the
    # compressed long-range CSA/HCA memory by compressed-block age.
    deepseek_split_stride: int = 4
    sw_window: int = 128
    csa_m: int = 4
    csa_top_k: int = 64
    csa_aux_loss_weight: float = 1.0
    hca_m_prime: int = 128
    groups: int = 8
    group_dim: int = 1024

    # DeepSeekMoE + activation routing.
    moe_num_shared_experts: int = 1
    moe_num_routed_experts: int = 2
    moe_top_k: int = 1
    moe_expert_hidden_dim: Optional[int] = 128
    activation_dictionary: str = "moa7"
    activation_gate_mode: str = "sigmoid"
    activation_top_k: int = 7
    activation_min_probability: float = 0.0
    activation_balance_weight: float = 0.0
    activation_balance_tolerance: float = 0.25
    activation_bias_update_speed: float = 0.0
    moe_aux_loss_weight: float = 0.0
    moe_bias_update_speed: float = 0.0
    expert_local_activation_routing: bool = False

    # Memory/performance.
    tie_embeddings: bool = True
    dropout: float = 0.0
    grad_checkpointing: bool = False
    exit_gate_loop_embed_dim: int = 16

    def __post_init__(self) -> None:
        errors: list[str] = []

        def check(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        check(self.vocab_size > 0, f"vocab_size must be > 0, got {self.vocab_size}")
        check(self.dim > 0, f"dim must be > 0, got {self.dim}")
        check(self.n_layers > 0, f"n_layers must be > 0, got {self.n_layers}")
        check(self.n_heads > 0, f"n_heads must be > 0, got {self.n_heads}")
        check(self.head_dim > 0, f"head_dim must be > 0, got {self.head_dim}")
        check(self.ffn_hidden_dim > 0, f"ffn_hidden_dim must be > 0, got {self.ffn_hidden_dim}")
        check(
            self.rope_dim > 0 and self.rope_dim % 2 == 0 and self.rope_dim <= self.head_dim,
            f"rope_dim must be positive, even, and <= head_dim; got {self.rope_dim} vs {self.head_dim}",
        )

        if self.embed_dim is not None:
            check(self.embed_dim > 0, f"embed_dim must be > 0, got {self.embed_dim}")
        if self.embed_dim_out is not None:
            check(self.embed_dim_out > 0, f"embed_dim_out must be > 0, got {self.embed_dim_out}")
        check(
            self.embed_dim_out is None or self.embed_dim is not None,
            "embed_dim_out requires embed_dim to also be set",
        )
        effective_in = self.embed_dim if self.embed_dim is not None else self.dim
        effective_out = self.embed_dim_out if self.embed_dim_out is not None else effective_in
        check(
            not self.tie_embeddings or effective_in == effective_out,
            "tie_embeddings=True requires input and output embedding dimensions to match",
        )

        check(self.max_loops >= 1, f"max_loops must be >= 1, got {self.max_loops}")
        check(self.min_loops >= 1, f"min_loops must be >= 1, got {self.min_loops}")
        check(self.min_loops <= self.max_loops, "min_loops must be <= max_loops")
        check(self.beta_entropy >= 0, f"beta_entropy must be >= 0, got {self.beta_entropy}")
        check(self.loop_sampling_lambda is None or self.loop_sampling_lambda > 0, "loop_sampling_lambda must be > 0 when set")
        for name in (
            "loop_supervision_weight", "joint_exit_loss_weight", "loop_monotonic_weight",
            "loop_monotonic_margin", "loop_refinement_weight", "loop_refinement_margin",
            "loop_task_weight", "shortcut_consistency_weight",
        ):
            check(getattr(self, name) >= 0, f"{name} must be >= 0")
        check(self.loop_task_mode in {"none", "horizon"}, "loop_task_mode must be 'none' or 'horizon'")
        check(self.shortcut_consistency_temperature > 0, "shortcut_consistency_temperature must be > 0")

        check(self.loop_positional_max_len > 0, "loop_positional_max_len must be > 0")
        check(self.loop_positional_scale >= 0, "loop_positional_scale must be >= 0")
        check(self.causal_registers >= 1, "causal_registers must be >= 1")
        check(self.causal_register_update_scale > 0, "causal_register_update_scale must be > 0")
        check(self.loop_transition_init_scale >= 0, "loop_transition_init_scale must be >= 0")
        check(0 < self.loop_controller_bound <= 1.0, "loop_controller_bound must be in (0, 1]")
        check(self.exit_gate_loop_embed_dim >= 0, "exit_gate_loop_embed_dim must be >= 0")

        check(self.attention_type in {"deepseek", "deepseek-split", "grassmann", "grassmann-new", "real-grassmann"},
              "attention_type must be 'deepseek', 'deepseek-split', 'grassmann', 'grassmann-new', or 'real-grassmann'")
        if self.attention_type == "deepseek-split":
            check(self.deepseek_split_stride >= 2,
                  "deepseek_split_stride must be >= 2")
        # Grassmann-only settings are deliberately ignored by the default
        # DeepSeek path, so older/small baseline configs remain valid.
        if is_grassmann_attention(self.attention_type):
            check(self.grassmann_positional_max_len > 0,
                  "grassmann_positional_max_len must be > 0")
            check(
                self.grassmann_positional_seed is None or self.grassmann_positional_seed >= 0,
                "grassmann_positional_seed must be >= 0 when set",
            )
            check(2 <= self.grassmann_rank <= self.dim,
                  f"grassmann_rank must be in [2, dim={self.dim}]")
            check(bool(self.grassmann_offsets), "grassmann_offsets must not be empty")
            check(all(v >= 1 for v in self.grassmann_offsets),
                  "grassmann_offsets must contain only positive integers")
            check(len(set(self.grassmann_offsets)) == len(self.grassmann_offsets),
                  "grassmann_offsets must be unique")
            check(self.grassmann_offset_count >= 1,
                  "grassmann_offset_count must be >= 1 (includes mandatory lag 1)")
            check(self.grassmann_offset_spacing_base >= 1,
                  "grassmann_offset_spacing_base must be >= 1")
            check(
                self.grassmann_offset_scheme
                in {"adaptive_sparse_global_v1", "legacy_local_only_v0"},
                "grassmann_offset_scheme must be adaptive_sparse_global_v1 "
                "or legacy_local_only_v0",
            )
            check(self.grassmann_long_range_stride >= 1,
                  "grassmann_long_range_stride must be >= 1")

        check(self.sw_window >= 1, "sw_window must be >= 1")
        check(self.csa_m >= 1, "csa_m must be >= 1")
        check(self.csa_top_k >= 1, "csa_top_k must be >= 1")
        check(self.csa_aux_loss_weight >= 0, "csa_aux_loss_weight must be >= 0")
        check(self.hca_m_prime >= 1, "hca_m_prime must be >= 1")
        check(self.groups >= 1, "groups must be >= 1")
        check(self.group_dim > 0, "group_dim must be > 0")
        check(
            (self.n_heads * self.head_dim) % self.groups == 0,
            f"n_heads*head_dim ({self.n_heads*self.head_dim}) must be divisible by groups ({self.groups})",
        )

        check(self.moe_num_shared_experts >= 0, "moe_num_shared_experts must be >= 0")
        check(self.moe_num_routed_experts >= 1, "moe_num_routed_experts must be >= 1")
        check(1 <= self.moe_top_k <= self.moe_num_routed_experts, "moe_top_k must be in [1, moe_num_routed_experts]")
        if self.moe_expert_hidden_dim is not None:
            check(self.moe_expert_hidden_dim >= 1, "moe_expert_hidden_dim must be >= 1")
        check(self.activation_dictionary in {"legacy4", "moa7"}, "activation_dictionary must be 'legacy4' or 'moa7'")
        check(self.activation_gate_mode in {"softmax", "sigmoid"}, "activation_gate_mode must be 'softmax' or 'sigmoid'")
        max_k = 7 if self.activation_dictionary == "moa7" else 4
        check(1 <= self.activation_top_k <= max_k, f"activation_top_k must be in [1, {max_k}]")
        check(0 <= self.activation_min_probability < 1.0 / max_k, "activation_min_probability out of range")
        check(self.activation_balance_weight >= 0, "activation_balance_weight must be >= 0")
        check(self.activation_balance_tolerance >= 0, "activation_balance_tolerance must be >= 0")
        check(self.activation_bias_update_speed >= 0, "activation_bias_update_speed must be >= 0")
        check(self.moe_aux_loss_weight >= 0, "moe_aux_loss_weight must be >= 0")
        check(self.moe_bias_update_speed >= 0, "moe_bias_update_speed must be >= 0")

        check(0 <= self.dropout < 1, "dropout must be in [0, 1)")

        if errors:
            raise LoopConfigError(
                "Invalid LoopConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )
