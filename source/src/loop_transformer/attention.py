"""Attention mechanisms -- the "Memory" layer of the architecture.

SlidingWindowAttention, CSAAttention, and HCAAttention preserve the original
DeepSeek-style layer routing. The experimental deepseek-split mode keeps the
same three mechanisms: SWA and the raw local windows stay complete, while only
the compressed long-range CSA/HCA memory banks are partitioned across loops.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PartialRoPE, RMSNorm, causal_mask, causal_windows, safe_eps

# Masking sentinel for CSAAttention's indexer auxiliary KL-divergence loss
# specifically -- see _indexer_aux_loss's docstring for why this needs to
# be a large finite value rather than float('-inf') here, unlike every
# other masked_fill in this file (which correctly use literal -inf).
#
# Sized dynamically to whatever dtype is actually being masked, rather
# than a single fixed constant: a Python float like -1e9 overflows
# float16's range (max magnitude ~65504) and raises "value cannot be
# converted to type c10::Half without overflow" -- reproduced directly
# with `torch.randn(4, dtype=torch.float16).masked_fill(mask, -1e9)` --
# the moment this runs under --dtype float16 (CSA's indexer aux loss is
# active whenever csa_aux_loss_weight > 0, which it is by default).
# torch.finfo(dtype).min is the same "large-but-finite" sentinel scaled
# to whatever's representable, matching the fix already applied to the
# main attention-mask overflow elsewhere in this project.
def _kl_mask_value(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


class AttentionSink(nn.Module):
    """Learnable per-head sink logit that absorbs probability mass.

    Callers MUST apply this before masking, not after: logsumexp(-inf,
    sink_logit) = sink_logit (finite), so applying the sink after
    masked_fill would silently resurrect masked-out positions with
    nonzero attention weight. Every call site in this module does
    sink-then-mask for this reason.
    """
    def __init__(self, n_heads: int):
        super().__init__()
        self.sink_logits = nn.Parameter(torch.zeros(n_heads))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        sink = self.sink_logits.view(1, -1, 1, 1)
        return torch.logsumexp(
            torch.stack([logits, sink.expand_as(logits)], dim=0), dim=0,
        )


class AttentionOutputGate(nn.Module):
    """Per-head sigmoid gate on attention OUTPUT, applied before the output
    projection -- the fix from LT2 (arXiv:2605.20670) for compounding
    attention sinks in weight-shared LOOPED architectures specifically.

    AttentionSink (above) gives a head somewhere to dump unneeded softmax
    mass instead of distorting real tokens -- standard practice, and
    correctly implemented here (sink-before-mask, verified). But LT2 found
    that in a looped model, the sink still forms and COMPOUNDS: the same
    softmax block gets re-applied to a residual stream that already carries
    the sink pattern from the previous loop pass, getting worse each
    iteration -- a failure mode a non-looped model never encounters, since
    it only ever sees each layer's softmax once. Their ablation: a
    head-specific sigmoid gate after SDPA, before the output projection,
    eliminates it. Confirmed in exactly this looped setting, not just
    inferred from the non-looped literature.

    Weight zero-init, bias=+4 (sigmoid(4)=0.982): starts close to fully
    open -- a near-identity, safe for warm-starting from an existing
    checkpoint -- while keeping the gradient at this bias large enough
    (sigmoid'(4)~=0.018) to actually move if training finds closing it
    useful. An exact-open bias like +10 would be a closer identity but its
    gradient (~4.5e-5) would make early learning very slow.

    Per-loop bias layered on top, zero-init: lets a loop learn to close its
    gate more than others, motivated by the compounding mechanism itself
    getting worse the more loops have already run. This specific extension
    -- loop-conditioning the gate -- isn't something LT2 tested; the base
    per-head gate is the evidenced part, this is a natural but unverified
    extrapolation of it, consistent with how every other attention/FFN
    mechanism in this codebase is already loop-conditioned.
    """

    def __init__(self, dim: int, n_heads: int, max_loops: int, init_bias: float = 4.0):
        super().__init__()
        self.proj = nn.Linear(dim, n_heads, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)
        self.loop_bias = nn.Parameter(torch.zeros(max_loops, n_heads))

    def forward(self, x: torch.Tensor, loop_idx: int) -> torch.Tensor:
        """x: block input [B, T, dim].  Returns gate in (0,1), [B, T, n_heads]
        -- caller reshapes to match its own attention-output layout."""
        loop_idx = int(max(0, min(loop_idx, self.loop_bias.size(0) - 1)))
        return torch.sigmoid(self.proj(x) + self.loop_bias[loop_idx])


class SlidingWindowAttention(nn.Module):
    """Full causal self-attention restricted to a local window."""

    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 window_size: int = 128, rope_dim: int = 64, max_loops: int = 4):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.window_size = window_size
        self.qkv      = nn.Linear(dim, 3 * n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # see mlp_attnres.py's zero-init note
        self.q_norm   = RMSNorm(head_dim)
        self.k_norm   = RMSNorm(head_dim)
        self.rope     = PartialRoPE(head_dim, rope_dim=rope_dim)
        self.sink     = AttentionSink(n_heads)
        self.out_gate = AttentionOutputGate(dim, n_heads, max_loops)

    def forward(self, x: torch.Tensor, loop_idx: int = 0, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope.apply(q), self.rope.apply(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is None:
            mask = causal_mask(T, x.device)
            if self.window_size > 0:
                idx = torch.arange(T, device=x.device)
                mask = mask & (idx[None, :] >= idx[:, None] - self.window_size + 1)
        else:
            if attention_mask.shape != (T, T):
                raise ValueError(f"attention_mask must be [{T},{T}], got {tuple(attention_mask.shape)}")
            mask = attention_mask.to(device=x.device, dtype=torch.bool)
        # Sink BEFORE mask: logsumexp(-inf, sink_logit) = sink_logit (finite),
        # so applying the sink after masked_fill would silently resurrect
        # masked-out (future/out-of-window) positions with nonzero weight --
        # masking must have the final say. Valid positions still get the
        # intended sink-boosted logit either way; only invalid ones differ.
        logits = self.sink(logits)
        logits = logits.masked_fill(~mask[None, None], float("-inf"))
        y = torch.matmul(F.softmax(logits, dim=-1), v)
        gate = self.out_gate(x, loop_idx).transpose(1, 2).unsqueeze(-1)  # [B,n_heads,T,1]
        y = y * gate
        return self.out_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


def deepseek_loop_split_lags(
    seq_len: int, loop_idx: int, stride: int = 4,
) -> tuple[int, ...]:
    """Return positive causal look-back lags assigned to one recurrent pass.

    The schedule is ``{1} union {stride*n + phase}``, where
    ``phase = loop_idx % stride`` and only lags in ``[1, seq_len)`` survive.
    The shared lag 1 is therefore present on every pass. With stride 4:

      loop 0: 1, 4, 8, 12, ...
      loop 1: 1, 5, 9, 13, ...
      loop 2: 1, 2, 6, 10, ...
      loop 3: 1, 3, 7, 11, ...
      loop 4: same as loop 0

    Note: the original experiment note also listed lag 2 on loop 1. That
    single value is not implied by the stated arithmetic formula; this helper
    follows the formula exactly rather than inserting a one-off exception.
    The current token (lag 0) is added separately by the attention module so
    every query has at least one valid key.
    """
    seq_len = int(seq_len)
    loop_idx = int(loop_idx)
    stride = int(stride)
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if loop_idx < 0:
        raise ValueError(f"loop_idx must be >= 0, got {loop_idx}")
    if stride < 2:
        raise ValueError(f"stride must be >= 2, got {stride}")

    phase = loop_idx % stride
    lags = {1} if seq_len > 1 else set()
    n = 0
    while True:
        lag = stride * n + phase
        if lag >= seq_len:
            break
        if lag >= 1:
            lags.add(lag)
        n += 1
    return tuple(sorted(lags))


def deepseek_split_block_mask(
    visible: torch.Tensor, loop_idx: int, stride: int = 4,
) -> torch.Tensor:
    """Partition causally visible compressed blocks across recurrent passes.

    ``visible`` is [T, N] and marks compressed blocks that are fully available
    to each query. For each query, compressed blocks are ranked from newest to
    oldest. Rank 1 is shared by every loop; older ranks follow
    ``{stride*n + (loop_idx % stride)}`` (with positive ranks only).

    Ranking compressed slots, rather than raw token offsets, is important:
    using raw offsets would collapse all blocks into the same phase whenever
    the compression factor is a multiple of the split stride.
    """
    if visible.ndim != 2:
        raise ValueError(f"visible must be [T,N], got {tuple(visible.shape)}")
    stride = int(stride)
    loop_idx = int(loop_idx)
    if stride < 2:
        raise ValueError(f"stride must be >= 2, got {stride}")
    if loop_idx < 0:
        raise ValueError(f"loop_idx must be >= 0, got {loop_idx}")
    if visible.size(1) == 0:
        return visible.to(dtype=torch.bool)

    visible = visible.to(dtype=torch.bool)
    block_ids = torch.arange(visible.size(1), device=visible.device)
    newest_visible = torch.where(
        visible, block_ids.unsqueeze(0), torch.full_like(block_ids.unsqueeze(0), -1)
    ).amax(dim=1)
    # One-based age: the newest causally visible compressed block is rank 1.
    age = newest_visible[:, None] - block_ids[None, :] + 1
    phase = loop_idx % stride
    assigned = (age == 1) | ((age >= 1) & ((age % stride) == phase))
    return visible & assigned


def gather_deepseek_split_blocks(
    comp: torch.Tensor,
    eligible: torch.Tensor,
    stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather each query's loop-assigned compressed blocks into a compact bank.

    Returns ``(selected, valid)`` with shapes [B,T,K,D] and [T,K]. K is a
    context-independent upper bound for the number of ranks assigned to one
    loop. Invalid padding slots are masked by the returned validity tensor.
    """
    if comp.ndim != 3 or eligible.ndim != 2:
        raise ValueError(
            f"expected comp=[B,N,D] and eligible=[T,N], got "
            f"{tuple(comp.shape)} and {tuple(eligible.shape)}"
        )
    batch, num_blocks, width = comp.shape
    if eligible.size(1) != num_blocks:
        raise ValueError("eligible block count must match compressed memory")
    if num_blocks == 0:
        return comp.new_empty(batch, eligible.size(0), 0, width), eligible

    stride = int(stride)
    if stride < 2:
        raise ValueError(f"stride must be >= 2, got {stride}")
    max_candidates = min(num_blocks, (num_blocks + stride - 1) // stride + 1)
    candidate_valid_values, candidate_indices = torch.topk(
        eligible.to(dtype=torch.uint8), k=max_candidates, dim=-1,
    )
    candidate_valid = candidate_valid_values.to(dtype=torch.bool)
    batch_indices = torch.arange(batch, device=comp.device)[:, None, None]
    selected = comp[
        batch_indices,
        candidate_indices.unsqueeze(0).expand(batch, -1, -1),
    ]
    return selected, candidate_valid


class GroupedOutputProjection(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, groups: int, group_dim: int):
        super().__init__()
        if dim_in % groups != 0:
            raise ValueError(
                f"GroupedOutputProjection: dim_in ({dim_in}) must be "
                f"divisible by groups ({groups}). dim_in is n_heads*head_dim "
                f"for the attention module that owns this projection -- "
                f"adjust n_heads/head_dim/groups so the product divides evenly."
            )
        group_in   = dim_in // groups
        self.groups = groups
        self.proj1  = nn.ModuleList([nn.Linear(group_in, group_dim, bias=False)
                                     for _ in range(groups)])
        self.proj2  = nn.Linear(groups * group_dim, dim_out, bias=False)
        nn.init.zeros_(self.proj2.weight)  # see mlp_attnres.py's zero-init note;
                                            # covers both CSA and HCA, which
                                            # both route output through here.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = x.chunk(self.groups, dim=-1)
        return self.proj2(torch.cat([p(c) for p, c in zip(self.proj1, parts)], dim=-1))


class CSAAttention(nn.Module):
    """
    Compressed Sparse Attention (DeepSeek-V4 §CSA) -- causal, per-query.

    When ``split_stride`` is set, loop passes partition only the compressed
    long-range block bank. CSA's raw local window and learned top-k selection
    remain active; top-k is applied within the current loop's eligible bank.

    Each query attends over two candidate sets, combined into one softmax:
      - up to top_k causally-visible compressed blocks (long-range, sparse)
      - a causal local window of raw tokens (recent, full detail)

    This computes attention per-query directly (einsum over each query's
    own gathered candidates), costing O(B * n_heads * T * (top_k+window)) --
    linear in T. An earlier version routed every query's picks through one
    shared memory pool attended by ALL queries densely, which cost
    O(B * n_heads * T * (T*top_k)) -- quadratic in T and WORSE than plain
    full attention -- and had no causal masking at all, so any position
    could see compressed summaries of future tokens. Both are fixed here.

    Indexer auxiliary loss
    ----------------------
    The indexer (index_down/index_up/index_weight/index_key) scores each
    (query, block) pair, but its scores only ever feed torch.topk's
    INDICES below -- index selection has no gradient, so without a
    dedicated training signal, the indexer never learns which blocks
    matter and stays at its random initialization forever.

    Fix, verified against DeepSeek-V3.2's public documentation for its
    own lightning indexer rather than guessed: train the indexer via a
    KL-divergence loss against a target built from the model's own dense
    attention -- specifically, sum PER-HEAD SOFTMAX PROBABILITIES (not
    averaged logits) across heads, then L1-normalize. Gradient isolation
    is enforced two ways: `comp` is detached before scoring, so the
    auxiliary loss can only update the indexer's own parameters, never
    the compression pathway (kv_a/kv_b/z_a/z_b); and the reference
    distribution is built under torch.no_grad() and detached, since it's
    a fixed target the indexer moves toward, not something the indexer's
    loss should reshape.

    One adaptation from DeepSeek's actual procedure: they retrofit
    sparsity onto an already-pretrained dense model, via a dedicated
    freeze-everything-but-the-indexer warm-up stage before jointly
    fine-tuning. We train this architecture from scratch -- there's no
    pretrained dense behavior to retrofit onto -- so the indexer and the
    reference it chases both develop jointly from step one, kept
    separated by the detach/no_grad boundaries above rather than by
    DeepSeek's literal staged parameter freezing, which assumes a
    different starting point than ours.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 m: int = 4, top_k: int = 64, window_size: int = 128,
                 index_heads: int = 32, index_dim: int = 64,
                 groups: int = 8, group_dim: int = 1024, rope_dim: int = 64,
                 max_loops: int = 4, split_stride: int | None = None):
        super().__init__()
        if split_stride is not None and int(split_stride) < 2:
            raise ValueError("split_stride must be >= 2 when enabled")
        self.n_heads, self.head_dim = n_heads, head_dim
        self.m, self.top_k, self.window_size = m, top_k, window_size
        self.split_stride = None if split_stride is None else int(split_stride)
        self.index_heads, self.index_dim = index_heads, index_dim

        # Overlapping compression (produces one summary vector per m-token block)
        self.kv_a = nn.Linear(dim, head_dim, bias=False)
        self.kv_b = nn.Linear(dim, head_dim, bias=False)
        self.z_a  = nn.Linear(dim, head_dim, bias=False)
        self.z_b  = nn.Linear(dim, head_dim, bias=False)

        # Lightweight indexer: scores each (query, block) pair for top-k selection.
        self.index_down   = nn.Linear(dim, index_dim, bias=False)
        self.index_up     = nn.Linear(index_dim, index_heads * index_dim, bias=False)
        self.index_weight = nn.Linear(dim, index_heads, bias=False)
        self.index_key    = nn.Linear(head_dim, index_dim, bias=False)
        # Per-loop FiLM on the index-head weighting, same (1+delta)/zero-init
        # pattern used for activation/expert routing and attn/ffn FiLM
        # elsewhere in this codebase: lets the SAME shared indexer shift
        # which blocks look relevant at different loop depths (e.g. loop 1
        # casts a wide net, loop 4 has learned to trust a narrower, more
        # precise selection) instead of every loop scoring blocks
        # identically. Applied to index_weight's output (a fixed
        # index_heads-size vector) rather than the final per-block score,
        # since the number of blocks varies with sequence length and can't
        # be a fixed FiLM target.
        self.index_loop_scale = nn.Parameter(torch.zeros(max_loops, index_heads))
        self.index_loop_bias = nn.Parameter(torch.zeros(max_loops, index_heads))

        # Local window (recent raw detail)
        self.local_kv = nn.Linear(dim, head_dim, bias=False)

        # Query + shared (MQA-style) memory norm/rope/sink
        self.q_down   = nn.Linear(dim, dim, bias=False)
        self.q_up     = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.q_norm   = RMSNorm(head_dim)
        self.mem_norm = RMSNorm(head_dim)
        self.rope     = PartialRoPE(head_dim, rope_dim=rope_dim)
        self.sink     = AttentionSink(n_heads)
        self.out_gate = AttentionOutputGate(dim, n_heads, max_loops)

        self.grouped_proj = GroupedOutputProjection(n_heads * head_dim, dim, groups, group_dim)

        # Transient per-forward-pass state, NOT a Parameter/Buffer -- plain
        # Python attributes aren't included in state_dict(), which is what
        # we want here (nothing persistent to checkpoint). Collected and
        # cleared by LoopTransformer.compute_loss(); see model.py.
        self._aux_losses: List[torch.Tensor] = []

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        """Block j (columns [j*m, (j+1)*m)) -> one summary vector. Purely
        intra-block; causal visibility of the RESULT is enforced in forward()."""
        B, T, _ = x.shape
        m   = self.m
        pad = (m - T % m) % m
        xp  = F.pad(x, (0, 0, 0, pad)) if pad else x
        nb  = xp.size(1) // m
        Ca  = self.kv_a(xp).view(B, nb, m, self.head_dim)
        Cb  = self.kv_b(xp).view(B, nb, m, self.head_dim)
        Za  = self.z_a(xp).view(B, nb, m, self.head_dim)
        Zb  = self.z_b(xp).view(B, nb, m, self.head_dim)
        comp = []
        for i in range(nb):
            b  = torch.zeros_like(Cb[:, i]) if i == 0 else Cb[:, i - 1]
            zb = torch.full_like(Zb[:, i], float("-inf")) if i == 0 else Zb[:, i - 1]
            w  = F.softmax(torch.cat([Za[:, i], zb], dim=1), dim=1)
            comp.append((w[:, :m] * Ca[:, i]).sum(1) + (w[:, m:] * b).sum(1))
        return torch.stack(comp, dim=1)  # [B, nb, head_dim]

    def _index_scores(self, x: torch.Tensor, comp: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        loop_idx = int(max(0, min(loop_idx, self.index_loop_scale.size(0) - 1)))
        B, T, _ = x.shape
        q = self.index_up(self.index_down(x)).view(B, T, self.index_heads, self.index_dim)
        w = self.index_weight(x)
        w = w * (1.0 + self.index_loop_scale[loop_idx]) + self.index_loop_bias[loop_idx]
        k = self.index_key(comp)
        return (F.relu(torch.einsum("bthd,bsd->bths", q, k)) * w.unsqueeze(-1)).sum(2)

    def _indexer_aux_loss(
        self,
        q:               torch.Tensor,  # [B, T, H, head_dim] -- same q used for the real attention
        comp:            torch.Tensor,  # [B, nb, head_dim] -- NOT detached (reference uses live comp)
        visible:         torch.Tensor,  # [T, nb] bool
        idx_scores_raw:  torch.Tensor,  # [B, T, nb] -- UNMASKED indexer scores (from detached comp)
    ) -> torch.Tensor:
        """KL(student=indexer || teacher=dense-attention-derived reference).
        See class docstring for the full rationale.

        Uses a large-but-finite masking sentinel (_KL_MASK_VALUE) rather
        than literal float('-inf') here specifically. Verified empirically
        while building this: whenever a query has only one visible block,
        BOTH the student's log-probability and the reference's probability
        hit an exact zero at every OTHER (masked) position. F.kl_div copes
        fine with an exact-zero target against a finite log-probability
        (0 * log(0) correctly contributes 0), but 0 * (log(0) - (-inf))
        is 0 * (-inf - (-inf)) = 0 * NaN = NaN -- literal -inf makes the
        student's log-probability at that position -inf too, not just
        very negative, and the two infinities collide. A finite sentinel
        keeps the student's log-probability very negative but not
        infinite, which resolves it without changing the result anywhere
        that isn't already this degenerate.
        """
        B, T, candidate_count = idx_scores_raw.shape
        has_visible = visible.any(dim=-1)  # [T]; false when no compressed block is causally available

        with torch.no_grad():
            comp_n = self.mem_norm(comp)
            if comp.ndim == 3:
                # Baseline path: shared compressed memory [B,N,D].
                ref_logits = torch.einsum("bthd,bsd->bths", q, comp_n) / math.sqrt(self.head_dim)
            elif comp.ndim == 4:
                # Split path: each query has its own compact [B,T,K,D] candidate bank.
                ref_logits = torch.einsum("bthd,btkd->bthk", q, comp_n) / math.sqrt(self.head_dim)
            else:
                raise ValueError(f"unsupported compressed-memory rank: {comp.ndim}")
            mask_4d = visible[None, :, None, :] | (~has_visible)[None, :, None, None]
            ref_logits = ref_logits.masked_fill(~mask_4d, _kl_mask_value(ref_logits.dtype))
            ref_probs = F.softmax(ref_logits, dim=-1)
            ref_probs = ref_probs.sum(dim=2)
            ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True).clamp(min=safe_eps(ref_probs.dtype))
            ref_probs = ref_probs.detach()

        mask_3d = visible[None, :, :] | (~has_visible)[None, :, None]  # [1,T,nb]
        student_log_probs = F.log_softmax(
            idx_scores_raw.masked_fill(~mask_3d, _kl_mask_value(idx_scores_raw.dtype)), dim=-1,
        )

        kl_per_position = F.kl_div(student_log_probs, ref_probs, reduction="none").sum(dim=-1)  # [B,T]
        kl_per_position = kl_per_position.masked_fill(~has_visible[None, :], 0.0)
        n_valid = has_visible.sum().clamp(min=1)
        return kl_per_position.sum() / (n_valid * B)

    def forward(self, x: torch.Tensor, loop_idx: int = 0, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        comp = self._compress(x)          # [B, nb, head_dim]
        nb   = comp.size(1)

        # ---- causal visibility: block j is visible to query t iff block
        # j's LAST covered position is <= t (fully in the past-or-present) ----
        block_last = (torch.arange(nb, device=x.device) + 1) * self.m - 1   # [nb]
        pos        = torch.arange(T, device=x.device)                        # [T]
        if attention_mask is None:
            visible = (block_last[None, :] <= pos[:, None])                  # [T, nb]
        else:
            if attention_mask.shape != (T, T):
                raise ValueError(f"attention_mask must be [{T},{T}], got {tuple(attention_mask.shape)}")
            visible_parts = []
            for j in range(nb):
                start = j * self.m
                end = min((j + 1) * self.m, T)
                visible_parts.append(attention_mask[:, start:end].all(dim=1))
            visible = torch.stack(visible_parts, dim=1) if visible_parts else torch.zeros(T, 0, device=x.device, dtype=torch.bool)

        # The default DeepSeek path remains exactly the original CSA selection.
        # In deepseek-split, only the compressed long-range bank is partitioned;
        # the raw local window below is deliberately complete on every loop.
        if self.split_stride is None:
            # comp.detach() isolates the indexer auxiliary loss from compression.
            idx_scores_raw = self._index_scores(x, comp.detach(), loop_idx)
            idx_scores = idx_scores_raw.masked_fill(~visible[None], float("-inf"))
            k = min(self.top_k, nb)
            topv, topi = torch.topk(idx_scores, k, dim=-1)
            sel_valid = topv > float("-inf")
            comp_expand = comp[:, None, :, :].expand(B, T, nb, self.head_dim)
            sel = torch.gather(
                comp_expand, 2,
                topi.clamp(min=0).unsqueeze(-1).expand(B, T, k, self.head_dim),
            )
            aux_comp, aux_visible = comp, visible
        else:
            split_visible = deepseek_split_block_mask(
                visible, loop_idx=loop_idx, stride=self.split_stride,
            )
            candidate_comp, candidate_valid = gather_deepseek_split_blocks(
                comp, split_visible, stride=self.split_stride,
            )
            candidate_count = candidate_comp.size(2)

            # Run the learned CSA indexer only over this loop's compressed
            # candidates, rather than scoring every compressed block first.
            index_loop = int(max(0, min(loop_idx, self.index_loop_scale.size(0) - 1)))
            q_index = self.index_up(self.index_down(x)).view(
                B, T, self.index_heads, self.index_dim,
            )
            index_weight = self.index_weight(x)
            index_weight = (
                index_weight * (1.0 + self.index_loop_scale[index_loop])
                + self.index_loop_bias[index_loop]
            )
            candidate_keys = self.index_key(candidate_comp.detach())
            idx_scores_raw = (
                F.relu(torch.einsum("bthd,btkd->bthk", q_index, candidate_keys))
                * index_weight.unsqueeze(-1)
            ).sum(dim=2)
            idx_scores = idx_scores_raw.masked_fill(
                ~candidate_valid[None], float("-inf"),
            )
            k = min(self.top_k, candidate_count)
            topv, topi = torch.topk(idx_scores, k, dim=-1)
            sel_valid = topv > float("-inf")
            sel = torch.gather(
                candidate_comp, 2,
                topi.clamp(min=0).unsqueeze(-1).expand(B, T, k, self.head_dim),
            )
            aux_comp, aux_visible = candidate_comp, candidate_valid

        # ---- causal local window (recent raw detail) ----
        w = min(self.window_size, T)
        local_x, local_valid = causal_windows(x, w)
        if attention_mask is not None:
            # causal_windows returns indices [t-w+1 .. t]; recompute their exact
            # validity from the supplied attention mask. The query's own key therefore
            # remains available even for noisy position 0.
            base = torch.arange(T, device=x.device)[:, None] - (w - 1) + torch.arange(w, device=x.device)[None, :]
            base = base.clamp(min=0)
            local_valid = local_valid & attention_mask.gather(1, base).to(dtype=torch.bool)
        local_mem = self.local_kv(local_x)                     # [B, T, w, head_dim]

        # ---- combine into one per-query candidate set, one softmax ----
        mem   = torch.cat([sel, local_mem], dim=2)              # [B, T, k+w, head_dim]
        valid = torch.cat([sel_valid, local_valid], dim=2)      # [B, T, k+w]

        q = self.q_up(self.q_down(x)).view(B, T, self.n_heads, self.head_dim)
        q = self.rope.apply(self.q_norm(q))

        mem_n  = self.mem_norm(mem)
        logits = torch.einsum("bthd,btsd->bths", q, mem_n) / math.sqrt(self.head_dim)
        # Sink BEFORE mask -- see AttentionSink docstring for why order matters.
        logits = self.sink(logits.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
        att    = F.softmax(logits, dim=-1)

        out = torch.einsum("bths,btsd->bthd", att, mem)
        gate = self.out_gate(x, loop_idx).unsqueeze(-1)  # [B,T,n_heads,1]
        out = out * gate
        result = self.grouped_proj(out.reshape(B, T, self.n_heads * self.head_dim))

        if self.training and nb > 0:
            self._aux_losses.append(
                self._indexer_aux_loss(q, aux_comp, aux_visible, idx_scores_raw)
            )

        return result




class HCAAttention(nn.Module):
    """
    Heavily Compressed Attention (DeepSeek-V4 §HCA) -- causal, per-query.

    When ``split_stride`` is set, loop passes partition only compressed
    long-range blocks. The raw local window remains complete on every pass.

    Like CSAAttention but without top-k selection: each query attends over
    ALL causally-visible compressed blocks (there are few -- roughly
    T/m_prime -- so this stays cheap without needing sparsification) plus
    the same causal local window. Same fix rationale as CSAAttention:
    the original routed everything through one shared, non-causal pool.
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 m_prime: int = 128, window_size: int = 128,
                 groups: int = 8, group_dim: int = 1024, rope_dim: int = 64,
                 max_loops: int = 4, split_stride: int | None = None):
        super().__init__()
        if split_stride is not None and int(split_stride) < 2:
            raise ValueError("split_stride must be >= 2 when enabled")
        self.n_heads, self.head_dim = n_heads, head_dim
        self.m_prime, self.window_size = m_prime, window_size
        self.split_stride = None if split_stride is None else int(split_stride)

        self.kv       = nn.Linear(dim, head_dim, bias=False)
        self.z        = nn.Linear(dim, head_dim, bias=False)
        self.local_kv = nn.Linear(dim, head_dim, bias=False)

        self.q_down   = nn.Linear(dim, dim, bias=False)
        self.q_up     = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.q_norm   = RMSNorm(head_dim)
        self.mem_norm = RMSNorm(head_dim)
        self.rope     = PartialRoPE(head_dim, rope_dim=rope_dim)
        self.sink     = AttentionSink(n_heads)
        self.out_gate = AttentionOutputGate(dim, n_heads, max_loops)

        self.grouped_proj = GroupedOutputProjection(n_heads * head_dim, dim, groups, group_dim)

    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        m   = self.m_prime
        pad = (m - T % m) % m
        xp  = F.pad(x, (0, 0, 0, pad)) if pad else x
        nb  = xp.size(1) // m
        kv  = self.kv(xp).view(B, nb, m, self.head_dim)
        z   = self.z(xp).view(B, nb, m, self.head_dim)
        return (F.softmax(z, dim=2) * kv).sum(2)   # [B, nb, head_dim]

    def forward(self, x: torch.Tensor, loop_idx: int = 0, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        comp = self._compress(x)
        nb   = comp.size(1)

        block_last = (torch.arange(nb, device=x.device) + 1) * self.m_prime - 1
        pos        = torch.arange(T, device=x.device)
        if attention_mask is None:
            visible = (block_last[None, :] <= pos[:, None])       # [T, nb]
        else:
            if attention_mask.shape != (T, T):
                raise ValueError(f"attention_mask must be [{T},{T}], got {tuple(attention_mask.shape)}")
            visible_parts = []
            for j in range(nb):
                start = j * self.m_prime
                end = min((j + 1) * self.m_prime, T)
                visible_parts.append(attention_mask[:, start:end].all(dim=1))
            visible = torch.stack(visible_parts, dim=1) if visible_parts else torch.zeros(T, 0, device=x.device, dtype=torch.bool)

        w = min(self.window_size, T)
        local_x, local_valid = causal_windows(x, w)
        if attention_mask is not None:
            base = torch.arange(T, device=x.device)[:, None] - (w - 1) + torch.arange(w, device=x.device)[None, :]
            base = base.clamp(min=0)
            local_valid = local_valid & attention_mask.gather(1, base).to(dtype=torch.bool)
        local_mem = self.local_kv(local_x)                        # [B, T, w, head_dim]

        if self.split_stride is None:
            compressed_mem = comp[:, None, :, :].expand(B, T, nb, self.head_dim)
            compressed_valid = visible[None].expand(B, T, nb)
        else:
            split_visible = deepseek_split_block_mask(
                visible, loop_idx=loop_idx, stride=self.split_stride,
            )
            compressed_mem, split_valid = gather_deepseek_split_blocks(
                comp, split_visible, stride=self.split_stride,
            )
            compressed_valid = split_valid[None].expand(B, T, split_valid.size(1))

        # The compressed long-range bank is loop-partitioned, but every pass
        # retains the full causal raw-token local window.
        mem = torch.cat([compressed_mem, local_mem], dim=2)
        valid = torch.cat([compressed_valid, local_valid], dim=2)

        q = self.q_up(self.q_down(x)).view(B, T, self.n_heads, self.head_dim)
        q = self.rope.apply(self.q_norm(q))

        mem_n  = self.mem_norm(mem)
        logits = torch.einsum("bthd,btsd->bths", q, mem_n) / math.sqrt(self.head_dim)
        # Sink BEFORE mask -- see AttentionSink docstring for why order matters.
        logits = self.sink(logits.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        logits = logits.masked_fill(~valid[:, :, None, :], float("-inf"))
        att    = F.softmax(logits, dim=-1)

        out = torch.einsum("bths,btsd->bthd", att, mem)
        gate = self.out_gate(x, loop_idx).unsqueeze(-1)  # [B,T,n_heads,1]
        out = out * gate
        return self.grouped_proj(out.reshape(B, T, self.n_heads * self.head_dim))


def progressive_grassmann_offsets(
    loop_idx: int,
    offset_count: int = 5,
    spacing_base: int = 2,
) -> tuple[int, ...]:
    """Generate the local per-loop offsets for ``grassmann-new``.

    ``offset_count`` is the number of local offsets, including mandatory lag 1.
    For zero-based loop n, the result is {1} U {k*(spacing_base+n):
    k=1,...,offset_count-1}.
    """
    loop_idx = int(loop_idx)
    offset_count = int(offset_count)
    spacing_base = int(spacing_base)
    if loop_idx < 0:
        raise ValueError(f"loop_idx must be >= 0, got {loop_idx}")
    if offset_count < 1:
        raise ValueError(f"offset_count must be >= 1, got {offset_count}")
    if spacing_base < 1:
        raise ValueError(f"spacing_base must be >= 1, got {spacing_base}")
    spacing = spacing_base + loop_idx
    return (1,) + tuple(k * spacing for k in range(1, offset_count))


def adaptive_progressive_grassmann_offsets(
    loop_idx: int,
    seq_len: int,
    offset_count: int = 5,
    spacing_base: int = 2,
    recurrent_loops: int = 4,
) -> tuple[int, ...]:
    """Resolve progressive local offsets plus sparse adaptive global offsets.

    The local offsets for recurrent pass ``n`` are::

        L_n = {1} union {k * (spacing_base + n): k=1,...,N-1}

    ``offset_count`` (N) counts local offsets only.  The global stride is
    chosen once for the whole recurrent stack so it cannot overlap any
    configured local offset in any pass::

        L_max = (N - 1) * (spacing_base + recurrent_loops - 1)
        S = max(L_max + 1, ceil((seq_len - 1) / N))
        G = {j*S: j=1,...,floor((seq_len - 1)/S)}

    The returned set is ``L_n union G``, sorted and unique.  Lags are bounded
    by ``seq_len - 1`` because a [B,T,D] input has no valid pair at a larger
    lag.  This function is pure Python and does not allocate tensors.
    """
    loop_idx = int(loop_idx)
    seq_len = int(seq_len)
    offset_count = int(offset_count)
    spacing_base = int(spacing_base)
    recurrent_loops = int(recurrent_loops)

    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if offset_count < 1:
        raise ValueError(f"offset_count must be >= 1, got {offset_count}")
    if spacing_base < 1:
        raise ValueError(f"spacing_base must be >= 1, got {spacing_base}")
    if recurrent_loops < 1:
        raise ValueError(f"recurrent_loops must be >= 1, got {recurrent_loops}")
    if not 0 <= loop_idx < recurrent_loops:
        raise ValueError(
            f"loop_idx must be in [0, {recurrent_loops - 1}], got {loop_idx}"
        )
    if seq_len == 1:
        return ()

    max_lag = seq_len - 1
    local_spacing = spacing_base + loop_idx
    # Cap local generation by the actual available history.  This also
    # prevents pathological N values from producing enormous Python tuples.
    max_valid_local_count = 1 + max_lag // local_spacing
    local_count = min(offset_count, max_valid_local_count)
    local = progressive_grassmann_offsets(
        loop_idx=loop_idx,
        offset_count=local_count,
        spacing_base=spacing_base,
    )

    max_local_across_stack = (offset_count - 1) * (
        spacing_base + recurrent_loops - 1
    )
    adaptive_stride = max(
        max_local_across_stack + 1,
        (max_lag + offset_count - 1) // offset_count,
    )
    global_offsets = range(adaptive_stride, max_lag + 1, adaptive_stride)
    return tuple(sorted(set(local).union(global_offsets)))


def real_grassmann_offsets(
    seq_len: int,
    local_offsets: tuple[int, ...] = (1, 2, 4, 8, 12, 16),
    long_range_stride: int = 16,
) -> tuple[int, ...]:
    """Local offsets plus regularly-spaced long-range offsets through context.

    For a sequence of T positions, append stride, 2*stride, ... while the
    offset is strictly less than T. Local offsets outside the actual sequence
    are omitted. The returned values are sorted and unique. This stride-based
    tail is a project extension; it is not specified in arXiv:2512.19428v1,
    whose method examples use local windows through 16.
    """
    seq_len = int(seq_len)
    long_range_stride = int(long_range_stride)
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if long_range_stride < 1:
        raise ValueError(
            f"long_range_stride must be >= 1, got {long_range_stride}"
        )
    local_values = {int(v) for v in local_offsets}
    if not local_values or any(v < 1 for v in local_values):
        raise ValueError("local_offsets must contain positive integers")
    # Only lags that can occur in this concrete [B,T,D] tensor are effective.
    values = {v for v in local_values if v < seq_len}
    values.update(range(long_range_stride, seq_len, long_range_stride))
    return tuple(sorted(values))


class GrassmannFlowAttention(nn.Module):
    """Causal multi-scale Grassmann/Pluecker sequence mixer.

    This replaces only the attention branch. For each token position ``t``
    and offset ``delta``, it forms the oriented 2-plane generated by
    ``(z_t, z_{t-delta})``. The source paper writes ``z_{t+delta}`` while
    calling the operation causal; that is a future-token leak for a
    left-to-right language model. This implementation deliberately uses the
    previous token ``z_{t-delta}`` and never reads a position greater than t.

    The paper describes the gated representation as the layer output. This
    project has an external pre-norm residual owner, so the module returns
    ``LayerNorm(mixed) - x`` as the attention-branch delta. The surrounding
    Transformer residual, FFN, recurrent loops, and dropout remain untouched.
    Positions with no valid previous pair return an exact zero delta.

    Modes:
      - grassmann: the original fixed offset tuple (backward-compatible).
      - grassmann-new: loop-dependent local offsets plus adaptive sparse
        global offsets shared across the configured recurrent depth.
      - real-grassmann: fixed local offsets plus stride-spaced offsets up to
        the actual sequence length.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 32,
        offsets: tuple[int, ...] = (1, 2, 4, 8, 12, 16),
        mode: str = "grassmann",
        offset_count: int = 5,
        offset_spacing_base: int = 2,
        long_range_stride: int = 16,
        max_loops: int = 4,
        offset_scheme: str = "adaptive_sparse_global_v1",
        eps: float = 1e-2,
    ):
        super().__init__()
        if dim < 2:
            raise ValueError(f"GrassmannFlowAttention requires dim >= 2, got {dim}")
        if not 2 <= rank <= dim:
            raise ValueError(
                f"GrassmannFlowAttention rank must be in [2, dim={dim}], got {rank}"
            )
        if mode not in {"grassmann", "grassmann-new", "real-grassmann"}:
            raise ValueError(f"Unsupported Grassmann mode: {mode!r}")
        normalized_offsets = tuple(int(v) for v in offsets)
        if not normalized_offsets or any(v < 1 for v in normalized_offsets):
            raise ValueError("Grassmann offsets must be a non-empty sequence of positive integers")
        if len(set(normalized_offsets)) != len(normalized_offsets):
            raise ValueError(f"Grassmann offsets must be unique, got {normalized_offsets}")

        self.dim = int(dim)
        self.rank = int(rank)
        self.offsets = normalized_offsets
        self.mode = mode
        self.offset_count = int(offset_count)
        self.offset_spacing_base = int(offset_spacing_base)
        self.long_range_stride = int(long_range_stride)
        self.max_loops = int(max_loops)
        self.offset_scheme = str(offset_scheme)
        if self.offset_count < 1:
            raise ValueError("offset_count must be >= 1")
        if self.offset_spacing_base < 1:
            raise ValueError("offset_spacing_base must be >= 1")
        if self.long_range_stride < 1:
            raise ValueError("long_range_stride must be >= 1")
        if self.max_loops < 1:
            raise ValueError("max_loops must be >= 1")
        if self.offset_scheme not in {
            "adaptive_sparse_global_v1", "legacy_local_only_v0"
        }:
            raise ValueError(
                f"Unsupported Grassmann offset scheme: {self.offset_scheme!r}"
            )
        self.eps = float(eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"Grassmann normalization eps must be finite and > 0, got {eps!r}")

        self.reduce = nn.Linear(dim, rank, bias=True)
        self.plucker_dim = rank * (rank - 1) // 2
        self.plucker_proj = nn.Linear(self.plucker_dim, dim, bias=True)
        self.gate = nn.Linear(2 * dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)

        pair_indices = torch.triu_indices(rank, rank, offset=1)
        self.register_buffer("plucker_i", pair_indices[0], persistent=False)
        self.register_buffer("plucker_j", pair_indices[1], persistent=False)

    def offsets_for(self, seq_len: int, loop_idx: int = 0) -> tuple[int, ...]:
        """Resolve effective offsets for this mode and recurrent pass."""
        if self.mode == "grassmann-new":
            if self.offset_scheme == "legacy_local_only_v0":
                if seq_len <= 1:
                    return ()
                spacing = self.offset_spacing_base + int(loop_idx)
                max_valid_count = 1 + (seq_len - 1) // spacing
                return progressive_grassmann_offsets(
                    loop_idx=loop_idx,
                    offset_count=min(self.offset_count, max_valid_count),
                    spacing_base=self.offset_spacing_base,
                )
            return adaptive_progressive_grassmann_offsets(
                loop_idx=loop_idx,
                seq_len=seq_len,
                offset_count=self.offset_count,
                spacing_base=self.offset_spacing_base,
                recurrent_loops=self.max_loops,
            )
        if self.mode == "real-grassmann":
            return real_grassmann_offsets(
                seq_len=seq_len,
                local_offsets=self.offsets,
                long_range_stride=self.long_range_stride,
            )
        return self.offsets

    def forward(
        self,
        x: torch.Tensor,
        loop_idx: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"GrassmannFlowAttention expects [B,T,D], got {tuple(x.shape)}")
        batch, seq_len, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected hidden dimension {self.dim}, got {dim}")
        if attention_mask is not None and attention_mask.shape != (seq_len, seq_len):
            raise ValueError(
                f"attention_mask must be [{seq_len},{seq_len}], got {tuple(attention_mask.shape)}"
            )

        active_offsets = self.offsets_for(seq_len=seq_len, loop_idx=loop_idx)
        z = self.reduce(x)
        positions = torch.arange(seq_len, device=x.device)
        z32 = z.float()
        g_sum = x.new_zeros(batch, seq_len, self.dim)
        valid_count = x.new_zeros(seq_len)

        for delta in active_offsets:
            if delta >= seq_len:
                continue

            # At position t, this is exactly z[t-delta]. Left padding is
            # zero-filled but those positions are excluded by `valid` below.
            z_prev = F.pad(z[:, : seq_len - delta, :], (0, 0, delta, 0))
            valid = positions >= delta
            if attention_mask is not None:
                prior_positions = (positions - delta).clamp_min(0)
                visible = attention_mask.to(device=x.device, dtype=torch.bool)[
                    positions, prior_positions
                ]
                valid = valid & visible

            # Pluecker coordinates for z_t wedge z_{t-delta}:
            # p_ij = z[t,i] z[t-delta,j] - z[t,j] z[t-delta,i].
            # Compute and normalize in fp32 to avoid low-precision
            # cancellation/underflow for nearly collinear pairs.
            prev32 = z_prev.float()
            p = (
                z32[..., self.plucker_i] * prev32[..., self.plucker_j]
                - z32[..., self.plucker_j] * prev32[..., self.plucker_i]
            )
            # Smooth normalization avoids the 1/||p|| derivative singularity
            # when the two reduced vectors are equal or nearly collinear.
            # eps=1e-2 gives p / sqrt(||p||^2 + 1e-4), matching the stable
            # regularizer validated for repeated-token sequences. Unlike a
            # hard clamp, this keeps the derivative bounded near p=0.
            inv_norm = torch.rsqrt(
                p.square().sum(dim=-1, keepdim=True) + self.eps * self.eps
            )
            p = p * inv_norm
            projected = self.plucker_proj(p.to(dtype=x.dtype))
            projected = projected * valid.view(1, seq_len, 1).to(dtype=x.dtype)

            g_sum = g_sum + projected
            valid_count = valid_count + valid.to(dtype=x.dtype)

        has_pair = valid_count > 0
        g = g_sum / valid_count.clamp_min(1).view(1, seq_len, 1)
        alpha = torch.sigmoid(self.gate(torch.cat((x, g), dim=-1)))
        mixed = alpha * x + (1.0 - alpha) * g

        # Match the paper's LayerNorm after gated fusion, expressed as a
        # branch delta to respect this model's existing external residual.
        delta = self.norm(mixed) - x
        return delta * has_pair.view(1, seq_len, 1).to(dtype=x.dtype)
