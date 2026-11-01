"""JAX / Flax implementation of Recursive Flint (FlintForCausalLM) for TPU v5p training.

Implements all 6 architectural components of RecursiveSI/flint:
  1. Per-head QK RMSNorm (q_norm, k_norm over head_dim=128) + pre/post RMSNorm.
  2. Learned attention sinks: per-head `softmax_offset` logit added to the attention
     softmax denominator without contributing to the value aggregation.
  3. Hybrid attention: 35 sliding-window layers (window=513, including current token)
     and 14 full-attention layers (`config.layer_types`).
  4. Selective RoPE: RoPE applied on sliding-window layers; skipped on the 14
     full-attention layers listed in `config.no_rope_layer_indices`.
  5. Dense SwiGLU MLPs on `config.mlp_only_layers` ([0, 1], intermediate_size=6144);
     Sparse MoE blocks on layers 2..48.
  6. Sigmoid Top-K routing (`k=6` of `256` experts, renormalized) with TPU v5p
     MaxText/Megablox `jax.lax.ragged_dot` grouped matmul kernel + always-on shared expert.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from flint_jax.config import FlintJAXConfig


class FlintRMSNorm(nn.Module):
    """RMSNorm computed in FP32 and cast back to activation dtype."""

    dim: int
    eps: float = 1e-6
    dtype: Any = jnp.bfloat16
    weight_dtype: Any = jnp.bfloat16
    sharding_axes: tuple[str | None, ...] = (None,)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param(
            "weight",
            nn.with_partitioning(nn.initializers.ones, self.sharding_axes),
            (self.dim,),
            self.weight_dtype,
        )
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        normed = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight.astype(jnp.float32)).astype(self.dtype)


def apply_rope(
    q: jnp.ndarray,
    k: jnp.ndarray,
    positions: jnp.ndarray,
    head_dim: int = 128,
    rope_theta: float = 10000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Applies NeoX/Llama-style half-rotated RoPE to query and key tensors.

    Args:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        positions: [batch, seq_len]
    """
    half_dim = head_dim // 2
    freq_seq = jnp.arange(0, half_dim, dtype=jnp.float32) / float(half_dim)
    inv_freq = 1.0 / (rope_theta ** freq_seq)  # [half_dim]
    # [batch, seq_len, 1, half_dim]
    angles = jnp.einsum("bs,d->bsd", positions.astype(jnp.float32), inv_freq)[:, :, None, :]
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    def _rotate(x: jnp.ndarray) -> jnp.ndarray:
        x_f32 = x.astype(jnp.float32)
        x1 = x_f32[..., :half_dim]
        x2 = x_f32[..., half_dim:]
        rotated = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        return rotated.astype(x.dtype)

    return _rotate(q), _rotate(k)


@functools.partial(jax.checkpoint, prevent_cse=True, static_argnums=(4, 5, 6))
def attention_with_learned_sinks(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    softmax_offset: jnp.ndarray,
    is_sliding: bool,
    sliding_window: int,
    dtype: Any = jnp.bfloat16,
) -> jnp.ndarray:
    """Grouped-Query Attention with per-head learned attention sinks (`softmax_offset`).

    In Flint, each attention head h has a learned scalar sink logit `s_h = softmax_offset[h]`.
    The sink logit participates in the softmax denominator `exp(s_h - max_logit)` with
    zero value contribution:
        m_i = max(s_h, max_{j in mask(i)} logits_{i,j})
        weights_{i,j} = exp(logits_{i,j} - m_i) / (exp(s_h - m_i) + sum_k exp(logits_{i,k} - m_i))

    Args:
        q: [B, T, H_q, D]
        k: [B, T, H_kv, D]
        v: [B, T, H_kv, D]
        softmax_offset: [H_q] learned sink logits per query head
        is_sliding: True for sliding_attention layers, False for full_attention layers
        sliding_window: Window size including current token (513 in Flint -> i - j <= 512)
    Returns:
        attn_out: [B, T, H_q, D]
    """
    bsz, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Reshape Q to [B, H_kv, G, T, D] for GQA
    q_grouped = q.reshape(bsz, seq_len, num_kv_heads, num_queries_per_kv, head_dim)
    q_grouped = jnp.transpose(q_grouped, (0, 2, 3, 1, 4))  # [B, H_kv, G, T, D]
    k_t = jnp.transpose(k, (0, 2, 1, 3))  # [B, H_kv, T, D]
    v_t = jnp.transpose(v, (0, 2, 1, 3))  # [B, H_kv, T, D]

    # Attention logits in FP32: [B, H_kv, G, T_q, T_k]
    logits = jnp.einsum(
        "bhgqd,bhkd->bhgqk",
        q_grouped.astype(jnp.float32),
        k_t.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ) * scale

    # Build causal (or causal + sliding-window) mask: [1, 1, 1, T_q, T_k]
    q_idx = jnp.arange(seq_len)[:, None]
    k_idx = jnp.arange(seq_len)[None, :]
    mask = q_idx >= k_idx
    if is_sliding and sliding_window is not None:
        # Flint's sliding_window=513 includes the current token (0 <= q_idx - k_idx < 513)
        mask = mask & ((q_idx - k_idx) < sliding_window)
    logits = jnp.where(mask[None, None, None, :, :], logits, -1e30)

    # Incorporate learned per-head sink logit s_h into stable softmax denominator
    # sink_logits shape: [1, H_kv, G, 1, 1]
    sink_logits = softmax_offset.astype(jnp.float32).reshape(1, num_kv_heads, num_queries_per_kv, 1, 1)
    max_key_logit = jnp.max(logits, axis=-1, keepdims=True)
    max_logit = jnp.maximum(max_key_logit, sink_logits)

    exp_logits = jnp.where(mask[None, None, None, :, :], jnp.exp(logits - max_logit), 0.0)
    exp_sink = jnp.exp(sink_logits - max_logit)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True) + exp_sink
    attn_weights = (exp_logits / (denom + 1e-20)).astype(dtype)

    # Aggregate values: [B, H_kv, G, T_q, D]
    out_grouped = jnp.einsum(
        "bhgqk,bhkd->bhgqd",
        attn_weights,
        v_t.astype(dtype),
        preferred_element_type=jnp.float32,
    ).astype(dtype)

    # Transpose back to [B, T, H_q, D]
    out = jnp.transpose(out_grouped, (0, 3, 1, 2, 4)).reshape(bsz, seq_len, num_q_heads, head_dim)
    return out


class FlintAttention(nn.Module):
    """Flint Grouped-Query Attention with QK RMSNorm, Learned Sinks, Sliding Window & Selective RoPE."""

    config: FlintJAXConfig
    layer_id: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        cfg = self.config
        bsz, seq_len, _ = hidden_states.shape
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim

        q_proj = nn.Dense(
            features=q_dim,
            use_bias=cfg.attention_bias,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("fsdp", "tensor"),
            ),
            name="q_proj",
        )
        k_proj = nn.Dense(
            features=kv_dim,
            use_bias=cfg.attention_bias,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("fsdp", "tensor"),
            ),
            name="k_proj",
        )
        v_proj = nn.Dense(
            features=kv_dim,
            use_bias=cfg.attention_bias,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("fsdp", "tensor"),
            ),
            name="v_proj",
        )
        o_proj = nn.Dense(
            features=cfg.hidden_size,
            use_bias=cfg.attention_bias,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("tensor", "fsdp"),
            ),
            name="o_proj",
        )

        q = q_proj(hidden_states).reshape(bsz, seq_len, cfg.num_attention_heads, cfg.head_dim)
        k = k_proj(hidden_states).reshape(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim)
        v = v_proj(hidden_states).reshape(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim)

        # Per-head QK RMSNorm over head_dim=128
        q = FlintRMSNorm(
            dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            name="q_norm",
        )(q)
        k = FlintRMSNorm(
            dim=cfg.head_dim,
            eps=cfg.rms_norm_eps,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            name="k_norm",
        )(k)

        # Selective RoPE: only applied on sliding-window layers (not in no_rope_layer_indices)
        if cfg.uses_rope(self.layer_id):
            q, k = apply_rope(
                q=q,
                k=k,
                positions=positions,
                head_dim=cfg.head_dim,
                rope_theta=cfg.rope_theta,
            )

        # Learned per-head attention sink parameter (`self_attn.softmax_offset`)
        softmax_offset = self.param(
            "softmax_offset",
            nn.with_partitioning(nn.initializers.zeros, ("tensor",)),
            (cfg.num_attention_heads,),
            cfg.weight_dtype,
        )

        attn_out = attention_with_learned_sinks(
            q,
            k,
            v,
            softmax_offset,
            cfg.is_sliding_layer(self.layer_id),
            cfg.sliding_window,
            cfg.dtype,
        )
        return o_proj(attn_out.reshape(bsz, seq_len, q_dim))


class FlintMLP(nn.Module):
    """SwiGLU MLP used for dense layers ([0, 1]) and the MoE shared expert."""

    hidden_size: int
    intermediate_size: int
    dtype: Any = jnp.bfloat16
    weight_dtype: Any = jnp.bfloat16
    initializer_range: float = 0.02

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate = nn.Dense(
            features=self.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(self.initializer_range),
                ("fsdp", "tensor"),
            ),
            name="gate_proj",
        )(x)
        up = nn.Dense(
            features=self.intermediate_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(self.initializer_range),
                ("fsdp", "tensor"),
            ),
            name="up_proj",
        )(x)
        hidden = nn.silu(gate) * up
        down = nn.Dense(
            features=self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(self.initializer_range),
                ("tensor", "fsdp"),
            ),
            name="down_proj",
        )(hidden)
        return down


_EP_IN_SPECS = (
    P(("data", "fsdp", "expert"), None, None),
    P(("data", "fsdp", "expert"), None, None),
    P(("data", "fsdp", "expert"), None, None),
    P(("expert", "fsdp"), None, "tensor"),
    P(("expert", "fsdp"), None, "tensor"),
    P(("expert", "fsdp"), "tensor", None),
)
_EP_OUT_SPEC = P(("data", "fsdp", "expert"), None, None)


def _ep_moe_impl(
    h_local: jnp.ndarray,
    ids_local: jnp.ndarray,
    w_local: jnp.ndarray,
    eg_local: jnp.ndarray,
    eu_local: jnp.ndarray,
    ed_local: jnp.ndarray,
) -> jnp.ndarray:
    compute_dtype = h_local.dtype
    h_all = jax.lax.all_gather(h_local, ("data", "fsdp", "expert"), axis=0, tiled=True)
    ids_all = jax.lax.all_gather(ids_local, ("data", "fsdp", "expert"), axis=0, tiled=True)
    w_all = jax.lax.all_gather(w_local, ("data", "fsdp", "expert"), axis=0, tiled=True)

    total_bsz, t_len, d_dim = h_all.shape
    n_toks = total_bsz * t_len
    k_top = ids_all.shape[-1]
    num_local_experts = eg_local.shape[0]
    ep_rank = jax.lax.axis_index(("expert", "fsdp"))
    exp_start = ep_rank * num_local_experts

    flat_toks = h_all.reshape(n_toks, d_dim)
    flat_ids = ids_all.reshape(-1)
    flat_w = w_all.reshape(-1)

    is_mine = (flat_ids >= exp_start) & (flat_ids < exp_start + num_local_experts)
    local_exp_id = jnp.where(is_mine, flat_ids - exp_start, num_local_experts)
    masked_w = jnp.where(is_mine, flat_w, 0.0)

    tok_indices = jnp.repeat(jnp.arange(n_toks), k_top)
    sort_order = jnp.argsort(local_exp_id)
    sorted_exp_ids = local_exp_id[sort_order]
    sorted_inputs = flat_toks[tok_indices[sort_order]]

    local_group_sizes = jnp.bincount(sorted_exp_ids, length=num_local_experts).astype(jnp.int32)

    g_out = jax.lax.ragged_dot(
        lhs=sorted_inputs.astype(compute_dtype),
        rhs=eg_local.astype(compute_dtype),
        group_sizes=local_group_sizes,
    )
    u_out = jax.lax.ragged_dot(
        lhs=sorted_inputs.astype(compute_dtype),
        rhs=eu_local.astype(compute_dtype),
        group_sizes=local_group_sizes,
    )
    inter = nn.silu(g_out) * u_out
    d_out = jax.lax.ragged_dot(
        lhs=inter.astype(compute_dtype),
        rhs=ed_local.astype(compute_dtype),
        group_sizes=local_group_sizes,
    )

    unsort_order = jnp.argsort(sort_order)
    routed_per_k = d_out[unsort_order].reshape(n_toks, k_top, d_dim)
    weighted = jnp.sum(
        routed_per_k * masked_w.reshape(n_toks, k_top, 1).astype(compute_dtype),
        axis=1,
    ).reshape(total_bsz, t_len, d_dim)

    return jax.lax.psum_scatter(
        weighted,
        ("data", "fsdp", "expert"),
        scatter_dimension=0,
        tiled=True,
    )


@jax.custom_vjp
def _ep_ragged_moe(
    hidden_states: jnp.ndarray,
    topk_ids: jnp.ndarray,
    topk_weights: jnp.ndarray,
    experts_gate: jnp.ndarray,
    experts_up: jnp.ndarray,
    experts_down: jnp.ndarray,
) -> jnp.ndarray:
    return jax.shard_map(
        _ep_moe_impl,
        in_specs=_EP_IN_SPECS,
        out_specs=_EP_OUT_SPEC,
        check_vma=False,
    )(hidden_states, topk_ids, topk_weights, experts_gate, experts_up, experts_down)


def _ep_ragged_moe_fwd(
    hidden_states: jnp.ndarray,
    topk_ids: jnp.ndarray,
    topk_weights: jnp.ndarray,
    experts_gate: jnp.ndarray,
    experts_up: jnp.ndarray,
    experts_down: jnp.ndarray,
):
    out = _ep_ragged_moe(
        hidden_states, topk_ids, topk_weights, experts_gate, experts_up, experts_down
    )
    return out, (
        hidden_states,
        topk_ids,
        topk_weights,
        experts_gate,
        experts_up,
        experts_down,
    )


def _ep_ragged_moe_bwd(res, g):
    res, g = jax.lax.optimization_barrier((res, g))
    hidden_states, topk_ids, topk_weights, experts_gate, experts_up, experts_down = res

    def _bwd_shard(h_local, ids_local, w_local, eg_local, eu_local, ed_local, g_local):
        def _diff_fn(h_, w_, eg_, eu_, ed_):
            return _ep_moe_impl(h_, ids_local, w_, eg_, eu_, ed_)

        _, vjp_fn = jax.vjp(_diff_fn, h_local, w_local, eg_local, eu_local, ed_local)
        return vjp_fn(g_local)

    dh, dw, deg, deu, ded = jax.shard_map(
        _bwd_shard,
        in_specs=(*_EP_IN_SPECS, _EP_OUT_SPEC),
        out_specs=(
            P(("data", "fsdp", "expert"), None, None),
            P(("data", "fsdp", "expert"), None, None),
            P(("expert", "fsdp"), None, "tensor"),
            P(("expert", "fsdp"), None, "tensor"),
            P(("expert", "fsdp"), "tensor", None),
        ),
        check_vma=False,
    )(hidden_states, topk_ids, topk_weights, experts_gate, experts_up, experts_down, g)

    return dh, None, dw, deg, deu, ded


_ep_ragged_moe.defvjp(_ep_ragged_moe_fwd, _ep_ragged_moe_bwd)


class FlintSparseMoeBlock(nn.Module):
    """Flint Sparse MoE block (256 routed experts, sigmoid top-6 routing + 1 shared expert).

    Uses MaxText / Megablox-style `jax.lax.ragged_dot` on TPU v5p to execute grouped
    matrix multiplications across the 256 experts without token dropping.
    """

    config: FlintJAXConfig
    layer_id: int

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        bsz, seq_len, hidden_dim = hidden_states.shape

        # 1. Always-active ungated shared expert (shared_expert_intermediate_size = 1024)
        shared_output = FlintMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.shared_expert_intermediate_size,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            initializer_range=cfg.initializer_range,
            name="shared_expert",
        )(hidden_states)

        # 2. Router gate: sigmoid scores + optional expert_bias, top-6 selection, renormalized
        gate_kernel = self.param(
            "gate_weight",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                (None, "fsdp"),
            ),
            (cfg.num_experts, cfg.hidden_size),
            cfg.weight_dtype,
        )
        expert_bias = self.param(
            "expert_bias",
            nn.with_partitioning(nn.initializers.zeros, (None,)),
            (cfg.num_experts,),
            jnp.float32,
        )

        router_logits = jnp.einsum(
            "btd,ed->bte",
            hidden_states.astype(jnp.float32),
            gate_kernel.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )  # [B, T, E=256]
        scores = jax.nn.sigmoid(router_logits)
        scores_for_choice = scores + expert_bias[None, None, :]
        _, topk_ids = jax.lax.top_k(scores_for_choice, k=cfg.num_experts_per_tok)  # [B, T, K=6]
        topk_weights = jnp.take_along_axis(scores, topk_ids, axis=-1)  # [B, T, K=6]
        if cfg.norm_topk_prob:
            topk_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20)

        # 3. 3D Stacked Expert Parameters: [E=256, D=2048, M] and [E=256, M, D=2048]
        # Sharded 16-way along the expert dimension ('expert', 'fsdp') -> 16 experts per TPU v5p chip
        experts_gate = self.param(
            "experts_gate_proj",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                (("expert", "fsdp"), None, "tensor"),
            ),
            (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
            cfg.weight_dtype,
        )
        experts_up = self.param(
            "experts_up_proj",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                (("expert", "fsdp"), None, "tensor"),
            ),
            (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
            cfg.weight_dtype,
        )
        experts_down = self.param(
            "experts_down_proj",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                (("expert", "fsdp"), "tensor", None),
            ),
            (cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
            cfg.weight_dtype,
        )

        routed_output = _ep_ragged_moe(
            hidden_states.astype(cfg.dtype),
            topk_ids,
            topk_weights,
            experts_gate,
            experts_up,
            experts_down,
        )

        num_tokens = bsz * seq_len
        global_group_sizes = jnp.bincount(topk_ids.reshape(-1), length=cfg.num_experts).astype(jnp.float32)
        tokens_per_expert = global_group_sizes / float(num_tokens)
        prob_per_expert = jnp.mean(
            scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-20),
            axis=(0, 1),
        )
        aux_loss = cfg.num_experts * jnp.sum(tokens_per_expert * prob_per_expert)

        return (routed_output + shared_output).astype(cfg.dtype), aux_loss


class FlintDecoderLayer(nn.Module):
    """Single Flint Transformer Decoder Layer (Pre-Norm + Attention + Dense MLP or Sparse MoE)."""

    config: FlintJAXConfig
    layer_id: int

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        positions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config

        normed_attn_in = FlintRMSNorm(
            dim=cfg.hidden_size,
            eps=cfg.rms_norm_eps,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            sharding_axes=(None,),
            name="input_layernorm",
        )(hidden_states)

        attn_out = FlintAttention(
            config=cfg,
            layer_id=self.layer_id,
            name="self_attn",
        )(normed_attn_in, positions)

        hidden_states = hidden_states + attn_out

        normed_mlp_in = FlintRMSNorm(
            dim=cfg.hidden_size,
            eps=cfg.rms_norm_eps,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            sharding_axes=(None,),
            name="post_attention_layernorm",
        )(hidden_states)

        if cfg.is_dense_layer(self.layer_id):
            mlp_out = FlintMLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                dtype=cfg.dtype,
                weight_dtype=cfg.weight_dtype,
                initializer_range=cfg.initializer_range,
                name="mlp",
            )(normed_mlp_in)
            aux_loss = jnp.array(0.0, dtype=jnp.float32)
        else:
            mlp_out, aux_loss = FlintSparseMoeBlock(
                config=cfg,
                layer_id=self.layer_id,
                name="mlp",
            )(normed_mlp_in)

        hidden_states = hidden_states + mlp_out
        return hidden_states, aux_loss


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _run_layer_custom_vjp(
    layer: FlintDecoderLayer,
    layer_params: Any,
    hidden_states: jnp.ndarray,
    positions: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return layer.apply({"params": layer_params}, hidden_states, positions)


def _run_layer_fwd(
    layer: FlintDecoderLayer,
    layer_params: Any,
    hidden_states: jnp.ndarray,
    positions: jnp.ndarray,
):
    out = layer.apply({"params": layer_params}, hidden_states, positions)
    return out, (layer_params, hidden_states, positions)


def _run_layer_bwd(layer: FlintDecoderLayer, res, g):
    res, g = jax.lax.optimization_barrier((res, g))
    layer_params, hidden_states, positions = res

    def _fn(p, h):
        return layer.apply({"params": p}, h, positions)

    _, vjp_fn = jax.vjp(_fn, layer_params, hidden_states)
    d_params, d_hidden = vjp_fn(g)
    d_params, d_hidden = jax.lax.optimization_barrier((d_params, d_hidden))
    return d_params, d_hidden, None


_run_layer_custom_vjp.defvjp(_run_layer_fwd, _run_layer_bwd)


class FlintForCausalLM(nn.Module):
    """Full 49-layer Recursive Flint (39.8B-A2.5B) Causal Language Model in JAX / Flax."""

    config: FlintJAXConfig

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        bsz, seq_len = input_ids.shape
        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], (bsz, seq_len))

        embed_tokens = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.hidden_size,
            dtype=cfg.dtype,
            param_dtype=cfg.weight_dtype,
            embedding_init=nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("fsdp", "tensor"),
            ),
            name="embed_tokens",
        )
        hidden_states = embed_tokens(input_ids)
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, P(("data", "fsdp", "expert"), None, None)
        )

        total_aux_loss = jnp.array(0.0, dtype=jnp.float32)

        for layer_id in range(cfg.num_hidden_layers):
            if self.is_initializing() or not cfg.remat_layers:
                hidden_states, layer_aux_loss = FlintDecoderLayer(
                    config=cfg,
                    layer_id=layer_id,
                    name=f"layers_{layer_id}",
                )(hidden_states, positions)
            else:
                detached_layer = FlintDecoderLayer(
                    config=cfg,
                    layer_id=layer_id,
                    parent=None,
                )
                layer_params = self.variables["params"][f"layers_{layer_id}"]
                hidden_states, layer_aux_loss = _run_layer_custom_vjp(
                    detached_layer,
                    layer_params,
                    hidden_states,
                    positions,
                )
            hidden_states = jax.lax.with_sharding_constraint(
                hidden_states, P(("data", "fsdp", "expert"), None, None)
            )
            hidden_states, layer_aux_loss = jax.lax.optimization_barrier(
                (hidden_states, layer_aux_loss)
            )
            total_aux_loss = total_aux_loss + layer_aux_loss

        hidden_states = FlintRMSNorm(
            dim=cfg.hidden_size,
            eps=cfg.rms_norm_eps,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            sharding_axes=(None,),
            name="norm",
        )(hidden_states)

        # Flint uses FP32 lm_head logits (--enable-fp32-lm-head) for numerical stability
        lm_head = nn.Dense(
            features=cfg.vocab_size,
            use_bias=False,
            dtype=jnp.float32,
            param_dtype=cfg.weight_dtype,
            kernel_init=nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("fsdp", "tensor"),
            ),
            name="lm_head",
        )
        logits = lm_head(hidden_states)
        logits = jax.lax.with_sharding_constraint(
            logits, P(("data", "fsdp", "expert"), None, "tensor")
        )
        return logits, total_aux_loss * cfg.router_aux_loss_coef
