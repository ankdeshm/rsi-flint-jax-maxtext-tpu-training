"""JAX / Flax implementation of Recursive Flint (FlintForCausalLM) for TPU v5p training.

Implements all 6 architectural components of RecursiveSI/flint with TPU-native kernels:
  1. Per-head QK RMSNorm (q_norm, k_norm over head_dim=128) + pre/post RMSNorm.
  2. Learned attention sinks: per-head `softmax_offset` logit passed natively to TPU Pallas
     `SplashAttention` (`sinks=softmax_offset`) with block-sparse Causal / Sliding-Window masks.
  3. Hybrid attention: 35 sliding-window layers (`window=513`, `LocalMask(512, 0)`)
     and 14 full-attention layers (`CausalMask`).
  4. Selective BF16 RoPE: RoPE applied in pure BF16 on sliding-window layers; skipped on the 14
     full-attention layers listed in `config.no_rope_layer_indices`.
  5. Dense SwiGLU MLPs on `config.mlp_only_layers` ([0, 1], intermediate_size=6144);
     Sparse MoE blocks on layers 2..48.
  6. Sigmoid Top-K routing (`k=6` of `256` experts, renormalized) with fused `W_gate_up`
     `jax.lax.ragged_dot` (zero token ICI under FSDP=16, or capacity-compacted NaN-free
     `jnp.nonzero` dispatch under EP>1) + chunked `@jax.checkpoint` `jax.lax.scan` Cross-Entropy.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import flax.linen as nn
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel,
    splash_attention_mask,
)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import optax

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
    """Applies NeoX/Llama-style half-rotated RoPE directly in activation dtype (BF16)."""
    half_dim = head_dim // 2
    freq_seq = jnp.arange(0, half_dim, dtype=jnp.float32) / float(half_dim)
    inv_freq = 1.0 / (rope_theta ** freq_seq)
    angles = jnp.einsum("bs,d->bsd", positions.astype(jnp.float32), inv_freq)[:, :, None, :]
    cos = jnp.cos(angles).astype(q.dtype)
    sin = jnp.sin(angles).astype(q.dtype)

    def _rotate(x: jnp.ndarray) -> jnp.ndarray:
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

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
    """Reference dense GQA with per-head learned attention sinks (`softmax_offset`)."""
    bsz, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    q_grouped = q.reshape(bsz, seq_len, num_kv_heads, num_queries_per_kv, head_dim)
    q_grouped = jnp.transpose(q_grouped, (0, 2, 3, 1, 4))
    k_t = jnp.transpose(k, (0, 2, 1, 3))
    v_t = jnp.transpose(v, (0, 2, 1, 3))

    logits = (
        jnp.einsum(
            "bhgqd,bhkd->bhgqk",
            q_grouped.astype(jnp.float32),
            k_t.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        )
        * scale
    )

    q_idx = jnp.arange(seq_len)[:, None]
    k_idx = jnp.arange(seq_len)[None, :]
    mask = q_idx >= k_idx
    if is_sliding and sliding_window is not None:
        mask = mask & ((q_idx - k_idx) < sliding_window)
    logits = jnp.where(mask[None, None, None, :, :], logits, -1e30)

    sink_logits = softmax_offset.astype(jnp.float32).reshape(
        1, num_kv_heads, num_queries_per_kv, 1, 1
    )
    max_logit = jnp.maximum(jnp.max(logits, axis=-1, keepdims=True), sink_logits)

    exp_logits = jnp.where(mask[None, None, None, :, :], jnp.exp(logits - max_logit), 0.0)
    exp_sink = jnp.exp(sink_logits - max_logit)
    denom = jnp.sum(exp_logits, axis=-1, keepdims=True) + exp_sink
    attn_weights = (exp_logits / (denom + 1e-20)).astype(dtype)

    out_grouped = jnp.einsum(
        "bhgqk,bhkd->bhgqd",
        attn_weights,
        v_t.astype(dtype),
        preferred_element_type=jnp.float32,
    ).astype(dtype)

    return jnp.transpose(out_grouped, (0, 3, 1, 2, 4)).reshape(
        bsz, seq_len, num_q_heads, head_dim
    )


def splash_attention_with_learned_sinks(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    softmax_offset: jnp.ndarray,
    *,
    mesh: Any,
    is_sliding: bool,
    sliding_window: int,
    block_size: int = 512,
    dtype: Any = jnp.bfloat16,
) -> jnp.ndarray:
    """TPU Pallas Block-Sparse SplashAttention with native learned `sinks` (`softmax_offset`)."""
    seq_len = q.shape[1]
    head_dim = q.shape[3]
    b_sz = min(block_size, seq_len)
    b_comp = min(256, b_sz)
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=b_sz,
        block_kv=b_sz,
        block_kv_compute=b_comp,
        block_q_dkv=b_sz,
        block_kv_dkv=b_sz,
        block_kv_dkv_compute=b_comp,
        block_q_dq=b_sz,
        block_kv_dq=b_sz,
    )
    is_cpu = jax.default_backend() == "cpu"

    def _shard_splash(
        q_local: jnp.ndarray,
        k_local: jnp.ndarray,
        v_local: jnp.ndarray,
        sinks_local: jnp.ndarray,
    ) -> jnp.ndarray:
        hq_local = q_local.shape[2]
        if is_sliding and sliding_window is not None:
            mask = splash_attention_mask.MultiHeadMask(
                [
                    splash_attention_mask.LocalMask(
                        shape=(seq_len, seq_len),
                        window_size=(sliding_window - 1, 0),
                        offset=0,
                    )
                    for _ in range(hq_local)
                ]
            )
        else:
            mask = splash_attention_mask.MultiHeadMask(
                [
                    splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
                    for _ in range(hq_local)
                ]
            )
        kernel = splash_attention_kernel.make_splash_mha(
            mask,
            block_sizes=block_sizes,
            head_shards=1,
            q_seq_shards=1,
            interpret=is_cpu,
        )
        scale = jnp.asarray(head_dim ** -0.5, dtype=dtype)
        qt = jnp.transpose(q_local.astype(dtype) * scale, (0, 2, 1, 3))
        kt = jnp.transpose(k_local.astype(dtype), (0, 2, 1, 3))
        vt = jnp.transpose(v_local.astype(dtype), (0, 2, 1, 3))
        sinks_f32 = sinks_local.astype(jnp.float32)
        out = jax.vmap(lambda qi, ki, vi: kernel(qi, ki, vi, sinks=sinks_f32))(qt, kt, vt)
        return jnp.transpose(out, (0, 2, 1, 3)).astype(dtype)

    return jax.shard_map(
        _shard_splash,
        mesh=mesh,
        in_specs=(
            P(("data", "fsdp", "expert"), None, "tensor", None),
            P(("data", "fsdp", "expert"), None, "tensor", None),
            P(("data", "fsdp", "expert"), None, "tensor", None),
            P("tensor"),
        ),
        out_specs=P(("data", "fsdp", "expert"), None, "tensor", None),
        check_vma=False,
    )(q, k, v, softmax_offset)


class FlintAttention(nn.Module):
    """Flint Grouped-Query Attention with QK RMSNorm, Learned Sinks, Sliding Window & Selective RoPE."""

    config: FlintJAXConfig
    layer_id: int
    mesh: Any = None

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

        if cfg.uses_rope(self.layer_id):
            q, k = apply_rope(
                q=q,
                k=k,
                positions=positions,
                head_dim=cfg.head_dim,
                rope_theta=cfg.rope_theta,
            )

        softmax_offset = self.param(
            "softmax_offset",
            nn.with_partitioning(nn.initializers.zeros, ("tensor",)),
            (cfg.num_attention_heads,),
            cfg.weight_dtype,
        )

        if cfg.use_splash_attention and self.mesh is not None and seq_len >= 128 and seq_len % 128 == 0:
            attn_out = splash_attention_with_learned_sinks(
                q,
                k,
                v,
                softmax_offset,
                mesh=self.mesh,
                is_sliding=cfg.is_sliding_layer(self.layer_id),
                sliding_window=cfg.sliding_window,
                block_size=cfg.splash_block_size,
                dtype=cfg.dtype,
            )
        else:
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


def _spmd_ragged_moe(
    hidden_states: jnp.ndarray,
    topk_ids: jnp.ndarray,
    topk_weights: jnp.ndarray,
    experts_gate: jnp.ndarray,
    experts_up: jnp.ndarray,
    experts_down: jnp.ndarray,
    *,
    mesh: Any,
) -> jnp.ndarray:
    """Unified FSDP / Compact-EP fused `W_gate_up` `jax.lax.ragged_dot` MoE kernel.

    - Outer GSPMD automatically pipelines async FSDP weight `all-gather` (fwd) and
      gradient `reduce-scatter` (bwd) across layers along the `'fsdp'` mesh axis.
    - When `ep == 1`: zero token ICI collectives, zero capacity padding, 100% valid rows.
    - When `ep > 1`: gathers/scatters tokens ONLY across the small `'expert'` axis (`2` or `4`
      chips) and compacts local assignments via `jnp.nonzero(..., size=cap)` with strict
      pre/post `jnp.where(valid, ..., 0.0)` masking so uninitialized `ragged_dot` tail rows
      can never introduce NaN/Inf in forward or backward passes.
    """
    ep_size = int(mesh.shape["expert"]) if mesh is not None else 1
    k_top = topk_ids.shape[-1]

    def _shard_moe(
        h_local: jnp.ndarray,
        ids_local: jnp.ndarray,
        w_local: jnp.ndarray,
        eg_local: jnp.ndarray,
        eu_local: jnp.ndarray,
        ed_local: jnp.ndarray,
    ) -> jnp.ndarray:
        compute_dtype = h_local.dtype
        egu_local = jnp.concatenate([eg_local, eu_local], axis=-1).astype(compute_dtype)
        ed_cast = ed_local.astype(compute_dtype)
        num_local_experts = eg_local.shape[0]

        if ep_size == 1:
            bsz_l, t_len, d_dim = h_local.shape
            n_toks = bsz_l * t_len
            flat_toks = h_local.reshape(n_toks, d_dim)
            flat_ids = ids_local.reshape(-1)
            flat_w = w_local.reshape(-1).astype(compute_dtype)

            tok_idx = jnp.repeat(jnp.arange(n_toks, dtype=jnp.int32), k_top)
            sort_order = jnp.argsort(flat_ids)
            sorted_exp = flat_ids[sort_order]
            sorted_tok = tok_idx[sort_order]
            sorted_in = flat_toks[sorted_tok]
            sorted_w = flat_w[sort_order][:, None]

            group_sizes = jnp.bincount(sorted_exp, length=num_local_experts).astype(jnp.int32)
            gu_out = jax.lax.ragged_dot(
                lhs=sorted_in,
                rhs=egu_local,
                group_sizes=group_sizes,
            )
            g_out, u_out = jnp.split(gu_out, 2, axis=-1)
            inter = (nn.silu(g_out) * u_out).astype(compute_dtype)
            d_out = jax.lax.ragged_dot(
                lhs=inter,
                rhs=ed_cast,
                group_sizes=group_sizes,
            )
            out_flat = jnp.zeros((n_toks, d_dim), dtype=compute_dtype).at[sorted_tok].add(
                d_out * sorted_w
            )
            return out_flat.reshape(bsz_l, t_len, d_dim)

        h_all = jax.lax.all_gather(h_local, "expert", axis=0, tiled=True)
        ids_all = jax.lax.all_gather(ids_local, "expert", axis=0, tiled=True)
        w_all = jax.lax.all_gather(w_local, "expert", axis=0, tiled=True)

        total_bsz, t_len, d_dim = h_all.shape
        n_toks = total_bsz * t_len
        ep_rank = jax.lax.axis_index("expert")
        exp_start = ep_rank * num_local_experts

        flat_toks = h_all.reshape(n_toks, d_dim)
        flat_ids = ids_all.reshape(-1)
        flat_w = w_all.reshape(-1).astype(compute_dtype)

        total_assigns = flat_ids.shape[0]
        cap = min(total_assigns, max(1024, ((2 * total_assigns) // ep_size + 255) // 256 * 256))
        is_mine = (flat_ids >= exp_start) & (flat_ids < exp_start + num_local_experts)
        mine_idx = jnp.nonzero(is_mine, size=cap, fill_value=total_assigns)[0]
        valid_slot = mine_idx < total_assigns
        safe_idx = jnp.minimum(mine_idx, total_assigns - 1)

        loc_exp = jnp.where(valid_slot, flat_ids[safe_idx] - exp_start, num_local_experts)
        loc_tok = jnp.where(valid_slot, safe_idx // k_top, 0)
        loc_w = jnp.where(valid_slot, flat_w[safe_idx], 0.0)

        sort_order = jnp.argsort(loc_exp)
        sorted_exp = loc_exp[sort_order]
        sorted_tok = loc_tok[sort_order]
        sorted_w = loc_w[sort_order]
        sorted_valid = (sorted_exp < num_local_experts)[:, None]
        sorted_in = jnp.where(sorted_valid, flat_toks[sorted_tok], 0.0).astype(compute_dtype)

        group_sizes = jnp.bincount(sorted_exp, length=num_local_experts).astype(jnp.int32)
        gu_out = jax.lax.ragged_dot(
            lhs=sorted_in,
            rhs=egu_local,
            group_sizes=group_sizes,
        )
        gu_out = jnp.where(sorted_valid, gu_out, 0.0)
        g_out, u_out = jnp.split(gu_out, 2, axis=-1)
        inter = (nn.silu(g_out) * u_out).astype(compute_dtype)
        d_out = jax.lax.ragged_dot(
            lhs=inter,
            rhs=ed_cast,
            group_sizes=group_sizes,
        )
        d_out = jnp.where(sorted_valid, d_out, 0.0)

        out_flat = jnp.zeros((n_toks, d_dim), dtype=compute_dtype).at[sorted_tok].add(
            d_out * sorted_w[:, None]
        )
        return jax.lax.psum_scatter(
            out_flat.reshape(total_bsz, t_len, d_dim),
            "expert",
            scatter_dimension=0,
            tiled=True,
        )

    if mesh is None:
        return _shard_moe(
            hidden_states, topk_ids, topk_weights, experts_gate, experts_up, experts_down
        )

    return jax.shard_map(
        _shard_moe,
        mesh=mesh,
        in_specs=(
            P(("data", "fsdp", "expert"), None, None),
            P(("data", "fsdp", "expert"), None, None),
            P(("data", "fsdp", "expert"), None, None),
            P("expert", None, "tensor"),
            P("expert", None, "tensor"),
            P("expert", "tensor", None),
        ),
        out_specs=P(("data", "fsdp", "expert"), None, None),
        check_vma=False,
    )(hidden_states, topk_ids, topk_weights, experts_gate, experts_up, experts_down)


class FlintSparseMoeBlock(nn.Module):
    """Flint Sparse MoE block (256 routed experts, sigmoid top-6 routing + 1 shared expert)."""

    config: FlintJAXConfig
    layer_id: int
    mesh: Any = None

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        bsz, seq_len, _ = hidden_states.shape

        shared_output = FlintMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.shared_expert_intermediate_size,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            initializer_range=cfg.initializer_range,
            name="shared_expert",
        )(hidden_states)

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
            hidden_states.astype(cfg.dtype),
            gate_kernel.astype(cfg.dtype),
            preferred_element_type=jnp.float32,
        )
        scores = jax.nn.sigmoid(router_logits)
        scores_for_choice = scores + expert_bias[None, None, :]
        _, topk_ids = jax.lax.top_k(scores_for_choice, k=cfg.num_experts_per_tok)
        topk_weights = jnp.take_along_axis(scores, topk_ids, axis=-1)
        if cfg.norm_topk_prob:
            topk_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-20)

        experts_gate = self.param(
            "experts_gate_proj",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("expert", "fsdp", "tensor"),
            ),
            (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
            cfg.weight_dtype,
        )
        experts_up = self.param(
            "experts_up_proj",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("expert", "fsdp", "tensor"),
            ),
            (cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size),
            cfg.weight_dtype,
        )
        experts_down = self.param(
            "experts_down_proj",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("expert", "tensor", "fsdp"),
            ),
            (cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size),
            cfg.weight_dtype,
        )

        routed_output = _spmd_ragged_moe(
            hidden_states.astype(cfg.dtype),
            topk_ids,
            topk_weights,
            experts_gate,
            experts_up,
            experts_down,
            mesh=self.mesh,
        )

        num_tokens = bsz * seq_len
        global_group_sizes = jnp.bincount(
            topk_ids.reshape(-1), length=cfg.num_experts
        ).astype(jnp.float32)
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
    mesh: Any = None

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
            mesh=self.mesh,
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
                mesh=self.mesh,
                name="mlp",
            )(normed_mlp_in)

        hidden_states = hidden_states + mlp_out
        return hidden_states, aux_loss


def _chunked_lm_cross_entropy(
    hidden_states: jnp.ndarray,
    lm_kernel: jnp.ndarray,
    labels: jnp.ndarray,
    loss_mask: jnp.ndarray,
    chunk_tokens: int = 256,
) -> jnp.ndarray:
    """Computes shifted FP32 Cross-Entropy in sequence chunks via `jax.lax.scan` to bound HBM."""
    bsz, seq_len, hidden_dim = hidden_states.shape
    shift_labels = jnp.pad(labels[:, 1:], ((0, 0), (0, 1)), constant_values=0)
    shift_mask = jnp.pad(loss_mask[:, 1:].astype(jnp.float32), ((0, 0), (0, 1)), constant_values=0.0)

    chunk_t = min(seq_len, chunk_tokens)
    while seq_len % chunk_t != 0 and chunk_t > 1:
        chunk_t //= 2
    num_chunks = seq_len // chunk_t

    h_chunks = jnp.swapaxes(hidden_states.reshape(bsz, num_chunks, chunk_t, hidden_dim), 0, 1)
    lbl_chunks = jnp.swapaxes(shift_labels.reshape(bsz, num_chunks, chunk_t), 0, 1)
    msk_chunks = jnp.swapaxes(shift_mask.reshape(bsz, num_chunks, chunk_t), 0, 1)

    @jax.checkpoint
    def _ce_chunk_step(carry: jnp.ndarray, xs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        h_c, l_c, m_c = xs
        logits_c = jnp.einsum(
            "bcd,dv->bcv",
            h_c,
            lm_kernel,
            preferred_element_type=jnp.float32,
        )
        tok_loss = optax.softmax_cross_entropy_with_integer_labels(logits_c, l_c)
        return carry + jnp.sum(tok_loss * m_c), None

    total_nll, _ = jax.lax.scan(
        _ce_chunk_step,
        jnp.array(0.0, dtype=jnp.float32),
        (h_chunks, lbl_chunks, msk_chunks),
    )
    denom = jnp.maximum(jnp.sum(shift_mask), 1.0)
    return total_nll / denom


class FlintForCausalLM(nn.Module):
    """Full 49-layer Recursive Flint (39.8B-A2.5B) Causal Language Model in JAX / Flax."""

    config: FlintJAXConfig
    mesh: Any = None

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray | None = None,
        labels: jnp.ndarray | None = None,
        loss_mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        bsz, seq_len = input_ids.shape
        if positions is None:
            positions = jnp.broadcast_to(
                jnp.arange(seq_len, dtype=jnp.int32)[None, :], (bsz, seq_len)
            )

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
        if self.mesh is not None:
            hidden_states = jax.lax.with_sharding_constraint(
                hidden_states, P(("data", "fsdp", "expert"), None, None)
            )

        total_aux_loss = jnp.array(0.0, dtype=jnp.float32)
        RematDecoderLayer = nn.remat(FlintDecoderLayer)

        for layer_id in range(cfg.num_hidden_layers):
            use_remat = (
                not self.is_initializing()
                and cfg.remat_layers
                and layer_id < cfg.remat_first_n_layers
            )
            LayerCls = RematDecoderLayer if use_remat else FlintDecoderLayer
            hidden_states, layer_aux_loss = LayerCls(
                config=cfg,
                layer_id=layer_id,
                mesh=self.mesh,
                name=f"layers_{layer_id}",
            )(hidden_states, positions)
            if self.mesh is not None:
                hidden_states = jax.lax.with_sharding_constraint(
                    hidden_states, P(("data", "fsdp", "expert"), None, None)
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

        scaled_aux = total_aux_loss * cfg.router_aux_loss_coef
        out = FlintLMHead(config=cfg, mesh=self.mesh, name="lm_head")(
            hidden_states, labels=labels, loss_mask=loss_mask
        )
        return out, scaled_aux


class FlintLMHead(nn.Module):
    """Vocabulary projection head supporting both full logits and chunked Cross-Entropy."""

    config: FlintJAXConfig
    mesh: Any = None

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        labels: jnp.ndarray | None = None,
        loss_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        cfg = self.config
        kernel = self.param(
            "kernel",
            nn.with_partitioning(
                nn.initializers.normal(cfg.initializer_range),
                ("fsdp", "tensor"),
            ),
            (cfg.hidden_size, cfg.vocab_size),
            cfg.weight_dtype,
        )
        if labels is not None and loss_mask is not None and not self.is_initializing():
            return _chunked_lm_cross_entropy(
                hidden_states.astype(cfg.dtype),
                kernel.astype(cfg.dtype),
                labels,
                loss_mask,
                chunk_tokens=cfg.ce_chunk_tokens,
            )
        logits = jnp.einsum(
            "btd,dv->btv",
            hidden_states,
            kernel,
            preferred_element_type=jnp.float32,
        )
        if self.mesh is not None:
            logits = jax.lax.with_sharding_constraint(
                logits, P(("data", "fsdp", "expert"), None, "tensor")
            )
        return logits
