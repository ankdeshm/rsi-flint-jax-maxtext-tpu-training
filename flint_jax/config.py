"""Flint model configuration for JAX / Flax on TPU v5p.

Matches RecursiveSI/flint/config.json and flint_sglang.configuration_flint.FlintConfig:
  - 39.8B total parameters, ~2.5B active parameters per token
  - 49 layers: 2 dense SwiGLU layers ([0, 1]) + 47 sparse MoE layers ([2..48])
  - 256 routed experts (top-6 sigmoid routing, renormalized) + 1 ungated shared expert
  - Hybrid attention: 35 sliding-window layers (window=513, with RoPE) +
    14 full-attention layers (no_rope_layer_indices, no positional encoding)
  - Per-head QK RMSNorm (head_dim=128) and per-head learned attention sinks (softmax_offset)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp

DEFAULT_LAYER_TYPES: tuple[str, ...] = (
    "full_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
)

DEFAULT_NO_ROPE_LAYER_INDICES: tuple[int, ...] = (
    0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48
)


@dataclass(frozen=True)
class FlintJAXConfig:
    vocab_size: int = 150144
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 49
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_sliding_window: bool = True
    sliding_window: int = 513
    layer_types: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LAYER_TYPES)
    no_rope_layer_indices: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_NO_ROPE_LAYER_INDICES
    )
    softmax_type: str = "learnable"
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 1024
    num_experts_per_tok: int = 6
    num_experts: int = 256
    norm_topk_prob: bool = True
    router_score_function: str = "sigmoid"
    output_router_logits: bool = True
    router_aux_loss_coef: float = 0.0005
    mlp_only_layers: tuple[int, ...] = (0, 1)
    pad_token_id: int = 150001
    bos_token_id: int = 150000
    eos_token_id: tuple[int, ...] = (150001, 150003, 150005, 150018)

    # JAX / TPU v5p training & kernel options
    dtype: Any = jnp.bfloat16
    weight_dtype: Any = jnp.bfloat16
    use_ragged_dot_moe: bool = True
    use_splash_attention: bool = True
    splash_block_size: int = 512
    remat_layers: bool = True
    remat_first_n_layers: int = 49
    ce_chunk_tokens: int = 256

    def is_dense_layer(self, layer_id: int) -> bool:
        return layer_id in self.mlp_only_layers

    def is_sliding_layer(self, layer_id: int) -> bool:
        return self.use_sliding_window and self.layer_types[layer_id] == "sliding_attention"

    def uses_rope(self, layer_id: int) -> bool:
        return layer_id not in self.no_rope_layer_indices

    @classmethod
    def from_json_file(cls, path: str | Path, **overrides: Any) -> "FlintJAXConfig":
        raw = json.loads(Path(path).read_text())
        known_fields = {f for f in cls.__dataclass_fields__}
        filtered: dict[str, Any] = {}
        for k, v in raw.items():
            if k in known_fields:
                if isinstance(v, list):
                    filtered[k] = tuple(v)
                else:
                    filtered[k] = v
        filtered.update(overrides)
        return cls(**filtered)
