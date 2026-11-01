"""Hugging Face Safetensors -> Sharded JAX/Flax PyTree loader for RecursiveSI/flint.

Maps all 36,738 checkpoint tensors across `model-00001-of-00015.safetensors` ..
`model-00015-of-00015.safetensors` from `RecursiveSI/flint` into the sharded
Flax parameter tree for TPU v5p training:
  - `model.embed_tokens.weight`              -> `params['embed_tokens']['embedding']`
  - `model.norm.weight`                      -> `params['norm']['weight']`
  - `lm_head.weight`                         -> `params['lm_head']['kernel']` (transposed)
  - `model.layers.{i}.input_layernorm.weight` -> `params['layers_{i}']['input_layernorm']['weight']`
  - `model.layers.{i}.post_attention_layernorm.weight` -> `params['layers_{i}']['post_attention_layernorm']['weight']`
  - `model.layers.{i}.self_attn.{q,k,v}_proj.weight`   -> reshaped to `[hidden, num_heads, head_dim]`
  - `model.layers.{i}.self_attn.o_proj.weight`         -> reshaped to `[num_heads, head_dim, hidden]`
  - `model.layers.{i}.self_attn.{q,k}_norm.weight`     -> `params['layers_{i}']['self_attn']['{q,k}_norm']['weight']`
  - `model.layers.{i}.self_attn.softmax_offset`        -> `params['layers_{i}']['self_attn']['softmax_offset']`
  - Dense layers (0, 1):
    `model.layers.{i}.mlp.{gate,up,down}_proj.weight`  -> transposed to `[in, out]`
  - MoE layers (2..48):
    `model.layers.{i}.mlp.gate.weight`                 -> `params['layers_{i}']['mlp']['gate_weight']`
    `model.layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight` -> transposed to `[in, out]`
    `model.layers.{i}.mlp.experts.{0..255}.{gate,up,down}_proj.weight` -> stacked 3D `[256, in, out]`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from flint_jax.config import FlintJAXConfig

logger = logging.getLogger(__name__)


def load_hf_flint_checkpoint(
    checkpoint_dir: str | Path,
    target_params: dict[str, Any],
    config: FlintJAXConfig,
) -> dict[str, Any]:
    """Loads a local Hugging Face `RecursiveSI/flint` safetensors directory into `target_params`."""
    from safetensors import safe_open

    ckpt_path = Path(checkpoint_dir)
    index_path = ckpt_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}. Download RecursiveSI/flint first.")

    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    handles = {
        shard: safe_open(str(ckpt_path / shard), framework="np", device="cpu")
        for shard in shard_files
    }

    def get_np_tensor(name: str) -> np.ndarray:
        shard = weight_map[name]
        return handles[shard].get_tensor(name)

    def to_sharded_jax(arr: np.ndarray, ref_leaf: jax.Array) -> jax.Array:
        return jax.device_put(jnp.asarray(arr, dtype=ref_leaf.dtype), ref_leaf.sharding)

    params = jax.tree_util.tree_map(lambda x: x, target_params)

    # Top-level embeddings, final RMSNorm, and LM head
    params["embed_tokens"]["embedding"] = to_sharded_jax(
        get_np_tensor("model.embed_tokens.weight"),
        params["embed_tokens"]["embedding"],
    )
    params["norm"]["weight"] = to_sharded_jax(
        get_np_tensor("model.norm.weight"),
        params["norm"]["weight"],
    )
    params["lm_head"]["kernel"] = to_sharded_jax(
        get_np_tensor("lm_head.weight").T,
        params["lm_head"]["kernel"],
    )

    for layer_id in range(config.num_hidden_layers):
        lkey = f"layers_{layer_id}"
        prefix = f"model.layers.{layer_id}"
        layer_params = params[lkey]

        layer_params["input_layernorm"]["weight"] = to_sharded_jax(
            get_np_tensor(f"{prefix}.input_layernorm.weight"),
            layer_params["input_layernorm"]["weight"],
        )
        layer_params["post_attention_layernorm"]["weight"] = to_sharded_jax(
            get_np_tensor(f"{prefix}.post_attention_layernorm.weight"),
            layer_params["post_attention_layernorm"]["weight"],
        )

        # Self-attention projections & QK norms + learned attention sinks
        attn_p = layer_params["self_attn"]
        q_w = get_np_tensor(f"{prefix}.self_attn.q_proj.weight").T.reshape(
            config.hidden_size, config.num_attention_heads, config.head_dim
        )
        k_w = get_np_tensor(f"{prefix}.self_attn.k_proj.weight").T.reshape(
            config.hidden_size, config.num_key_value_heads, config.head_dim
        )
        v_w = get_np_tensor(f"{prefix}.self_attn.v_proj.weight").T.reshape(
            config.hidden_size, config.num_key_value_heads, config.head_dim
        )
        o_w = get_np_tensor(f"{prefix}.self_attn.o_proj.weight").T.reshape(
            config.num_attention_heads, config.head_dim, config.hidden_size
        )

        attn_p["q_proj"]["kernel"] = to_sharded_jax(q_w, attn_p["q_proj"]["kernel"])
        attn_p["k_proj"]["kernel"] = to_sharded_jax(k_w, attn_p["k_proj"]["kernel"])
        attn_p["v_proj"]["kernel"] = to_sharded_jax(v_w, attn_p["v_proj"]["kernel"])
        attn_p["o_proj"]["kernel"] = to_sharded_jax(o_w, attn_p["o_proj"]["kernel"])
        attn_p["q_norm"]["weight"] = to_sharded_jax(
            get_np_tensor(f"{prefix}.self_attn.q_norm.weight"),
            attn_p["q_norm"]["weight"],
        )
        attn_p["k_norm"]["weight"] = to_sharded_jax(
            get_np_tensor(f"{prefix}.self_attn.k_norm.weight"),
            attn_p["k_norm"]["weight"],
        )
        attn_p["softmax_offset"] = to_sharded_jax(
            get_np_tensor(f"{prefix}.self_attn.softmax_offset"),
            attn_p["softmax_offset"],
        )

        mlp_p = layer_params["mlp"]
        if config.is_dense_layer(layer_id):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mlp_p[proj]["kernel"] = to_sharded_jax(
                    get_np_tensor(f"{prefix}.mlp.{proj}.weight").T,
                    mlp_p[proj]["kernel"],
                )
        else:
            mlp_p["gate_weight"] = to_sharded_jax(
                get_np_tensor(f"{prefix}.mlp.gate.weight"),
                mlp_p["gate_weight"],
            )
            bias_key = f"{prefix}.mlp.gate.expert_bias"
            if bias_key in weight_map:
                mlp_p["expert_bias"] = to_sharded_jax(
                    get_np_tensor(bias_key),
                    mlp_p["expert_bias"],
                )

            for proj in ("gate_proj", "up_proj", "down_proj"):
                mlp_p["shared_expert"][proj]["kernel"] = to_sharded_jax(
                    get_np_tensor(f"{prefix}.mlp.shared_expert.{proj}.weight").T,
                    mlp_p["shared_expert"][proj]["kernel"],
                )

            # Stack 256 routed experts into [256, in_dim, out_dim]
            gate_stack = np.stack(
                [get_np_tensor(f"{prefix}.mlp.experts.{e}.gate_proj.weight").T for e in range(config.num_experts)],
                axis=0,
            )
            up_stack = np.stack(
                [get_np_tensor(f"{prefix}.mlp.experts.{e}.up_proj.weight").T for e in range(config.num_experts)],
                axis=0,
            )
            down_stack = np.stack(
                [get_np_tensor(f"{prefix}.mlp.experts.{e}.down_proj.weight").T for e in range(config.num_experts)],
                axis=0,
            )
            mlp_p["experts_gate_proj"] = to_sharded_jax(gate_stack, mlp_p["experts_gate_proj"])
            mlp_p["experts_up_proj"] = to_sharded_jax(up_stack, mlp_p["experts_up_proj"])
            mlp_p["experts_down_proj"] = to_sharded_jax(down_stack, mlp_p["experts_down_proj"])

        logger.info("Loaded Flint layer %d/%d into sharded TPU HBM", layer_id + 1, config.num_hidden_layers)

    return params
