"""JAX / Flax TPU v5p training package for Recursive Flint (39.8B-A2.5B MoE)."""

from flint_jax.config import FlintJAXConfig
from flint_jax.model import FlintForCausalLM

__all__ = ["FlintJAXConfig", "FlintForCausalLM"]
