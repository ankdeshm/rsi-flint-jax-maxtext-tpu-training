"""Distributed JAX BF16 Pre-Training / Continual Pre-Training Script for Recursive Flint on TPU v5p.

Supports both baseline (`--sweep-v5p16`) and high-MFU optimized (`--sweep-v5p16-opt`) sweeps across
4-host TPU v5p slices (`2x2x4`, 16 chips) with:
  - Native TPU Pallas `SplashAttention` (`sinks=softmax_offset`, block-sparse `LocalMask(512, 0)`)
  - Fused `W_gate_up` `jax.lax.ragged_dot` (zero token ICI under `FSDP=16, EP=1` or compacted `EP>1`)
  - Overlapped async FSDP weight `all-gather` / gradient `reduce-scatter` across all 49 layers
  - Chunked `@jax.checkpoint` `jax.lax.scan` Cross-Entropy & `jax.lax.scan` Gradient Accumulation
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import socket
import time
from typing import Any


def _wait_for_gke_tpu_worker_dns() -> None:
    """Ensures insecure SliceBuilder gRPC on GKE and waits for headless Service DNS records before libtpu loads."""
    os.environ.setdefault("GRPC_DNS_RESOLVER", "native")
    init_args = os.environ.get("LIBTPU_INIT_ARGS", "")
    required_flags = [
        "--slicebuilder_use_insecure_grpc=true",
        "--undefok=slicebuilder_use_insecure_grpc",
        "--xla_tpu_use_enhanced_launch_barrier=false",
    ]
    for flag in required_flags:
        flag_key = flag.split("=")[0]
        if flag_key not in init_args:
            init_args = f"{init_args} {flag}".strip()
    os.environ["LIBTPU_INIT_ARGS"] = init_args

    tpu_hostnames_str = os.environ.get("TPU_WORKER_HOSTNAMES", "")
    if not tpu_hostnames_str or "," not in tpu_hostnames_str:
        return
    for h in [x.strip() for x in tpu_hostnames_str.split(",") if x.strip()]:
        for _ in range(90):
            try:
                socket.gethostbyname(h)
                break
            except Exception:
                time.sleep(1.0)


_wait_for_gke_tpu_worker_dns()

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

from flint_jax.checkpoint_loader import load_hf_flint_checkpoint
from flint_jax.config import FlintJAXConfig
from flint_jax.model import FlintForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [process=%(process)d] %(message)s",
)
logger = logging.getLogger("flint_jax.train")


def create_tpu_mesh(
    data_parallel: int,
    fsdp_parallel: int,
    expert_parallel: int,
    tensor_parallel: int,
) -> Mesh:
    """Creates a topology-aware 4D (data, fsdp, expert, tensor) JAX Mesh over all visible TPU devices."""
    from jax.experimental import mesh_utils

    devices = jax.devices()
    num_devices = len(devices)
    expected = data_parallel * fsdp_parallel * expert_parallel * tensor_parallel
    if expected != num_devices:
        raise ValueError(
            f"Mesh product ({data_parallel} * {fsdp_parallel} * {expert_parallel} * {tensor_parallel} = {expected}) "
            f"must equal total TPU devices ({num_devices})."
        )
    try:
        device_mesh = mesh_utils.create_device_mesh(
            (data_parallel, fsdp_parallel, expert_parallel, tensor_parallel),
            devices=devices,
            allow_split_physical_axes=True,
        )
    except Exception as exc:
        logger.warning("mesh_utils.create_device_mesh fallback to flat reshape (%s)", exc)
        device_mesh = np.array(devices).reshape(
            data_parallel, fsdp_parallel, expert_parallel, tensor_parallel
        )
    return Mesh(device_mesh, axis_names=("data", "fsdp", "expert", "tensor"))


def build_optimizer(
    learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    weight_decay: float,
    grad_clip_norm: float,
) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(max_steps, warmup_steps + 1),
        end_value=learning_rate * 0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay,
            mu_dtype=jnp.bfloat16,
        ),
    )


def create_sharded_train_state(
    rng: jax.Array,
    mesh: Mesh,
    model: FlintForCausalLM,
    tx: optax.GradientTransformation,
    seq_len: int,
) -> tuple[train_state.TrainState, Any]:
    """Initializes and shards Flint's parameters and AdamW optimizer state directly across TPU HBM."""
    num_devs = jax.device_count()
    token_sharding = NamedSharding(mesh, P(("data", "fsdp", "expert"), None))

    def init_fn(key: jax.Array, tokens: jnp.ndarray) -> train_state.TrainState:
        variables = model.init(key, tokens)
        return train_state.TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=tx,
        )

    with jax.set_mesh(mesh):
        dummy_tokens = jax.jit(
            lambda: jnp.ones((num_devs, seq_len), dtype=jnp.int32),
            out_shardings=token_sharding,
        )()
        abstract_vars = jax.eval_shape(init_fn, rng, dummy_tokens)
        state_sharding = nn.get_sharding(abstract_vars, mesh)
        sharded_init = jax.jit(
            init_fn,
            in_shardings=(NamedSharding(mesh, P()), token_sharding),
            out_shardings=state_sharding,
        )
        state = sharded_init(rng, dummy_tokens)
    return state, state_sharding


def train_step(
    state: train_state.TrainState,
    batch: dict[str, jnp.ndarray],
    grad_accum_steps: int = 1,
) -> tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
    """Compiled SPMD training step in BF16 with chunked LM cross-entropy + scan gradient accumulation."""

    def micro_loss_fn(
        params: Any,
        input_ids: jnp.ndarray,
        labels: jnp.ndarray,
        loss_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        lm_loss, aux_loss = state.apply_fn(
            {"params": params},
            input_ids,
            labels=labels,
            loss_mask=loss_mask,
        )
        total_loss = lm_loss + aux_loss
        return total_loss, {
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "moe_aux_loss": aux_loss,
        }

    grad_fn = jax.value_and_grad(micro_loss_fn, has_aux=True)

    if grad_accum_steps <= 1:
        (_, metrics), grads = grad_fn(
            state.params,
            batch["input_ids"],
            batch["labels"],
            batch["loss_mask"],
        )
    else:
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        zero_metrics = {
            "total_loss": jnp.array(0.0, dtype=jnp.float32),
            "lm_loss": jnp.array(0.0, dtype=jnp.float32),
            "moe_aux_loss": jnp.array(0.0, dtype=jnp.float32),
        }

        def _accum_body(carry, micro_batch):
            acc_g, acc_m = carry
            in_ids, lbls, msk = micro_batch
            (_, m_step), g_step = grad_fn(state.params, in_ids, lbls, msk)
            new_g = jax.tree_util.tree_map(lambda a, b: a + b, acc_g, g_step)
            new_m = {k: acc_m[k] + m_step[k] for k in acc_m}
            return (new_g, new_m), None

        (sum_grads, sum_metrics), _ = jax.lax.scan(
            _accum_body,
            (zero_grads, zero_metrics),
            (batch["input_ids"], batch["labels"], batch["loss_mask"]),
        )
        inv_ga = 1.0 / float(grad_accum_steps)
        grads = jax.tree_util.tree_map(lambda g: (g * inv_ga).astype(g.dtype), sum_grads)
        metrics = {k: v * inv_ga for k, v in sum_metrics.items()}

    grad_norm = optax.global_norm(grads)
    metrics["grad_norm"] = grad_norm
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


def compute_active_flops_per_token(config: FlintJAXConfig, seq_len: int) -> float:
    """Computes exact active training FLOPs/token (6*N_active + causal/SWA attention FLOPs)."""
    h = config.hidden_size
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim
    n_dense = len(config.mlp_only_layers)
    n_moe = config.num_hidden_layers - n_dense

    n_lm_head = h * config.vocab_size
    n_attn = config.num_hidden_layers * h * (q_dim + 2 * kv_dim + q_dim)
    n_dense_mlp = n_dense * 3 * h * config.intermediate_size
    n_moe_mlp = n_moe * (
        h * config.num_experts
        + 3 * h * config.shared_expert_intermediate_size
        + config.num_experts_per_tok * 3 * h * config.moe_intermediate_size
    )
    n_active = n_lm_head + n_attn + n_dense_mlp + n_moe_mlp
    attn_flops = 0.0
    for layer_id in range(config.num_hidden_layers):
        eff_ctx = min(seq_len, config.sliding_window) if config.is_sliding_layer(layer_id) else seq_len
        attn_flops += 12.0 * q_dim * eff_ctx
    return 6.0 * n_active + attn_flops


def _upload_file_to_gcs(
    local_path: str,
    gcs_object_name: str,
    bucket: str = "YOUR_GCS_BUCKET",
) -> None:
    """Uploads a local file to GCS via GCE metadata token without external CLI dependencies."""
    import json as _json
    import pathlib as _pathlib
    import urllib.parse as _up
    import urllib.request as _ur

    try:
        req = _ur.Request(
            "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        tok = _json.loads(_ur.urlopen(req, timeout=10).read().decode())["access_token"]
        obj = _up.quote(gcs_object_name, safe="")
        data = _pathlib.Path(local_path).read_bytes()
        up_req = _ur.Request(
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={obj}",
            data=data,
            method="POST",
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
        )
        _ur.urlopen(up_req, timeout=30).read()
        logger.info("Uploaded %s -> gs://%s/%s", local_path, bucket, gcs_object_name)
    except Exception as exc:
        logger.warning("GCS upload skipped/failed for %s (%s)", local_path, exc)


def run_single_experiment(
    *,
    exp_name: str,
    config: FlintJAXConfig,
    seq_len: int,
    per_device_batch_size: int,
    grad_accum_steps: int = 1,
    steps: int,
    learning_rate: float,
    warmup_steps: int,
    weight_decay: float,
    grad_clip_norm: float,
    data_parallel: int,
    fsdp_parallel: int,
    expert_parallel: int,
    tensor_parallel: int,
    hf_checkpoint_dir: str | None = None,
    summary_json: str | None = None,
    gcs_results_prefix: str | None = None,
) -> dict[str, Any]:
    """Runs one full SPMD compile + training benchmark inside an active JAX session."""
    import gc
    import json as _json
    import math as _math
    import statistics as _stats
    from jax.experimental import multihost_utils

    num_devs = jax.device_count()
    micro_global_bs = per_device_batch_size * num_devs
    tokens_per_step = micro_global_bs * seq_len * grad_accum_steps
    flops_per_token = compute_active_flops_per_token(config, seq_len)
    dev_kind = jax.devices()[0].device_kind
    peak_tflops_per_chip = (
        459.0 if "v5p" in dev_kind.lower() or "v5" in dev_kind.lower() else 1153.5
    )

    logger.info(
        "=== [%s] START: seq=%d, mb=%d, ga=%d (tok/step=%d), remat_n=%d, mesh=(dp=%d, fsdp=%d, ep=%d, tp=%d), flops/tok=%.3fG ===",
        exp_name,
        seq_len,
        per_device_batch_size,
        grad_accum_steps,
        tokens_per_step,
        config.remat_first_n_layers if config.remat_layers else 0,
        data_parallel,
        fsdp_parallel,
        expert_parallel,
        tensor_parallel,
        flops_per_token / 1e9,
    )

    mesh = create_tpu_mesh(
        data_parallel=data_parallel,
        fsdp_parallel=fsdp_parallel,
        expert_parallel=expert_parallel,
        tensor_parallel=tensor_parallel,
    )
    model = FlintForCausalLM(config=config, mesh=mesh)
    tx = build_optimizer(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_steps=steps,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
    )

    rng = jax.random.PRNGKey(42)
    t_init0 = time.perf_counter()
    state = None
    state_sharding = None
    compiled_make_batch = None
    compiled_train_step = None
    step_records: list[dict[str, Any]] = []

    try:
        state, state_sharding = create_sharded_train_state(
            rng=rng,
            mesh=mesh,
            model=model,
            tx=tx,
            seq_len=seq_len,
        )
        init_time_s = time.perf_counter() - t_init0
        logger.info("[%s] Initialized sharded TrainState in %.2fs", exp_name, init_time_s)

        if hf_checkpoint_dir:
            logger.info(
                "[%s] Loading pretrained RecursiveSI/flint BF16 weights from %s ...",
                exp_name,
                hf_checkpoint_dir,
            )
            loaded_params = load_hf_flint_checkpoint(hf_checkpoint_dir, state.params, config)
            state = state.replace(params=loaded_params)

        if grad_accum_steps <= 1:
            batch_sharding = NamedSharding(mesh, P(("data", "fsdp", "expert"), None))
            batch_shape = (micro_global_bs, seq_len)
        else:
            batch_sharding = NamedSharding(mesh, P(None, ("data", "fsdp", "expert"), None))
            batch_shape = (grad_accum_steps, micro_global_bs, seq_len)

        def _make_batch(step_idx: jnp.ndarray) -> dict[str, jnp.ndarray]:
            # Cycle over 2 synthetic mini-corpus batches so loss convergence is directly observable
            step_rng = jax.random.fold_in(rng, step_idx % 2)
            tokens = jax.random.randint(
                step_rng,
                shape=batch_shape,
                minval=0,
                maxval=config.vocab_size,
                dtype=jnp.int32,
            )
            return {
                "input_ids": tokens,
                "labels": tokens,
                "loss_mask": jnp.ones_like(tokens, dtype=jnp.int32),
            }

        compiled_make_batch = jax.jit(
            _make_batch,
            in_shardings=(NamedSharding(mesh, P()),),
            out_shardings=batch_sharding,
        )

        compiled_train_step = jax.jit(
            functools.partial(train_step, grad_accum_steps=grad_accum_steps),
            in_shardings=(state_sharding, batch_sharding),
            out_shardings=(state_sharding, NamedSharding(mesh, P())),
            donate_argnums=(0,),
        )

        with jax.set_mesh(mesh):
            for step in range(1, steps + 1):
                batch = compiled_make_batch(jnp.asarray(step, dtype=jnp.int32))
                jax.block_until_ready(batch["input_ids"])
                step_t0 = time.perf_counter()

                state, metrics = compiled_train_step(state, batch)
                jax.block_until_ready(metrics["total_loss"])
                dt = time.perf_counter() - step_t0
                tokens_per_sec = tokens_per_step / dt
                tps_per_chip = tokens_per_sec / num_devs
                mfu_pct = (tps_per_chip * flops_per_token) / (peak_tflops_per_chip * 1e12) * 100.0
                mem_stats = jax.local_devices()[0].memory_stats() or {}
                cur_hbm_gib = mem_stats.get("bytes_in_use", 0) / (1024 ** 3)
                peak_hbm_gib = mem_stats.get("peak_bytes_in_use", 0) / (1024 ** 3)

                rec = {
                    "step": step,
                    "total_loss": round(float(metrics["total_loss"]), 5),
                    "lm_loss": round(float(metrics["lm_loss"]), 5),
                    "moe_aux_loss": round(float(metrics["moe_aux_loss"]), 6),
                    "grad_norm": round(float(metrics["grad_norm"]), 4),
                    "step_time_s": round(dt, 4),
                    "cluster_tok_s": round(tokens_per_sec, 1),
                    "chip_tok_s": round(tps_per_chip, 1),
                    "mfu_pct": round(mfu_pct, 2),
                    "cur_hbm_gib": round(cur_hbm_gib, 2),
                    "peak_hbm_gib": round(peak_hbm_gib, 2),
                }
                step_records.append(rec)

                if jax.process_index() == 0:
                    logger.info(
                        "[%s] Step %04d/%04d | loss=%.4f (lm=%.4f aux=%.5f) | gnorm=%.4f | dt=%.3fs | chip_tps=%.1f | slice_tps=%.1f | MFU=%.2f%% | HBM=%.2f/%.2fGiB",
                        exp_name,
                        step,
                        steps,
                        rec["total_loss"],
                        rec["lm_loss"],
                        rec["moe_aux_loss"],
                        rec["grad_norm"],
                        dt,
                        tps_per_chip,
                        tokens_per_sec,
                        mfu_pct,
                        cur_hbm_gib,
                        peak_hbm_gib,
                    )

        steady = step_records[min(3, len(step_records) - 1) :]
        has_nan = any(
            _math.isnan(r["lm_loss"]) or _math.isnan(r["grad_norm"]) for r in step_records
        )
        summary = {
            "exp_name": exp_name,
            "status": "PASS_WITH_NAN" if has_nan else "PASS",
            "device_kind": dev_kind,
            "num_devices": num_devs,
            "seq_len": seq_len,
            "per_device_batch_size": per_device_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "remat_first_n_layers": config.remat_first_n_layers if config.remat_layers else 0,
            "save_splash_attn_layers": config.save_splash_attn_layers,
            "splash_block_size": config.splash_block_size,
            "use_splash_attention": config.use_splash_attention,
            "tokens_per_step": tokens_per_step,
            "flops_per_token_g": round(flops_per_token / 1e9, 4),
            "peak_tflops_per_chip": peak_tflops_per_chip,
            "mesh": {
                "data": data_parallel,
                "fsdp": fsdp_parallel,
                "expert": expert_parallel,
                "tensor": tensor_parallel,
            },
            "compile_step1_s": step_records[0]["step_time_s"],
            "steady_step_time_s": round(_stats.median([r["step_time_s"] for r in steady]), 4),
            "steady_chip_tok_s": round(_stats.median([r["chip_tok_s"] for r in steady]), 1),
            "steady_cluster_tok_s": round(_stats.median([r["cluster_tok_s"] for r in steady]), 1),
            "steady_mfu_pct": round(_stats.median([r["mfu_pct"] for r in steady]), 2),
            "peak_mfu_pct": round(max(r["mfu_pct"] for r in steady), 2),
            "cur_hbm_gib": round(max(r["cur_hbm_gib"] for r in steady), 2),
            "peak_hbm_gib": round(max(r["peak_hbm_gib"] for r in step_records), 2),
            "loss_start": step_records[0]["lm_loss"],
            "loss_end": step_records[-1]["lm_loss"],
            "steps": step_records,
        }
    except Exception as exc:
        logger.error("[%s] FAILED with %s: %s", exp_name, type(exc).__name__, exc)
        summary = {
            "exp_name": exp_name,
            "status": (
                "OOM"
                if "RESOURCE_EXHAUSTED" in str(exc) or "Out of memory" in str(exc)
                else "FAIL"
            ),
            "error": f"{type(exc).__name__}: {str(exc)[:600]}",
            "device_kind": dev_kind,
            "num_devices": num_devs,
            "seq_len": seq_len,
            "per_device_batch_size": per_device_batch_size,
            "grad_accum_steps": grad_accum_steps,
            "save_splash_attn_layers": config.save_splash_attn_layers,
            "splash_block_size": config.splash_block_size,
            "tokens_per_step": tokens_per_step,
            "mesh": {
                "data": data_parallel,
                "fsdp": fsdp_parallel,
                "expert": expert_parallel,
                "tensor": tensor_parallel,
            },
            "steps": step_records,
        }
    finally:
        del state, state_sharding, compiled_make_batch, compiled_train_step
        gc.collect()
        jax.clear_caches()
        if jax.process_count() > 1:
            multihost_utils.sync_global_devices(f"cleanup_{exp_name}")

    if jax.process_index() == 0 and summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
        with open(summary_json, "w") as f:
            _json.dump(summary, f, indent=2)
        logger.info("[%s] Wrote summary to %s (status=%s)", exp_name, summary_json, summary["status"])
        if gcs_results_prefix:
            _upload_file_to_gcs(
                summary_json, f"{gcs_results_prefix.rstrip('/')}/{os.path.basename(summary_json)}"
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Recursive Flint (39.8B MoE) in BF16 with JAX on TPU v5p"
    )
    parser.add_argument("--config-json", type=str, default=None, help="Path to RecursiveSI/flint config.json")
    parser.add_argument("--hf-checkpoint-dir", type=str, default=None, help="Optional local path to RecursiveSI/flint safetensors")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("BASE_OUTPUT_DIR", "gs://YOUR_GCS_BUCKET/flint-jax-training/checkpoints"))
    parser.add_argument("--seq-len", type=int, default=2048, help="Training sequence length (Flint supports up to 262144)")
    parser.add_argument("--per-device-batch-size", type=int, default=16, help="Micro-batch size per TPU chip")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation microbatches per optimizer step")
    parser.add_argument("--remat-first-n-layers", type=int, default=49, help="Number of leading layers to activation-checkpoint (49 = full remat)")
    parser.add_argument("--save-splash-attn-layers", type=int, default=0, help="Number of layers (full-attn prioritized) that skip attention remat while still rematting MoE")
    parser.add_argument("--splash-block-size", type=int, default=512, help="SplashAttention block size on full-attention layers")
    parser.add_argument("--steps", type=int, default=12, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--fsdp-parallel", type=int, default=16)
    parser.add_argument("--expert-parallel", type=int, default=1)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--num-layers-override", type=int, default=None, help="Optional layer count override for fast smoke test")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional path to write JSON benchmark summary")
    parser.add_argument("--sweep-v5p16", action="store_true", help="Run baseline 6-config 2x2x4 SPMD mesh sweep")
    parser.add_argument("--sweep-v5p16-opt", action="store_true", help="Run optimized high-MFU SplashAttention + fused ragged_dot hillclimb sweep on 2x2x4 TPU v5p")
    parser.add_argument("--sweep-v5p16-r6", action="store_true", help="Run Round 6 high-batch (mb=64..96) + selective MLP-only remat + sa=1024 frontier sweep on 2x2x4 TPU v5p")
    parser.add_argument("--sweep-v5p32-r6", action="store_true", help="Run Round 6 32-chip (2x4x4) TPU v5p-32 FSDP=32 + HSDP 2x16 + FSDP=8xEP=4 frontier sweep")
    parser.add_argument("--sweep-v5p16-r7", action="store_true", help="Run Round 7 16-chip (2x2x4) HSDP 2x8 + mb=96..112 + ga=4 + 4K frontier sweep")
    parser.add_argument("--sweep-v5p32-r7", action="store_true", help="Run Round 7 32-chip (2x4x4) HSDP 2x16 mb=80..112 + ga=4 + HSDP 4x8 + 4K frontier sweep")
    parser.add_argument("--sweep-v7x16-r7", action="store_true", help="Run Round 7 Ironwood v7x-16 (2x2x4, 32 TensorCores) FSDP=32 + HSDP 2x16 + HSDP 4x8 frontier sweep")
    parser.add_argument("--sweep-v7x16-r8", action="store_true", help="Run Round 8 Ironwood v7x-16 (2x2x4, 32 TensorCores) HBM-calibrated mb=32..64 + HSDP 2x16 + EP=4 sweep")
    parser.add_argument("--sweep-v5p16-r8", action="store_true", help="Run Round 8 16-chip (2x2x4) high-AI 4K/8K/16K long-context FSDP=16 MFU frontier sweep")
    parser.add_argument("--sweep-v5p32-r8", action="store_true", help="Run Round 8 32-chip (2x4x4) high-AI 4K/8K/16K long-context HSDP 2x16 MFU frontier sweep")
    parser.add_argument("--sweep-v7x16-r9", action="store_true", help="Run Round 9 Ironwood v7x-16 4K/8K/16K/32K/64K long-context MFU frontier sweep")
    parser.add_argument("--sweep-v5p16-r9", action="store_true", help="Run Round 9 16-chip (2x2x4) 32K/64K/128K ultra-long-context + shared_mem=32% MFU frontier sweep")
    parser.add_argument("--sweep-v5p32-r9", action="store_true", help="Run Round 9 32-chip (2x4x4) 32K/64K/128K ultra-long-context + shared_mem=32% MFU frontier sweep")
    parser.add_argument("--sweep-output-dir", type=str, default="/scratch/results", help="Local directory for sweep JSON outputs")
    parser.add_argument("--gcs-results-prefix", type=str, default="flint-jax-training/results", help="GCS prefix for immediate result upload")
    args = parser.parse_args()

    try:
        coord_addr = os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")
        num_slices = os.environ.get("MEGASCALE_NUM_SLICES")
        slice_id = os.environ.get("MEGASCALE_SLICE_ID", os.environ.get("JOB_COMPLETION_INDEX"))
        if coord_addr and num_slices and slice_id is not None and int(num_slices) > 1:
            jax.distributed.initialize(
                coordinator_address=coord_addr,
                num_processes=int(num_slices),
                process_id=int(slice_id),
            )
        else:
            tpu_hostnames_str = os.environ.get("TPU_WORKER_HOSTNAMES", "")
            tpu_worker_id = os.environ.get("TPU_WORKER_ID")
            if tpu_hostnames_str and "," in tpu_hostnames_str and tpu_worker_id is not None:
                raw_hosts = [h.strip() for h in tpu_hostnames_str.split(",") if h.strip()]
                coord_host = raw_hosts[0]
                for _ in range(90):
                    try:
                        coord_ip = socket.gethostbyname(coord_host)
                        break
                    except Exception:
                        time.sleep(1.0)
                else:
                    coord_ip = coord_host
                logger.info(
                    "Initializing GKE TPU slice JAX runtime: worker_id=%s/%d, coordinator=%s:8482",
                    tpu_worker_id,
                    len(raw_hosts),
                    coord_ip,
                )
                jax.distributed.initialize(
                    coordinator_address=f"{coord_ip}:8482",
                    num_processes=len(raw_hosts),
                    process_id=int(tpu_worker_id),
                )
            else:
                jax.distributed.initialize()
    except Exception as exc:
        logger.info("Single-process or already-initialized JAX runtime (%s)", exc)

    logger.info(
        "JAX process %d/%d initialized | Local devices: %d | Global devices: %d (%s)",
        jax.process_index(),
        jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
        jax.devices()[0].device_kind,
    )

    base_overrides: dict[str, Any] = {
        "dtype": jnp.bfloat16,
        "weight_dtype": jnp.bfloat16,
        "remat_first_n_layers": args.remat_first_n_layers,
        "save_splash_attn_layers": args.save_splash_attn_layers,
        "splash_block_size": args.splash_block_size,
    }
    if args.num_layers_override is not None:
        base_overrides["num_hidden_layers"] = args.num_layers_override
        base_overrides["layer_types"] = FlintJAXConfig.layer_types[: args.num_layers_override]

    def _build_cfg(**extra: Any) -> FlintJAXConfig:
        merged = {**base_overrides, **extra}
        if args.config_json:
            return FlintJAXConfig.from_json_file(args.config_json, **merged)
        return FlintJAXConfig(**merged)

    if (
        args.sweep_v5p16
        or args.sweep_v5p16_opt
        or args.sweep_v5p16_r6
        or args.sweep_v5p32_r6
        or args.sweep_v5p16_r7
        or args.sweep_v5p32_r7
        or args.sweep_v7x16_r7
        or args.sweep_v7x16_r8
        or args.sweep_v5p16_r8
        or args.sweep_v5p32_r8
        or args.sweep_v7x16_r9
        or args.sweep_v5p16_r9
        or args.sweep_v5p32_r9
    ):
        import json as _json

        if args.sweep_v5p16_r9:
            # Ultra-high arithmetic-intensity 32K/64K/128K SplashAttention + ragged_dot frontier on v5p-16 (FSDP=16)
            sweep_specs = [
                ("jax_v5p16_r9_01_fsdp16_s32768_mb6_ga2_sa1024", 32768, 6, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r9_02_fsdp16_s65536_mb3_ga2_sa1024", 65536, 3, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r9_03_fsdp16_s131072_mb1_ga4_sa1024", 131072, 1, 4, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r9_04_fsdp16_s4096_mb56_ga2_sa1024", 4096, 56, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r9_05_fsdp16_s8192_mb28_ga2_sa1024", 8192, 28, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r9_06_fsdp16_s32768_mb8_ga2_sa1024", 32768, 8, 2, 49, 0, 1024, 1, 16, 1, 1),
            ]
            summary_filename = "jax_v5p16_r9_sweep_summary.json"
        elif args.sweep_v5p32_r9:
            # Ultra-high arithmetic-intensity 32K/64K/128K SplashAttention + ragged_dot frontier on v5p-32 (HSDP 2x16)
            sweep_specs = [
                ("jax_v5p32_r9_01_dp2_fsdp16_s32768_mb6_ga2_sa1024", 32768, 6, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r9_02_dp2_fsdp16_s65536_mb3_ga2_sa1024", 65536, 3, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r9_03_dp2_fsdp16_s131072_mb1_ga4_sa1024", 131072, 1, 4, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r9_04_dp2_fsdp16_s4096_mb56_ga2_sa1024", 4096, 56, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r9_05_dp2_fsdp16_s8192_mb28_ga2_sa1024", 8192, 28, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r9_06_dp2_fsdp16_s32768_mb8_ga2_sa1024", 32768, 8, 2, 49, 0, 1024, 2, 16, 1, 1),
            ]
            summary_filename = "jax_v5p32_r9_sweep_summary.json"
        elif args.sweep_v5p16_r8:
            # High-arithmetic-intensity 4K/8K/16K SplashAttention + ragged_dot frontier on v5p-16 (FSDP=16)
            sweep_specs = [
                ("jax_v5p16_r8_01_fsdp16_s4096_mb56_ga2_sa1024", 4096, 56, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r8_02_fsdp16_s4096_mb64_ga2_sa1024", 4096, 64, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r8_03_fsdp16_s8192_mb24_ga2_sa1024", 8192, 24, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r8_04_fsdp16_s8192_mb32_ga2_sa1024", 8192, 32, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r8_05_fsdp16_s16384_mb12_ga2_sa1024", 16384, 12, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_v5p16_r8_06_fsdp16_s16384_mb16_ga2_sa1024", 16384, 16, 2, 49, 0, 1024, 1, 16, 1, 1),
            ]
            summary_filename = "jax_v5p16_r8_sweep_summary.json"
        elif args.sweep_v5p32_r8:
            # High-arithmetic-intensity 4K/8K/16K SplashAttention + ragged_dot frontier on v5p-32 (HSDP 2x16)
            sweep_specs = [
                ("jax_v5p32_r8_01_dp2_fsdp16_s4096_mb56_ga2_sa1024", 4096, 56, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r8_02_dp2_fsdp16_s4096_mb64_ga2_sa1024", 4096, 64, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r8_03_dp2_fsdp16_s8192_mb24_ga2_sa1024", 8192, 24, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r8_04_dp2_fsdp16_s8192_mb32_ga2_sa1024", 8192, 32, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r8_05_dp2_fsdp16_s16384_mb12_ga2_sa1024", 16384, 12, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r8_06_dp2_fsdp16_s16384_mb16_ga2_sa1024", 16384, 16, 2, 49, 0, 1024, 2, 16, 1, 1),
            ]
            summary_filename = "jax_v5p32_r8_sweep_summary.json"
        elif args.sweep_v7x16_r9:
            # Ironwood v7x-16 (32 TensorCores) HSDP 2x16 2K -> 4K -> 8K -> 16K -> 32K -> 64K MFU frontier
            sweep_specs = [
                ("jax_v7x16_r9_01_dp2_fsdp16_s4096_mb32_ga2_sa1024", 4096, 32, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r9_02_dp2_fsdp16_s8192_mb16_ga2_sa1024", 8192, 16, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r9_03_dp2_fsdp16_s16384_mb8_ga2_sa1024", 16384, 8, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r9_04_dp2_fsdp16_s32768_mb4_ga2_sa1024", 32768, 4, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r9_05_dp2_fsdp16_s65536_mb2_ga2_sa1024", 65536, 2, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r9_06_dp2_fsdp16_s131072_mb1_ga2_sa1024", 131072, 1, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r9_07_fsdp16_ep2_s2048_mb32_ga2_sa1024", 2048, 32, 2, 49, 0, 1024, 1, 16, 2, 1),
            ]
            summary_filename = "jax_v7x16_r9_sweep_summary.json"
        elif args.sweep_v7x16_r8:
            # (exp_name, seq_len, mb, ga, remat_n, save_splash, sa_block, dp, fsdp, ep, tp)
            sweep_specs = [
                ("jax_v7x16_r8_01_dp2_fsdp16_s2048_mb32_ga2_sa1024", 2048, 32, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r8_02_dp2_fsdp16_s2048_mb48_ga2_sa1024", 2048, 48, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r8_03_fsdp32_s2048_mb48_ga2_sa1024", 2048, 48, 2, 49, 0, 1024, 1, 32, 1, 1),
                ("jax_v7x16_r8_04_dp2_fsdp16_s2048_mb48_ga4_sa1024", 2048, 48, 4, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r8_05_dp2_fsdp16_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r8_06_dp4_fsdp8_s2048_mb48_ga2_sa1024", 2048, 48, 2, 49, 0, 1024, 4, 8, 1, 1),
                ("jax_v7x16_r8_07_fsdp8_ep4_s2048_mb32_ga2_sa1024", 2048, 32, 2, 49, 0, 1024, 1, 8, 4, 1),
            ]
            summary_filename = "jax_v7x16_r8_sweep_summary.json"
        elif args.sweep_v7x16_r7:
            # (exp_name, seq_len, mb, ga, remat_n, save_splash, sa_block, dp, fsdp, ep, tp)
            sweep_specs = [
                ("jax_v7x16_r7_01_fsdp32_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 1, 32, 1, 1),
                ("jax_v7x16_r7_02_dp2_fsdp16_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r7_03_dp2_fsdp16_s2048_mb80_ga2_sa1024", 2048, 80, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r7_04_dp2_fsdp16_s2048_mb96_ga2_sa1024", 2048, 96, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r7_05_dp2_fsdp16_s2048_mb96_ga4_sa1024", 2048, 96, 4, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v7x16_r7_06_dp4_fsdp8_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 4, 8, 1, 1),
            ]
            summary_filename = "jax_v7x16_r7_sweep_summary.json"
        elif args.sweep_v5p32_r7:
            # (exp_name, seq_len, mb, ga, remat_n, save_splash, sa_block, dp, fsdp, ep, tp)
            sweep_specs = [
                ("jax_v5p32_r7_01_dp2_fsdp16_s2048_mb80_ga2_sa1024", 2048, 80, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r7_02_dp2_fsdp16_s2048_mb96_ga2_sa1024", 2048, 96, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r7_03_dp2_fsdp16_s2048_mb96_ga4_sa1024", 2048, 96, 4, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r7_04_dp2_fsdp16_s2048_mb112_ga2_sa1024", 2048, 112, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r7_05_dp2_fsdp16_s2048_mb112_ga4_sa1024", 2048, 112, 4, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r7_06_dp2_fsdp16_s4096_mb48_ga2_sa1024", 4096, 48, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_r7_07_dp4_fsdp8_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 4, 8, 1, 1),
            ]
            summary_filename = "jax_v5p32_r7_sweep_summary.json"
        elif args.sweep_v5p16_r7:
            # (exp_name, seq_len, mb, ga, remat_n, save_splash, sa_block, dp, fsdp, ep, tp)
            sweep_specs = [
                ("jax_r7_01_fsdp16_s2048_mb112_ga2_sa1024", 2048, 112, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_r7_02_fsdp16_s2048_mb96_ga4_sa1024", 2048, 96, 4, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_r7_03_dp2_fsdp8_s2048_mb80_ga2_sa1024", 2048, 80, 2, 49, 0, 1024, 2, 8, 1, 1),
                ("jax_r7_04_dp2_fsdp8_s2048_mb96_ga2_sa1024", 2048, 96, 2, 49, 0, 1024, 2, 8, 1, 1),
                ("jax_r7_05_dp2_fsdp8_s2048_mb96_ga4_sa1024", 2048, 96, 4, 49, 0, 1024, 2, 8, 1, 1),
                ("jax_r7_06_fsdp16_s4096_mb48_ga2_sa1024", 4096, 48, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_r7_07_dp2_fsdp8_s4096_mb48_ga2_sa1024", 4096, 48, 2, 49, 0, 1024, 2, 8, 1, 1),
            ]
            summary_filename = "jax_v5p16_r7_sweep_summary.json"
        elif args.sweep_v5p32_r6:
            # (exp_name, seq_len, mb, ga, remat_n, save_splash, sa_block, dp, fsdp, ep, tp)
            sweep_specs = [
                ("jax_v5p32_01_fsdp32_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 1, 32, 1, 1),
                ("jax_v5p32_02_dp2_fsdp16_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 2, 16, 1, 1),
                ("jax_v5p32_03_dp2_fsdp16_s2048_mb80_ga2_sa1024_noattnremat12", 2048, 80, 2, 49, 12, 1024, 2, 16, 1, 1),
                ("jax_v5p32_04_fsdp32_s2048_mb96_ga2_sa1024_noattnremat12", 2048, 96, 2, 49, 12, 1024, 1, 32, 1, 1),
                ("jax_v5p32_05_dp2_fsdp16_s4096_mb32_ga2_sa1024_noattnremat12", 4096, 32, 2, 49, 12, 1024, 2, 16, 1, 1),
                ("jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024", 2048, 48, 2, 49, 0, 1024, 1, 8, 4, 1),
            ]
            summary_filename = "jax_v5p32_r6_sweep_summary.json"
        elif args.sweep_v5p16_r6:
            # (exp_name, seq_len, mb, ga, remat_n, save_splash, sa_block, dp, fsdp, ep, tp)
            sweep_specs = [
                ("jax_r6_01_fsdp16_s2048_mb64_ga2_sa1024", 2048, 64, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_r6_02_fsdp16_s2048_mb64_ga2_sa1024_noattnremat12", 2048, 64, 2, 49, 12, 1024, 1, 16, 1, 1),
                ("jax_r6_03_fsdp16_s2048_mb80_ga2_sa1024", 2048, 80, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_r6_04_fsdp16_s2048_mb96_ga2_sa1024", 2048, 96, 2, 49, 0, 1024, 1, 16, 1, 1),
                ("jax_r6_05_fsdp16_s2048_mb64_ga4_sa1024_noattnremat24", 2048, 64, 4, 49, 24, 1024, 1, 16, 1, 1),
                ("jax_r6_06_fsdp16_s4096_mb32_ga2_sa1024_noattnremat12", 4096, 32, 2, 49, 12, 1024, 1, 16, 1, 1),
            ]
            summary_filename = "jax_v5p16_r6_sweep_summary.json"
        elif args.sweep_v5p16_opt:
            sweep_specs = [
                ("jax_opt01_fsdp16_ep1_s2048_mb8_ga1", 2048, 8, 1, 49, 0, 512, 1, 16, 1, 1),
                ("jax_opt02_fsdp16_ep1_s2048_mb16_ga2", 2048, 16, 2, 49, 0, 512, 1, 16, 1, 1),
                ("jax_opt03_fsdp16_ep1_s2048_mb24_ga2", 2048, 24, 2, 49, 0, 512, 1, 16, 1, 1),
                ("jax_opt04_fsdp16_ep1_s2048_mb32_ga2", 2048, 32, 2, 49, 0, 512, 1, 16, 1, 1),
                ("jax_opt05_fsdp16_ep1_s2048_mb32_ga4_hybac32", 2048, 32, 4, 32, 0, 512, 1, 16, 1, 1),
                ("jax_opt06_fsdp16_ep1_s2048_mb48_ga2", 2048, 48, 2, 49, 0, 512, 1, 16, 1, 1),
                ("jax_opt07_fsdp4_ep4_s2048_mb16_ga2", 2048, 16, 2, 49, 0, 512, 1, 4, 4, 1),
                ("jax_opt08_fsdp16_ep1_s4096_mb16_ga2", 4096, 16, 2, 49, 0, 512, 1, 16, 1, 1),
            ]
            summary_filename = "jax_v5p16_opt_sweep_summary.json"
        else:
            sweep_specs = [
                ("jax_v5p16_fsdp4_ep4_s2048_mb2", 2048, 2, 1, 49, 0, 512, 1, 4, 4, 1),
                ("jax_v5p16_fsdp4_ep4_s2048_mb4", 2048, 4, 1, 49, 0, 512, 1, 4, 4, 1),
                ("jax_v5p16_fsdp8_ep2_s2048_mb4", 2048, 4, 1, 49, 0, 512, 1, 8, 2, 1),
                ("jax_v5p16_fsdp16_ep1_s2048_mb4", 2048, 4, 1, 49, 0, 512, 1, 16, 1, 1),
                ("jax_v5p16_fsdp4_ep4_s4096_mb2", 4096, 2, 1, 49, 0, 512, 1, 4, 4, 1),
                ("jax_v5p16_fsdp4_ep4_s4096_mb4", 4096, 4, 1, 49, 0, 512, 1, 4, 4, 1),
            ]
            summary_filename = "jax_v5p16_sweep_summary.json"

        all_results: list[dict[str, Any]] = []
        for exp_name, seq_len, mb, ga, remat_n, save_splash, sa_block, dp, fsdp, ep, tp in sweep_specs:
            cfg = _build_cfg(
                remat_first_n_layers=remat_n,
                save_splash_attn_layers=save_splash,
                splash_block_size=sa_block,
            )
            out_json = os.path.join(args.sweep_output_dir, f"{exp_name}.json")
            res = run_single_experiment(
                exp_name=exp_name,
                config=cfg,
                seq_len=seq_len,
                per_device_batch_size=mb,
                grad_accum_steps=ga,
                steps=args.steps,
                learning_rate=args.learning_rate,
                warmup_steps=min(args.warmup_steps, max(1, args.steps // 3)),
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                data_parallel=dp,
                fsdp_parallel=fsdp,
                expert_parallel=ep,
                tensor_parallel=tp,
                hf_checkpoint_dir=args.hf_checkpoint_dir,
                summary_json=out_json,
                gcs_results_prefix=args.gcs_results_prefix,
            )
            all_results.append(res)

        if jax.process_index() == 0:
            combined_path = os.path.join(args.sweep_output_dir, summary_filename)
            with open(combined_path, "w") as f:
                _json.dump({"experiments": all_results}, f, indent=2)
            _upload_file_to_gcs(
                combined_path, f"{args.gcs_results_prefix.rstrip('/')}/{summary_filename}"
            )
        return

    config = _build_cfg()
    run_single_experiment(
        exp_name=os.path.splitext(os.path.basename(args.summary_json))[0]
        if args.summary_json
        else "flint_jax_run",
        config=config,
        seq_len=args.seq_len,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        steps=args.steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        data_parallel=args.data_parallel,
        fsdp_parallel=args.fsdp_parallel,
        expert_parallel=args.expert_parallel,
        tensor_parallel=args.tensor_parallel,
        hf_checkpoint_dir=args.hf_checkpoint_dir,
        summary_json=args.summary_json,
        gcs_results_prefix=args.gcs_results_prefix if args.summary_json else None,
    )


if __name__ == "__main__":
    main()
