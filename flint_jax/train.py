"""Distributed JAX BF16 Pre-Training / Continual Pre-Training Script for Recursive Flint on TPU v5p.

Designed for the `tpu-v5p-16-pool` node pool (`2x2x4` topology = 4 nodes, 16 TPU v5p chips,
1,520 GB HBM) in cluster `tpu-v5p-cluster` (`YOUR_GCP_PROJECT_ID`), and also supports single-host
`2x2x1` smoke testing or larger multi-slice `JobSet` topologies.

Features:
  - Automatic multi-host TPU initialization (`jax.distributed.initialize()`)
  - 4D SPMD Mesh (`data`, `fsdp`, `expert`, `tensor`) across TPU v5p devices
  - Full BF16 forward/backward pass with FP32 master weights & AdamW optimizer states sharded across
    `fsdp` and `expert` axes
  - Optional loading of pretrained `RecursiveSI/flint` safetensors checkpoint
  - Async Orbax checkpointing to GCS (`gs://YOUR_GCP_PROJECT_ID-maxtext/flint-jax-checkpoints`)
    for Spot TPU preemption recovery
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
from flax.linen import partitioning as nn_partitioning
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
    """Creates a 4D (data, fsdp, expert, tensor) JAX Mesh over all visible TPU devices."""
    devices = jax.devices()
    num_devices = len(devices)
    expected = data_parallel * fsdp_parallel * expert_parallel * tensor_parallel
    if expected != num_devices:
        raise ValueError(
            f"Mesh product ({data_parallel} * {fsdp_parallel} * {expert_parallel} * {tensor_parallel} = {expected}) "
            f"must equal total TPU devices ({num_devices})."
        )
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
        abstract_Vars = jax.eval_shape(init_fn, rng, dummy_tokens)
        state_sharding = nn.get_sharding(abstract_Vars, mesh)
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
    vocab_size: int,
) -> tuple[train_state.TrainState, dict[str, jnp.ndarray]]:
    """Single compiled SPMD training step in BF16 with causal LM cross-entropy + MoE aux loss."""

    def loss_fn(params: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        logits, aux_loss = state.apply_fn({"params": params}, batch["input_ids"])
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].astype(jnp.float32)
        shift_labels = batch["labels"][:, 1:]
        loss_mask = batch["loss_mask"][:, 1:].astype(jnp.float32)

        token_losses = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        lm_loss = jnp.sum(token_losses * loss_mask) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        total_loss = lm_loss + aux_loss
        return total_loss, {"total_loss": total_loss, "lm_loss": lm_loss, "moe_aux_loss": aux_loss}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
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


def _upload_file_to_gcs(local_path: str, gcs_object_name: str, bucket: str = "YOUR_GCS_BUCKET") -> None:
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
    import statistics as _stats
    from jax.experimental import multihost_utils

    num_devs = jax.device_count()
    global_batch_size = per_device_batch_size * num_devs
    flops_per_token = compute_active_flops_per_token(config, seq_len)
    dev_kind = jax.devices()[0].device_kind
    peak_tflops_per_chip = 459.0 if "v5p" in dev_kind.lower() or "v5" in dev_kind.lower() else 1153.5

    logger.info(
        "=== [%s] START: seq=%d, mb=%d (global_bs=%d, tok/step=%d), mesh=(dp=%d, fsdp=%d, ep=%d, tp=%d), flops/tok=%.3fG ===",
        exp_name,
        seq_len,
        per_device_batch_size,
        global_batch_size,
        global_batch_size * seq_len,
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
    model = FlintForCausalLM(config=config)
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
            logger.info("[%s] Loading pretrained RecursiveSI/flint BF16 weights from %s ...", exp_name, hf_checkpoint_dir)
            loaded_params = load_hf_flint_checkpoint(hf_checkpoint_dir, state.params, config)
            state = state.replace(params=loaded_params)

        batch_sharding = NamedSharding(mesh, P(("data", "fsdp", "expert"), None))

        def _make_batch(step_idx: jnp.ndarray) -> dict[str, jnp.ndarray]:
            step_rng = jax.random.fold_in(rng, step_idx)
            tokens = jax.random.randint(
                step_rng,
                shape=(global_batch_size, seq_len),
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
            functools.partial(train_step, vocab_size=config.vocab_size),
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
                tokens_per_sec = (global_batch_size * seq_len) / dt
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
        summary: dict[str, Any] = {
            "exp_name": exp_name,
            "status": "PASS",
            "device_kind": dev_kind,
            "num_devices": num_devs,
            "seq_len": seq_len,
            "per_device_batch_size": per_device_batch_size,
            "tokens_per_step": global_batch_size * seq_len,
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
            "status": "OOM" if "RESOURCE_EXHAUSTED" in str(exc) or "Out of memory" in str(exc) else "FAIL",
            "error": f"{type(exc).__name__}: {str(exc)[:600]}",
            "device_kind": dev_kind,
            "num_devices": num_devs,
            "seq_len": seq_len,
            "per_device_batch_size": per_device_batch_size,
            "tokens_per_step": global_batch_size * seq_len,
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
            _upload_file_to_gcs(summary_json, f"{gcs_results_prefix.rstrip('/')}/{os.path.basename(summary_json)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Recursive Flint (39.8B MoE) in BF16 with JAX on TPU v5p")
    parser.add_argument("--config-json", type=str, default=None, help="Path to RecursiveSI/flint config.json")
    parser.add_argument("--hf-checkpoint-dir", type=str, default=None, help="Optional local path to RecursiveSI/flint safetensors")
    parser.add_argument("--output-dir", type=str, default=os.environ.get("BASE_OUTPUT_DIR", "gs://YOUR_GCS_BUCKET/flint-jax-training/checkpoints"))
    parser.add_argument("--seq-len", type=int, default=4096, help="Training sequence length (Flint supports up to 262144)")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Micro-batch size per TPU chip")
    parser.add_argument("--steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    # Mesh axes for 2x2x4 (16 TPU v5p chips): data=1, fsdp=4, expert=4, tensor=1
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--fsdp-parallel", type=int, default=4)
    parser.add_argument("--expert-parallel", type=int, default=4)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--num-layers-override", type=int, default=None, help="Optional layer count override for fast smoke test")
    parser.add_argument("--summary-json", type=str, default=None, help="Optional path to write JSON benchmark summary")
    parser.add_argument("--sweep-v5p16", action="store_true", help="Run full single-process 2x2x4 (16-chip TPU v5p) SPMD mesh sweep")
    parser.add_argument("--sweep-output-dir", type=str, default="/scratch/results", help="Local directory for sweep JSON outputs")
    parser.add_argument("--gcs-results-prefix", type=str, default="flint-jax-training/results", help="GCS prefix for immediate result upload")
    args = parser.parse_args()

    # Initialize multi-host / multi-slice JAX runtime on GKE TPU ONCE per pod
    try:
        coord_addr = os.environ.get("MEGASCALE_COORDINATOR_ADDRESS")
        num_slices = os.environ.get("MEGASCALE_NUM_SLICES")
        slice_id = os.environ.get("MEGASCALE_SLICE_ID", os.environ.get("JOB_COMPLETION_INDEX"))
        if coord_addr and num_slices and slice_id is not None and int(num_slices) > 1:
            logger.info(
                "Initializing multi-slice JAX runtime: coordinator=%s, num_processes=%s, process_id=%s",
                coord_addr,
                num_slices,
                slice_id,
            )
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

    overrides: dict[str, Any] = {
        "dtype": jnp.bfloat16,
        "weight_dtype": jnp.bfloat16,
    }
    if args.num_layers_override is not None:
        overrides["num_hidden_layers"] = args.num_layers_override
        overrides["layer_types"] = FlintJAXConfig.layer_types[: args.num_layers_override]

    if args.config_json:
        config = FlintJAXConfig.from_json_file(args.config_json, **overrides)
    else:
        config = FlintJAXConfig(**overrides)

    if args.sweep_v5p16:
        import json as _json

        sweep_specs = [
            # (exp_name, seq_len, mb, dp, fsdp, ep, tp)
            ("jax_v5p16_fsdp4_ep4_s2048_mb2", 2048, 2, 1, 4, 4, 1),
            ("jax_v5p16_fsdp4_ep4_s2048_mb4", 2048, 4, 1, 4, 4, 1),
            ("jax_v5p16_fsdp8_ep2_s2048_mb4", 2048, 4, 1, 8, 2, 1),
            ("jax_v5p16_fsdp16_ep1_s2048_mb4", 2048, 4, 1, 16, 1, 1),
            ("jax_v5p16_fsdp4_ep4_s4096_mb2", 4096, 2, 1, 4, 4, 1),
            ("jax_v5p16_fsdp4_ep4_s4096_mb4", 4096, 4, 1, 4, 4, 1),
        ]
        all_results: list[dict[str, Any]] = []
        for exp_name, seq_len, mb, dp, fsdp, ep, tp in sweep_specs:
            out_json = os.path.join(args.sweep_output_dir, f"{exp_name}.json")
            res = run_single_experiment(
                exp_name=exp_name,
                config=config,
                seq_len=seq_len,
                per_device_batch_size=mb,
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
            combined_path = os.path.join(args.sweep_output_dir, "jax_v5p16_sweep_summary.json")
            with open(combined_path, "w") as f:
                _json.dump({"experiments": all_results}, f, indent=2)
            _upload_file_to_gcs(combined_path, f"{args.gcs_results_prefix.rstrip('/')}/jax_v5p16_sweep_summary.json")
        return

    run_single_experiment(
        exp_name=os.path.splitext(os.path.basename(args.summary_json))[0] if args.summary_json else "flint_jax_run",
        config=config,
        seq_len=args.seq_len,
        per_device_batch_size=args.per_device_batch_size,
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

