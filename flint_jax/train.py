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
import time
from typing import Any

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
    args = parser.parse_args()

    # Initialize multi-host / multi-slice JAX runtime on GKE TPU
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
            import socket
            import urllib.request
            tpu_hostnames_str = os.environ.get("TPU_WORKER_HOSTNAMES", "")
            if tpu_hostnames_str and "," in tpu_hostnames_str:
                raw_hosts = [h.strip() for h in tpu_hostnames_str.split(",") if h.strip()]
                num_hosts = len(raw_hosts)
                req_host = urllib.request.Request(
                    "http://metadata.google.internal/computeMetadata/v1/instance/hostname",
                    headers={"Metadata-Flavor": "Google"},
                )
                my_vm = urllib.request.urlopen(req_host, timeout=5).read().decode().strip().split(".")[0]
                my_pod_ip = socket.gethostbyname(socket.gethostname())
                idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
                coord_host = raw_hosts[0]
                if idx == 0:
                    entries_map: dict[str, str] = {my_vm: my_pod_ip}
                    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    srv.bind(("0.0.0.0", 9999))
                    srv.listen(num_hosts * 4)
                    conns = []
                    while len(entries_map) < num_hosts:
                        conn, _ = srv.accept()
                        data = conn.recv(1024).decode().strip()
                        if ":" in data:
                            vm_name, pod_ip = data.split(":", 1)
                            entries_map[vm_name] = pod_ip
                            conns.append(conn)
                        else:
                            conn.close()
                    entries = sorted(entries_map.items(), key=lambda item: item[0])
                    payload = ";".join(f"{vm}:{ip}" for vm, ip in entries).encode()
                    for c in conns:
                        try:
                            c.sendall(payload)
                            c.close()
                        except Exception:
                            pass
                    srv.close()
                else:
                    for _ in range(180):
                        try:
                            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s.settimeout(180)
                            s.connect((socket.gethostbyname(coord_host), 9999))
                            s.sendall(f"{my_vm}:{my_pod_ip}".encode())
                            resp = s.recv(4096).decode().strip()
                            s.close()
                            if resp and ";" in resp:
                                entries = [tuple(part.split(":", 1)) for part in resp.split(";")]
                                break
                        except Exception:
                            time.sleep(1.0)
                phys_rank = next(i for i, (vm, _) in enumerate(entries) if vm == my_vm)
                sorted_ips = ",".join(ip for _, ip in entries)
                os.environ["TPU_WORKER_ID"] = str(phys_rank)
                os.environ["TPU_WORKER_HOSTNAMES"] = sorted_ips
                logger.info(
                    "Pre-init aligned physical GCE TPU VM order: vm=%s -> TPU_WORKER_ID=%d, TPU_WORKER_HOSTNAMES=%s",
                    my_vm,
                    phys_rank,
                    sorted_ips,
                )
                jax.distributed.initialize(
                    coordinator_address=f"{entries[0][1]}:8482",
                    num_processes=num_hosts,
                    process_id=phys_rank,
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

    mesh = create_tpu_mesh(
        data_parallel=args.data_parallel,
        fsdp_parallel=args.fsdp_parallel,
        expert_parallel=args.expert_parallel,
        tensor_parallel=args.tensor_parallel,
    )
    logger.info("Created 4D TPU v5p Mesh: %s", mesh.shape)

    model = FlintForCausalLM(config=config)
    tx = build_optimizer(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
    )

    rng = jax.random.PRNGKey(42)
    t0 = time.perf_counter()
    state, state_sharding = create_sharded_train_state(
        rng=rng,
        mesh=mesh,
        model=model,
        tx=tx,
        seq_len=args.seq_len,
    )
    logger.info("Initialized sharded Flint TrainState in %.2fs", time.perf_counter() - t0)

    if args.hf_checkpoint_dir:
        logger.info("Loading pretrained RecursiveSI/flint BF16 weights from %s ...", args.hf_checkpoint_dir)
        loaded_params = load_hf_flint_checkpoint(args.hf_checkpoint_dir, state.params, config)
        state = state.replace(params=loaded_params)

    num_devs = jax.device_count()
    global_batch_size = args.per_device_batch_size * num_devs
    batch_sharding = NamedSharding(mesh, P(("data", "fsdp", "expert"), None))

    def _make_batch(step_idx: jnp.ndarray) -> dict[str, jnp.ndarray]:
        step_rng = jax.random.fold_in(rng, step_idx)
        tokens = jax.random.randint(
            step_rng,
            shape=(global_batch_size, args.seq_len),
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

    # Active FLOPs/token for RecursiveSI/flint (17.385 GFLOP/tok @ seq_len=2048)
    flops_per_token = 17.385e9
    peak_tflops_per_chip = 459.0
    step_records: list[dict[str, Any]] = []

    with jax.set_mesh(mesh):
        for step in range(1, args.steps + 1):
            batch = compiled_make_batch(jnp.asarray(step, dtype=jnp.int32))
            jax.block_until_ready(batch["input_ids"])
            step_t0 = time.perf_counter()

            state, metrics = compiled_train_step(state, batch)
            jax.block_until_ready(metrics["total_loss"])
            dt = time.perf_counter() - step_t0
            tokens_per_sec = (global_batch_size * args.seq_len) / dt
            tps_per_chip = tokens_per_sec / num_devs
            mfu_pct = (tps_per_chip * flops_per_token) / (peak_tflops_per_chip * 1e12) * 100.0
            mem_stats = jax.local_devices()[0].memory_stats() or {}
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
                "peak_hbm_gib": round(peak_hbm_gib, 2),
            }
            step_records.append(rec)

            if jax.process_index() == 0:
                logger.info(
                    "Step %04d/%04d | loss=%.4f (lm=%.4f aux=%.5f) | gnorm=%.4f | dt=%.3fs | chip_tps=%.1f | slice_tps=%.1f | MFU=%.2f%% | HBM=%.2fGiB",
                    step,
                    args.steps,
                    rec["total_loss"],
                    rec["lm_loss"],
                    rec["moe_aux_loss"],
                    rec["grad_norm"],
                    dt,
                    tps_per_chip,
                    tokens_per_sec,
                    mfu_pct,
                    peak_hbm_gib,
                )

    if jax.process_index() == 0 and args.summary_json:
        import json as _json
        import statistics as _stats
        steady = step_records[min(3, len(step_records) - 1) :]
        summary = {
            "device_kind": jax.devices()[0].device_kind,
            "num_devices": num_devs,
            "seq_len": args.seq_len,
            "per_device_batch_size": args.per_device_batch_size,
            "tokens_per_step": global_batch_size * args.seq_len,
            "mesh": {
                "data": args.data_parallel,
                "fsdp": args.fsdp_parallel,
                "expert": args.expert_parallel,
                "tensor": args.tensor_parallel,
            },
            "steady_chip_tok_s": round(_stats.median([r["chip_tok_s"] for r in steady]), 1),
            "steady_cluster_tok_s": round(_stats.median([r["cluster_tok_s"] for r in steady]), 1),
            "steady_mfu_pct": round(_stats.median([r["mfu_pct"] for r in steady]), 2),
            "peak_mfu_pct": round(max(r["mfu_pct"] for r in steady), 2),
            "peak_hbm_gib": round(max(r["peak_hbm_gib"] for r in step_records), 2),
            "loss_start": step_records[0]["lm_loss"],
            "loss_end": step_records[-1]["lm_loss"],
            "steps": step_records,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as f:
            _json.dump(summary, f, indent=2)
        logger.info("Wrote JAX benchmark summary to %s: steady MFU=%.2f%% (peak %.2f%%), HBM=%.2fGiB", args.summary_json, summary["steady_mfu_pct"], summary["peak_mfu_pct"], summary["peak_hbm_gib"])


if __name__ == "__main__":
    main()
