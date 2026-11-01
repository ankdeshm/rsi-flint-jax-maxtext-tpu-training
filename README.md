# Recursive Flint (39.8B-A2.5B MoE) — JAX / MaxText BF16 Training on GKE TPU v5p

This directory contains a complete JAX / Flax & [MaxText](https://github.com/AI-Hypercomputer/maxtext) BF16 pre-training and continual pre-training stack for **[Recursive Flint (`RecursiveSI/flint`)](https://huggingface.co/RecursiveSI/flint)** targeting cluster **`tpu-v5p-cluster`** (`us-central1`, project `YOUR_GCP_PROJECT_ID`).

---

## 1. Architecture & TPU v5p Kernel Mapping

Flint (`FlintForCausalLM`, 39.8B total / ~2.5B active parameters) combines several custom architectural features that are implemented in [`flint_jax/model.py`](./flint_jax/model.py):

| Flint Component (`RecursiveSI/flint`) | Specification | JAX / MaxText TPU v5p Implementation |
| :--- | :--- | :--- |
| **Backbone** | 49 layers, `hidden_size=2048`, `vocab_size=150,144` | `FlintForCausalLM` with per-layer `jax.checkpoint` (`nn.remat`) |
| **Hybrid Dense + MoE Layers** | Layers `0, 1`: Dense SwiGLU (`intermediate_size=6144`)<br>Layers `2..48` (47 layers): Sparse MoE | `FlintMLP` on `mlp_only_layers=(0, 1)` + `FlintSparseMoeBlock` on layers `2..48` |
| **Sparse MoE + Shared Expert** | `256` routed experts (`moe_intermediate_size=512`), `top_k=6`, `router_score_function="sigmoid"`, `norm_topk_prob=True`, `router_aux_loss_coef=0.0005` + 1 ungated shared expert (`1024`) | Sigmoid top-6 routing with `expert_bias` + dropless grouped matmul via **`jax.lax.ragged_dot`** (Megablox/MaxText kernel) sharded across `("expert", "fsdp", "tensor")` |
| **Hybrid Attention & Selective RoPE** | 35 `sliding_attention` layers (`sliding_window=513`, with RoPE $\theta=10000$) + 14 `full_attention` layers (`no_rope_layer_indices`, no positional encoding) | `FlintAttention` with GQA (`32` Q heads, `4` KV heads, `head_dim=128`), per-head QK `FlintRMSNorm`, and selective `apply_rope` |
| **Learned Attention Sinks** | `softmax_type="learnable"` (`self_attn.softmax_offset` of shape `[32]`) | `attention_with_learned_sinks`: adds $\exp(s_h - m_i)$ to the attention softmax denominator without contributing to $V$ |

---

## 2. Directory Structure

- [`flint_jax/config.py`](./flint_jax/config.py): Dataclass configuration matching [`RecursiveSI/flint/config.json`](https://huggingface.co/RecursiveSI/flint/blob/main/config.json).
- [`flint_jax/model.py`](./flint_jax/model.py): Complete JAX/Flax implementation of `FlintForCausalLM` with 4D SPMD partitioning (`data`, `fsdp`, `expert`, `tensor`) and `jax.lax.ragged_dot` MoE execution.
- [`flint_jax/checkpoint_loader.py`](./flint_jax/checkpoint_loader.py): Streams the 15 Hugging Face safetensors shards (`model-00001-of-00015.safetensors` .. `model-00015-of-00015.safetensors`) into sharded TPU HBM arrays.
- [`flint_jax/train.py`](./flint_jax/train.py): Multi-host JAX BF16 training loop with Optax AdamW (`b1=0.9, b2=0.95, wd=0.1`), cosine warmup, global norm clipping, and MoE auxiliary load-balancing loss.
- [`maxtext/flint_40b_a2_5b_v5p.yml`](./maxtext/flint_40b_a2_5b_v5p.yml): MaxText configuration file using TPU v5p Megablox (`megablox: true`, `sparse_matmul: true`) and XLA compiler flags from [`AI-Hypercomputer/tpu-recipes/tree/main/training/v5p`](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p).
- [`k8s/flint-v5p-2x2x4-job.yaml`](./k8s/flint-v5p-2x2x4-job.yaml): Kubernetes Indexed `Job` + Headless `Service` targeting the **`tpu-v5p-16-pool`** node pool (`2x2x4` topology = 4 nodes, 16 TPU v5p chips, 1,520 GB HBM).
- [`k8s/flint-v5p-multislice-jobset.yaml`](./k8s/flint-v5p-multislice-jobset.yaml): Multi-Slice `JobSet` (`4 × 2x2x1` = 16 TPU v5p chips, 1,520 GB HBM) that runs **immediately** across 4 of the 14 already-running `2x2x1` Spot nodes in `tpu-v5p-cluster`.
- [`Dockerfile`](./Dockerfile): Builds the training image on top of `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/maxtext/maxtext:latest`.

---

## 3. Deploying from `gs://YOUR_GCS_BUCKET/flint-jax-training` on `tpu-v5p-cluster`

All code, configs, manifests, and training outputs live in **`gs://YOUR_GCS_BUCKET/flint-jax-training`**. At startup, each GKE TPU worker pulls the latest code directly from `gs://YOUR_GCS_BUCKET/flint-jax-training` and writes checkpoints and logs to `gs://YOUR_GCS_BUCKET/flint-jax-training/checkpoints`.

### Step 1: Authenticate to Cluster `tpu-v5p-cluster`
```bash
gcloud container clusters get-credentials tpu-v5p-cluster \
  --location=us-central1 \
  --project=YOUR_GCP_PROJECT_ID
```

### Step 2A: Deploy Option A on the `2x2x4` Node Pool (`tpu-v5p-16-pool`)
Submitting this Job places 4 indexed pods in `Pending` state for topology `2x2x4`, triggering GKE Cluster Autoscaler on `tpu-v5p-16-pool` to scale `0 → 4` nodes atomically:
```bash
gcloud storage cat gs://YOUR_GCS_BUCKET/flint-jax-training/k8s/flint-v5p-2x2x4-job.yaml | kubectl apply -f -
kubectl get pods -l app=flint-v5p-train -w
```

### Step 2B: Deploy Option B Immediately Across 4 Existing Healthy `2x2x1` Nodes
If `us-central1-a` experiences a stockout for a contiguous `2x2x4` slice, deploy across **4 of the 14 already-running `2x2x1` Spot nodes** (16 TPU v5p chips = 1,520 GB HBM total) right away:
```bash
gcloud storage cat gs://YOUR_GCS_BUCKET/flint-jax-training/k8s/flint-v5p-multislice-jobset.yaml | kubectl apply -f -
kubectl get pods -l app=flint-v5p-multislice-train -w
```

