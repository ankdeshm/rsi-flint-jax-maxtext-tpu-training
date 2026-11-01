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

## 3. Empirical Benchmark Results & Optimization Progression on TPU `v5p` (`16` Chips, `2x2x4`)

All raw per-step JSON summaries are stored in [`results/`](./results/) and persisted to `gs://YOUR_GCS_BUCKET/flint-jax-training/results_r4/` and `results_r5/`.

### 3.1 Key JAX / Pallas / XLA Engineering Optimizations (`r4` Baseline → `r5` Hillclimb)

1. **Zero-NaN Custom Softmax-with-Learned-Sinks Backward (`softmax_with_sinks`):**
   - Naive `jnp.where(mask, scores, -inf)` produces `0 * (-inf) = NaN` in XLA's backward pass on masked sliding-window (`W=513`) positions when combined with learned sink denominators (`loss -> NaN` on Step 2 in `r4`). Masking pre-exp logits cleanly and zeroing masked probabilities post-division eliminated all gradient NaNs (`grad_norm = 0.20..0.47` across all steps).
2. **TPU Pallas `SplashAttention` (`jax.experimental.pallas.ops.tpu.splash_attention`) with Learned Sinks (`softmax_offset`):**
   - Replaced eager `[B, 32, S, S]` materialized attention tensors with fused block-sparse Pallas `SplashAttention` (`CausalMask` on the 14 global layers + `LocalMask(window_size=(512, 0))` on the 35 sliding-window layers, passing `sinks=self.softmax_offset` natively into `SplashAttention` VMEM tiles).
3. **Single Fused Gate+Up (`W1/W3`) `jax.lax.ragged_dot` MoE Kernel:**
   - Concatenated `gate_proj` and `up_proj` into a single `[256, D, 2 * moe_inter]` weight tensor (`("expert", "fsdp", "tensor")`), cutting 47-layer MoE `ragged_dot` kernel launches and token gather/scatter passes in half (`94 -> 47` forward grouped matmuls per microbatch).
4. **256-Token Chunked `jax.lax.scan` Cross-Entropy Loss:**
   - Replaced full-vocabulary `[B * S, 150,144]` FP32 logit materialization (`~39.4 GiB` at `mb=32`) with a 256-token `jax.checkpoint`-wrapped `jax.lax.scan` loop (`<0.6 GiB` peak logit buffer).
5. **Compiled `jax.lax.scan` Gradient Accumulation + Full-Layer Rematerialization (`remat_n=49`):**
   - Accumulating microbatch gradients inside a single XLA-compiled `jax.lax.scan` loop with `nn.remat(FlintDecoderLayer)` holds peak HBM nearly flat (**`14.32 GiB` at `mb=8` → `15.26 GiB` at `mb=32`** out of `95 GiB` `v5p` HBM!), converting FSDP `all-gather` / `reduce-scatter` bandwidth tax directly into **`20.30%` Peak MFU (`7.96×` over `r4` `FSDP=16`!)**.

### 3.2 Benchmark Leaderboard (`16` TPU `v5p` Chips, `459.0` BF16 TFLOP/s/chip, `17.383` GFLOP/token)

| Pass / Experiment ID | Mesh (`dp, fsdp, ep, tp`) | `seq_len` | `mb` / `ga` | Global Batch (tok/step) | Remat Layers | Steady chip tok/s | Cluster tok/s (16 chips) | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (`start -> end`) | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **`r5` `jax_opt04_fsdp16_ep1_s2048_mb32_ga2`** | `(1, 16, 1, 1)` | 2048 | `32 / 2` | **2,097,152** | 49 | **5,357.1** | **85,713.7** *(85,762 pk)* | **20.29%** | **20.30%** | **15.26** | **`12.3283 -> 11.9035`** | **`PASS` (`7.96×` vs `r4`!)** |
| **`r5` `jax_opt03_fsdp16_ep1_s2048_mb24_ga2`** | `(1, 16, 1, 1)` | 2048 | `24 / 2` | **1,572,864** | 49 | **4,910.6** | **78,569.3** *(78,691 pk)* | **18.60%** | **18.63%** | **14.66** | **`12.3286 -> 11.8774`** | **`PASS`** |
| **`r5` `jax_opt02_fsdp16_ep1_s2048_mb16_ga2`** | `(1, 16, 1, 1)` | 2048 | `16 / 2` | **1,048,576** | 49 | **4,173.7** | **66,779.5** *(66,960 pk)* | **15.81%** | **15.85%** | **14.34** | **`12.3294 -> 11.7538`** | **`PASS`** |
| **`r5` `jax_opt01_fsdp16_ep1_s2048_mb8_ga1`** | `(1, 16, 1, 1)` | 2048 | `8 / 1` | **262,144** | 49 | **2,858.8** | **45,741.5** *(45,805 pk)* | **10.83%** | **10.84%** | **14.32** | **`12.3281 -> 11.3851`** | **`PASS`** |
| `r5` `jax_opt05_fsdp16_ep1_s2048_mb32_ga4_hybac32` | `(1, 16, 1, 1)` | 2048 | `32 / 4` | 4,194,304 | 32 *(17 unremat)* | — | — | — | — | >95.7 | — | `OOM` (`191.1G` HLO temp) |
| `r4` `jax_v5p16_fsdp4_ep4_s2048_mb4` *(Eager Attn)* | `(1, 4, 4, 1)` | 2048 | `4 / 1` | 131,072 | 49 | 965.6 | 15,449.4 | 3.66% | 3.81% | 16.78 | `12.3253 -> nan` | Baseline (`r4`) |
| `r4` `jax_v5p16_fsdp4_ep4_s4096_mb4` *(Eager Attn)* | `(1, 4, 4, 1)` | 4096 | `4 / 1` | 262,144 | 49 | 646.7 | 10,347.0 | 2.65% | 2.65% | 16.79 | `12.3272 -> nan` | Baseline (`r4`) |
| `r4` `jax_v5p16_fsdp8_ep2_s2048_mb4` *(Eager Attn)* | `(1, 8, 2, 1)` | 2048 | `4 / 1` | 131,072 | 49 | 676.0 | 10,816.3 | 2.56% | 2.56% | 16.78 | `12.3253 -> nan` | Baseline (`r4`) |
| `r4` `jax_v5p16_fsdp16_ep1_s2048_mb4` *(Eager Attn)* | `(1, 16, 1, 1)` | 2048 | `4 / 1` | 131,072 | 49 | 674.2 | 10,787.5 | 2.55% | 2.55% | 16.78 | `12.3253 -> nan` | Baseline (`r4`) |

---

## 4. Deploying from `gs://YOUR_GCS_BUCKET/flint-jax-training` on `tpu-v5p-cluster`

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

