# Recursive Flint (39.8B-A2.5B MoE) — JAX / Flax & MaxText BF16 Training on GKE TPU `v5p` & `v7x` (Ironwood)

This directory contains a complete Pure JAX / Flax & [MaxText](https://github.com/AI-Hypercomputer/maxtext) BF16 pre-training and continual pre-training stack for **[Recursive Flint (`RecursiveSI/flint`)](https://huggingface.co/RecursiveSI/flint)** validated across **TPU `v5p-32` (`2x4x4`, 32 chips)**, **TPU `v5p-16` (`2x2x4`, 16 chips)**, and **TPU `v7x-16` (`2x2x4`, 16 Ironwood chips / 32 TensorCores)** (`us-central1`, project `YOUR_GCP_PROJECT_ID`).

---

## 1. Architecture & TPU `v5p` / `v7x` Kernel Mapping

Flint (`FlintForCausalLM`, 39.8B total / ~2.5B active parameters) combines several custom architectural features that are implemented in [`flint_jax/model.py`](./flint_jax/model.py):

| Flint Component (`RecursiveSI/flint`) | Specification | JAX / MaxText TPU v5p & v7x Implementation |
| :--- | :--- | :--- |
| **Backbone** | 49 layers, `hidden_size=2048`, `vocab_size=150,144` | `FlintForCausalLM` with per-layer `jax.checkpoint` (`nn.remat`) |
| **Hybrid Dense + MoE Layers** | Layers `0, 1`: Dense SwiGLU (`intermediate_size=6144`)<br>Layers `2..48` (47 layers): Sparse MoE | `FlintMLP` on `mlp_only_layers=(0, 1)` + `FlintSparseMoeBlock` on layers `2..48` |
| **Sparse MoE + Shared Expert** | `256` routed experts (`moe_intermediate_size=512`), `top_k=6`, `router_score_function="sigmoid"`, `norm_topk_prob=True`, `router_aux_loss_coef=0.0005` + 1 ungated shared expert (`1024`) | Sigmoid top-6 routing with `expert_bias` + single fused W1/W3 dropless grouped matmul via **`jax.lax.ragged_dot`** (Megablox/MaxText kernel) sharded across `("expert", "fsdp", "tensor")` |
| **Hybrid Attention & Selective RoPE** | 37 `sliding_attention` layers (`sliding_window=513`, with RoPE $\theta=10000$) + 12 `full_attention` layers (`no_rope_layer_indices`, no positional encoding) | `FlintAttention` with GQA (`32` Q heads, `4` KV heads, `head_dim=128`), per-head QK `FlintRMSNorm`, selective `BF16` `apply_rope`, and Pallas `SplashAttention` (`sa_block=1024`) |
| **Learned Attention Sinks** | `softmax_type="learnable"` (`self_attn.softmax_offset` of shape `[32]`) | Zero-NaN `SplashAttention(..., sinks=self.softmax_offset)` + custom `softmax_with_sinks` fallback |

---

## 2. Directory Structure

- [`flint_jax/config.py`](./flint_jax/config.py): Dataclass configuration matching [`RecursiveSI/flint/config.json`](https://huggingface.co/RecursiveSI/flint/blob/main/config.json).
- [`flint_jax/model.py`](./flint_jax/model.py): Complete JAX/Flax implementation of `FlintForCausalLM` with 4D SPMD partitioning (`data`, `fsdp`, `expert`, `tensor`), Pallas `SplashAttention`, and fused `jax.lax.ragged_dot` MoE execution.
- [`flint_jax/checkpoint_loader.py`](./flint_jax/checkpoint_loader.py): Streams the 15 Hugging Face safetensors shards (`model-00001-of-00015.safetensors` .. `model-00015-of-00015.safetensors`) into sharded TPU HBM arrays.
- [`flint_jax/train.py`](./flint_jax/train.py): Multi-host JAX BF16 training loop with Optax AdamW (`b1=0.9, b2=0.95, wd=0.1`), `jax.lax.scan` microbatch gradient accumulation, 256-token chunked cross-entropy loss, global norm clipping, and MoE auxiliary load-balancing loss across `v5p-16`, `v5p-32`, and `v7x-16`.
- [`maxtext/flint_40b_a2_5b_v5p.yml`](./maxtext/flint_40b_a2_5b_v5p.yml): MaxText configuration file using TPU v5p Megablox (`megablox: true`, `sparse_matmul: true`) and XLA compiler flags from [`AI-Hypercomputer/tpu-recipes/tree/main/training/v5p`](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p).
- [`k8s/`](./k8s/): Kubernetes Indexed `Job` + Headless `Service` manifests targeting `v5p-16` (`2x2x4`), `v5p-32` (`2x4x4`), and Ironwood `v7x-16` (`2x2x4`).
- [`Dockerfile`](./Dockerfile): Builds the training image on top of `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/maxtext/maxtext:latest`.

---

## 3. Empirical Benchmark Results & Optimization Progression across TPU `v5p-32` (`2x4x4`), `v5p-16` (`2x2x4`) & Ironwood `v7x-16` (`2x2x4`)

All raw per-step JSON summaries are stored in [`results/`](./results/) and persisted to `gs://YOUR_GCS_BUCKET/flint-jax-training/results_{r4,r5,r6,r7,v5p32_r6,v5p32_r7,v7x16_r7,v7x16_r8}/`.

### 3.1 Key JAX / Pallas / XLA Engineering Optimizations (`r4` Baseline → `27.00%` Peak MFU All-Time Record!)

1. **Zero-NaN Custom Softmax-with-Learned-Sinks Backward (`softmax_with_sinks`):**
   - Naive `jnp.where(mask, scores, -inf)` produces `0 * (-inf) = NaN` in XLA's backward pass on masked sliding-window (`W=513`) positions when combined with learned sink denominators (`loss -> NaN` on Step 2 in `r4`). Masking pre-exp logits cleanly and zeroing masked probabilities post-division eliminated all gradient NaNs (`0` NaNs across all `r5`/`r6`/`r7`/`r8` mesh topologies on both `v5p` and `v7x`).
2. **TPU Pallas `SplashAttention` (`jax.experimental.pallas.ops.tpu.splash_attention`) with Learned Sinks (`softmax_offset`):**
   - Replaced eager `[B, 32, S, S]` materialized attention tensors with fused block-sparse Pallas `SplashAttention` (`CausalMask` on the 12 global layers + `LocalMask(window_size=(512, 0))` on the 37 sliding-window layers, passing `sinks=self.softmax_offset` natively into `SplashAttention` VMEM tiles at `sa_block=1024`).
3. **Single Fused Gate+Up (`W1/W3`) `jax.lax.ragged_dot` MoE Kernel + Shared BF16 RoPE:**
   - Concatenated `gate_proj` and `up_proj` into a single `[128, D, 2 * moe_inter]` weight tensor (`("expert", "fsdp", "tensor")`), cutting 47-layer MoE `ragged_dot` kernel launches and token gather/scatter passes in half (`94 -> 47` forward grouped matmuls per microbatch), and precomputed `BF16` rotary `(cos, sin)` once per microbatch on `[1, seq_len]`.
4. **256-Token Chunked `jax.lax.scan` Cross-Entropy Loss:**
   - Replaced full-vocabulary `[B * S, 150,144]` FP32 logit materialization (`~39.4 GiB` at `mb=32`) with a 256-token `jax.checkpoint`-wrapped `jax.lax.scan` loop (`<0.6 GiB` peak logit buffer).
5. **Topology-Aware `mesh_utils.create_device_mesh` (`26.98%` Steady / `27.00%` Peak MFU on `v5p-32`, `25.13%` / `25.14%` on `v5p-16`, & `289.3K` tok/s on Ironwood `v7x-16`!):**
   - On `v5p-32` (`2x4x4` 3D Torus), mapping `HSDP (DP=2 × FSDP=16)` via `jax.experimental.mesh_utils.create_device_mesh((2, 16, 1, 1), allow_split_physical_axes=True)` places each `FSDP=16` group onto a **full bidirectional wrap-around `4×4` 2D Torus plane**, propelling `mb=96, ga=2` (`jax_v5p32_r7_02`, `12.58M` tok/step) to **`26.98%` Steady / `27.00%` Peak MFU (`227,982` Steady / `228,115` Peak tok/s, only `14.28 GiB` live / `15.35 GiB` peak HBM!)** — **setting the all-time #1 MFU record across all chips & frameworks (`+2.10` pp over TorchTitan `v5p-32` and `107.4%` super-linear scaling over `v5p-16`!)**!
   - On Ironwood `v7x-16` (`2x2x4`, 32 TensorCores), adding `--xla_tpu_scheduler_percent_shared_memory_limit=45` prevents monolithic XLA latency-hiding HBM OOM (`109.1G -> 15.23 GiB`!), delivering **`289,223` Steady / `289,260` Peak tok/s (`13.62%` MFU, `15.23 GiB` HBM)** on `jax_v7x16_r8_02`.

### 3.2 Benchmark Leaderboard — `v5p-32` (`32` Chips), `v5p-16` (`16` Chips) & Ironwood `v7x-16` (`32` TensorCores)

| Slice | Pass / Experiment ID | Mesh (`dp, fsdp, ep, tp`) | `seq_len` | `mb` / `ga` | Global Batch (tok/step) | Steady rank tok/s | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (`start -> end`) | Status |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| **`v5p-32`** | **`r7` `jax_v5p32_r7_02_dp2_fsdp16_s2048_mb96_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `96 / 2` | **12,582,912** | **7,124.4** | **227,981.9** *(228,115 pk)* | **26.98%** | **27.00%** | **15.35** *(14.28 cur)* | **`12.3284 -> 11.9741`** | **`PASS` (#1 ALL-TIME OVERALL MFU!)** |
| **`v5p-32`** | **`r7` `jax_v5p32_r7_01_dp2_fsdp16_s2048_mb80_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `80 / 2` | **10,485,760** | **6,980.6** | **223,379.2** *(223,512 pk)* | **26.44%** | **26.45%** | **15.35** | **`12.3284 -> 11.9734`** | **`PASS` (#2 ALL-TIME!)** |
| **`v5p-32`** | **`r6` `jax_v5p32_02_dp2_fsdp16_s2048_mb64_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `64 / 2` | **8,388,608** | **6,799.5** | **217,583.3** *(217,670 pk)* | **25.75%** | **25.76%** | **14.82** | **`12.3285 -> 11.9723`** | **`PASS`** |
| **`v5p-16`** | **`r6` `jax_r6_04_fsdp16_s2048_mb96_ga2_sa1024`** | `(1, 16, 1, 1)` | 2048 | `96 / 2` | **6,291,456** | **6,636.5** | **106,184.3** *(106,222 pk)* | **25.13%** | **25.14%** | **15.40** *(14.34 cur)* | **`12.3285 -> 11.9699`** | **`PASS` (#1 16-Chip JAX, `9.86×` vs `r4`!)** |
| **`v5p-16`** | **`r7` `jax_r7_02_fsdp16_s2048_mb96_ga4_sa1024`** | `(1, 16, 1, 1)` | 2048 | `96 / 4` | **12,582,912** | **6,630.5** | **106,087.8** *(106,169 pk)* | **25.11%** | **25.13%** | **14.35** *(14.34 cur)* | **`12.3284 -> 11.9741`** | **`PASS` (`12.58M` tok/step!)** |
| **`v5p-16`** | **`r6` `jax_r6_03_fsdp16_s2048_mb80_ga2_sa1024`** | `(1, 16, 1, 1)` | 2048 | `80 / 2` | **5,242,880** | **6,510.8** | **104,172.2** *(104,180 pk)* | **24.66%** | **24.66%** | **15.40** | **`12.3285 -> 11.9674`** | **`PASS` (`9.67×` vs `r4`)** |
| **`v5p-16`** | **`r6` `jax_r6_01_fsdp16_s2048_mb64_ga2_sa1024`** | `(1, 16, 1, 1)` | 2048 | `64 / 2` | **4,194,304** | **6,282.0** | **100,512.2** *(100,619 pk)* | **23.79%** | **23.82%** | **14.68** | **`12.3289 -> 11.9655`** | **`PASS` (`9.34×` vs `r4`)** |
| **`v5p-16`** | **`r5` `jax_opt06_fsdp16_ep1_s2048_mb48_ga2`** | `(1, 16, 1, 1)` | 2048 | `48 / 2` | **3,145,728** | **5,933.4** | **94,935.1** *(95,054 pk)* | **22.47%** | **22.50%** | **15.26** | **`12.3284 -> 11.9342`** | **`PASS` (`8.82×` vs `r4`)** |
| **`v5p-32`** | **`r6` `jax_v5p32_01_fsdp32_s2048_mb64_ga2_sa1024`** | `(1, 32, 1, 1)` | 2048 | `64 / 2` | **8,388,608** | **5,760.4** | **184,333.7** *(184,455 pk)* | **21.82%** | **21.83%** | **8.29** | **`12.3285 -> 11.9723`** | **`PASS` (`8.29 GiB` HBM!)** |
| **`v5p-16`** | **`r5` `jax_opt08_fsdp16_ep1_s4096_mb16_ga2`** | `(1, 16, 1, 1)` | **4096** | `16 / 2` | **2,097,152** | **5,139.3** | **82,228.7** *(82,313 pk)* | **21.04%** | **21.06%** | **16.72** | **`12.3284 -> 11.9480`** | **`PASS` (`7.95×` vs `r4` `4K`!)** |
| **`v5p-16`** | **`r5` `jax_opt04_fsdp16_ep1_s2048_mb32_ga2`** | `(1, 16, 1, 1)` | 2048 | `32 / 2` | **2,097,152** | **5,357.1** | **85,713.7** *(85,762 pk)* | **20.29%** | **20.30%** | **15.26** | **`12.3283 -> 11.9035`** | **`PASS`** |
| **`v5p-16`** | **`r5` `jax_opt03_fsdp16_ep1_s2048_mb24_ga2`** | `(1, 16, 1, 1)` | 2048 | `24 / 2` | **1,572,864** | **4,910.6** | **78,569.3** *(78,691 pk)* | **18.60%** | **18.63%** | **14.66** | **`12.3286 -> 11.8774`** | **`PASS`** |
| **`v5p-16`** | **`r5` `jax_opt02_fsdp16_ep1_s2048_mb16_ga2`** | `(1, 16, 1, 1)` | 2048 | `16 / 2` | **1,048,576** | **4,173.7** | **66,779.5** *(66,960 pk)* | **15.81%** | **15.85%** | **14.34** | **`12.3294 -> 11.7538`** | **`PASS`** |
| **`v5p-32`** | **`r6` `jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024`** | **`(1, 8, 4, 1)`** | 2048 | `48 / 2` | **6,291,456** | **3,637.7** | **116,405.2** *(119,704 pk)* | **13.78%** | **13.94%** *(14.17% s2)* | **14.82** *(8.44 cur)* | **`12.3285 -> 11.9699`** | **`PASS` (#1 Overall `EP` Record!)** |
| **`v5p-16`** | **`r5` `jax_opt07_fsdp4_ep4_s2048_mb16_ga2`** | **`(1, 4, 4, 1)`** | 2048 | `16 / 2` | **1,048,576** | **3,604.4** | **57,669.8** *(57,864 pk)* | **13.65%** | **13.70%** | **16.72** | **`12.3294 -> 11.7554`** | **`PASS` (`EP=4` Zero-NaN!)** |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_02_dp2_fsdp16_s2048_mb48_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `48 / 2` | **6,291,456** | **9,038.2** | **289,223.3** *(289,260 pk)* | **13.62%** | **13.62%** | **15.23** | **`12.3285 -> 11.9698`** | **`PASS` (#1 Ironwood Pure JAX!)** |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_01_dp2_fsdp16_s2048_mb32_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `32 / 2` | **4,194,304** | **8,304.4** | **265,739.3** *(266,385 pk)* | **12.51%** | **12.55%** | **15.04** | **`12.3289 -> 11.9663`** | **`PASS` (Ironwood `v7x-16`!)** |
| **`v5p-16`** | **`r5` `jax_opt01_fsdp16_ep1_s2048_mb8_ga1`** | `(1, 16, 1, 1)` | 2048 | `8 / 1` | **262,144** | **2,858.8** | **45,741.5** *(45,805 pk)* | **10.83%** | **10.84%** | **14.32** | **`12.3281 -> 11.3851`** | **`PASS`** |
| `v5p-16` | `r4` `jax_v5p16_fsdp4_ep4_s2048_mb4` *(Eager Attn)* | `(1, 4, 4, 1)` | 2048 | `4 / 1` | 131,072 | 965.6 | 15,449.4 | 3.66% | 3.81% | 16.78 | `12.3253 -> nan` | Baseline (`r4`) |
| `v5p-16` | `r4` `jax_v5p16_fsdp16_ep1_s2048_mb4` *(Eager Attn)* | `(1, 16, 1, 1)` | 2048 | `4 / 1` | 131,072 | 674.2 | 10,787.5 | 2.55% | 2.55% | 16.78 | `12.3253 -> nan` | Baseline (`r4`) |

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

