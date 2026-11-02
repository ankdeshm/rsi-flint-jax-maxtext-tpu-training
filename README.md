# `RecursiveSI/flint` (`39.8B` MoE) JAX / MaxText Training on TPU `v5p` & `v7x`

Pure JAX / Flax and [MaxText](https://github.com/AI-Hypercomputer/maxtext) BF16 pre-training and continual pre-training stack for **[Recursive Flint (`RecursiveSI/flint`)](https://huggingface.co/RecursiveSI/flint)** validated across **TPU `v5p-32` (`2x4x4`, 32 chips)**, **TPU `v5p-16` (`2x2x4`, 16 chips)**, and **TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores)** (`us-central1`, project `YOUR_GCP_PROJECT_ID`).

---

## Key Results & Script Reference

> Each row links to the JAX/Pallas implementation (`#line`), Kubernetes job manifest, and per-step JSON output in [`results/`](./results/README.md):

| Configuration | Baseline MFU → Optimized MFU (`Throughput`) | Run ID | Code & Kernel Anchors | K8s Manifest | Result Files | Key Changes Over `2.55%` `r4` Eager Baseline |
|---|---|---|---|---|---|---|
| **TPU `v5p-32` (`16K` Context)** | `2.55%` (`r4` `v5p-16`) → **`33.72%` / `33.76%`** (`181,763` tok/s) | `jax_v5p32_r8_05` | [`splash_attn#L159`](./flint_jax/model.py#L159-L240) · [`ragged_moe#L417`](./flint_jax/model.py#L417-L560) · [`mesh#L88`](./flint_jax/train.py#L88-L98) | [`flint-v5p-2x4x4-bench-r8.yaml`](./k8s/flint-v5p-2x4x4-bench-r8.yaml) | [`JSON`](./results/v5p32_r8/jax_v5p32_r8_05_dp2_fsdp16_s16384_mb12_ga2_sa1024.json) · [`Sweep`](./results/v5p32_r8/jax_v5p32_r8_sweep_summary.json) | Zero-NaN sink masking + Pallas `SplashAttention(sa=1024)` + fused W1/W3 `ragged_dot` + 256-tok scan CE + torus-aligned `HSDP 2×16` @ `16K` (`14.28 GiB` flat HBM) |
| **TPU `v5p-16` (`16K` Context)** | `2.55%` → **`31.91%` / `31.97%`** (`85,998` tok/s) | `jax_v5p16_r8_05` | [`FlintForCausalLM#L776`](./flint_jax/model.py#L776-L885) · [`train_step#L163`](./flint_jax/train.py#L163-L227) | [`flint-v5p-2x2x4-bench-r8.yaml`](./k8s/flint-v5p-2x2x4-bench-r8.yaml) | [`JSON`](./results/r8/jax_v5p16_r8_05_fsdp16_s16384_mb12_ga2_sa1024.json) · [`Sweep`](./results/r8/jax_v5p16_r8_sweep_summary.json) | `r8` stack on `16`-chip `FSDP=16` (`mb=12, ga=2`, `6.29M` tok/step, `14.34 GiB` flat HBM) |
| **TPU `v5p-32` (`2K` High Tok/s)** | `21.82%` (`FSDP=32`) → **`27.02%` / `27.03%`** (**`228,313` tok/s**) | `jax_v5p32_r7_03` | [`chunked_ce#L734`](./flint_jax/model.py#L734-L774) · [`train_step#L163`](./flint_jax/train.py#L163-L227) | [`flint-v5p-2x4x4-bench-r7.yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) | [`JSON`](./results/v5p32_r7/jax_v5p32_r7_03_dp2_fsdp16_s2048_mb96_ga4_sa1024.json) · [`Sweep`](./results/v5p32_r7/jax_v5p32_r7_sweep_summary.json) | `HSDP 2×16` (`4×4` torus plane) + `mb=96, ga=4` (`25.17M` tok/step) at `15.35 GiB` peak HBM |
| **TPU `v7x-16` (`2K` Context)** | `11.80%` (`FSDP=32`) → **`14.28%` / `14.29%`** (**`303,251` tok/s**) | `jax_v7x16_r8_05` | [`train.py#L546`](./flint_jax/train.py#L546-L620) | [`flint-v7x-2x2x4-bench-r8.yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) | [`JSON`](./results/v7x16_r8/jax_v7x16_r8_05_dp2_fsdp16_s2048_mb64_ga2_sa1024.json) · [`Sweep`](./results/v7x16_r8/jax_v7x16_r8_sweep_summary.json) | `--xla_tpu_scheduler_percent_shared_memory_limit=45` (`109.1 GiB OOM -> 15.24 GiB` HBM) + `HSDP 2×16`, `mb=64, ga=2` |
| **Expert Parallelism (`EP=4` `v5p-32`)** | `3.66%` (`r4` `EP=4`) → **`13.78%` / `13.94%`** (`116,405` tok/s) | `jax_v5p32_06` | [`_spmd_ragged_moe#L417`](./flint_jax/model.py#L417-L560) | [`flint-v5p-2x4x4-bench-r6.yaml`](./k8s/flint-v5p-2x4x4-bench-r6.yaml) | [`JSON`](./results/v5p32_r6/jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024.json) · [`Sweep`](./results/v5p32_r6/jax_v5p32_r6_sweep_summary.json) | Capacity-compacted NaN-free `FSDP=8 × EP=4` `ragged_dot` (`8.44 GiB` live HBM) |

---

## 1. Architecture & TPU Kernel Mapping

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
- [`flint_jax/model.py`](./flint_jax/model.py): JAX/Flax implementation of `FlintForCausalLM` with 4D SPMD partitioning (`data`, `fsdp`, `expert`, `tensor`), Pallas `SplashAttention`, and fused `jax.lax.ragged_dot` MoE execution.
- [`flint_jax/checkpoint_loader.py`](./flint_jax/checkpoint_loader.py): Streams the 15 Hugging Face safetensors shards (`model-00001-of-00015.safetensors` .. `model-00015-of-00015.safetensors`) into sharded TPU HBM arrays.
- [`flint_jax/train.py`](./flint_jax/train.py): Multi-host JAX BF16 training loop with Optax AdamW (`b1=0.9, b2=0.95, wd=0.1`), `jax.lax.scan` microbatch gradient accumulation, 256-token chunked cross-entropy loss, global norm clipping, and MoE auxiliary load-balancing loss across `v5p-16`, `v5p-32`, and `v7x-16`.
- [`maxtext/flint_40b_a2_5b_v5p.yml`](./maxtext/flint_40b_a2_5b_v5p.yml): MaxText configuration file using TPU v5p Megablox (`megablox: true`, `sparse_matmul: true`) and XLA compiler flags from [`AI-Hypercomputer/tpu-recipes/tree/main/training/v5p`](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p).
- [`k8s/`](./k8s/): Kubernetes Indexed `Job` + Headless `Service` manifests targeting `v5p-16` (`2x2x4`), `v5p-32` (`2x4x4`), and `v7x-16` (`2x2x4`).
- [`Dockerfile`](./Dockerfile): Builds the training image on top of `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/maxtext/maxtext:latest`.

---

## 3. Benchmark Results & Optimization Summary

All per-step JSON summaries are stored in [`results/`](./results/) and persisted to `gs://YOUR_GCS_BUCKET/flint-jax-training/results_{r4,r5,r6,r7,r8,v5p32_r6,v5p32_r7,v5p32_r8,v7x16_r7,v7x16_r8}/`.

### 3.1 Optimization Progression (`2.55%` `r4` Baseline to `33.76%` MFU)

1. **Zero-NaN Custom Softmax-with-Learned-Sinks Backward (`softmax_with_sinks`):**
   - Naive `jnp.where(mask, scores, -inf)` produces `0 * (-inf) = NaN` in XLA's backward pass on masked sliding-window (`W=513`) positions when combined with learned sink denominators (`loss -> NaN` on Step 2 in `r4`). Masking pre-exp logits cleanly and zeroing masked probabilities post-division eliminated gradient NaNs across `r5`/`r6`/`r7`/`r8` mesh topologies on `v5p` and `v7x`.
2. **TPU Pallas `SplashAttention` (`jax.experimental.pallas.ops.tpu.splash_attention`) with Learned Sinks (`softmax_offset`):**
   - Replaced eager `[B, 32, S, S]` materialized attention tensors with fused block-sparse Pallas `SplashAttention` (`CausalMask` on the global layers + `LocalMask(window_size=(512, 0))` on the sliding-window layers, passing `sinks=self.softmax_offset` into `SplashAttention` VMEM tiles at `sa_block=1024`).
3. **Single Fused Gate+Up (`W1/W3`) `jax.lax.ragged_dot` MoE Kernel + Shared BF16 RoPE:**
   - Concatenated `gate_proj` and `up_proj` into a single `[128, D, 2 * moe_inter]` weight tensor (`("expert", "fsdp", "tensor")`), reducing 47-layer MoE `ragged_dot` kernel launches and token gather/scatter passes from `94` to `47` forward grouped matmuls per microbatch, and precomputed `BF16` rotary `(cos, sin)` once per microbatch on `[1, seq_len]`.
4. **256-Token Chunked `jax.lax.scan` Cross-Entropy Loss:**
   - Replaced full-vocabulary `[B * S, 150,144]` FP32 logit materialization (`~39.4 GiB` at `mb=32`) with a 256-token `jax.checkpoint`-wrapped `jax.lax.scan` loop (`<0.6 GiB` peak logit buffer).
5. **Topology-Aware `mesh_utils.create_device_mesh` & Sequence Scaling (`v5p-32`, `v5p-16`, `v7x-16`):**
   - On `v5p-32` (`2x4x4` 3D Torus), mapping `HSDP (DP=2 × FSDP=16)` via `jax.experimental.mesh_utils.create_device_mesh((2, 16, 1, 1), allow_split_physical_axes=True)` places each `FSDP=16` group onto a bidirectional `4×4` 2D Torus plane. Holding tokens/chip/microbatch constant at `196,608` (`12.58M` tok/step across 32 chips, `~14.28 GiB` HBM) scales MFU from `seq=2048` (`27.02%` / `27.03%` MFU, `228,383` pk tok/s, `r7_03`) to `seq=4096` (`28.31%` / `28.33%` MFU, `r7_06`), `seq=8192` (`30.20%` / `30.23%` MFU, `r8_03`), and `seq=16384` (**`33.72%` Steady / `33.76%` Peak MFU**, `181,763` Steady / `181,959` Peak tok/s, `14.28 GiB` HBM, `jax_v5p32_r8_05`).
   - On `v5p-16` (`2x2x4`), scaling sequence length at `196,608` tok/chip/microbatch (`6.29M` tok/step) improves MFU from `seq=2048` (`25.13%` / `25.14%`) to `seq=4096` (`26.28%` / `26.31%` MFU, `jax_r7_06`), `seq=8192` (`28.47%` / `28.52%` MFU, `jax_v5p16_r8_03`), and `seq=16384` (**`31.91%` Steady / `31.97%` Peak MFU**, `85,998` Steady / `86,178` Peak tok/s, `14.34 GiB` HBM, `jax_v5p16_r8_05`).
   - On `v7x-16` (`2x2x4`, 32 TensorCores), adding `--xla_tpu_scheduler_percent_shared_memory_limit=45` reduces XLA latency-hiding HBM usage from `109.1 GiB` OOM to `15.24 GiB`, achieving **`303,251` Steady / `303,394` Peak tok/s (`14.28%` Steady / `14.29%` Peak MFU)** on `HSDP 2×16` (`jax_v7x16_r8_05`, `mb=64`), `289,592` tok/s (`13.64%` MFU) at `12.58M` tok/step (`r8_04`), and `250,573` tok/s (`11.80%` MFU, `8.14 GiB` live HBM) on `FSDP=32` (`r8_03`).

### 3.2 Benchmark Results Table (`v5p-32`, `v5p-16` & `v7x-16`)

| Slice | Pass / Experiment ID | Mesh (`dp, fsdp, ep, tp`) | `seq_len` | `mb` / `ga` | Global Batch (tok/step) | Steady rank tok/s | Cluster tok/s | Steady MFU (%) | Peak MFU (%) | Peak HBM (GiB) | Loss (`start -> end`) | Status | Manifest & Result |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| **`v5p-32`** | **`r8` `jax_v5p32_r8_05_dp2_fsdp16_s16384_mb12_ga2_sa1024`** | **`(2, 16, 1, 1)`** | **16384** | `12 / 2` | **12,582,912** | **5,680.1** | **181,763.2** *(181,959 pk)* | **33.72%** | **33.76%** | **14.28** *(14.28 cur)* | **`12.3279 -> 11.9991`** | `PASS` (`16K` `v5p-32`) | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r8.yaml) · [`JSON`](./results/v5p32_r8/jax_v5p32_r8_05_dp2_fsdp16_s16384_mb12_ga2_sa1024.json) |
| **`v5p-16`** | **`r8` `jax_v5p16_r8_05_fsdp16_s16384_mb12_ga2_sa1024`** | **`(1, 16, 1, 1)`** | **16384** | `12 / 2` | **6,291,456** | **5,374.9** | **85,997.9** *(86,178 pk)* | **31.91%** | **31.97%** | **14.34** *(14.34 cur)* | **`12.3277 -> 11.9988`** | `PASS` (`16K` `v5p-16`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r8.yaml) · [`JSON`](./results/r8/jax_v5p16_r8_05_fsdp16_s16384_mb12_ga2_sa1024.json) |
| **`v5p-32`** | **`r8` `jax_v5p32_r8_03_dp2_fsdp16_s8192_mb24_ga2_sa1024`** | **`(2, 16, 1, 1)`** | **8192** | `24 / 2` | **12,582,912** | **6,414.6** | **205,268.5** *(205,460 pk)* | **30.20%** | **30.23%** | **14.28** *(14.27 cur)* | **`12.3280 -> 11.9789`** | `PASS` (`8K` `v5p-32`) | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r8.yaml) · [`JSON`](./results/v5p32_r8/jax_v5p32_r8_03_dp2_fsdp16_s8192_mb24_ga2_sa1024.json) |
| **`v5p-16`** | **`r8` `jax_v5p16_r8_03_fsdp16_s8192_mb24_ga2_sa1024`** | **`(1, 16, 1, 1)`** | **8192** | `24 / 2` | **6,291,456** | **6,046.4** | **96,742.0** *(96,907 pk)* | **28.47%** | **28.52%** | **14.34** *(14.34 cur)* | **`12.3279 -> 11.9756`** | `PASS` (`8K` `v5p-16`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r8.yaml) · [`JSON`](./results/r8/jax_v5p16_r8_03_fsdp16_s8192_mb24_ga2_sa1024.json) |
| **`v5p-32`** | **`r7` `jax_v5p32_r7_06_dp2_fsdp16_s4096_mb48_ga2_sa1024`** | **`(2, 16, 1, 1)`** | **4096** | `48 / 2` | **12,582,912** | **6,914.4** | **221,259.3** *(221,424 pk)* | **28.31%** | **28.33%** | **15.35** *(14.28 cur)* | **`12.3284 -> 11.9756`** | `PASS` (`4K` `v5p-32`) | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) · [`JSON`](./results/v5p32_r7/jax_v5p32_r7_06_dp2_fsdp16_s4096_mb48_ga2_sa1024.json) |
| **`v5p-32`** | **`r7` `jax_v5p32_r7_03_dp2_fsdp16_s2048_mb96_ga4_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `96 / 4` | **25,165,824** | **7,134.8** | **228,312.7** *(228,383 pk)* | **27.02%** | **27.03%** | **15.35** *(14.28 cur)* | **`12.3285 -> 11.9763`** | `PASS` (`2K` `v5p-32`) | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) · [`JSON`](./results/v5p32_r7/jax_v5p32_r7_03_dp2_fsdp16_s2048_mb96_ga4_sa1024.json) |
| **`v5p-32`** | **`r7` `jax_v5p32_r7_02_dp2_fsdp16_s2048_mb96_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `96 / 2` | **12,582,912** | **7,124.4** | **227,981.9** *(228,115 pk)* | **26.98%** | **27.00%** | **15.35** *(14.28 cur)* | **`12.3284 -> 11.9741`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) · [`JSON`](./results/v5p32_r7/jax_v5p32_r7_02_dp2_fsdp16_s2048_mb96_ga2_sa1024.json) |
| **`v5p-32`** | **`r7` `jax_v5p32_r7_01_dp2_fsdp16_s2048_mb80_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `80 / 2` | **10,485,760** | **6,980.6** | **223,379.2** *(223,512 pk)* | **26.44%** | **26.45%** | **15.35** | **`12.3284 -> 11.9734`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) · [`JSON`](./results/v5p32_r7/jax_v5p32_r7_01_dp2_fsdp16_s2048_mb80_ga2_sa1024.json) |
| **`v5p-16`** | **`r7` `jax_r7_06_fsdp16_s4096_mb48_ga2_sa1024`** | **`(1, 16, 1, 1)`** | **4096** | `48 / 2` | **6,291,456** | **6,419.8** | **102,716.4** *(102,812 pk)* | **26.28%** | **26.31%** | **27.82** *(14.33 cur)* | **`12.3282 -> 11.9719`** | `PASS` (`4K` `v5p-16`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r7.yaml) · [`JSON`](./results/r7/jax_r7_06_fsdp16_s4096_mb48_ga2_sa1024.json) |
| **`v5p-32`** | **`r6` `jax_v5p32_02_dp2_fsdp16_s2048_mb64_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `64 / 2` | **8,388,608** | **6,799.5** | **217,583.3** *(217,670 pk)* | **25.75%** | **25.76%** | **14.82** | **`12.3285 -> 11.9723`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r6.yaml) · [`JSON`](./results/v5p32_r6/jax_v5p32_02_dp2_fsdp16_s2048_mb64_ga2_sa1024.json) |
| **`v5p-16`** | **`r6` `jax_r6_04_fsdp16_s2048_mb96_ga2_sa1024`** | `(1, 16, 1, 1)` | 2048 | `96 / 2` | **6,291,456** | **6,636.5** | **106,184.3** *(106,222 pk)* | **25.13%** | **25.14%** | **15.40** *(14.34 cur)* | **`12.3285 -> 11.9699`** | `PASS` (`2K` `v5p-16`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r6.yaml) · [`JSON`](./results/r6/jax_r6_04_fsdp16_s2048_mb96_ga2_sa1024.json) |
| **`v5p-16`** | **`r7` `jax_r7_02_fsdp16_s2048_mb96_ga4_sa1024`** | `(1, 16, 1, 1)` | 2048 | `96 / 4` | **12,582,912** | **6,630.5** | **106,087.8** *(106,169 pk)* | **25.11%** | **25.13%** | **14.35** *(14.34 cur)* | **`12.3284 -> 11.9741`** | `PASS` (`12.58M` tok/step) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r7.yaml) · [`JSON`](./results/r7/jax_r7_02_fsdp16_s2048_mb96_ga4_sa1024.json) |
| **`v5p-16`** | **`r6` `jax_r6_03_fsdp16_s2048_mb80_ga2_sa1024`** | `(1, 16, 1, 1)` | 2048 | `80 / 2` | **5,242,880** | **6,510.8** | **104,172.2** *(104,180 pk)* | **24.66%** | **24.66%** | **15.40** | **`12.3285 -> 11.9674`** | `PASS` (`9.67×` vs `r4`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r6.yaml) · [`JSON`](./results/r6/jax_r6_03_fsdp16_s2048_mb80_ga2_sa1024.json) |
| **`v5p-16`** | **`r6` `jax_r6_01_fsdp16_s2048_mb64_ga2_sa1024`** | `(1, 16, 1, 1)` | 2048 | `64 / 2` | **4,194,304** | **6,282.0** | **100,512.2** *(100,619 pk)* | **23.79%** | **23.82%** | **14.68** | **`12.3289 -> 11.9655`** | `PASS` (`9.34×` vs `r4`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r6.yaml) · [`JSON`](./results/r6/jax_r6_01_fsdp16_s2048_mb64_ga2_sa1024.json) |
| **`v5p-16`** | **`r5` `jax_opt06_fsdp16_ep1_s2048_mb48_ga2`** | `(1, 16, 1, 1)` | 2048 | `48 / 2` | **3,145,728** | **5,933.4** | **94,935.1** *(95,054 pk)* | **22.47%** | **22.50%** | **15.26** | **`12.3284 -> 11.9342`** | `PASS` (`8.82×` vs `r4`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r5.yaml) · [`JSON`](./results/r5/jax_opt06_fsdp16_ep1_s2048_mb48_ga2.json) |
| **`v5p-32`** | **`r6` `jax_v5p32_01_fsdp32_s2048_mb64_ga2_sa1024`** | `(1, 32, 1, 1)` | 2048 | `64 / 2` | **8,388,608** | **5,760.4** | **184,333.7** *(184,455 pk)* | **21.82%** | **21.83%** | **8.29** | **`12.3285 -> 11.9723`** | `PASS` (`8.29 GiB` HBM) | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r6.yaml) · [`JSON`](./results/v5p32_r6/jax_v5p32_01_fsdp32_s2048_mb64_ga2_sa1024.json) |
| **`v5p-16`** | **`r5` `jax_opt08_fsdp16_ep1_s4096_mb16_ga2`** | `(1, 16, 1, 1)` | **4096** | `16 / 2` | **2,097,152** | **5,139.3** | **82,228.7** *(82,313 pk)* | **21.04%** | **21.06%** | **16.72** | **`12.3284 -> 11.9480`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r5.yaml) · [`JSON`](./results/r5/jax_opt08_fsdp16_ep1_s4096_mb16_ga2.json) |
| **`v5p-16`** | **`r5` `jax_opt04_fsdp16_ep1_s2048_mb32_ga2`** | `(1, 16, 1, 1)` | 2048 | `32 / 2` | **2,097,152** | **5,357.1** | **85,713.7** *(85,762 pk)* | **20.29%** | **20.30%** | **15.26** | **`12.3283 -> 11.9035`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r5.yaml) · [`JSON`](./results/r5/jax_opt04_fsdp16_ep1_s2048_mb32_ga2.json) |
| **`v5p-16`** | **`r5` `jax_opt03_fsdp16_ep1_s2048_mb24_ga2`** | `(1, 16, 1, 1)` | 2048 | `24 / 2` | **1,572,864** | **4,910.6** | **78,569.3** *(78,691 pk)* | **18.60%** | **18.63%** | **14.66** | **`12.3286 -> 11.8774`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r5.yaml) · [`JSON`](./results/r5/jax_opt03_fsdp16_ep1_s2048_mb24_ga2.json) |
| **`v5p-16`** | **`r5` `jax_opt02_fsdp16_ep1_s2048_mb16_ga2`** | `(1, 16, 1, 1)` | 2048 | `16 / 2` | **1,048,576** | **4,173.7** | **66,779.5** *(66,960 pk)* | **15.81%** | **15.85%** | **14.34** | **`12.3294 -> 11.7538`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r5.yaml) · [`JSON`](./results/r5/jax_opt02_fsdp16_ep1_s2048_mb16_ga2.json) |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_05_dp2_fsdp16_s2048_mb64_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `64 / 2` | **8,388,608** | **9,476.6** | **303,250.6** *(303,394 pk)* | **14.28%** | **14.29%** | **15.24** *(14.66 cur)* | **`12.3285 -> 11.9723`** | `PASS` (`v7x-16` JAX) | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_05_dp2_fsdp16_s2048_mb64_ga2_sa1024.json) |
| **`v5p-32`** | **`r6` `jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024`** | **`(1, 8, 4, 1)`** | 2048 | `48 / 2` | **6,291,456** | **3,637.7** | **116,405.2** *(119,704 pk)* | **13.78%** | **13.94%** *(14.17% s2)* | **14.82** *(8.44 cur)* | **`12.3285 -> 11.9699`** | `PASS` (`EP=4` `v5p-32`) | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r6.yaml) · [`JSON`](./results/v5p32_r6/jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024.json) |
| **`v5p-16`** | **`r5` `jax_opt07_fsdp4_ep4_s2048_mb16_ga2`** | **`(1, 4, 4, 1)`** | 2048 | `16 / 2` | **1,048,576** | **3,604.4** | **57,669.8** *(57,864 pk)* | **13.65%** | **13.70%** | **16.72** | **`12.3294 -> 11.7554`** | `PASS` (`EP=4` `v5p-16`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r5.yaml) · [`JSON`](./results/r5/jax_opt07_fsdp4_ep4_s2048_mb16_ga2.json) |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_04_dp2_fsdp16_s2048_mb48_ga4_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `48 / 4` | **12,582,912** | **9,045.3** | **289,449.5** *(289,592 pk)* | **13.63%** | **13.64%** | **15.24** | **`12.3284 -> 11.9741`** | `PASS` (`12.58M` `v7x-16`) | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_04_dp2_fsdp16_s2048_mb48_ga4_sa1024.json) |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_02_dp2_fsdp16_s2048_mb48_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `48 / 2` | **6,291,456** | **9,038.2** | **289,223.3** *(289,260 pk)* | **13.62%** | **13.62%** | **15.23** | **`12.3285 -> 11.9698`** | `PASS` (`v7x-16` `HSDP 2×16`) | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_02_dp2_fsdp16_s2048_mb48_ga2_sa1024.json) |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_06_dp4_fsdp8_s2048_mb48_ga2_sa1024`** | **`(4, 8, 1, 1)`** | 2048 | `48 / 2` | **6,291,456** | **8,969.5** | **287,023.9** *(287,072 pk)* | **13.52%** | **13.52%** | **29.12** | **`12.3285 -> 11.9699`** | `PASS` (`v7x-16` `HSDP 4×8`) | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_06_dp4_fsdp8_s2048_mb48_ga2_sa1024.json) |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_01_dp2_fsdp16_s2048_mb32_ga2_sa1024`** | **`(2, 16, 1, 1)`** | 2048 | `32 / 2` | **4,194,304** | **8,304.4** | **265,739.3** *(266,385 pk)* | **12.51%** | **12.55%** | **15.04** | **`12.3289 -> 11.9663`** | `PASS` (`v7x-16`) | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_01_dp2_fsdp16_s2048_mb32_ga2_sa1024.json) |
| **`v7x-16`** | **`r8` `jax_v7x16_r8_03_fsdp32_s2048_mb48_ga2_sa1024`** | **`(1, 32, 1, 1)`** | 2048 | `48 / 2` | **6,291,456** | **7,828.4** | **250,509.4** *(250,573 pk)* | **11.80%** | **11.80%** | **15.23** *(8.14 cur)* | **`12.3285 -> 11.9698`** | `PASS` (`8.14 GiB` Live HBM) | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_03_fsdp32_s2048_mb48_ga2_sa1024.json) |
| **`v5p-16`** | **`r5` `jax_opt01_fsdp16_ep1_s2048_mb8_ga1`** | `(1, 16, 1, 1)` | 2048 | `8 / 1` | **262,144** | **2,858.8** | **45,741.5** *(45,805 pk)* | **10.83%** | **10.84%** | **14.32** | **`12.3281 -> 11.3851`** | `PASS` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r5.yaml) · [`JSON`](./results/r5/jax_opt01_fsdp16_ep1_s2048_mb8_ga1.json) |
| `v5p-16` | `r4` `jax_v5p16_fsdp4_ep4_s2048_mb4` *(Eager Attn)* | `(1, 4, 4, 1)` | 2048 | `4 / 1` | 131,072 | 965.6 | 15,449.4 | 3.66% | 3.81% | 16.78 | `12.3253 -> nan` | Baseline (`r4`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench.yaml) · [`JSON`](./results/jax_v5p16_fsdp4_ep4_s2048_mb4.json) |
| `v5p-16` | `r4` `jax_v5p16_fsdp16_ep1_s2048_mb4` *(Eager Attn)* | `(1, 16, 1, 1)` | 2048 | `4 / 1` | 131,072 | 674.2 | 10,787.5 | 2.55% | 2.55% | 16.78 | `12.3253 -> nan` | Baseline (`r4`) | [`yaml`](./k8s/flint-v5p-2x2x4-bench.yaml) · [`JSON`](./results/jax_v5p16_fsdp16_ep1_s2048_mb4.json) |

---

## 4. Reproducing the `r8` Sweeps (`v5p-32`, `v5p-16` & `v7x-16`)

```bash
# 1. Authenticate to your GKE TPU cluster
gcloud container clusters get-credentials tpu-v5p-cluster \
  --location=us-central1 \
  --project=YOUR_GCP_PROJECT_ID

# 2A. Run r8 sweep on TPU v5p-32 2x4x4 (33.72% / 33.76% MFU at 16K)
kubectl apply -f k8s/flint-v5p-2x4x4-bench-r8.yaml

# 2B. Run r8 sweep on TPU v5p-16 2x2x4 (31.91% / 31.97% MFU at 16K)
kubectl apply -f k8s/flint-v5p-2x2x4-bench-r8.yaml

# 2C. Run r8 sweep on TPU v7x-16 2x2x4 (303,251 tok/s, 14.28% MFU at 2K)
kubectl apply -f k8s/flint-v7x-2x2x4-bench-r8.yaml

# 2D. Continuous Pre-Training Job (2x2x4 or Multi-Slice JobSet)
kubectl apply -f k8s/flint-v5p-2x2x4-job.yaml
```
