# `RecursiveSI/flint` (`39.8B` MoE) JAX / MaxText Training on TPU `v5p` & `v7x`

Pure JAX / Flax and [MaxText](https://github.com/AI-Hypercomputer/maxtext) BF16 pre-training and continual pre-training stack for **[Recursive Flint (`RecursiveSI/flint`)](https://huggingface.co/RecursiveSI/flint)** validated across **TPU `v5p-32` (`2x4x4`, 32 chips)**, **TPU `v5p-16` (`2x2x4`, 16 chips)**, and **TPU `v7x-16` (`2x2x4`, 16 physical chips / 32 TensorCores)** (`us-central1`, project `YOUR_GCP_PROJECT_ID`).

---

## Key Results & Script Reference

> **Throughput & MFU Accounting Note:** All rows report **Steady-State Median Throughput (`tokens/sec/chip` and `Slice tokens/sec`)** first, followed by **Steady Causal-Masked MFU** (`6·H·D` lower-triangle on the 14 full causal layers + exact causal ramp on the 35 `W=513` sliding-window layers; `16.57` GF/tok @ `2K`, `17.33` @ `4K`, `18.77` @ `8K`, `21.60` @ `16K`) alongside the **PaLM / MaxText Full-Square Convention** (`12·H·D` unhalved; `17.38` GF/tok @ `2K` → `27.25` @ `16K`). All JAX sweeps use contiguous `[mb, seq_len]` sequences (zero cross-document SplashAttention block skipping). On dual-TensorCore TPU `v7x`, `1` physical chip = `2` JAX devices (`2,307` BF16 TFLOP/s/chip; `1,153.5` BF16 TFLOP/s/TensorCore).

| Configuration | **Steady `tok/s/chip`** (`Slice tok/s`) | **Steady Causal MFU** (`PaLM MFU`) | Run ID | Code & Kernel Anchors | K8s Manifest | Result Files | Key Changes Over Same-Slice Baseline |
|---|---:|---:|---|---|---|---|---|
| **TPU `v7x-16` (`2K` Context, 16 Chips / 32 TC)** | **`18,953.1` tok/s/chip** (`303,251` tok/s; `9,476.6`/TC) | **`13.61%`** (`14.28%` PaLM) | `jax_v7x16_r8_05` | [`train.py#L546`](./flint_jax/train.py#L546-L620) | [`flint-v7x-2x2x4-bench-r8.yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) | [`JSON`](./results/v7x16_r8/jax_v7x16_r8_05_dp2_fsdp16_s2048_mb64_ga2_sa1024.json) · [`Sweep`](./results/v7x16_r8/jax_v7x16_r8_sweep_summary.json) | `--xla_tpu_scheduler_percent_shared_memory_limit=45` (`109.1 GiB OOM -> 15.24 GiB` HBM) + `HSDP 2×16`, `mb=64, ga=2` (`+21.0%` vs `FSDP=32` `15,657` tok/s/chip) |
| **TPU `v5p-32` (`2K` Context, 32 Chips)** | **`7,134.8` tok/s/chip** (`228,313` tok/s) | **`25.75%`** (`27.02%` PaLM) | `jax_v5p32_r7_03` | [`chunked_ce#L734`](./flint_jax/model.py#L734-L774) · [`train_step#L163`](./flint_jax/train.py#L163-L227) | [`flint-v5p-2x4x4-bench-r7.yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) | [`JSON`](./results/v5p32_r7/jax_v5p32_r7_03_dp2_fsdp16_s2048_mb96_ga4_sa1024.json) · [`Sweep`](./results/v5p32_r7/jax_v5p32_r7_sweep_summary.json) | Torus-aligned `HSDP 2×16` (`4×4` plane) + `mb=96, ga=4` (`25.17M` tok/step, `15.35 GiB` HBM; `+23.9%` vs `FSDP=32` `5,760` tok/s/chip) |
| **TPU `v5p-32` (`16K` Context, 32 Chips)** | **`5,680.1` tok/s/chip** (`181,763` tok/s) | **`26.73%`** (`33.72%` PaLM) | `jax_v5p32_r8_05` | [`splash_attn#L159`](./flint_jax/model.py#L159-L240) · [`ragged_moe#L417`](./flint_jax/model.py#L417-L560) · [`mesh#L88`](./flint_jax/train.py#L88-L98) | [`flint-v5p-2x4x4-bench-r8.yaml`](./k8s/flint-v5p-2x4x4-bench-r8.yaml) | [`JSON`](./results/v5p32_r8/jax_v5p32_r8_05_dp2_fsdp16_s16384_mb12_ga2_sa1024.json) · [`Sweep`](./results/v5p32_r8/jax_v5p32_r8_sweep_summary.json) | Iso-token (`196,608` tok/chip/mb) `16K` scaling on `HSDP 2×16`: `-20.4%` tok/s/chip vs `2K` due to `+30.3%` causal FLOPs/tok (`14.28 GiB` HBM) |
| **TPU `v5p-16` (`2K` Context, 16 Chips)** | **`6,636.5` tok/s/chip** (`106,184` tok/s) | **`23.95%`** (`25.13%` PaLM) | `jax_r6_04` | [`FlintForCausalLM#L776`](./flint_jax/model.py#L776-L885) · [`train_step#L163`](./flint_jax/train.py#L163-L227) | [`flint-v5p-2x2x4-bench-r6.yaml`](./k8s/flint-v5p-2x2x4-bench-r6.yaml) | [`JSON`](./results/r6/jax_r6_04_fsdp16_s2048_mb96_ga2_sa1024.json) · [`Sweep`](./results/r6/jax_v5p16_r6_sweep_summary.json) | Zero-NaN sink masking + `SplashAttention(sa=1024)` + fused W1/W3 `ragged_dot` + 256-tok scan CE (`9.84×` vs `r4` eager baseline `674.2` tok/s/chip) |
| **TPU `v5p-16` (`16K` Context, 16 Chips)** | **`5,374.9` tok/s/chip** (`85,998` tok/s) | **`25.29%`** (`31.91%` PaLM) | `jax_v5p16_r8_05` | [`FlintForCausalLM#L776`](./flint_jax/model.py#L776-L885) · [`train_step#L163`](./flint_jax/train.py#L163-L227) | [`flint-v5p-2x2x4-bench-r8.yaml`](./k8s/flint-v5p-2x2x4-bench-r8.yaml) | [`JSON`](./results/r8/jax_v5p16_r8_05_fsdp16_s16384_mb12_ga2_sa1024.json) · [`Sweep`](./results/r8/jax_v5p16_r8_sweep_summary.json) | Iso-token (`196,608` tok/chip/mb) `16K` scaling on `FSDP=16`: `-19.0%` tok/s/chip vs `2K` (`14.34 GiB` HBM) |
| **Expert Parallelism (`EP=4` `v5p-32`, `2K`)** | **`3,637.7` tok/s/chip** (`116,405` tok/s) | **`13.13%`** (`13.78%` PaLM) | `jax_v5p32_06` | [`_spmd_ragged_moe#L417`](./flint_jax/model.py#L417-L560) | [`flint-v5p-2x4x4-bench-r6.yaml`](./k8s/flint-v5p-2x4x4-bench-r6.yaml) | [`JSON`](./results/v5p32_r6/jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024.json) · [`Sweep`](./results/v5p32_r6/jax_v5p32_r6_sweep_summary.json) | Capacity-compacted NaN-free `FSDP=8 × EP=4` `ragged_dot` (`8.44 GiB` live HBM; `3.77×` vs `r4` `EP=4` baseline `965.6` tok/s/chip) |

---

## 1. Architecture & TPU Kernel Mapping

Flint (`FlintForCausalLM`, 39.8B total / `2.52B` active non-embedding parameters) combines several custom architectural features matching [`RecursiveSI/flint/config.json`](https://huggingface.co/RecursiveSI/flint/blob/main/config.json) in [`flint_jax/model.py`](./flint_jax/model.py):

| Flint Component (`RecursiveSI/flint`) | Specification | JAX / MaxText TPU v5p & v7x Implementation |
| :--- | :--- | :--- |
| **Backbone** | `49` layers, `hidden_size=2048`, `vocab_size=150,144` (`tie_word_embeddings=false`) | `FlintForCausalLM` with per-layer `jax.checkpoint` (`nn.remat`) |
| **Hybrid Dense + MoE Layers** | Layers `0, 1`: Dense SwiGLU (`intermediate_size=6144`)<br>Layers `2..48` (`47` layers): Sparse MoE | `FlintMLP` on `mlp_only_layers=(0, 1)` + `FlintSparseMoeBlock` on layers `2..48` |
| **Sparse MoE + Shared Expert** | `256` routed experts (`moe_intermediate_size=512`), `top_k=6`, `router_score_function="sigmoid"`, `norm_topk_prob=True`, `router_aux_loss_coef=0.0005` + `1` ungated shared expert (`1024`) | Sigmoid top-6 routing + single fused W1/W3 dropless grouped matmul via **`jax.lax.ragged_dot`** (Megablox/MaxText kernel) sharded across `("expert", "fsdp", "tensor")` |
| **Hybrid Attention & Selective RoPE** | **`35` `sliding_attention` layers** (`sliding_window=513`, with RoPE $\theta=10000$) + **`14` `full_attention` layers** (`no_rope_layer_indices=(0,1,5,9,13,17,21,25,29,33,37,41,45,48)`, NoPE) | `FlintAttention` with GQA (`32` Q heads, `4` KV heads, `head_dim=128`), per-head QK `FlintRMSNorm`, selective `BF16` `apply_rope`, and Pallas `SplashAttention` (`sa_block=1024`) |
| **Learned Attention Sinks** | `softmax_type="learnable"` (`self_attn.softmax_offset` of shape `[32]`) | Zero-NaN `SplashAttention(..., sinks=self.softmax_offset)` + custom `softmax_with_sinks` fallback |

---

## 2. Context-Length Scaling Ablation (`2K → 4K → 8K → 16K` at Fixed Recipe & Iso-Microbatch Tokens)

Holding microbatch tokens per chip constant at `196,608` (`mb × seq_len = 196,608`) and keeping all kernel/mesh settings fixed isolates the true hardware cost of causal attention on the `14` global layers (while the `35` sliding-window `W=513` layers stay capped at `513`):

| Slice & Mesh | `seq_len` (`mb`, `ga`) | **Steady `tok/s/chip`** | Throughput vs `2K` | Slice `tok/s` | Causal GF/tok | **Steady Causal MFU** | PaLM GF/tok | Steady PaLM MFU | Peak HBM (GiB) | Run ID |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`v5p-32` (`HSDP 2×16`)** | **`2,048`** (`96`, `2`) | **`7,124.4`** | `1.00×` *(ref)* | `227,982` | `16.57` | **`25.72%`** | `17.38` | `26.98%` | `15.35` | `jax_v5p32_r7_02` |
| **`v5p-32` (`HSDP 2×16`)** | **`4,096`** (`48`, `2`) | **`6,914.4`** | `0.971×` *(`-2.9%`)* | `221,259` | `17.33` | **`26.10%`** | `18.79` | `28.31%` | `15.35` | `jax_v5p32_r7_06` |
| **`v5p-32` (`HSDP 2×16`)** | **`8,192`** (`24`, `2`) | **`6,414.6`** | `0.900×` *(`-10.0%`)* | `205,269` | `18.77` | **`26.22%`** | `21.61` | `30.20%` | `14.28` | `jax_v5p32_r8_03` |
| **`v5p-32` (`HSDP 2×16`)** | **`16,384`** (`12`, `2`) | **`5,680.1`** | `0.797×` *(`-20.3%`)* | `181,763` | `21.60` | **`26.73%`** | `27.25` | `33.72%` | `14.28` | `jax_v5p32_r8_05` |
| **`v5p-16` (`FSDP=16`)** | **`2,048`** (`96`, `2`) | **`6,636.5`** | `1.00×` *(ref)* | `106,184` | `16.57` | **`23.95%`** | `17.38` | `25.13%` | `15.40` | `jax_r6_04` |
| **`v5p-16` (`FSDP=16`)** | **`4,096`** (`48`, `2`) | **`6,419.8`** | `0.967×` *(`-3.3%`)* | `102,716` | `17.33` | **`24.23%`** | `18.79` | `26.28%` | `27.82` | `jax_r7_06` |
| **`v5p-16` (`FSDP=16`)** | **`8,192`** (`24`, `2`) | **`6,046.4`** | `0.911×` *(`-8.9%`)* | `96,742` | `18.77` | **`24.72%`** | `21.61` | `28.47%` | `14.34` | `jax_v5p16_r8_03` |
| **`v5p-16` (`FSDP=16`)** | **`16,384`** (`12`, `2`) | **`5,374.9`** | `0.810×` *(`-19.0%`)* | `85,998` | `21.60` | **`25.29%`** | `27.25` | `31.91%` | `14.34` | `jax_v5p16_r8_05` |

---

## 3. Optimization Progression & Full Benchmark Table

### 3.1 Same-Context (`2K`) Kernel & Mesh Progression

1. **Zero-NaN Custom Softmax-with-Learned-Sinks Backward (`softmax_with_sinks`):**
   - Naive `jnp.where(mask, scores, -inf)` produces `0 * (-inf) = NaN` in XLA's backward pass on masked sliding-window (`W=513`) positions when combined with learned sink denominators (`loss -> NaN` on Step 2 in `r4`). Masking pre-exp logits cleanly and zeroing masked probabilities post-division eliminated gradient NaNs across all `r5`–`r8` runs.
2. **TPU Pallas `SplashAttention` (`jax.experimental.pallas.ops.tpu.splash_attention`) with Learned Sinks (`softmax_offset`):**
   - Replaced eager `[B, 32, S, S]` materialized attention tensors with block-sparse Pallas `SplashAttention` (`CausalMask` on the 14 global layers + `LocalMask(window_size=(512, 0))` on the 35 sliding-window layers, passing `sinks=self.softmax_offset` in VMEM at `sa_block=1024`).
3. **Single Fused Gate+Up (`W1/W3`) `jax.lax.ragged_dot` MoE Kernel + Shared BF16 RoPE:**
   - Concatenated `gate_proj` and `up_proj` into `[256, D, 2 * moe_inter]`, cutting 47-layer MoE `ragged_dot` kernel launches from `94` to `47` forward grouped matmuls per microbatch.
4. **256-Token Chunked `jax.lax.scan` Cross-Entropy Loss:**
   - Replaced full-vocabulary `[B * S, 150,144]` FP32 logit materialization (`~39.4 GiB` at `mb=32`) with a 256-token `jax.checkpoint`-wrapped `jax.lax.scan` loop (`<0.6 GiB` peak logit buffer).
5. **Topology-Aware `HSDP (2×16)` & Ironwood `v7x` Shared-Memory Scheduler Tuning:**
   - On `v5p-32` (`2x4x4` 3D Torus), `HSDP (DP=2 × FSDP=16)` places each `FSDP=16` shard group onto a single `4×4` 2D torus plane, improving `2K` throughput by **`+23.9%`** (`5,760.4 -> 7,134.8 tok/s/chip`).
   - On `v7x-16` (`2x2x4`, `16` physical chips / `32` TensorCores), `--xla_tpu_scheduler_percent_shared_memory_limit=45` eliminates `109.1 GiB` HBM scheduler OOMs (`15.24 GiB` peak HBM), reaching **`18,953.1 tok/s/chip`** (`9,476.6 tok/s/TensorCore`, `303,251 tok/s` slice).

### 3.2 Complete Benchmark Results (`v5p-32`, `v5p-16` & `v7x-16`)

> Note: Existing JSON artifacts in [`results/`](./results/) store `steady_mfu_pct` under the upstream PaLM full-square convention (`12·H·D`); both **Steady Causal MFU** and **Steady PaLM MFU** are shown below, while updated [`flint_jax/train.py`](./flint_jax/train.py) emits both fields explicitly.

| Slice | Pass / Experiment ID | Mesh (`dp, fsdp, ep, tp`) | `seq_len` | `mb` / `ga` | Global Batch (tok/step) | **Steady `tok/s/chip`** | Steady `tok/s/device` | Slice `tok/s` | **Steady Causal MFU** | Steady PaLM MFU | Peak HBM (GiB) | Loss (`start -> end`) | Manifest & Result |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **`v7x-16`** | `r8` `jax_v7x16_r8_05_dp2_fsdp16_s2048_mb64_ga2_sa1024` | `(2, 16, 1, 1)` | 2048 | `64 / 2` | 8,388,608 | **18,953.1** | 9,476.6 | **303,250.6** | **13.61%** | 14.28% | 15.24 | `12.3285 -> 11.9723` | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_05_dp2_fsdp16_s2048_mb64_ga2_sa1024.json) |
| **`v7x-16`** | `r8` `jax_v7x16_r8_04_dp2_fsdp16_s2048_mb48_ga4_sa1024` | `(2, 16, 1, 1)` | 2048 | `48 / 4` | 12,582,912 | **18,090.6** | 9,045.3 | **289,449.5** | **12.99%** | 13.63% | 15.24 | `12.3284 -> 11.9741` | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_04_dp2_fsdp16_s2048_mb48_ga4_sa1024.json) |
| **`v7x-16`** | `r8` `jax_v7x16_r8_02_dp2_fsdp16_s2048_mb48_ga2_sa1024` | `(2, 16, 1, 1)` | 2048 | `48 / 2` | 6,291,456 | **18,076.5** | 9,038.2 | **289,223.3** | **12.98%** | 13.62% | 15.23 | `12.3285 -> 11.9698` | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_02_dp2_fsdp16_s2048_mb48_ga2_sa1024.json) |
| **`v7x-16`** | `r8` `jax_v7x16_r8_06_dp4_fsdp8_s2048_mb48_ga2_sa1024` | `(4, 8, 1, 1)` | 2048 | `48 / 2` | 6,291,456 | **17,939.0** | 8,969.5 | **287,023.9** | **12.89%** | 13.52% | 29.12 | `12.3285 -> 11.9699` | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_06_dp4_fsdp8_s2048_mb48_ga2_sa1024.json) |
| **`v7x-16`** | `r8` `jax_v7x16_r8_01_dp2_fsdp16_s2048_mb32_ga2_sa1024` | `(2, 16, 1, 1)` | 2048 | `32 / 2` | 4,194,304 | **16,608.7** | 8,304.4 | **265,739.3** | **11.93%** | 12.51% | 15.04 | `12.3289 -> 11.9663` | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_01_dp2_fsdp16_s2048_mb32_ga2_sa1024.json) |
| **`v7x-16`** | `r8` `jax_v7x16_r8_03_fsdp32_s2048_mb48_ga2_sa1024` | `(1, 32, 1, 1)` | 2048 | `48 / 2` | 6,291,456 | **15,656.8** | 7,828.4 | **250,509.4** | **11.25%** | 11.80% | 15.23 | `12.3285 -> 11.9698` | [`yaml`](./k8s/flint-v7x-2x2x4-bench-r8.yaml) · [`JSON`](./results/v7x16_r8/jax_v7x16_r8_03_fsdp32_s2048_mb48_ga2_sa1024.json) |
| **`v5p-32`** | `r7` `jax_v5p32_r7_03_dp2_fsdp16_s2048_mb96_ga4_sa1024` | `(2, 16, 1, 1)` | 2048 | `96 / 4` | 25,165,824 | **7,134.8** | 7,134.8 | **228,312.7** | **25.75%** | 27.02% | 15.35 | `12.3285 -> 11.9763` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) · [`JSON`](./results/v5p32_r7/jax_v5p32_r7_03_dp2_fsdp16_s2048_mb96_ga4_sa1024.json) |
| **`v5p-32`** | `r7` `jax_v5p32_r7_02_dp2_fsdp16_s2048_mb96_ga2_sa1024` | `(2, 16, 1, 1)` | 2048 | `96 / 2` | 12,582,912 | **7,124.4** | 7,124.4 | **227,981.9** | **25.72%** | 26.98% | 15.35 | `12.3284 -> 11.9741` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) · [`JSON`](./results/v5p32_r7/jax_v5p32_r7_02_dp2_fsdp16_s2048_mb96_ga2_sa1024.json) |
| **`v5p-32`** | `r7` `jax_v5p32_r7_06_dp2_fsdp16_s4096_mb48_ga2_sa1024` | `(2, 16, 1, 1)` | 4096 | `48 / 2` | 12,582,912 | **6,914.4** | 6,914.4 | **221,259.3** | **26.10%** | 28.31% | 15.35 | `12.3284 -> 11.9756` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r7.yaml) · [`JSON`](./results/v5p32_r7/jax_v5p32_r7_06_dp2_fsdp16_s4096_mb48_ga2_sa1024.json) |
| **`v5p-32`** | `r8` `jax_v5p32_r8_03_dp2_fsdp16_s8192_mb24_ga2_sa1024` | `(2, 16, 1, 1)` | 8192 | `24 / 2` | 12,582,912 | **6,414.6** | 6,414.6 | **205,268.5** | **26.22%** | 30.20% | 14.28 | `12.3280 -> 11.9789` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r8.yaml) · [`JSON`](./results/v5p32_r8/jax_v5p32_r8_03_dp2_fsdp16_s8192_mb24_ga2_sa1024.json) |
| **`v5p-32`** | `r8` `jax_v5p32_r8_05_dp2_fsdp16_s16384_mb12_ga2_sa1024` | `(2, 16, 1, 1)` | 16384 | `12 / 2` | 12,582,912 | **5,680.1** | 5,680.1 | **181,763.2** | **26.73%** | 33.72% | 14.28 | `12.3279 -> 11.9991` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r8.yaml) · [`JSON`](./results/v5p32_r8/jax_v5p32_r8_05_dp2_fsdp16_s16384_mb12_ga2_sa1024.json) |
| **`v5p-32`** | `r6` `jax_v5p32_01_fsdp32_s2048_mb64_ga2_sa1024` | `(1, 32, 1, 1)` | 2048 | `64 / 2` | 8,388,608 | **5,760.4** | 5,760.4 | **184,333.7** | **20.80%** | 21.82% | 8.29 | `12.3285 -> 11.9723` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r6.yaml) · [`JSON`](./results/v5p32_r6/jax_v5p32_01_fsdp32_s2048_mb64_ga2_sa1024.json) |
| **`v5p-32`** | `r6` `jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024` | `(1, 8, 4, 1)` | 2048 | `48 / 2` | 6,291,456 | **3,637.7** | 3,637.7 | **116,405.2** | **13.13%** | 13.78% | 14.82 | `12.3285 -> 11.9699` | [`yaml`](./k8s/flint-v5p-2x4x4-bench-r6.yaml) · [`JSON`](./results/v5p32_r6/jax_v5p32_06_fsdp8_ep4_s2048_mb48_ga2_sa1024.json) |
| **`v5p-16`** | `r6` `jax_r6_04_fsdp16_s2048_mb96_ga2_sa1024` | `(1, 16, 1, 1)` | 2048 | `96 / 2` | 6,291,456 | **6,636.5** | 6,636.5 | **106,184.3** | **23.95%** | 25.13% | 15.40 | `12.3285 -> 11.9699` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r6.yaml) · [`JSON`](./results/r6/jax_r6_04_fsdp16_s2048_mb96_ga2_sa1024.json) |
| **`v5p-16`** | `r7` `jax_r7_06_fsdp16_s4096_mb48_ga2_sa1024` | `(1, 16, 1, 1)` | 4096 | `48 / 2` | 6,291,456 | **6,419.8** | 6,419.8 | **102,716.4** | **24.23%** | 26.28% | 27.82 | `12.3282 -> 11.9719` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r7.yaml) · [`JSON`](./results/r7/jax_r7_06_fsdp16_s4096_mb48_ga2_sa1024.json) |
| **`v5p-16`** | `r8` `jax_v5p16_r8_03_fsdp16_s8192_mb24_ga2_sa1024` | `(1, 16, 1, 1)` | 8192 | `24 / 2` | 6,291,456 | **6,046.4** | 6,046.4 | **96,742.0** | **24.72%** | 28.47% | 14.34 | `12.3279 -> 11.9756` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r8.yaml) · [`JSON`](./results/r8/jax_v5p16_r8_03_fsdp16_s8192_mb24_ga2_sa1024.json) |
| **`v5p-16`** | `r8` `jax_v5p16_r8_05_fsdp16_s16384_mb12_ga2_sa1024` | `(1, 16, 1, 1)` | 16384 | `12 / 2` | 6,291,456 | **5,374.9** | 5,374.9 | **85,997.9** | **25.29%** | 31.91% | 14.34 | `12.3277 -> 11.9988` | [`yaml`](./k8s/flint-v5p-2x2x4-bench-r8.yaml) · [`JSON`](./results/r8/jax_v5p16_r8_05_fsdp16_s16384_mb12_ga2_sa1024.json) |
| `v5p-16` | `r4` `jax_v5p16_fsdp4_ep4_s2048_mb4` *(Eager Attn)* | `(1, 4, 4, 1)` | 2048 | `4 / 1` | 131,072 | 965.6 | 965.6 | 15,449.4 | 3.49% | 3.66% | 16.78 | `12.3253 -> nan` | [`yaml`](./k8s/flint-v5p-2x2x4-bench.yaml) · [`JSON`](./results/jax_v5p16_fsdp4_ep4_s2048_mb4.json) |
| `v5p-16` | `r4` `jax_v5p16_fsdp16_ep1_s2048_mb4` *(Eager Attn)* | `(1, 16, 1, 1)` | 2048 | `4 / 1` | 131,072 | 674.2 | 674.2 | 10,787.5 | 2.43% | 2.55% | 16.78 | `12.3253 -> nan` | [`yaml`](./k8s/flint-v5p-2x2x4-bench.yaml) · [`JSON`](./results/jax_v5p16_fsdp16_ep1_s2048_mb4.json) |

---

## 4. Reproducing the `r8` Sweeps (`v5p-32`, `v5p-16` & `v7x-16`)

```bash
# 1. Authenticate to your GKE TPU cluster
gcloud container clusters get-credentials tpu-v5p-cluster \
  --location=us-central1 \
  --project=YOUR_GCP_PROJECT_ID

# 2A. Run r8 sweep on TPU v5p-32 2x4x4 (228,313 tok/s @ 2K; 181,763 tok/s @ 16K)
kubectl apply -f k8s/flint-v5p-2x4x4-bench-r8.yaml

# 2B. Run r8 sweep on TPU v5p-16 2x2x4 (106,184 tok/s @ 2K; 85,998 tok/s @ 16K)
kubectl apply -f k8s/flint-v5p-2x2x4-bench-r8.yaml

# 2C. Run r8 sweep on TPU v7x-16 2x2x4 (18,953 tok/s/chip = 303,251 tok/s @ 2K)
kubectl apply -f k8s/flint-v7x-2x2x4-bench-r8.yaml

# 2D. Continuous Pre-Training Job (2x2x4 or Multi-Slice JobSet)
kubectl apply -f k8s/flint-v5p-2x2x4-job.yaml
```
