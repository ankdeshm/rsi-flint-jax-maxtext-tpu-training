# `results/` Guide — `RecursiveSI/flint` Pure JAX / Flax & MaxText TPU Training

| Folder / File | Slice & Pass | Key Results | Summary JSON |
|---|---|---|---|
| [`v5p32_r8/`](./v5p32_r8/) | **TPU `v5p-32` (`2x4x4`) `r8`** | **`33.72%` Steady / `33.76%` Peak MFU @ `16K` (`jax_v5p32_r8_05`), `30.20%` @ `8K` (`r8_03`)** | [`jax_v5p32_r8_sweep_summary.json`](./v5p32_r8/jax_v5p32_r8_sweep_summary.json) |
| [`r8/`](./r8/) | **TPU `v5p-16` (`2x2x4`) `r8`** | **`31.91%` Steady / `31.97%` Peak MFU @ `16K` (`jax_v5p16_r8_05`), `28.47%` @ `8K` (`r8_03`)** | [`jax_v5p16_r8_sweep_summary.json`](./r8/jax_v5p16_r8_sweep_summary.json) |
| [`v7x16_r8/`](./v7x16_r8/) | **TPU `v7x-16` (`2x2x4`) `r8`** | **`303,251` Steady / `303,394` Peak tok/s (`14.28%` MFU, `jax_v7x16_r8_05`)** | [`jax_v7x16_r8_sweep_summary.json`](./v7x16_r8/jax_v7x16_r8_sweep_summary.json) |
| [`v5p32_r7/`](./v5p32_r7/) | TPU `v5p-32` (`2x4x4`) `r7` | `28.31%` MFU @ `4K` (`r7_06`) & **`228,313` tok/s (`27.02%` MFU) @ `2K` (`r7_03`)** | [`jax_v5p32_r7_sweep_summary.json`](./v5p32_r7/jax_v5p32_r7_sweep_summary.json) |
| [`r7/`](./r7/) | TPU `v5p-16` (`2x2x4`) `r7` | `26.28%` MFU @ `4K` (`jax_r7_06`) & `25.11%` @ `2K` `12.58M` tok/step (`jax_r7_02`) | [`jax_v5p16_r7_sweep_summary.json`](./r7/jax_v5p16_r7_sweep_summary.json) |
| [`v5p32_r6/`](./v5p32_r6/) | TPU `v5p-32` (`2x4x4`) `r6` | **`13.78%` / `13.94%` MFU `FSDP=8 × EP=4` (`jax_v5p32_06`)** & `8.29 GiB` `FSDP=32` (`01`) | [`jax_v5p32_r6_sweep_summary.json`](./v5p32_r6/jax_v5p32_r6_sweep_summary.json) |
| [`r6/`](./r6/) & [`r5/`](./r5/) | TPU `v5p-16` (`2x2x4`) `r5–r6` | `SplashAttention sa=1024` + fused `ragged_dot` progression (`10.83% -> 25.13%` @ `2K`) | [`r6 summary`](./r6/jax_v5p16_r6_sweep_summary.json) · [`r5 summary`](./r5/jax_v5p16_opt_sweep_summary.json) |
