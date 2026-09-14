---
name: benchmark-kernel
description: Guide for benchmarking FlashInfer+ROCm kernels on AMD Instinct (CDNA3/CDNA4)
---

# Benchmarking FlashInfer+ROCm Kernels

For a real driver script to copy, see
[`benchmarks/rocm/bench_fa2_prefill.py`](../../../benchmarks/rocm/bench_fa2_prefill.py) and [`benchmarks/rocm/bench_aiter_prefill.py`](../../../benchmarks/rocm/bench_aiter_prefill.py)
For the in-repo profiler wrapper, see [`profiler/rocm/rocm_profiler.py`](../../../profiler/rocm/rocm_profiler.py).

## Timing method matrix

| Method | When | How |
| --- | --- | --- |
| `flashinfer.testing.bench_gpu_time` | Quick in-loop check (kernels ≳ 50 µs) | Falls through to PyTorch `torch.cuda.Event` (HIP events under ROCm) automatically. |
| `rocm_profiler` (`RocmProfiler`) | Anything you intend to optimize | Two-phase: in-process median timing, then re-execs the same script under `rocprofv3` (sentinel: `_ROCM_PROFILER_INTERNAL`) for hardware counters. Produces roofline PNG. |
| `rocprofv3` directly | Full control over counter set | `rocprofv3 --stats --kernel-trace -- python script.py`; or `-i pmc.txt` for custom counters. |
| `omnitrace` | Host + device timeline when Python overhead is suspect | Installed separately. |

## Non-obvious gotchas

- **CUPTI is NVIDIA-only — `enable_cupti=True` on ROCm warns and falls back.** [`flashinfer/testing/utils.py:1010`](../../../flashinfer/testing/utils.py) routes through `bench_gpu_time_with_cupti`, which `try/except`s the `cupti` import, emits a `UserWarning`, and reverts to CUDA/HIP event timing. No functional benefit on ROCm; just leave `enable_cupti=False` (the default) so `bench_gpu_time` uses `torch.cuda.Event` (HIP events) directly without the warning.
- **AITER backend constraints, accurately:**
  - Explicit `backend="aiter"` + `kv_layout != "NHD"` → `ValueError`. Grep [`rocm/prefill.py`](../../../flashinfer/rocm/prefill.py) for `only supports kv_layout`; single prefill raises at the call, both batch wrappers at `plan()`. Not raised by auto-selection — that path silently falls back to `fa2`.
  - Explicit `backend="aiter"` on non-gfx942/gfx950 → `RuntimeError`.
  - `amd-aiter` not importable → `ImportError`.
  - **"Native" page sizes** (no flat-gather): `_aiter_native_page_sizes()` in [`rocm/prefill.py`](../../../flashinfer/rocm/prefill.py) returns `{1, 16, 1024}` for any AITER >= 0.1.10; the `{16, 1024}` arm is the fallback for when `amd-aiter` metadata is unreadable, not dead code. What is *routed* native is narrower still: `_aiter_paged_route_page_sizes()` gives fp8 all three and fp16/bf16 only `{1024}`, because the gather measured equal or faster elsewhere. **Non-native page sizes are NOT rejected** -- they flat-gather.
  - Measured on amd-aiter 0.1.21.post2, gfx950: native paged prefill at `page_size=1024` runs (62 TFLOPS at `kv=1024`). Beyond `kv=1024` it hits a GPU memory fault in `FmhaBatchPrefillWithPagedKVCacheKernel`, reproducible on 0.1.20 as well -- so bound the paged sweep at `kv=1024` rather than reading the fault as a regression.
  - Auto-selection (no explicit `backend=`) silently falls back to `fa2` for any of: `kv_layout != "NHD"`, custom mask, dtype not in `{fp16, bf16}`, `dtype_q != dtype_kv`, `head_dim_qk != head_dim_vo`, `pos_encoding_mode != "NONE"`, or `amd-aiter` not importable. See `_auto_select_prefill_backend()` in [`rocm/prefill.py`](../../../flashinfer/rocm/prefill.py) for the authoritative list; it returns `(backend, reason)`, and the reason names the constraint that forced the fallback.
- **Always verify numerical parity before trusting perf numbers.** Compare default-HIP vs AITER outputs with `torch.testing.assert_close(rtol=1e-2, atol=1e-2)` for BF16/FP16 first.
- **`gcnArchName` is the unambiguous arch marker.** Device strings show `cuda:0` on AMD too. Record `torch.cuda.get_device_properties(0).gcnArchName` and `torch.version.hip` alongside every number — a `gfx942` / ROCm 7.2 result is not comparable to a `gfx950` / ROCm 7.0.2 result.

## What can actually be benchmarked on ROCm

Only the APIs in the `IS_HIP` branch of [`flashinfer/__init__.py`](../../../flashinfer/__init__.py) are callable — which is more than it used to be: MLA, cascade, POD, block-sparse and AITER MoE are all exported now. The generated support matrix in [`README.md`](../../../README.md) is the list; **not** available are FP4, cuDNN backends, upstream's CUTLASS MoE, and everything else `flashinfer/rocm/__init__.py` gates.

`auto` tries AITER first for single prefill, batch prefill (paged + ragged), decode and MLA, and AITER is the only backend for MoE. It stays native for norm, rope, activation and paged append — native measured faster there, so reaching AITER needs an explicit `backend="aiter"`. The per-op column of the README matrix is authoritative; it is generated from `flashinfer/rocm/arch_caps.py`.

Two families of driver live in [`benchmarks/rocm/`](../../../benchmarks/rocm/): the `rocm_profiler` ones that answer "where is this kernel on the roofline", and the standalone A/B ones (`bench_norm.py`, `bench_block_sparse_attention.py`, `bench_mixed_attention.py`) that answer "which arm should I use". Copy whichever matches the question.

## `rocm_profiler` counter presets

Pass via `RocmProfiler(counters=...)` or `--counters` on the driver script.

| Preset | What it shows | Use for |
| --- | --- | --- |
| `roofline` (default) | `FetchSize`, `WriteSize`, MFMA ops, TCC DRAM requests | "Am I compute- or memory-bound?" |
| `compute` | MFMA ops + cycle counters | Matrix-core throughput |
| `memory` | L2 + DRAM breakdown | L2 hit-rate, HBM traffic |
| `occupancy` | `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_INSTS_LDS` | Wavefront density |
| `stall` | `SQ_WAIT_INST_VMEM`, `SQ_WAIT_INST_LDS` | Diagnose memory stalls |
| `basic` | `FetchSize` / `WriteSize` | Minimal baseline |

Or pass a path to a `rocprofv3`-native YAML for a custom counter set.

Driver script flags: `--timing-only` (skip rocprofv3), `--skip-roofline`, `--replot` (regen PNG from existing CSVs, no GPU), `--list-presets`.

Output (under `benchmarks/rocm/`, gitignored):

```text
<label>_timing.csv             # median + std per config
<label>_counter_collection.csv # raw counters
<label>_roofline.png           # only for counters=roofline
```

## Reproducibility checklist

1. **Warm up.** `dry_run_iters >= 5`; raise to 10–20 if std is high. First call includes JIT compile.
2. **Pin clocks** for sub-100-µs kernels:

   ```bash
   rocm-smi --showclocks
   sudo rocm-smi --setsclk 7
   sudo rocm-smi --setmclk 3
   ```

3. **Record arch + ROCm version** in the log: `print(props.name, props.gcnArchName, torch.version.hip)`.
4. **Isolate the GPU:** `HIP_VISIBLE_DEVICES=N` (or `ROCR_VISIBLE_DEVICES=N`, one layer deeper).

## Troubleshooting `rocm_profiler`

- **Empty `_counter_collection.csv`:** `kernel_name_regex` doesn't match the mangled name. Run `rocprofv3 --stats --kernel-trace -- python my_bench.py` first and copy the prefix from `*_kernel_stats.csv`.
- **Hang or no output:** confirm `which rocprofv3` is on `PATH`; the wrapper uses script `print()` output as a heartbeat — make sure the `if __name__ == "__main__":` block prints something.
- **Use `--timing-only` first** to verify the kernel path works before involving `rocprofv3`.

## Before tuning: check whether a library already wins

For any op, enumerate the ROCm libraries that already implement it — AITER, CK /
CK-Tile, hipBLASLt, rocBLAS, MIOpen, Triton — and benchmark each against the
in-tree `hip`/`fa2`/`native` kernel before writing or tuning one. Route `auto` to
whichever wins, on evidence, per arch.

Both outcomes are on the record here, so assume neither: AITER prefill beats fa2
by 3.67x (gfx942) and 4.51x (gfx950), while the in-tree HIP kernel beats AITER on
`rmsnorm`/`fused_add_rmsnorm` (1.6-1.8x) and on `append_paged_kv_cache`.

Three things that make this measurement wrong more often than the stopwatch:

- **Contract before speed.** A library can be faster and still unusable. CK-Tile's
  `layernorm2d` reads this API's fp32 `gamma`/`beta` as the input dtype and returns
  garbage with no error — it was rejected on contract, not on time.
- **Name the arm.** A library may dispatch internally between its own backends, so
  "AITER" is not one number. Say which module and codegen path ran.
- **Both boards.** gfx942 and gfx950 have diverged; a win on one is not a win.

Record the outcome as a `Capability` row in `flashinfer/rocm/arch_caps.py` with an
`evidence=` string naming board, versions and date, so the next person inherits the
measurement instead of repeating it.

## External tuning references

When optimizing a CDNA kernel, consult these in order:

- **Composable Kernel (CK)** — ground truth for LDS layout, `sched_group_barrier`
  ratios, and tiling on CDNA. For a specific hdim/dtype combination, read
  `qr_ks_vs.hpp` in [CK](https://github.com/ROCmSoftwarePlatform/composable_kernel)
  first.
- **AITER** — performance reference for fp16/hdim=256 attention on gfx942.
  [AITER repo](https://github.com/ROCm/aiter)
- **HipKittens** (arxiv 2511.08083) — producer/consumer patterns underperform on
  CDNA; **4-wave interleave** is the recommended approach instead.
