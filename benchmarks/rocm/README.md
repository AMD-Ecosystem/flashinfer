# Benchmarking on ROCm (gfx942 / gfx950)

On AMD the supported-backend table in
[`../README.md`](../README.md#flashinfer_benchmarkpy-routine--backend-support-matrix)
does not apply: gfx942 and gfx950 report compute capability 9.4 and 9.5, which
match no entry in it. The ROCm branch keys off the GPU architecture instead and
derives the backend list from
[`flashinfer/rocm/arch_caps.py`](../../flashinfer/rocm/arch_caps.py), so it
tracks the arch-support matrix rather than restating it.

The attention, norm, rope and sampling routines are available. `gemm` and `moe`
are not built on ROCm, and `BatchMLAPagedAttentionWrapper` is not wired up yet;
requesting any of them reports "not supported" rather than failing at import.

| Routines | gfx942 | gfx950 |
|---|---|---|
| **BatchDecodeWithPagedKVCacheWrapper** | fa2, auto | fa2, auto |
| **BatchPrefillWithPagedKVCacheWrapper** | fa2, auto | fa2, auto |
| **BatchPrefillWithRaggedKVCacheWrapper** | fa2, auto | fa2, auto |
| **norm** — `rmsnorm`, `fused_add_rmsnorm`, `gemma_rmsnorm`, `gemma_fused_add_rmsnorm` | cuda | cuda |
| **rope** — `apply_rope`, `apply_rope_pos_ids`, `apply_llama31_rope`, `apply_llama31_rope_pos_ids`, `rope_quantize_fp8`, `mla_rope_quantize_fp8`, `rope_quantize_fp8_append_paged_kv_cache` | cuda | cuda |
| **sampling** — `softmax`, `sampling_from_probs`, `sampling_from_logits`, `top_k_sampling_from_probs`, `top_p_sampling_from_probs`, `top_k_top_p_sampling_from_probs`, `top_k_top_p_sampling_from_logits`, `min_p_sampling_from_probs`, `top_k_renorm_probs`, `top_p_renorm_probs`, `top_k_mask_logits`, `chain_speculative_sampling` | cuda | cuda |

- **fa2** — the in-tree HIP kernel.
- **auto** — what the library picks in production: [AITER](https://github.com/ROCm/aiter)
  when `amd-aiter` is installed and the call satisfies its constraints, otherwise fa2.
  `backend_resolved` records which one ran.
- **cuda** — upstream's name for the library's own kernel, kept because it is
  what those CLIs accept. On ROCm it is the in-tree HIP kernel: `auto` stays
  native for norm and rope, so no row here can reach AITER. Those ops do take
  `backend="aiter"`, but these routines never pass one — use
  [`bench_norm.py`](bench_norm.py) and [`bench_rope.py`](bench_rope.py)
  for that comparison.

`--backends aiter` is deliberately not offered. AITER is reached through `auto`,
which reports what it resolved to, so an explicit name would add no coverage.

The table above is the whole registered set; anything else in those groups
reports "not supported" and writes no row. Most are absent because the op has no
ROCm kernel — every `*_quant` and `*_fp4quant` norm variant, `fused_rmsnorm_silu`,
`fused_dit_layernorm`, `fused_qk_rmsnorm_rope`, and the three `top_k*` routines
that call `flashinfer.topk`. One is absent for a different reason:
`apply_rope_with_cos_sin_cache` builds its cache in `--input_dtype` while the op
requires float32, so it fails on CUDA too.

## Running

```bash
cd benchmarks
python flashinfer_benchmark.py --testlist rocm/testlist_rocm.txt \
    --output_path run-$(date +%F).csv
```

Diff successive CSVs **by column name**, not position — the ROCm columns below
were appended and shift the index of everything after them.

## Reading the output

Two columns are present in every CSV, but only ever populated on ROCm — the
schema is the same on both platforms, so a CUDA run leaves them empty rather
than omitting them:

- **`backend_resolved`** — what `auto` actually became (`aiter` or `fa2`). Without
  it an `auto` row is uninterpretable, since both possibilities look identical.
- **`backend_fallback_reason`** — why `auto` declined AITER. This is what separates
  "AITER correctly rejected this shape" from "the benchmark got in the way".

A row with `backend_resolved=fa2` and an **empty** reason came from neither: it
means `auto` was short-circuited because `use_tensor_cores` was set or CUDA-graph
capture was enabled. The testlist's attention lines avoid both, so treat it as
worth investigating.

`auto` rows are always timed eagerly. Under graph capture `auto` stays on fa2
*unless* the wrapper was given a `max_seq_len` capacity: AITER's launch grid is
fixed at capture, so without a declared capacity replaying at a longer sequence
is silently wrong. Declare one and `auto` may select AITER, replaying for any
`seq_len <= max_seq_len`. See
[`bench_decode_graph.py`](bench_decode_graph.py).

## Two tiers

This runner answers "did anything regress" across model-shaped configs. For
"why is this kernel slow", use [`rocm_profiler`](../../profiler/rocm/rocm_profiler.py)
and the per-op drivers in this directory, which collect
`rocprofv3` hardware counters and plot a roofline.

Some drivers here instead answer "which of these two arms should I
use", and print a ratio rather than a roofline:
[`bench_block_sparse_attention.py`](bench_block_sparse_attention.py) sweeps
block density against dense prefill, and
[`bench_mixed_attention.py`](bench_mixed_attention.py) sweeps the
prefill/decode mix for POD attention against running the two separately.

Several upstream benchmarks in the parent directory also run unmodified on ROCm:
`bench_append_paged_kv_cache.py`, `bench_append_paged_mla_kv_cache.py`,
`bench_fused_add_rmsnorm.py`, `bench_renorm.py`, `bench_sampling.py` and
`bench_sliding_window.py`.

## Comparing numbers

Record the architecture and toolchain with every result — `gcnArchName`,
`torch.version.hip`, and the `amd-aiter` version. A number from one
architecture is not comparable to another's, and routing can differ between
them, so a run is only meaningful against the stack it was taken on.
