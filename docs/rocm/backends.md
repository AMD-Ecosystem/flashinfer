# Backends on ROCm

FlashInfer+ROCm ships an in-tree HIP kernel for most ops and can dispatch
a subset to AMD's [AITER](https://github.com/ROCm/aiter). This page is the
long form of the support matrix in [README.md](../../README.md): what each
backend can do, what makes `backend="auto"` decline AITER, and the per-op
constraints that are easy to trip over.

* [Choosing a backend](#choosing-a-backend)
* [Installing AITER](#installing-aiter)
* [How `backend="auto"` resolves](#how-backendauto-resolves)
* [Known limitations](#known-limitations)
* [Per-op notes](#per-op-notes)

## Choosing a backend

Four backend strings are accepted by the ops that take a `backend=`
argument. Which one names the in-tree kernel depends on the op:

| `backend=` | Meaning |
| :--- | :--- |
| `"auto"` (default) | Pick per call — see [below](#how-backendauto-resolves) |
| `"aiter"` | Require AITER; raise if the call is not compatible |
| `"fa2"` | The in-tree HIP kernel, for the attention wrappers (single/batch prefill and decode) |
| `"native"` | The in-tree HIP kernel, for everything else (`append_paged_kv_cache`, `rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`, `rope`) |

`mla` accepts only `"auto"` and `"aiter"` — there is no in-tree MLA kernel.

## Installing AITER

Unless you are using the prebuilt Docker image, AITER is a separate
install. `amd-aiter` is **not** on the top-level `pypi.amd.com/simple`
index, and the ROCm-versioned channels carry no wheel at or above the
supported floor, so install from the nightlies index:

```bash
pip install amd-aiter==0.1.16.post3.dev0+g620287969.d20260725 \
  --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch/vllm-cdna/
```

Use `--extra-index-url`, not `--index-url`, so AITER's own dependencies
still resolve from PyPI, and spell the version out in full including the
local `+g...` segment — pip will not select a local version from a loose
specifier.

**`aiter_utils.AITER_MIN_VERSION` (0.1.16) is a hard floor**, enforced
before routing. FlashInfer links AITER's C++ symbols by mangled name
(`flashinfer/csrc/rocm/aiter_loader.cc`) and vendors its argument structs
(`include/flashinfer/rocm/attention/aiter/`) at the 0.1.16 layout, so an older
release shifts field offsets instead of failing to load. Below the floor
`auto` will not select AITER and an explicit `backend="aiter"` raises.

That rules out `pypi.amd.com/rocm-7.1.1/simple`, which carries only
`0.1.10` and only cp310/cp312 wheels. The CI image
(`docker/Dockerfile.rocm_ci`) still installs `0.1.10` and so runs without
the AITER backends; the devcontainer bundles the wheel above, on CPython
3.14, and needs no separate install.

A source build tracks master, which is many releases ahead of the pin
**with a different C ABI** — symbols the shim expects are renamed, hidden
rather than `extern "C"`, or absent:

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter && python3 setup.py develop
```

Nothing stops you running one, but treat it as untested here.

### C++-level integration

The `rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`, and `rope`
(cos/sin-cache) AITER backends are integrated at the **C++ level**: the
JIT compiles a small HIP shim that calls AITER's C++ kernels
(`rmsnorm2d`, `rmsnorm2d_with_add`, `aiter::silu_and_mul`,
`rope_cached_positions_2c_fwd_impl`) and links a symbol-visible AITER
`.so`. There is no runtime `import aiter` on these paths.

The first JIT build of each op builds the corresponding AITER module once
with `AITER_SYMBOL_VISIBLE=1` and caches it under
`~/.cache/flashinfer/aiter_libs/`. The CK `module_rmsnorm` build is large
and can take many minutes the first time.

### `mha_fwd` ships no prebuilt kernels at all

The 0.1.10 wheel carries 58 prebuilt `mha_varlen_fwd_*.so` files and zero
`mha_fwd*` — only `mha_fwd_kernels.cu` source. Single prefill is the op that
routes through the non-varlen `mha_fwd` template (batch=1 needs no seqstart
plumbing, see PR #246), so **every** one of its `(dtype, causal, has_lse)`
variants JIT-builds on first call.

**Absence of `mha_fwd*.so` is expected, not a broken install.** AITER
prebuilds what vLLM and SGLang call, which is the varlen path; the
non-varlen variant space is not in that set.

Those 58 varlen files are not full coverage either, so this is a difference
of degree rather than a unique case: batch prefill lazily builds the varlen
variants that are missing — on 0.1.10, bf16 ships only `nmask_lse` and
`mask_nlse`, so both remaining `nlogits` arms build on first use. Single
prefill builds every variant; batch prefill builds the gaps.

Two consequences worth planning for:

* Budget **20+ minutes** for a cold variant. This is the same first-build
  cost as the C++ AITER modules above, not a separate surprise.
* A read-only or foreign-owned `site-packages/aiter/jit/` lets the build
  succeed but the install step fail. That error currently propagates out of
  `backend="auto"` instead of falling back to `fa2`.

## How `backend="auto"` resolves

There are three policies. Which one applies is a property of the op, not of
the call:

| Policy | Ops |
| :--- | :--- |
| AITER when the call is compatible, else the in-tree kernel | `single_prefill`, `batch_prefill`, `batch_decode` |
| Always the in-tree `native` kernel — AITER is opt-in | `rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`, `rope`, `append_paged_kv_cache` |
| AITER only — no HIP kernel exists | `mla` |

`single_decode_with_kv_cache` (HIP-only) and `aiter_fused_moe`
(AITER-only) take no `backend=` argument at all.

When `auto` declines AITER for the attention ops it falls back to the
in-tree kernel and usually warns once with the reason. A few `batch_decode`
short-circuits (CUDA-graph capture, `use_tensor_cores=True`) fall back
silently. The always-native ops make no per-call decision, so they neither
warn nor report a reason.

The "always native" ops are that way because measurement said so,
not because AITER is unavailable:

* **`append_paged_kv_cache`** — AITER's `reshape_and_cache_flash` is
  bit-exact against the in-tree kernel but sustains 2.86 TB/s to its 3.62
  on identical work (gfx942, nnz=262144). Explicit `backend="aiter"`
  additionally requires fp16/bf16 and `NHD`.
* **`silu_and_mul`** and **`rope`** (cos/sin-cache) — the in-tree kernel
  was the better default in experiment. The AITER C++ path is reachable
  only via an explicit `backend="aiter"`.
* **`rmsnorm`** and **`fused_add_rmsnorm`** — at every 8-aligned
  `hidden_size`, AITER won 0 of 230 configs on gfx942 and 0 of 226 on
  gfx950. `fused_add_rmsnorm` is 1.6-1.8x slower, because CK cannot alias
  its output onto its input and the shim must stage two extra buffers on a
  bandwidth-bound kernel; `rmsnorm` is level on speed and less accurate.
  AITER does win up to 2x at `hidden_size` 111 and 500, where the native
  kernel's `vec_size` (`gcd(16/sizeof(T), d)`) collapses to 1 or 4 — but no
  model uses those widths, and the win is absent on gfx942 for the shapes
  where gfx950 shows it. Re-run with
  `python benchmarks/rocm_benchmarks/bench_norm.py --aa` for the noise floor
  and then without `--aa`; read the A/A first, since a margin inside it is
  not a result.

## Known limitations

AITER constraints fall into two groups. The first errors out under
`backend="aiter"` and triggers fallback under `backend="auto"`. The second
is worse: the call runs, but the flag is silently dropped.

`append_paged_kv_cache` sits outside both — its `auto` already resolves to
`native`, so there is no fallback to trigger, and none of the ignored
kwargs below are parameters of it.

### Falls back to the in-tree kernel under `auto`, raises under `aiter`

* GPU is not gfx942 or gfx950
* `kv_layout` is not `NHD`
* a custom attention mask tensor is supplied
* `q_dtype` is not `float16` / `bfloat16` (no fp32, fp8, or int8)
* `q_dtype != kv_dtype` — mixed-precision Q/KV is unsupported
* `head_dim_qk != head_dim_vo` (e.g. DeepSeek-style MLA with 192/128)
* `pos_encoding_mode != "NONE"` — AITER attention supports only `"NONE"`
* batch decode: `use_tensor_cores=True`, or `use_cuda_graph=True` under
  `auto`
* the `aiter` Python package is not importable

### Accepted but not honoured

These kwargs are accepted by the wrapper and never reach the kernel, so
passing them can produce **wrong results**. Switching to the in-tree
backend fixes only the first group.

**Dropped on the AITER path; pass `backend="fa2"` if you need them:**

* attention sinks (`sinks`)
* FP8 dequant scales (`scale_q` / `scale_k` / `scale_v`)
* `use_fp16_qk_reduction`
* RoPE scaling kwargs (`rope_scale`, `rope_theta`) — only meaningful
  alongside `pos_encoding_mode != "NONE"`, which AITER attention rejects
  outright, so in practice they pass through when the mode is `"NONE"`

**Dropped on *both* ROCm paths — there is no backend that honours them.**
These at least log a warning rather than failing silently:

* `enable_pdl`
* multi-modal / prefix-cache helpers (`maybe_prefix_len_ptr`,
  `maybe_token_pos_in_items_ptr`, `maybe_max_item_len_ptr`)

ALiBi is not in either list: `maybe_alibi_slopes` is filled internally
from the head count, and the user-facing route to it is
`pos_encoding_mode="ALIBI"`, which raises on the AITER path rather than
being ignored.

## Per-op notes

### `fused_add_rmsnorm` and `gemma_fused_add_rmsnorm` at large `hidden_size`

The `native` fused kernels stage the fp32 row in shared memory, costing
`hidden_size` floats. Above 16352 on gfx942 (40928 on gfx950) that exceeds the
per-block limit, so the kernel re-reads the row from `residual` instead — one
extra dtype round-trip of precision, on those sizes only.

Both `fused_add_rmsnorm` and `gemma_fused_add_rmsnorm` always take this path
above the threshold: the Gemma variant has no `backend=` argument, and
`fused_add_rmsnorm` now resolves `auto` to native. Only an explicit
`backend="aiter"` avoids it.

### Batch prefill: page size and the flat-gather path

AITER's CK FMHA kernels natively serve page sizes `{16, 1024}`, or
`{128, 256, 1024}` on `amd-aiter >= 0.1.10`. Other sizes still work but go
through an extra GPU gather that flattens the paged KV cache before the
AITER call — inside the timed region, which matters when benchmarking.

That list is a starting point, not a guarantee. `plan()` confirms it by
building the kernel, and a page size the installed AITER cannot actually
serve also falls back to the gather with a warning naming the reason.
Builds installed from an AITER source commit (as SGLang and vLLM do) are
the usual case where a "native" page size is rejected.

Set `FLASHINFER_AITER_STRICT=1` to raise instead of falling back — useful
in CI, where a silent slowdown is easy to absorb and hard to notice.

Ragged (non-paged) batch prefill is supported through
`BatchPrefillWithRaggedKVCacheWrapper`, with the same auto-routing rules.

### Batch decode: CUDA-graph capture

Graph capture on the AITER decode path is available via an explicit
`backend="aiter"`, not `auto`. **Capture at your maximum sequence
length**: the launch grid and `.so` variant are fixed at capture-time
shapes and the kernel early-exits per sequence on `context_lens`, so
replays *shorter* than captured are correct but *longer* ones are not.

`fa2`'s graph path is capacity-based and carries no such constraint, which
is why `auto` uses it under capture.

### MLA

* `q_data_type` must be `float16` or `bfloat16`, and must equal
  `kv_data_type`.
* `page_size` is not restricted by the code, but only `page_size=1` is
  covered by the numerical tests. Treat anything larger as unverified.
* `plan(causal=False)` **raises** when `max_seqlen_q > 1`. AITER's
  `mla_prefill_fwd` has no causal flag and unconditionally dispatches the
  causal ASM kernel, so honouring `causal=False` would silently return
  causal results. This is a deliberate divergence from the CUDA wrapper,
  which does honour it.
* `run()` hands AITER a zero-copy view when `ckv_cache` and `kpe_cache`
  are adjacent halves of one allocation (the vLLM / SGLang layout).
  Otherwise it builds the combined cache with `torch.cat` on every call
  and warns once *per wrapper* — a 60-layer model with a wrapper per layer
  emits 60 warnings. 4-D caches are rejected outright.

### Fused MoE

`flashinfer.aiter_fused_moe` is AITER-only — there is no HIP MoE kernel.

```python
from flashinfer import aiter_fused_moe, shuffle_moe_weight

w1s = shuffle_moe_weight(w1)
w2s = shuffle_moe_weight(w2)
out = aiter_fused_moe(hidden_states, w1s, w2s, topk_ids, topk_weights)
```

* `hidden_states` and the weights must be `bfloat16` or `float16`.
* `activation` is `"silu"` or `"gelu"`. `block_m` is the CK tile height —
  32, 64, or 128, or `"auto"` (the default), which picks from the expected
  tokens *per expert*, not the total token count.
* `topk_ids` is int32 in `[0, num_experts)` — there is no drop marker.
  `topk_weights` is float32.
* **Weights must be pre-shuffled** with `shuffle_moe_weight`, which lays
  them out for the 16x16 MFMA tile. Passing unshuffled weights does not
  raise: shape and dtype are unchanged, so nothing can detect it, and the
  output is silently wrong.

### HIP-only kernels

These have no AITER path at all:

* **Cascade attention** — two-level shared-prefix attention. A fused
  single-kernel HIP variant is gated behind
  `FLASHINFER_HIP_FUSED_CASCADE=1` (experimental).
* **POD attention** — `PODWithPagedKVCacheWrapper` and
  `BatchPODWithPagedKVCacheWrapper`. JIT-only, excluded from AOT builds,
  matching upstream CUDA.
* **LayerNorm / Gemma RMSNorm**, **sampling** (Top-K / Top-P / Min-P /
  OnlineSoftmax / SamplingFromLogits), the **logits processor** pipeline,
  and **quantization** (`packbits`, `segment_packbits`).

### fp8 on the HIP path

* Batch decode accepts an fp8 KV-cache (`float8_e4m3fnuz`).
* RoPE has a fused RoPE + fp8-quantize + paged-KV-append path covering
  `float8_e4m3fnuz` and `float8_e5m2fnuz`, alongside LLaMA and LLaMA 3.1
  scaling.

fp8 on the AITER attention paths is work in progress.

## Tests

Every row in the README matrix is covered by the default `pytest`
selection (`testpaths` in `pyproject.toml`). Most have a matching
`tests/rocm/test_*.py`. Two are covered from elsewhere:
`single_decode` from `test_batch_decode_kernels.py`,
`test_sliding_window.py`, and `test_logits_cap.py`; `quantization`
(`packbits`, `segment_packbits`) from `tests/utils/test_quantization.py`,
which is shared with the CUDA suite rather than ROCm-specific.
