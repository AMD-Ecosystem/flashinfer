# Backends on ROCm

FlashInfer+ROCm ships an in-tree HIP kernel for most ops and can dispatch
a subset to AMD's [AITER](https://github.com/ROCm/aiter). This page is the
long form of the support matrix in [README.md](../../README.md): what each
backend can do, what makes `backend="auto"` decline AITER, and the per-op
constraints that are easy to trip over.

* [Choosing a backend](#choosing-a-backend)
* [Not available on ROCm](#not-available-on-rocm)
  * [Not ported yet](#not-ported-yet-imports-but-has-no-rocm-kernel)
  * [NVIDIA-only, but not gated](#nvidia-only-but-not-gated)
  * [Blocked on a missing package](#blocked-on-a-missing-package-not-on-rocm)
* [Installing AITER](#installing-aiter)
  * [Prebuilding the variants](#prebuilding-the-variants)
* [How `backend="auto"` resolves](#how-backendauto-resolves)
* [CUDA-only arguments](#cuda-only-arguments)
* [Known limitations](#known-limitations)
* [Per-op notes](#per-op-notes)
* [Tests](#tests)

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

## Not available on ROCm

Some upstream modules wrap NVIDIA-only libraries or kernels. Importing one
raises `ImportError` naming the module, rather than exposing a stub that
fails later:

| Module | Why |
| :--- | :--- |
| `flashinfer.gemm`, `flashinfer.grouped_mm`, `flashinfer.trtllm_low_latency_gemm` | CUTLASS / TensorRT-LLM GEMM kernels |
| `flashinfer.cudnn`, `flashinfer.attention` | cuDNN attention |
| `flashinfer.fp4_quantization`, `flashinfer.fp8_quantization`, and the same two under `flashinfer.quantization` | NVFP4 / trtllm-gen quantization |
| `flashinfer.fused_moe` | The upstream CUTLASS MoE. **ROCm has MoE** — use `flashinfer.aiter_fused_moe`, below |
| `flashinfer.dsv3_ops` | DeepSeek-V3 fusions built on the above |
| `flashinfer.comm.*` — `cuda_ipc`, `mixed_comm`, `mnnvl`, `nvshmem`, `nvshmem_allreduce`, `trtllm_alltoall`, `trtllm_ar`, `trtllm_mnnvl_ar`, `vllm_ar` | NVLink / NVSHMEM transports |
| `flashinfer.deep_gemm`, `flashinfer.green_ctx` | Both reach `flashinfer.cuda_utils`, which requires `cuda-python`; CUDA green contexts have no HIP analogue |
| `flashinfer.gdn_prefill`, `flashinfer.mamba.ssd_combined` | CuTe DSL — both import `cutlass` eagerly. The rest of `flashinfer.mamba` imports fine |
| `flashinfer.parallel_attention` | Needs `prefill.fmha_varlen`, upstream's CUTLASS varlen FMHA |
| `flashinfer.aot`, `flashinfer.__main__` | CUDA-only AOT build and CLI — they shell out to `nvcc`. **ROCm has AOT** — use `flashinfer.rocm.aot` |
| `flashinfer.tactics_blocklist_gen` | Enumerates CUDA backend tactics |

`importlib.util.find_spec` still reports these as present — the files ship,
the import is what is gated. Feature-detect with `hasattr(flashinfer, ...)`
or a `try: import ... except ImportError`, not `find_spec`.

A module that merely *imports* a gated one — `flashinfer.comm.trtllm_moe_alltoall`,
for instance — still fails, but transitively, so the error names the gated
dependency rather than the module you asked for. `flashinfer.cuda_utils` is
deliberately ungated: it is private, and the modules that reach it are listed
above.

`flashinfer.quantization`'s `packbits` and `segment_packbits` are unaffected
and have in-tree HIP kernels.

### Not ported yet: imports, but has no ROCm kernel

Gating is not the whole story. These import cleanly and fail on first call,
while resolving or building their JIT sources, naming the file that does not
exist. Usually that surfaces from ninja:

```text
ninja: error: '.../csrc/rocm/topk.cu', needed by '.../topk.cuda.o', missing
```

* `flashinfer.topk`, `flashinfer.concat_ops` and `flashinfer.mhc` — ordinary
  kernels with no `csrc/rocm` source.
* `flashinfer.topk_varlen` — its optimized backends admit only NVIDIA compute
  capabilities and the general fallback calls `get_topk_module()`, so no path
  is available.
* `flashinfer.mamba`'s `selective_state_update` and `checkpointing_ssu`. Its
  `ssd_combined` is gated instead: it imports `cutlass` eagerly and so never
  reaches a kernel.
* `flashinfer.kda_prefill` and the `flashinfer.kda_kernels` entry points —
  `csrc/rocm` has no `kda/` tree. `flashinfer.kda` does not even import, for
  want of `tvm_ffi`, which the image does not ship.
* `flashinfer.diffusion_ops` — it re-exports four fused DiT entry points from
  `flashinfer.norm`, and `csrc/rocm/norm.cu` defines none of them.
* `flashinfer.norm`'s `layernorm`, `layernorm_quant`, `rmsnorm_quant` and
  `fused_add_rmsnorm_quant`. These fail differently: the JIT source exists and
  builds, but `csrc/rocm/flashinfer_norm_binding.cu` binds only `rmsnorm`,
  `fused_add_rmsnorm`, `gemma_rmsnorm` and `gemma_fused_add_rmsnorm`, so the
  call dies on a bare `AttributeError`. AITER's CK `layernorm2d` cannot stand in
  for `layernorm`: it reads fp32 `gamma`/`beta` as the input dtype and returns
  NaN, and casting them down misses the op's 1e-2 bf16 tolerance.
* `flashinfer.trace.templates.gemm`, through the `nv_internal` FP4 sources.
* Three side paths of otherwise supported modules: `utils.set_log_level()`,
  the opt-in GPU stats counter in `api_logging`, and `norm`'s fused
  rmsnorm+silu variant.

They stay importable because working code reaches them — `topk_varlen` imports
`topk` at module scope — so gating would break more than it documents.

Some fail earlier, as a plain `FileNotFoundError`: the Mamba generators open
their templates directly, and fused rmsnorm+silu copies its source before any
build starts.

`tests/rocm/test_kernel_source_coverage.py` holds this list. It fails when a
newly vendored op names a kernel source absent from `csrc/rocm`.

It only sees missing *files*, though, which is how the unbound `norm` entry
points above went unnoticed — there the file exists and the symbol does not.
`tests/rocm/test_norm_entry_points.py` covers that case for `norm`, asserting
the unresolvable set in both directions so porting one is also a test change.

### NVIDIA-only, but not gated

These import cleanly and fail only when called, so the gate never sees them —
but a ROCm equivalent would be a rewrite, not a port:

* `flashinfer.cute_dsl` and `flashinfer.cutile` — CuTe DSL. Neither fails on
  call: `cute_dsl/__init__.py:37` imports its kernels only when
  `is_cute_dsl_available()`, so on ROCm the module offers its availability
  helpers and no kernels, and `cutile` exports `is_cuda_tile_available()`
  alone, which answers `False`. Probe those rather than the import.
* `flashinfer.msa_ops` and `flashinfer.moe_ep` — `cutlass.cute` plus, for
  `moe_ep`, NVSHMEM.
* `flashinfer.gdn_decode` — every kernel under `gdn_kernels/` is `@cute.jit`.
  It imports and fails when called, where `gdn_prefill` reaches `import
  cutlass` at module scope and is gated in the table above instead.
* `flashinfer.nvfp4_attention_sm120` — `supported_compute_capability([120, 121])`.
* `flashinfer.xqa` — `csrc/xqa/{mma.cuh,gmma.cuh,gmma_impl.cuh,tma.h}` emit
  NVIDIA PTX directly (`mma.sync`, `wgmma`, `cp.async.bulk`), so an AMD path is
  a rewrite rather than a port.
* `flashinfer.tllm_utils` — `delay_kernel()` builds five `nv_internal`
  TensorRT-LLM sources. `flashinfer.autotuner` imports the module, which is why
  it is not gated.

### Blocked on a missing package, not on ROCm

`flashinfer.profiler` needs `tg4perfetto` and `flashinfer.artifacts` needs
`requests`; `docker/Dockerfile.rocm` ships neither, so neither imports here.
That says nothing about their ROCm support either way.

## Installing AITER

Unless you are using the prebuilt Docker image, AITER is a separate
install. `amd-aiter` is **not** on the top-level `pypi.amd.com/simple`
index, and the ROCm-versioned channels carry no wheel at or above the
supported floor, so install from the nightlies index:

```bash
pip install amd-aiter==0.1.20+rocm10.1.0a20260819.3135022 \
  --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch/
```

Use `--extra-index-url`, not `--index-url`, so AITER's own dependencies
still resolve from PyPI, and spell the version out in full including the
local `+rocm...` segment — pip will not select a local version from a loose
specifier. Check afterwards that the install did not pull a CPU-only torch
over your ROCm one: `python -c "import torch; assert torch.version.hip"`.

**`aiter_utils.AITER_MIN_VERSION` (0.1.20) is a hard floor**, enforced
before routing. FlashInfer links AITER's C++ symbols by mangled name
(`csrc/rocm/aiter_loader.cc`) and vendors its argument structs
(`include/flashinfer/rocm/attention/aiter/`) at the 0.1.20 layout, so an older
release shifts field offsets instead of failing to load -- and 0.1.20 renamed
the RMSNorm entry points, so an older one cannot resolve them. Below the floor
`auto` will not select AITER and an explicit `backend="aiter"` raises.

The development image (`docker/Dockerfile.rocm`) bundles that wheel, on
CPython 3.12, and needs no separate install -- it is the one supported
configuration, so in practice this section is only for someone assembling
their own.

Every 0.1.20 wheel is cp312 only, which is what fixes the interpreter. None
is built against ROCm 10.0 directly; the pin above is the nearest retarget
of the same source revision (build id `3135022`).

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
(`aiter::rmsnorm`, `aiter::add_rmsnorm`, `aiter::silu_and_mul`,
`rope_cached_positions_2c_fwd_impl`) and links a symbol-visible AITER
`.so`. There is no runtime `import aiter` on these paths.

The first JIT build of each op builds the corresponding AITER module once
with `AITER_SYMBOL_VISIBLE=1` and caches it under
`~/.cache/flashinfer/aiter_libs/`. The `module_rmsnorm_quant` build is large
and can take many minutes the first time.

### Not all of AITER is Composable Kernel

Worth knowing before comparing backends: the modules linked where `auto` picks
the in-tree kernel are AITER's own hand-written HIP, not CK.

| AITER module | backing | used for |
| :--- | :--- | :--- |
| `module_rmsnorm_quant`, `module_activation`, `module_rope_*` | AITER HIP (no `ck_tile` references) | norm, activation, rope |
| `module_norm` | CK-tile | `layernorm2d` only — it exports no rmsnorm |
| `mha_fwd`, `mha_varlen_fwd`, `mha_batch_prefill` | CK-tile | prefill |

So an "AITER lost to the in-tree kernel" result for norm, activation or rope is
a HIP-vs-HIP comparison and says nothing about CK. CK-tile `layernorm2d` was
measured for `layernorm` and rejected on contract, not speed: it derives one
dtype from the input and reads this API's fp32 `gamma`/`beta` as bf16, which
returns silent garbage rather than an error (see *Not ported yet*).

### `mha_fwd` ships no prebuilt kernels at all

AITER ships prebuilt `mha_varlen_fwd_*.so` files and no `mha_fwd*` — only
`mha_fwd_kernels.cu` source. Single prefill is the op that routes through the
non-varlen `mha_fwd` template (batch=1 needs no seqstart plumbing, see
[PR 246](https://github.com/AMD-Ecosystem/flashinfer/pull/246)), so **every**
one of its `(dtype, needs_mask, has_lse)` variants JIT-builds on first call. `needs_mask` is `causal or window_left >= 0`: AITER
splits that `.so` on whether anything is masked, not on causality.

**Absence of `mha_fwd*.so` is expected, not a broken install.** AITER
prebuilds what vLLM and SGLang call, which is the varlen path; the
non-varlen variant space is not in that set.

The shipped varlen set is not full coverage either, so this is a difference
of degree rather than a unique case: batch prefill lazily builds whichever
varlen variants are missing. Single prefill builds every variant; batch
prefill builds the gaps.

Two consequences worth planning for:

* A cold variant costs a CK-tile compile. Measured at `MAX_JOBS=32` on
  gfx942 (MI300X, ROCm 10.0), one variant per family: `mha_fwd` 280-360 s,
  `mha_batch_prefill` 119 s, `mha_varlen_fwd` 74 s. A 16-core gfx950 box took
  roughly 2x each, tracking core count rather than architecture. It is paid once
  per *variant* -- one `.so` serves every head dimension -- and again on any
  restart that does not persist `site-packages`.
* A read-only or foreign-owned `site-packages/aiter/jit/` lets the build
  succeed but the install step fail. Under `auto` that demotes to `fa2` with a
  warning; under `backend="aiter"` it raises.

Both are what the variant store below exists to remove.

### Prebuilding the variants

`plan()` resolves a variant from FlashInfer's own store before asking AITER to
build one, so the whole set can be built once, ahead of time:

```bash
python -m flashinfer.rocm.prebuild_aiter_variants --list    # what the default set covers
python -m flashinfer.rocm.prebuild_aiter_variants           # build the missing ones
python -m flashinfer.rocm.prebuild_aiter_variants --prune   # report stores from an older pin
```

The default set is `mha_fwd` and `mha_varlen_fwd` — 24 of the 40 variants, from
16 builds. `mha_batch_prefill` is excluded because its bootstrap is also the
`page_size` capability probe: the variant filename has no page-size axis, so the
probe has to run whether or not the artifact is in the store, and prebuilding
that family costs ~32 min on gfx942 for files nothing ever saves time on. Pass
`--only mha_batch_prefill` to build them anyway.

**It needs a GPU**: the only supported way to make AITER emit a variant is to
call the op. So it cannot be a `docker build` step — run it as a GPU-attached
job and copy the resulting directory into the image.

The store lives under FlashInfer's cache directory, at
`aiter_variants/<arch>__aiter-<version>__rocm-<version>/`. That cache is
`~/.cache/flashinfer` unless `FLASHINFER_WORKSPACE_BASE` moves it; there is no
`FLASHINFER_CACHE_DIR` environment variable.
All three components are in the tag on purpose: these are CK-tile objects that
travel between machines, so bumping AITER or ROCm names a directory that does
not exist, and the lookup misses into a rebuild rather than loading a
mismatched artifact. Nothing prunes the old ones — each is ~165 MB per
architecture — hence `--prune`, which is dry-run unless you add `--yes`.

`FLASHINFER_AITER_VARIANT_DIR` points the loader at a store somewhere else, for
an image that ships one. `AITER_JIT_DIR` still takes precedence over both, so an
operator pointing at a custom AITER build is unaffected.

The `amd-flashinfer-jit-cache` wheel carries a store when one was built before
the wheel was, under `aiter_variants/<tag>/`. The layout allows one subdirectory
per tag so a consumer can pick its own, but a single wheel build packages only
the architecture it ran on — a variant can only be produced on the device it
targets, so a two-architecture wheel is assembled from two builds. The tag is the whole compatibility check, which is why this is not recorded
in the AOT manifest: that names a comma-joined list of architectures, and a
store is always single-arch. A wheel built without running the prebuild simply
carries none, and variants are built on demand exactly as before.

40 variants are reachable per architecture, not 64: `has_alibi` is hard-coded
false at every call site, and `mha_fwd` has no `_logits` arm. One `.so` serves
every head dim.

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
short-circuits (CUDA-graph capture without a declared `max_seq_len`,
`use_tensor_cores=True`) fall back silently. The always-native ops make no per-call decision, so they neither
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
  `python benchmarks/rocm/bench_norm.py --aa` for the noise floor
  and then without `--aa`; read the A/A first, since a margin inside it is
  not a result.

## CUDA-only arguments

The decode, prefill and MLA wrappers declare upstream's full parameter list, in
upstream's order, so a caller written against the CUDA API binds correctly and
an argument left at its default costs nothing. An argument that actually asks
for a CUDA feature — NVFP4 KV-cache scale factors, trtllm-gen skip-softmax, the
split-KV scheduler knobs, CUDA-graph MLA capture — raises `NotImplementedError`
naming itself rather than being ignored. A value that means "not requested"
(`False` for an enable flag, `1.0` for a calibration scale) is accepted.

`q_scale`, `k_scale` and `v_scale` are *not* in that set: every attention entry
point folds them into `sm_scale` and the output.

`CUDA_ONLY_PARAMS` in `scripts/rocm_api_parity.py` is the full list. The script
also fails on any undeclared signature divergence from upstream — a missing
parameter, a reordered one, or a drifted default; `tests/rocm/test_api_parity.py`
runs it in CI.

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
* batch decode: `use_tensor_cores=True`, or `use_cuda_graph=True` without a
  declared `max_seq_len`
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

### Single prefill takes AITER's asm kernel on CDNA4 only

`aiter::mha_fwd` is a two-arm dispatcher in AITER's source — hand-written
gfx9 asm first, CK Tile second — but the arms are compiled into separate
modules. The per-variant `.so` files this port loads are built `-DFAV2_ON=1`
with no `-DFAV3_ON`, so they contain CK Tile only; the asm arm lives in a
prebuilt `module_fmha_v3_fwd.so` that has to be opened explicitly.

It is worth opening only where it wins, and that differs by architecture.
Over a 60-cell sweep at bf16 `head_dim` 128 (batch 1 and 4 × 16/32/64 q-heads
× `seqlen` 256-6144), asm against CK Tile:

* **gfx950** — the per-`seqlen` geomean climbs with length above 1024: 1.01,
  1.09, 1.15, 1.14, 1.19, 1.19. The one step down (3072) is inside the A/A
  floor. At and above `qo_len` 2048, 23 of 24 cells win or hold: geomean 1.17,
  worst cell 0.98 (batch 1, 16 heads, `seqlen` 2048). That is the shipping
  threshold.
* **gfx942** — non-monotonic, and the oscillation does not damp with length:
  1.08 at `seqlen` 2048, 0.97 at 3072, 1.06 at 4096, 0.99 at 6144. Every one of
  those steps is outside the A/A floor, so no threshold holds and the arm stays
  unreachable. The same 2048 cut scores geomean 1.02 with 9 of 24 cells
  regressing.

Measured on an idle node against an A/A floor of 0.987-1.013 (gfx942) and
0.989-1.012 (gfx950), so the one gfx950 cell at 0.98 is a real if small loss
rather than noise.

The mechanism is occupancy: the asm kernel tiles 256 rows of Q at a time, so at
batch 1 with few heads there are too few workgroups to fill the device, and it
loses. That is also why the gate reads `qo_len` rather than `kv_len` — a
prefill-with-history shape (`qo` 512, `kv` 4096) measures 0.73× on gfx950 and
must stay on CK Tile.

Re-run with `python benchmarks/rocm/bench_asm_vs_cktile.py --aa` for the noise
floor and then without `--aa`; read the A/A first, since a margin inside it is
not a result. `FLASHINFER_AITER_ASM_PREFILL=0` pins CK Tile.

The sweep is square (`qo_len == kv_len`) apart from one point, so a long cached
context with `qo_len >= 2048` — chunked prefill — takes the asm arm on speed
evidence from square shapes only. Correctness there is covered:
`test_aiter_asm_routing.py` runs `kv_len == 4 * qo_len` above the threshold and
checks output and LSE against fp32 on the execution that reached asm. If the
speedup matters for that shape, measure it, and use
`FLASHINFER_AITER_ASM_VERBOSE=1` to confirm which arm ran.

### Soft-capped causal prefill avoids one AITER kernel

AITER's
`mha_varlen_fwd` miscomputes `logits_soft_cap` for causal prefill at
`head_dim=128` (through amd-aiter 0.1.21) — on **gfx950 only**, and there at
every length, so which architectures are affected lives in `arch_caps.py`
rather than at the call sites. Single and ragged prefill always dispatch
through that kernel, so on gfx950 `auto` serves them with `fa2` and
`backend="aiter"` raises rather than returning wrong numbers. gfx942 measures
clean over a `qo_len` × `kv_len` sweep at every cap and uses AITER as normal.

Paged prefill keeps AITER at a native page size, since that route takes
`mha_batch_prefill` instead — measured exact on amd-aiter 0.1.20 against an
fp32 reference on both architectures — and falls back only when the run-time
probe demotes it to a flat gather. Every other soft-cap shape is unaffected.

### Short-query paged prefill avoids AITER's flat gather

At a page size AITER cannot page natively, its paged prefill first
`index_select`s the whole KV cache into a contiguous buffer. That copy is
`O(kv)` against an `O(q·kv)` attention, so its cost decays as `1/q`: a long
query amortises it, a short one pays it in full. `auto` therefore serves
short queries with `fa2`, and `backend_fallback_reason` names the threshold.

One exception: once a cudagraph-enabled wrapper has planned AITER's
flat-gather path, later short-query plans stay on AITER. Demoting would null
the gather buffers a captured graph still points at. Use a separate wrapper
per query-length regime if you need both routed correctly under capture.

| arch | routed to `fa2` when `max_q_len` is | AITER slower by |
| :--- | :--- | :--- |
| gfx942 | ≤ 16 | 1.35–4.6× |
| gfx950 | ≤ 8 | 1.25–4.6× |

Three paths are deliberately **not** gated. Native page sizes have no gather
and beat `fa2` even at one query row (0.93× on gfx942, 0.54× on gfx950).
Ragged prefill dispatches through `mha_varlen_fwd` on already-contiguous KV.
Decode is genuinely one query row, and AITER wins there. An explicit
`backend="aiter"` is also honoured — this is a routing preference, not a
wrong answer, so it stays measurable.

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

AITER's CK FMHA kernels natively serve page sizes `{128, 256, 1024}` at
every release at or above the supported floor. Other sizes still work but go
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

AITER's launch grid and `.so` variant are fixed at capture time, so the
contract depends on whether you declare a capacity.

Without `max_seq_len`, graph capture is opt-in via an explicit
`backend="aiter"` and you must **capture at your maximum sequence length**:
the kernel early-exits per sequence on `context_lens`, so replays *shorter*
than captured are correct but *longer* ones are not. `auto` stays on `fa2`.

Passing `max_seq_len` to the wrapper sizes the grid, `.so` variant and
partition workspace from that capacity instead, so a graph captured at any
shape replays for any `seq_len <= max_seq_len` — the same capacity-based
contract `fa2` has, which is what lets `auto` select AITER under capture.

Two caveats. The partition workspace scales with the declared capacity, so
an over-generous value costs memory on every `run()`. And the capacity is
enforced in `plan()`: a replay that writes the persistent buffers directly
is not checked, and an over-length sequence there is silently truncated to
the capacity rather than attending to its full context.

`run(..., return_lse=True)` raises on this backend under capture — PA v1
emits no LSE and the FA2 shadow plan it borrows is not capture-safe.

Multi-token decode (`q_len_per_req > 1`), for speculative-decode verify,
needs `use_tensor_cores=True` — that path runs the batch-prefill kernel,
which is what supplies the causal mask, and AITER decode requires
`use_tensor_cores=False`. Every request needs `kv_len >= q_len_per_req`,
since the draft tokens must already be in the KV cache. Under graph
capture the value is part of the frozen shape: use one wrapper per
`q_len_per_req`.

At `head_dim < 256` the cost steps with `q_len_per_req * gqa_group_size`,
not with `q_len_per_req` alone — the query tile is 16 at or below 16 and 64
above (`FA2DetermineCtaTileQ` in `include/flashinfer/rocm/utils.cuh`). At GQA 32/8 a draft length
of 4 is free and 8 costs a step (1.2-1.6x on gfx942, 1.4-2.0x on gfx950); at
64/8 the step arrives at 2. `head_dim >= 256` takes the 64 tile
unconditionally, so no draft length is free there.

That rule describes **eager** planning only. Under `enable_cuda_graph` the
scheduler cannot see per-request lengths, so it bounds them by
`total_num_rows - batch_size + 1` and sizes the tile from that
(`PrefillSplitQOKVIndptr` in `include/flashinfer/rocm/attention/scheduler.cuh`).
The tile is therefore 64 whenever
`(batch_size * (q_len_per_req - 1) + 1) * gqa_group_size > 16` — at batch 4 /
draft 4 / GQA 4 that is `13 * 4 = 52`, so capture takes the 64 tile where eager
takes 16. Any draft length above 1 crosses it at a large enough batch, so
measure under capture rather than carrying the eager figures over.

**Chain drafts only.** The verify mask is causal, so the draft tokens are
taken as one linear sequence per request — which fits vanilla speculative
decoding, DeepSeek MTP, n-gram/prompt-lookup, and EAGLE run in chain mode.
It does **not** fit tree drafts (EAGLE-2, Medusa's multi-head tree,
SpecInfer): a causal mask lets a draft token attend to a sibling on another
branch, so a tree would return plausible but wrong numbers rather than an
error. Tree verification needs a topology mask, which this path does not
accept — build it with `custom_mask` on
`BatchPrefillWithPagedKVCacheWrapper` instead. Note the acceptance step is
chain-only too: `chain_speculative_sampling` takes
`(batch_size, num_speculate_tokens)`, not a tree.

An output dtype that differs from the query dtype still raises.

### MLA

* `use_cuda_graph=True` and `run(..., return_lse=True)` both raise
  `NotImplementedError` — neither is supported on the AITER MLA path.
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

* `hidden_states` must be `bfloat16` or `float16` — fp8 activations are
  rejected; the shim quantizes per token itself.
* The **weights** may be fp8, matching `hidden_states` otherwise. fp8 needs
  both `w1_scale` and `w2_scale` (neither alone) and two shape constraints,
  not one: `model_dim % 128 == 0` (CK steps stage-1 K by 128 on every tile
  and both architectures, so no `block_m` rescues it), **and** `inter_dim`
  divisible by the stage-2 K tile, which depends on `block_m` and the
  architecture. With `block_m="auto"` the shim tries the other legal tiles
  before giving up; pin `block_m` and an indivisible `inter_dim` raises.
  **The fp8 encoding is architecture-dependent** — `float8_e4m3fnuz` on
  gfx942, `float8_e4m3fn` on gfx950. Ask `moe_fp8_dtype()` rather than
  hard-coding one; the shim raises `ValueError` on the wrong encoding, which
  is what stops the hardware reinterpreting the bits.
* `activation` is `"silu"` or `"gelu"`. `block_m` is the CK tile height —
  32, 64, or 128, or `"auto"` (the default), which picks from the expected
  tokens *per expert*, not the total token count.
* `topk_ids` is int32 in `[0, num_experts)` — there is no drop marker.
  `topk_weights` is float32.
* **Weights must be pre-shuffled** with `shuffle_moe_weight`, which lays
  them out for the 16x16 MFMA tile. Passing unshuffled weights does not
  raise: shape and dtype are unchanged, so nothing can detect it, and the
  output is silently wrong.

### No AITER backend

These take the in-tree HIP kernel. Cascade is listed first because it is the
partial case — its own kernels are HIP, but what it calls is not:

* **Cascade attention** — the *merge* kernels only.
  `FLASHINFER_HIP_FUSED_CASCADE=1` narrows to `MultiLevelCascadeAttentionWrapper`:
  it passes each level's partial state into the next prefill call rather than
  merging afterwards. A level that resolves to AITER still merges post-hoc --
  the AITER kernel takes no partial state -- and the two shared-prefix wrappers
  never consult the flag. Opt-in, both paths tested, read once at import. **The attention underneath is not HIP-only**: each of the
  three wrappers builds an ordinary entry point internally at
  `backend="auto"` — `BatchPrefillWithPagedKVCacheWrapper` for the
  multi-level and shared-prefix-prefill wrappers,
  `BatchDecodeWithPagedKVCacheWrapper` plus `single_prefill_with_kv_cache`
  for the shared-prefix decode one — so each level routes to AITER whenever
  a plain call of that shape would. Measured: a two-level
  `MultiLevelCascadeAttentionWrapper.plan()` at page_size 128 resolves both
  levels to `aiter`. None of the three takes a `backend=` argument, so
  there is no supported way to pin the levels to `fa2` short of the
  capability table declining AITER.
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
