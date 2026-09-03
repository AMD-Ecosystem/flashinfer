# FlashInfer+ROCm: An AMD ROCm port of FlashInfer

FlashInfer+ROCm brings the
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) inference kernel
library to AMD Instinct GPUs — CDNA3 (gfx942, MI300X / MI325X) and CDNA4
(gfx950, MI350X / MI355X). It ships in-tree HIP ports of the attention,
KV-cache, RoPE, normalization, sampling, and logits-processor kernels, and
transparently dispatches a subset of ops to AMD's
[AITER](https://github.com/ROCm/aiter) backend when that is the faster or
only path.

The port is in active development and is aimed at developers embedding
FlashInfer kernels into their own training or serving stack. See
[Releases](https://github.com/AMD-Ecosystem/flashinfer/releases) for the release history.

**Versioning.** Release tags are `<upstream_version>+amd.<n>`, tying each
FlashInfer+ROCm release to the upstream tag it is based on — `0.6.18+amd.1`
is the first AMD release based on upstream `v0.6.18`.

## Quick start

There is no published wheel or image for this release — build from the
repository. The development image carries a matched ROCm, PyTorch, Python
and AITER set, so it is the shortest path to a working environment:

```bash
docker build -t flashinfer-dev:rocm10.0 -f docker/Dockerfile.rocm . \
  --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g)
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v "$PWD":/workspace -w /workspace flashinfer-dev:rocm10.0
```

The image does not contain the source — `-v "$PWD":/workspace` is what puts
it there. The `--build-arg` trio matches the container user to yours; without
it the image runs as UID 1003 and the editable install cannot write to your
mounted tree. `render` must be the host's **numeric** GID, since the name
resolves against the image's own group. Then, inside the container:

```bash
python -m pip install --no-build-isolation -ve .
python -c "import flashinfer; print(flashinfer.__version__)"
```

[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md) has the full recipe: the
`docker run` flags the GPU devices need, the wheel build, and the
ahead-of-time kernel build.

**Bringing your own environment?** Torch must come from `repo.radeon.com`,
via `-f` and **not** `--index-url` — that repo is a flat wheel listing
rather than a PEP 503 index, so `--index-url` fails with "No matching
distribution found". pip still prefers the ROCm wheel over a same-version
PyPI wheel because its `+rocm<X.Y>` local version ranks higher:

```bash
pip install torch==2.9.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
python -c "import torch; assert torch.version.hip, 'not a ROCm build'"
```

Kernels are JIT-compiled on first use — a minute or so for an in-tree HIP
kernel, and up to 20+ minutes for a cold AITER variant. The optional
[`amd-flashinfer-jit-cache`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/amd-flashinfer-jit-cache/README.md)
package ships them prebuilt — one wheel per architecture, gfx942 and
gfx950, with the architecture in the version's local segment
(`0.6.18+amd.1.gfx942`). Pin it in full; an unqualified requirement
resolves to whichever architecture sorts highest, which need not be yours.

## Basic usage

```python
import torch
import flashinfer

# PyTorch+ROCm still uses device="cuda" for AMD GPUs.
q = torch.randn(1024, 32, 128, dtype=torch.float16, device="cuda")
k = torch.randn(1024,  8, 128, dtype=torch.float16, device="cuda")  # GQA 4:1
v = torch.randn(1024,  8, 128, dtype=torch.float16, device="cuda")

# backend="auto" (default) routes to AITER when supported and falls back
# to the in-tree fa2 HIP kernel otherwise.
output = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
```

Runnable scripts for single/batch prefill and batch decode, plus
`amd_flashinfer_rocm_tutorial.ipynb` walking through the public API on
ROCm, are in
[`examples/`](https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/examples):

```bash
python examples/single_prefill_example.py
```

## Supported hardware and toolchain

| | Supported |
| :--- | :--- |
| GPUs | gfx942 (CDNA3 — MI300X, MI325X), gfx950 (CDNA4 — MI350X, MI355X) |
| ROCm | 7.0.2, 7.1.1, 7.2, 7.14, 10.0 |
| PyTorch+ROCm | 2.8.0, 2.9.1, 2.12.0 |
| Python | 3.10+ (the development image uses 3.12, which is what the AITER pin requires) |

The combination this release is developed and tested against is the
development image's: **ROCm 10.0, PyTorch 2.12.0, Python 3.12,
Ubuntu 24.04, `amd-aiter` 0.1.20** (`docker/Dockerfile.rocm`). The other
rows are known to build but are not covered by every run. Replace `7.2` in
the torch install command with the ROCm version you need; see
<https://repo.radeon.com/rocm/manylinux/> for what is available.

ROCm 7.14 and 10.0 are the exceptions: `repo.radeon.com` publishes no
`rocm-rel-` directory for either, so there is no pip recipe. Take torch from
the `rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.12.0` image
instead, as `docker/Dockerfile.rocm` does. Stay on torch 2.12: 2.13 removes a
`c10` symbol that every published `amd-aiter` build's prebuilt prefill kernels
need, so AITER prefill fails to load there.

## Support matrix

Every op has an in-tree HIP kernel unless noted; a subset also has an
AITER backend, selected by a `backend=` argument that defaults to
`"auto"`. There are three policies for what `auto` picks:

| `backend="auto"` picks | Ops |
| :--- | :--- |
| AITER when the call is compatible, else the in-tree kernel | `single_prefill`, `batch_prefill`, `batch_decode` |
| Always the in-tree `native` kernel — AITER is opt-in | `rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`, `rope`, `append_paged_kv_cache` |
| AITER only — no HIP kernel exists | `mla` |

To override, pass `backend="aiter"`, or name the in-tree kernel:
`backend="fa2"` for the attention wrappers, `backend="native"` for
everything else.

Two ops take no `backend=` argument: `single_decode_with_kv_cache` is
HIP-only, and `aiter_fused_moe` is AITER-only. On gfx950 with ROCm 7.2.x,
`batch_prefill` never resolves to AITER — see the footnote under the
table.

Beyond the routed ops, this release also carries block-sparse attention
(`BlockSparseAttentionWrapper` and the variable-block variant), POD
attention (`PODWithPagedKVCacheWrapper`, `BatchPODWithPagedKVCacheWrapper`),
cascade attention, the sampling and logits-processor pipelines, and fp8
fused MoE via `aiter_fused_moe`. Batch decode can reach AITER under CUDA-graph
capture once you declare a `max_seq_len` capacity on the wrapper.

**The full routing rules, per-op constraints, AITER install instructions,
and the list of upstream modules that are not available on ROCm are in
[`docs/rocm/backends.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md).**
Read it before relying on an AITER path — several attention kwargs are
silently ignored there rather than rejected.

The table below is generated from
[`flashinfer/rocm/arch_caps.py`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/flashinfer/rocm/arch_caps.py), which is what
`backend="auto"` consults at runtime, so it cannot drift from the routing
decisions the library makes. Do not edit it by hand; run
`python3 scripts/gen_arch_support_matrix.py`.

<!-- BEGIN GENERATED: arch support matrix -- scripts/gen_arch_support_matrix.py -->

| Op | Backend | gfx942 (CDNA3) | gfx950 (CDNA4) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `batch_decode` | `aiter` | ✅ | ✅ | MHA / GQA / MQA with sliding window; fp16/bf16 + NHD. Under graph capture `auto` needs a declared `max_seq_len`, else it stays on fa2. |
| `single_prefill` | `aiter` | ✅ | ✅ | MHA / GQA / MQA with sliding window; fp16/bf16 + NHD, equal Q/KV dtypes and head dims, no custom mask. fp8 WIP. |
| `batch_prefill` | `aiter` | ✅ | ⚠️[^kb1] | Paged and ragged, with sliding window. Page sizes 128/256/1024 are served natively; others take a flat gather. |
| `mla` | `aiter` | ✅ | ✅ | DeepSeek-style 192/128 head-dim split; fp16/bf16. No HIP kernel exists, so `auto` resolves here. |
| `rope` | `aiter` | ✅ | ✅ | `apply_rope_with_cos_sin_cache` and its inplace variant, linked at the C++ level. Opt-in. |
| `append_paged_kv_cache` | `aiter` | ✅ | ✅ | fp16/bf16 + NHD. Bit-exact with the in-tree kernel but slower, so `auto` picks `native`. |
| `rmsnorm` | `aiter` | ✅ | ✅ | `aiter::rmsnorm`; 2-D fp16/bf16, hidden size even and <= 8192, weight dtype must match. Opt-in: level with native on speed and less accurate. |
| `fused_add_rmsnorm` | `aiter` | ✅ | ✅ | `aiter::add_rmsnorm`; 2-D, hidden size even and <= 8192, weight dtype must match. Opt-in: 1.6-1.8x slower, since correctness needs two staging buffers. |
| `silu_and_mul` | `aiter` | ✅ | ✅ | `aiter::silu_and_mul`, linked at the C++ level. Opt-in; matches native in fp16, lower in bf16. |
| `fused_moe` | `aiter` | ✅ | ✅ | `aiter_fused_moe`; bf16/fp16. Weights must be pre-shuffled with `shuffle_moe_weight` or results are silently wrong. |
| `fused_moe_fp8` | `aiter` | ✅ | ✅ | `aiter_fused_moe` with fp8 weights in `moe_fp8_dtype()` plus both scales; activations are quantized per token in the shim. |
| `single_decode` | `hip` | ◻️ | ◻️ | MHA / GQA / MQA. |
| `batch_decode` | `hip` | ◻️ | ◻️ | MHA / GQA / MQA; fp8 KV-cache (E4M3FNUZ) and CUDA-graph capture. |
| `single_prefill` | `hip` | ◻️ | ◻️ | MHA / GQA / MQA, including custom attention masks. |
| `batch_prefill` | `hip` | ◻️ | ◻️ | Paged and ragged; MHA / GQA / MQA, including custom attention masks. |
| `block_sparse` | `hip` | ◻️ | ◻️ | `BlockSparseAttentionWrapper` and the variable-block variant. Native HIP FA2 only -- `determine_attention_backend` never returns `aiter` here. |
| `cascade` | `hip` | ◻️ | ◻️ | Two-level shared-prefix attention; a fused single-kernel variant is gated behind `FLASHINFER_HIP_FUSED_CASCADE=1`. |
| `pod` | `hip` | ◻️ | ◻️ | `PODWithPagedKVCacheWrapper` and the batch variant. JIT-only, excluded from AOT as upstream. |
| `rope` | `hip` | ◻️ | ◻️ | LLaMA and LLaMA 3.1 scaling; fused RoPE + fp8 quant + paged-KV append (E4M3FNUZ, E5M2FNUZ). |
| `append_paged_kv_cache` | `hip` | ◻️ | ◻️ | fp8 KV-cache supported. Sustains 3.62 TB/s against AITER's 2.86 on gfx942, so `auto` picks this. |
| `rmsnorm` | `hip` | ◻️ | ◻️ | What `auto` always picks: level with AITER on speed and more accurate. |
| `fused_add_rmsnorm` | `hip` | ◻️ | ◻️ | What `auto` always picks: 1.6-1.8x faster than AITER on both arches. |
| `layernorm` | `hip` | ◻️ | ◻️ | `layernorm` plus the Gemma RMSNorm variants. No AITER path. |
| `sampling` | `hip` | ◻️ | ◻️ | Top-K / Top-P / Min-P / OnlineSoftmax / SamplingFromLogits. |
| `logits_processor` | `hip` | ◻️ | ◻️ | Composable processor pipeline (cap, mask, temperature, ...). |
| `silu_and_mul` | `hip` | ◻️ | ◻️ | SiLU and GELU with fused gating; the default for `auto`. |
| `quantization` | `hip` | ◻️ | ◻️ | `packbits` and `segment_packbits`. |

* ✅ **validated** — this op has been exercised on this architecture.
* ◻️ **supported** — the op runs here and the test suite covers it, but no run has been recorded against this specific op and architecture.
* ⚠️ **broken on some toolchains** — usable, but not on every ROCm/AITER version; see the footnote.

[^kb1]: `batch_prefill/aiter` on gfx950, ROCm [7.2, 7.3): ROCm 7.2.x miscompiles AITER's causal batch-prefill kernel on gfx950: causal=True with logits_soft_cap=0.0 returns wrong numbers (not an error), 97.6% of elements off. Use ROCm 7.1, or backend='fa2'. Override with `FLASHINFER_ARCH_ALLOW_KNOWN_BAD=1` if you have validated it yourself. <https://github.com/ROCm/aiter/blob/main/op_tests/test_batch_prefill.py>

<!-- END GENERATED: arch support matrix -->

Every row is covered by the default `pytest` selection. Most have a
matching `tests/rocm/test_*.py`; `single_decode` is exercised
from the batch-decode, sliding-window, and logits-cap files, and
`quantization` by `tests/utils/test_quantization.py`.

**Soft-capped causal prefill avoids one AITER kernel.** AITER's
`mha_varlen_fwd` miscomputes `logits_soft_cap` for causal prefill with
`head_dim=128` (through amd-aiter 0.1.21). The affected lengths differ by
architecture — from `kv_len >= 512` on gfx942, but at *every* length on gfx950 —
so the threshold lives in `flashinfer/rocm/arch_caps.py` rather than in the call
sites. Single and ragged prefill always dispatch through that kernel, so
`backend="auto"` serves those calls with `fa2` and `backend="aiter"` raises
rather than returning wrong numbers.

Paged prefill keeps using AITER when the page size is native, because that route
takes `mha_batch_prefill` instead — measured exact on amd-aiter 0.1.20 against an
fp32 reference on both architectures. It falls back only when the run-time probe
demotes the call to a flat gather. Every other soft-cap shape — non-causal, other
head dims — is unaffected.

## `torch.compile`

Set `FLASHINFER_USE_TORCH_CUSTOM_OPS=1` **before** importing `flashinfer`
to wrap the kernels in `torch.library.custom_op` so Dynamo can trace them.
Requires PyTorch ≥ 2.4 and adds a small per-call dispatch overhead.
Without it, `torch.compile` raises a clear error if it traces into a
FlashInfer op rather than silently producing a wrong graph.

## Running the tests

```bash
pytest -n auto --reruns 2 -m "not slow"
```

**`-n auto` is derived from the GPU count, not the CPU count** — half the
physical supported cards, minimum one, so a single-GPU host runs a single
worker. Each worker sets `HIP_VISIBLE_DEVICES` to one card, which scopes the
subprocesses a test spawns — not the worker itself, where HIP is already
initialized by the time the value is set. Pass an explicit `-n N` to
override; the project's hook takes precedence over
`PYTEST_XDIST_AUTO_NUM_WORKERS`, so that variable has no effect here.
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md) covers the `slow` marker and the
rerun policy.

## Benchmarking

The unified runner drives every ROCm attention path from one testlist:

```bash
cd benchmarks
python flashinfer_benchmark.py --testlist rocm/testlist_rocm.txt \
    --output_path run-$(date +%F).csv
```

Each line requests both `fa2` and `auto` and carries its own `--refcheck`,
so the two are compared side by side. **Read the `backend_resolved` column** — `auto` is a
request, not a result, and `backend_fallback_reason` says why AITER was
declined. Per-op drivers live in
[`benchmarks/rocm/`](https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/benchmarks/rocm), and
[`benchmarks/README.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/benchmarks/README.md) documents the output columns.

## Environment variables

Read at runtime or import time:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FLASHINFER_USE_TORCH_CUSTOM_OPS` | `0` | Wrap kernels for `torch.compile`; set before importing `flashinfer`. See above. |
| `FLASHINFER_AITER_STRICT` | `0` | Raise instead of degrading when AITER cannot serve a page size natively. Set in CI to catch AITER coverage regressions rather than absorb them as a slowdown. |
| `FLASHINFER_ARCH_ALLOW_KNOWN_BAD` | `0` | Run an (op, backend, arch) combination the capability table marks known-broken on your toolchain. Only if you have validated it yourself. |
| `FLASHINFER_HIP_FUSED_CASCADE` | `0` | Use the fused single-kernel HIP cascade path instead of the two-level merge. Experimental. |
| `FLASHINFER_WORKSPACE_BASE` | `$HOME` | Parent of the JIT cache directory (`.cache/flashinfer/`). Point it at fast local disk when `$HOME` is on NFS. Pass an absolute path — the value is not tilde-expanded, so `~` becomes a literal `./~` directory. |
| `FLASHINFER_DISABLE_JIT` | unset | Set to **any non-empty value** — including `0` — to skip JIT compilation. Useful with an AOT-built install, to fail loudly on a missing kernel rather than trigger a build. |
| `FLASHINFER_DISABLE_VERSION_CHECK` | unset | Any non-empty value skips the JIT-cache package version check. |
| `FLASHINFER_LOGGING_LEVEL` | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, …). Affects AITER fallback warnings and JIT build messages. |
| `FLASHINFER_DISABLE_AOT_ARCH_CHECK` | unset | Use the prebuilt kernels even when their architecture does not match the running GPU. By default a mismatch discards them, with a warning, and everything JIT-compiles instead. |
| `ROCM_PATH` / `ROCM_HOME` | `/opt/rocm` | Where `flashinfer.rocm.hip_utils` looks for ROCm. Override only for non-standard layouts. |
| `AITER_JIT_DIR` | AITER's own | Where the C++ shim `dlopen`s AITER's built `.so` files, overriding the path compiled in at build time. |
| `GPU_ARCHS` | autodetected | AITER's own JIT architecture. An explicit value is preserved — a shim build overrides it from `FLASHINFER_ROCM_ARCH_LIST` for the build and restores yours afterwards. Left set to the derived architecture only when you had not set it. |

Build-time variables — `FLASHINFER_ROCM_ARCH_LIST`, `PYTORCH_ROCM_ARCH`,
`FLASHINFER_JIT_VERBOSE`, `FLASHINFER_EXTRA_LDFLAGS`,
`FLASHINFER_EXTRA_CFLAGS`, `FLASHINFER_EXTRA_CUDAFLAGS`,
`FLASHINFER_OWN_HEADERS_NON_SYSTEM`, `MAX_JOBS` — are documented in
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md). Note `FLASHINFER_JIT_DEBUG` is a
**no-op on ROCm/HIP**; CONTRIBUTING.md explains how to get a debug build
instead.

## Runtime helpers

```python
import torch

from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.hip_utils import check_torch_rocm_compatibility

# True on gfx942/gfx950 with a ROCm torch build. Does *not* verify the
# `aiter` package is importable — wrap the call in try/except ImportError
# if you need that guarantee.
if is_aiter_supported(torch.device("cuda")):
    ...

# Raises a clear error if PyTorch + ROCm are incompatible, e.g. a CPU-only
# torch wheel was picked up from PyPI.
check_torch_rocm_compatibility()
```

## Building from source

See [CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md) for the development container, the
editable and wheel builds, the ahead-of-time kernel build, and how to run
the test suite.

## License and acknowledgements

Apache-2.0 — see [LICENSE](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/LICENSE) and [NOTICE](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/NOTICE). Upstream
project: [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).

Contributions are welcome. Please run `pre-commit run -a` and the relevant
`pytest` selection before opening a PR.
