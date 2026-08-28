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
[CHANGELOG.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CHANGELOG.md) for the release history.

**Versioning.** Release tags are `<upstream_version>+amd.<n>`, tying each
FlashInfer+ROCm release to the upstream tag it is based on — `0.5.3+amd.1`
is the first AMD release based on upstream `v0.5.3`.

## Quick start

### Docker

AMD validates and publishes FlashInfer images on Docker Hub. The latest
validated tag:

| Docker image | ROCm | FlashInfer | PyTorch | Ubuntu | Python | GPU |
| ------------ | ---- | ---------- | ------- | ------ | ------ | --- |
| `rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1` | 7.2.0 | v0.5.3 | 2.9.1 | 24.04 | 3.12 | MI355X, MI325X, MI300X |

Older ROCm / PyTorch / FlashInfer combinations are at
<https://hub.docker.com/r/rocm/flashinfer/tags>.

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --name=flashinfer-rocm \
  rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
```

Then, inside the container:

```bash
python -c "import flashinfer; print(flashinfer.__version__)"
```

The container's micromamba environment activates on shell start, so no
manual `micromamba activate` is required.

### pip

```bash
pip install amd-flashinfer --index-url https://pypi.amd.com/simple/
pip install torch==2.9.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
```

**Torch must come from `repo.radeon.com`, via `-f` and not `--index-url`.**
That repo is a flat wheel listing rather than a PEP 503 index, so
`--index-url` fails with "No matching distribution found". pip still
prefers the ROCm wheel over a same-version PyPI wheel because its
`+rocm<X.Y>` local version ranks higher. Confirm you got one:

```bash
python -c "import torch; assert torch.version.hip, 'not a ROCm build'"
```

Kernels are JIT-compiled on first use, which takes a few minutes. The
optional [`amd-flashinfer-jit-cache`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/amd-flashinfer-jit-cache/README.md)
package ships them prebuilt for gfx942.

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
| ROCm | 7.0.2, 7.1.1, 7.2, 7.14 |
| PyTorch+ROCm | 2.8.0, 2.9.1, 2.12.0 |
| Python | 3.10+ (the published images use 3.12; the devcontainer uses 3.14) |

Other versions may work but are untested. Replace `7.2` in the torch
install command with the ROCm version you need; see
<https://repo.radeon.com/rocm/manylinux/> for what is available.

ROCm 7.14 is the exception: `repo.radeon.com` publishes no `rocm-rel-7.14/`
directory, so there is no pip recipe for it. Take torch from the
`rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0` image
instead, as the devcontainer does.

## Support matrix

Every op has an in-tree HIP kernel unless noted; a subset also has an
AITER backend, selected by a `backend=` argument that defaults to
`"auto"`. There are four policies for what `auto` picks:

| `backend="auto"` picks | Ops |
| :--- | :--- |
| AITER when the call is compatible, else the in-tree kernel | `single_prefill`, `batch_prefill`, `batch_decode`, `fused_add_rmsnorm` |
| AITER for 2-D fp16/bf16 whose weight dtype matches, else `native` | `rmsnorm` |
| Always the in-tree `native` kernel — AITER is opt-in | `silu_and_mul`, `rope`, `append_paged_kv_cache` |
| AITER only — no HIP kernel exists | `mla` |

To override, pass `backend="aiter"`, or name the in-tree kernel:
`backend="fa2"` for the attention wrappers, `backend="native"` for
everything else.

Two ops take no `backend=` argument: `single_decode_with_kv_cache` is
HIP-only, and `aiter_fused_moe` is AITER-only. On gfx950 with ROCm 7.2.x,
`batch_prefill` never resolves to AITER — see the footnote under the
table.

**The full routing rules, per-op constraints, and AITER install
instructions are in [`docs/rocm/backends.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md).**
Read it before relying on an AITER path — several attention kwargs are
silently ignored there rather than rejected.

The table below is generated from
[`flashinfer/arch_caps.py`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/flashinfer/arch_caps.py), which is what
`backend="auto"` consults at runtime, so it cannot drift from the routing
decisions the library makes. Do not edit it by hand; run
`python3 scripts/gen_arch_support_matrix.py`.

<!-- BEGIN GENERATED: arch support matrix -- scripts/gen_arch_support_matrix.py -->

| Op | Backend | gfx942 (CDNA3) | gfx950 (CDNA4) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `batch_decode` | `aiter` | ✅ | ✅ | MHA / GQA / MQA with sliding window; fp16/bf16 + NHD. Graph capture is opt-in via `backend="aiter"`. |
| `single_prefill` | `aiter` | ✅ | ✅ | MHA / GQA / MQA; fp16/bf16 + NHD, equal Q/KV dtypes and head dims, no custom mask. fp8 WIP. |
| `batch_prefill` | `aiter` | ✅ | ⚠️[^kb1] | Paged and ragged. Page sizes 128/256/1024 are native on amd-aiter >= 0.1.10; others take a flat gather. |
| `mla` | `aiter` | ✅ | ✅ | DeepSeek-style 192/128 head-dim split; fp16/bf16. No HIP kernel exists, so `auto` resolves here. |
| `rope` | `aiter` | ✅ | ✅ | `apply_rope_with_cos_sin_cache` and its inplace variant, linked at the C++ level. Opt-in. |
| `append_paged_kv_cache` | `aiter` | ✅ | ✅ | fp16/bf16 + NHD. Bit-exact with the in-tree kernel but slower, so `auto` picks `native`. |
| `rmsnorm` | `aiter` | ✅ | ✅ | CK `rmsnorm2d`. `auto` routes only 2-D fp16/bf16 with a matching weight dtype here. |
| `fused_add_rmsnorm` | `aiter` | ✅ | ✅ | CK `rmsnorm2d_with_add`; 2-D only. `auto` does NOT check weight dtype — a mismatch silently yields garbage. |
| `silu_and_mul` | `aiter` | ✅ | ✅ | `aiter::silu_and_mul`, linked at the C++ level. Opt-in; matches native in fp16, lower in bf16. |
| `fused_moe` | `aiter` | ✅ | ✅ | `aiter_fused_moe`; bf16/fp16. Weights must be pre-shuffled with `shuffle_moe_weight` or results are silently wrong. |
| `fused_moe_fp8` | `aiter` | ✅ | ✅ | `aiter_fused_moe` with fp8 weights in `moe_fp8_dtype()` plus both scales; activations are quantized per token in the shim. |
| `single_decode` | `hip` | ◻️ | ◻️ | MHA / GQA / MQA. |
| `batch_decode` | `hip` | ◻️ | ◻️ | MHA / GQA / MQA; fp8 KV-cache (E4M3FNUZ) and CUDA-graph capture. |
| `single_prefill` | `hip` | ◻️ | ◻️ | MHA / GQA / MQA, including custom attention masks. |
| `batch_prefill` | `hip` | ◻️ | ◻️ | Paged and ragged; MHA / GQA / MQA, including custom attention masks. |
| `cascade` | `hip` | ◻️ | ◻️ | Two-level shared-prefix attention; a fused single-kernel variant is gated behind `FLASHINFER_HIP_FUSED_CASCADE=1`. |
| `pod` | `hip` | ◻️ | ◻️ | `PODWithPagedKVCacheWrapper` and the batch variant. JIT-only, excluded from AOT as upstream. |
| `rope` | `hip` | ◻️ | ◻️ | LLaMA and LLaMA 3.1 scaling; fused RoPE + fp8 quant + paged-KV append (E4M3FNUZ, E5M2FNUZ). |
| `append_paged_kv_cache` | `hip` | ◻️ | ◻️ | fp8 KV-cache supported. Sustains 3.62 TB/s against AITER's 2.86 on gfx942, so `auto` picks this. |
| `rmsnorm` | `hip` | ◻️ | ◻️ | The fallback for 3-D inputs, fp32, or a weight dtype that does not match the input. |
| `fused_add_rmsnorm` | `hip` | ◻️ | ◻️ | The fallback whenever the AITER path is unavailable. |
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
matching `tests/rocm_tests/test_*_hip.py`; `single_decode` is exercised
from the batch-decode, sliding-window, and logits-cap files, and
`quantization` by `tests/utils/test_quantization.py`.

**Soft-capped causal prefill does not use AITER.** AITER miscomputes
`logits_soft_cap` for causal prefill with `head_dim=128` and `kv_len >= 512`
(through amd-aiter 0.1.21), so `backend="auto"` serves those calls with `fa2`
and `backend="aiter"` raises rather than returning wrong numbers. Every other
soft-cap shape — non-causal, other head dims, shorter contexts — still goes to
AITER.

## `torch.compile`

Set `FLASHINFER_USE_TORCH_CUSTOM_OPS=1` **before** importing `flashinfer`
to wrap the kernels in `torch.library.custom_op` so Dynamo can trace them.
Requires PyTorch ≥ 2.4 and adds a small per-call dispatch overhead.
Without it, `torch.compile` raises a clear error if it traces into a
FlashInfer op rather than silently producing a wrong graph.

<!--
## Benchmarking

The unified runner drives every ROCm attention path from one testlist:

```bash
cd benchmarks
python flashinfer_benchmark.py --testlist rocm_benchmarks/testlist_rocm.txt \
    --output_path run-$(date +%F).csv
```

Each line requests both `fa2` and `auto`, so `--refcheck` compares them
side by side. **Read the `backend_resolved` column** — `auto` is a
request, not a result, and `backend_fallback_reason` says why AITER was
declined. Per-op drivers live in
[`benchmarks/rocm_benchmarks/`](https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/benchmarks/rocm_benchmarks).
-->

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
| `ROCM_PATH` / `ROCM_HOME` | `/opt/rocm` | Where `flashinfer.hip_utils` looks for ROCm. Override only for non-standard layouts. |

Build-time variables — `FLASHINFER_ROCM_ARCH_LIST`, `FLASHINFER_JIT_VERBOSE`,
`FLASHINFER_EXTRA_LDFLAGS`, `MAX_JOBS` — are documented in
[CLAUDE.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CLAUDE.md). Note `FLASHINFER_JIT_DEBUG` is a **no-op on
ROCm/HIP**; CLAUDE.md explains how to get a debug build instead.

## Runtime helpers

```python
import torch

from flashinfer.aiter_utils import is_aiter_supported
from flashinfer.hip_utils import check_torch_rocm_compatibility

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
