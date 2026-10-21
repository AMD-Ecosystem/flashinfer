# FlashInfer+ROCm: An AMD ROCm port of FlashInfer

FlashInfer+ROCm brings the
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) inference kernel
library to AMD Instinct GPUs — CDNA3 (gfx942, MI300X / MI325X) and CDNA4
(gfx950, MI350X / MI355X). It ships in-tree HIP ports of the attention,
KV-cache, RoPE, normalization, sampling, and logits-processor kernels, and
transparently dispatches a subset of ops to AMD's
[AITER](https://github.com/ROCm/aiter) backend when that is the faster or
only path.

In active development, aimed at developers embedding FlashInfer kernels into
their own training or serving stack. Release tags are
`<upstream_version>+amd.<n>` — `0.6.18+amd.1` is the first AMD release based on
upstream `v0.6.18`. History is under
[Releases](https://github.com/AMD-Ecosystem/flashinfer/releases).

## Quick start

No wheel or image is published for this release — build from the repository.
The development image carries a matched ROCm, PyTorch, Python and AITER set,
so it is the shortest path to a working environment:

```bash
docker build -t flashinfer-dev:rocm10.0 -f docker/Dockerfile.rocm . \
  --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g)
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v "$PWD":/workspace -w /workspace flashinfer-dev:rocm10.0
```

The image carries no source — `-v "$PWD":/workspace` is what puts it there.
The `--build-arg` trio matches the container user to yours; without it the
image runs as UID 1003 and the editable install cannot write to your mounted
tree. `render` must be the host's **numeric** GID: the name resolves against
the image's own group. Then, inside the container:

```bash
python -m pip install --no-build-isolation -ve .
python -c "import flashinfer; print(flashinfer.__version__)"
```

[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md)
has the full recipe: the `docker run` GPU flags, the wheel build, and the
ahead-of-time kernel build.

**Bringing your own environment?** The image is the supported path because of
torch: `repo.radeon.com` publishes no `rocm-rel-` directory for ROCm 10.0, so
no pip command installs the torch 2.12 build this release is tested against —
it comes from the base image
(`rocm/pytorch:rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.12.0`). Whatever
you assemble, check you did not land on a CPU-only wheel:

```bash
python -c "import torch; assert torch.version.hip, 'not a ROCm build'"
```

Kernels JIT-compile on first use — a minute or so for an in-tree HIP kernel,
20+ minutes for a cold AITER variant. The optional
[`amd-flashinfer-jit-cache`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/amd-flashinfer-jit-cache/README.md)
ships them prebuilt, one wheel per architecture with the architecture in the
version's local segment (`0.6.18+amd.1.gfx942`). Pin it in full: an
unqualified requirement resolves to whichever architecture sorts highest.

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

[`examples/`](https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/examples)
has runnable single/batch prefill and batch decode scripts, plus
`amd_flashinfer_rocm_tutorial.ipynb` walking the public API on ROCm:

```bash
python examples/single_prefill_example.py
```

## Supported hardware and toolchain

**One configuration is supported: the one `docker/Dockerfile.rocm` builds
and this release is tested on.**

| | Supported |
| :--- | :--- |
| GPUs | gfx942 (CDNA3 — MI300X, MI325X), gfx950 (CDNA4 — MI350X, MI355X) |
| ROCm | 10.0 |
| PyTorch+ROCm | 2.12.0 |
| Python | 3.12 |
| OS | Ubuntu 24.04 |
| `amd-aiter` | 0.1.20 |

Nothing rejects another combination at install time, and older ROCm and torch
releases have worked here before — but they are untested, uncovered by the
matrix below, and not what a bug report will be reproduced against.

The pins move together, which is why the supported configuration is an image
rather than a list of versions: every `amd-aiter` 0.1.20 wheel is cp312 only,
fixing the interpreter; torch must stay at 2.12, since 2.13 drops a `c10`
symbol those wheels' prebuilt prefill kernels need; and its ROCm 10.0 build
exists only in the base image.

## Support matrix

Every op has an in-tree HIP kernel unless noted; a subset also has an AITER
backend, selected by a `backend=` argument defaulting to `"auto"`, which
follows one of three policies:

| `backend="auto"` picks | Ops |
| :--- | :--- |
| AITER when the call is compatible, else the in-tree kernel | `single_prefill`, `batch_prefill`, `batch_decode` |
| Always the in-tree `native` kernel — AITER is opt-in | `rmsnorm`, `fused_add_rmsnorm`, `silu_and_mul`, `rope`, `append_paged_kv_cache` |
| AITER only — no HIP kernel exists | `mla` |

To override, pass `backend="aiter"`, or name the in-tree kernel —
`backend="fa2"` for the attention wrappers, `backend="native"` for everything
else. Some entry points take no `backend=` at all:
`single_decode_with_kv_cache` (HIP-only), `aiter_fused_moe` (AITER-only), and
the three cascade wrappers, whose per-level attention is still auto-routed and
so can reach AITER without being pinnable.

Beyond the routed ops this release carries block-sparse attention
(`BlockSparseAttentionWrapper` and the variable-block variant), POD attention
(`PODWithPagedKVCacheWrapper` and its batch variant), cascade attention, the
sampling and logits-processor pipelines, and fp8 fused MoE via
`aiter_fused_moe`. Batch decode reaches AITER under CUDA-graph capture once
you declare a `max_seq_len` capacity on the wrapper.

**[`docs/rocm/backends.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md)
has the full routing rules, per-op constraints, AITER install instructions and
unavailable-module lists.** Read it before relying on an AITER path — several
attention kwargs are silently ignored there rather than rejected.

**One row per (op, backend) pair, not per op.** The eight ops with both
backends appear twice, which is why the `Backend` column also says whether
`auto` takes that row by default. `cascade` is the odd one: only its merge
kernels are HIP, and each attention level routes on its own.

The table is generated from
[`flashinfer/rocm/arch_caps.py`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/flashinfer/rocm/arch_caps.py) —
what `backend="auto"` consults at runtime, so it cannot drift from the
library's actual routing. Do not edit it by hand; run
`python3 scripts/gen_arch_support_matrix.py`.

<!-- BEGIN GENERATED: arch support matrix -- scripts/gen_arch_support_matrix.py -->

| Op | Backend | gfx942 (CDNA3) | gfx950 (CDNA4) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `batch_decode` | `aiter` -- auto picks this when compatible | ✅ | ✅ | MHA / GQA / MQA with sliding window; fp16/bf16 + NHD. Under graph capture `auto` needs a declared `max_seq_len`, else it stays on fa2. |
| `single_prefill` | `aiter` -- auto picks this when compatible | ✅ | ✅ | MHA / GQA / MQA with sliding window; fp16/bf16 + NHD, equal Q/KV dtypes and head dims, no custom mask. fp8 WIP. |
| `batch_prefill` | `aiter` -- auto picks this when compatible | ✅ | ✅ | Paged and ragged, with sliding window. Page sizes 128/256/1024 are served natively; others take a flat gather. |
| `mla` | `aiter` -- only backend | ✅ | ✅ | DeepSeek-style 192/128 head-dim split; fp16/bf16. No HIP kernel exists, so `auto` resolves here. |
| `rope` | `aiter` -- opt-in | ✅ | ✅ | `apply_rope_with_cos_sin_cache` and its inplace variant, linked at the C++ level. Opt-in. |
| `append_paged_kv_cache` | `aiter` -- opt-in | ✅ | ✅ | fp16/bf16 + NHD. Bit-exact with the in-tree kernel but slower, so `auto` picks `native`. |
| `rmsnorm` | `aiter` -- opt-in | ✅ | ✅ | `aiter::rmsnorm`; 2-D fp16/bf16, hidden size even and <= 8192, weight dtype must match. Opt-in: level with native on speed and less accurate. |
| `fused_add_rmsnorm` | `aiter` -- opt-in | ✅ | ✅ | `aiter::add_rmsnorm`; 2-D, hidden size even and <= 8192, weight dtype must match. Opt-in: 1.6-1.8x slower, since correctness needs two staging buffers. |
| `silu_and_mul` | `aiter` -- opt-in | ✅ | ✅ | `aiter::silu_and_mul`, linked at the C++ level. Opt-in; matches native in fp16, lower in bf16. |
| `fused_moe` | `aiter` -- only backend | ✅ | ✅ | `aiter_fused_moe`; bf16/fp16. Weights must be pre-shuffled with `shuffle_moe_weight` or results are silently wrong. |
| `fused_moe_fp8` | `aiter` -- only backend | ✅ | ✅ | `aiter_fused_moe` with fp8 weights in `moe_fp8_dtype()` plus both scales; activations are quantized per token in the shim. |
| `single_decode` | `hip` -- only backend | ✅ | ✅ | MHA / GQA / MQA. |
| `batch_decode` | `hip` -- fallback; auto tries `aiter` first | ✅ | ✅ | MHA / GQA / MQA; fp8 KV-cache (E4M3FNUZ) and CUDA-graph capture. |
| `single_prefill` | `hip` -- fallback; auto tries `aiter` first | ✅ | ✅ | MHA / GQA / MQA, including custom attention masks. |
| `batch_prefill` | `hip` -- fallback; auto tries `aiter` first | ✅ | ✅ | Paged and ragged; MHA / GQA / MQA, including custom attention masks. |
| `block_sparse` | `hip` -- only backend | ✅ | ✅ | `BlockSparseAttentionWrapper` and the variable-block variant. Native HIP FA2 only -- `determine_attention_backend` never returns `aiter` here. |
| `cascade` | `hip` -- merge only; levels are auto-routed and can be `aiter` | ✅ | ✅ | Two-level shared-prefix attention. The `hip` backend is the merge kernels only: each level runs through the ordinary prefill/decode entry points at `backend="auto"`, so it can reach AITER, and no cascade wrapper exposes `backend=` to override that. `FLASHINFER_HIP_FUSED_CASCADE=1` threads partial state through the levels of `MultiLevelCascadeAttentionWrapper` only; AITER levels and both shared-prefix wrappers still merge post-hoc. |
| `pod` | `hip` -- only backend | ✅ | ✅ | `PODWithPagedKVCacheWrapper` and the batch variant. JIT-only, excluded from AOT as upstream. |
| `rope` | `hip` -- auto picks this | ✅ | ✅ | LLaMA and LLaMA 3.1 scaling; fused RoPE + fp8 quant + paged-KV append (E4M3FNUZ, E5M2FNUZ). |
| `append_paged_kv_cache` | `hip` -- auto picks this | ✅ | ✅ | fp8 KV-cache supported. Sustains 3.62 TB/s against AITER's 2.86 on gfx942, so `auto` picks this. |
| `rmsnorm` | `hip` -- auto picks this | ✅ | ✅ | What `auto` always picks: level with AITER on speed and more accurate. |
| `fused_add_rmsnorm` | `hip` -- auto picks this | ✅ | ✅ | What `auto` always picks: 1.6-1.8x faster than AITER on both arches. |
| `layernorm` | `hip` -- only backend | ✅ | ✅ | `layernorm` plus the Gemma RMSNorm variants. No AITER path. |
| `sampling` | `hip` -- only backend | ✅ | ✅ | Top-K / Top-P / Min-P / OnlineSoftmax / SamplingFromLogits. |
| `logits_processor` | `hip` -- only backend | ✅ | ✅ | Composable processor pipeline (cap, mask, temperature, ...). |
| `silu_and_mul` | `hip` -- auto picks this | ✅ | ✅ | SiLU and GELU with fused gating; the default for `auto`. |
| `quantization` | `hip` -- only backend | ✅ | ✅ | `packbits` and `segment_packbits`. |

* ✅ **supported** — this op runs on this architecture and the test suite covers it.

<!-- END GENERATED: arch support matrix -->

Every row is covered by the default `pytest` selection — most by a matching
`tests/rocm/test_*.py`, `single_decode` from the batch-decode, sliding-window
and logits-cap files, and `quantization` by `tests/utils/test_quantization.py`.

### What upstream has that this does not

The table above is what works. Upstream v0.6.18 is larger, and the rest of it
splits by *why* it is absent, not by how it fails:

| | Why | How it fails | Examples |
| :--- | :--- | :--- | :--- |
| **NVIDIA-only** | Wraps a vendor library, NVIDIA PTX, an SM-gated kernel, the CUDA driver API, or CuTe DSL. A ROCm equivalent means a rewrite, not a port | Gated where the import already breaks; otherwise the kernel surface is simply absent (`cute_dsl`, `cutile`) or the call fails | `flashinfer.gemm`, `flashinfer.fused_moe` (upstream's CUTLASS MoE — the `fused_moe` *op* above is AITER's and works), `flashinfer.cudnn`, `flashinfer.deep_gemm`, `flashinfer.green_ctx`, `flashinfer.aot`, `flashinfer.cute_dsl`, `flashinfer.cutile`, `flashinfer.msa_ops`, `flashinfer.moe_ep`, `flashinfer.gdn_decode`, `flashinfer.gdn_prefill`, `flashinfer.xqa`, `flashinfer.nvfp4_attention_sm120`, `flashinfer.tllm_utils`, `flashinfer.mamba.ssd_combined`, `flashinfer.comm`'s NVLink/NVSHMEM transports |
| **Not ported yet** | An ordinary kernel with no `csrc/rocm` source. Portable in principle; nobody has done it | On first call, from ninja or the binding lookup | `flashinfer.topk`, `flashinfer.topk_varlen`, `flashinfer.mhc`, `flashinfer.concat_ops`, `flashinfer.diffusion_ops`, `flashinfer.kda*`, `flashinfer.mamba`'s `selective_state_update` and `checkpointing_ssu` |

Gating is deliberate: it turns an obscure failure from inside the JIT into one
catchable error naming the module. For those entries, feature-detect with
`try: import ... except ImportError` rather than `importlib.util.find_spec` —
the files ship, the import is what is gated. **A successful import is not a
support check**: most rows above import fine and fail later, so probe the
symbol you actually need (`hasattr`, or `is_cute_dsl_available()` and
`is_cuda_tile_available()` for those two).

Two modules are simply uninstallable here rather than unsupported:
`flashinfer.profiler` needs `tg4perfetto` and `flashinfer.artifacts` needs
`requests`, neither of which `docker/Dockerfile.rocm` ships.

[`docs/rocm/backends.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md)
has the complete lists and the evidence for each entry;
`tests/rocm/test_kernel_source_coverage.py` fails when a newly vendored op
names a kernel source `csrc/rocm` does not have, so the second group cannot
grow unnoticed.

**Soft-capped causal prefill falls back to `fa2`.** AITER's `mha_varlen_fwd`
miscomputes `logits_soft_cap` at `head_dim=128`, so `auto` declines it and
`backend="aiter"` raises rather than returning wrong numbers — see
[per-op notes](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md#per-op-notes).

## `torch.compile`

Set `FLASHINFER_USE_TORCH_CUSTOM_OPS=1` **before** importing `flashinfer` to
wrap the kernels in `torch.library.custom_op` so TorchDynamo can trace them.
Requires PyTorch ≥ 2.4 and adds a small per-call dispatch overhead. Without it,
`torch.compile` raises a clear error on tracing into a FlashInfer op rather
than silently producing a wrong graph.

## Running the tests

```bash
pytest -n auto --reruns 2 -m "not slow"
```

**`-n auto` counts GPUs, not CPUs** — half the physical supported cards,
minimum one. Pass `-n N` to override.
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md)
covers worker pinning, the `slow` marker and the rerun policy.

## Benchmarking

The unified runner drives batch decode and paged/ragged batch prefill from one
testlist — the routed paths, not every op in the matrix above:

```bash
cd benchmarks
python flashinfer_benchmark.py --testlist rocm/testlist_rocm.txt \
    --output_path run-$(date +%F).csv
```

Each line requests both `fa2` and `auto` with its own `--refcheck`, so the two
compare side by side **where both survive capability filtering** — the
Llama-3.1-405B rows lose `fa2` to its GQA group-size set and run unverified, so
count rows per config rather than assuming two. **Read `backend_resolved`**:
`auto` is a request, not a result, and `backend_fallback_reason` says why AITER
was declined. Per-op drivers are in
[`benchmarks/rocm/`](https://github.com/AMD-Ecosystem/flashinfer/tree/amd-integration/benchmarks/rocm);
[`benchmarks/README.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/benchmarks/README.md)
documents the output columns.

## Environment variables

Read at runtime or import time:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FLASHINFER_USE_TORCH_CUSTOM_OPS` | `0` | Wrap kernels for `torch.compile`; set before importing `flashinfer`. See above. |
| `FLASHINFER_AITER_STRICT` | `0` | Raise instead of degrading when AITER cannot serve a page size natively. Set in CI to catch coverage regressions rather than absorb them as a slowdown. |
| `FLASHINFER_ARCH_ALLOW_KNOWN_BAD` | `0` | Run an (op, backend, arch) combination the capability table marks known-broken on your toolchain. Only if you have validated it yourself. |
| `FLASHINFER_HIP_FUSED_CASCADE` | `0` | In `MultiLevelCascadeAttentionWrapper` only, pass each level's partial state into the next prefill call instead of merging afterwards; AITER levels and the shared-prefix wrappers ignore it. Both paths tested. Read once at import, so set it first. |
| `FLASHINFER_WORKSPACE_BASE` | `$HOME` | Parent of the JIT cache (`.cache/flashinfer/`); point at fast local disk when `$HOME` is on NFS. Prefer an absolute path: the value is wrapped in `pathlib.Path` unvalidated, so a relative one resolves against the working directory and `~` becomes a literal `./~` directory. |
| `FLASHINFER_DISABLE_JIT` | unset | **Any non-empty value** — including `0` — skips JIT compilation. Use with an AOT-built install to fail loudly on a missing kernel rather than trigger a build. |
| `FLASHINFER_DISABLE_VERSION_CHECK` | unset | Any non-empty value skips the JIT-cache package version check. |
| `FLASHINFER_LOGGING_LEVEL` | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, …). Affects AITER fallback warnings and JIT build messages. |
| `FLASHINFER_DISABLE_AOT_ARCH_CHECK` | unset | Use prebuilt kernels even when their architecture does not match the running GPU. By default a mismatch discards them, with a warning, and everything JIT-compiles. |
| `ROCM_PATH` / `ROCM_HOME` | `/opt/rocm` | Where `flashinfer.rocm.hip_utils` looks for ROCm. Override only for non-standard layouts. |
| `AITER_JIT_DIR` | AITER's own | Where the C++ shim `dlopen`s AITER's built `.so` files, overriding the path compiled in at build time. |
| `GPU_ARCHS` | autodetected | AITER's own JIT architecture. A shim build overrides it from `FLASHINFER_ROCM_ARCH_LIST` and restores your value afterwards, leaving the derived one only if you had not set it. |

Build-time variables — `FLASHINFER_ROCM_ARCH_LIST`, `PYTORCH_ROCM_ARCH`,
`FLASHINFER_JIT_VERBOSE`, `FLASHINFER_EXTRA_{LDFLAGS,CFLAGS,CUDAFLAGS}`,
`FLASHINFER_OWN_HEADERS_NON_SYSTEM`, `MAX_JOBS` — live in
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md),
which also explains how to get a debug build, since `FLASHINFER_JIT_DEBUG` is a
**no-op on ROCm/HIP**.

## Runtime helpers

```python
import torch

from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.hip_utils import check_torch_rocm_compatibility

# True on gfx942/gfx950 with a ROCm torch build. Does *not* verify the
# `aiter` package imports — wrap in try/except ImportError if you need that.
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
