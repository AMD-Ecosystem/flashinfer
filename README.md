# FlashInfer+ROCm: An AMD ROCm port of FlashInfer

FlashInfer+ROCm brings the
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) inference
kernel library to AMD Instinct GPUs — currently
[CDNA3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
(gfx942 — MI300X / MI325X) and
[CDNA4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
(gfx950 — MI355X). It ships in-tree HIP ports of the attention,
KV-cache, RoPE, normalization, sampling, and logits-processor kernels,
and transparently dispatches a subset of attention paths to AMD's
[AITER](https://github.com/ROCm/aiter) backend when its compatibility
conditions hold (see [Feature Support Matrix](#feature-support-matrix)).

The port is in active development and is aimed at developers embedding
FlashInfer kernels into their own training or serving stack. See
[CHANGELOG.md](CHANGELOG.md) for the full release history.

**Versioning:** The release tag format `<upstream_version>+amd.<n>` ties
each FlashInfer+ROCm release to its corresponding upstream tag (e.g.
`0.5.3+amd.1` is the first AMD release based on upstream `v0.5.3`).

## Table of Contents

* [Feature Support Matrix](#feature-support-matrix)
* [GPU, ROCm, and PyTorch Support](#gpu-rocm-and-pytorch-support)
* [Getting Started](#getting-started)
  * [Option 1: Get a Pre-built Docker Image](#option-1-get-a-pre-built-docker-image)
  * [Option 2: Install from a Wheel Package](#option-2-install-from-a-wheel-package)
  * [Trying the Examples](#trying-the-examples)
* [Install from Source](#install-from-source)
  * [Setting up a Development Environment](#setting-up-a-development-environment)
  * [Building and Installing a Wheel Package](#building-and-installing-a-wheel-package)
  * [Running Tests](#running-tests)
* [AITER Support](#aiter-support)
  * [Install AITER from source](#install-aiter-from-source)
  * [Install AITER wheel package](#install-aiter-wheel-package)
  * [Known Limitations](#known-limitations)
* [Environment Variables](#environment-variables)
* [Runtime Helpers](#runtime-helpers)
* [Basic Usage](#basic-usage)
* [License and Acknowledgements](#license-and-acknowledgements)

## Feature Support Matrix

Most kernels ship with an in-tree HIP implementation. A subset also has
an AITER backend; for those, `backend="auto"` picks AITER when its
compatibility conditions hold and falls back to HIP otherwise. The one
AITER-only kernel today (MLA) has no HIP path — `backend="auto"`
resolves directly to `"aiter"`.

Legend: **HIP** = in-tree kernel (`fa2` for attention, `native` JIT
kernel for non-attention ops). **AITER** = ROCm AITER backend.

| Kernel | HIP | AITER | `backend="auto"` resolves to | Notes |
| :--- | :---: | :---: | :--- | :--- |
| **Single decode attention** | ✅ `fa2` | — | HIP | MHA / GQA / MQA |
| **Batch decode attention (paged)** | ✅ `fa2` | ✅ | **AITER** when `fp16/bf16` + `NHD` + `pos_encoding_mode="NONE"` + no CUDA-graph + `use_tensor_cores=False`; else **HIP** | MHA / GQA / MQA; **fp8 KV-cache (E4M3FNUZ)** on the HIP path; sliding-window on the AITER path; CUDA-graph auto-routes back to HIP |
| **Single prefill attention** | ✅ `fa2` | ✅ | **AITER** when `fp16/bf16` + `NHD` + no custom mask + equal Q/KV dtypes & head dims + `pos_encoding_mode="NONE"`; else **HIP** | MHA / GQA / MQA; fp8 WIP |
| **Batch prefill attention (paged + ragged)** | ✅ `fa2` | ✅ | Same auto criteria as single prefill | MHA / GQA / MQA; fp8 WIP. AITER native page sizes: `{16, 1024}` (`{128, 256, 1024}` on `amd-aiter==0.1.10`); other sizes go through a gather on the AITER path |
| **Cascade attention** | ✅ | — | HIP | Two-level shared-prefix attention; a fused single-kernel HIP variant is gated behind `FLASHINFER_HIP_FUSED_CASCADE=1` |
| **MLA (Multi-Latent Attention)** | — | ✅ | **AITER** (no HIP fallback) | DeepSeek-style 192/128 head-dim split; bf16 + `page_size=1`; `backend="auto"` (default) resolves to `"aiter"` |
| **POD attention** | ✅ `fa2` | — | HIP | MHA / GQA / MQA; single + batch variants (`PODWithPagedKVCacheWrapper`, `BatchPODWithPagedKVCacheWrapper`); JIT-only (excluded from AOT, same as upstream CUDA) |
| **RoPE (positional encoding)** | ✅ | — | HIP | LLaMA-style + LLaMA 3.1 scaling; fused RoPE + fp8 quant + paged-KV append (E4M3FNUZ, E5M2FNUZ) |
| **Paged KV-cache append** | ✅ `native` | ✅ | **AITER** when `fp16/bf16` + `NHD` + gfx942/gfx950 + AITER importable; else **HIP `native`** | `append_paged_kv_cache`; fp8 KV-cache supported on the HIP path |
| **RMSNorm** | ✅ `native` | ✅ | **HIP `native`** (auto stays on HIP — AITER is opt-in via `backend="aiter"`) | AITER path is fp16/bf16, 2-D only; slightly lower precision at `hidden_size >= 1024` |
| **Fused add RMSNorm** | ✅ `native` | ✅ | **AITER** when 2-D + `>= 4M` elements + gfx942/gfx950 + AITER importable; else **HIP `native`** | `fused_add_rmsnorm`; AITER (CK `rmsnorm2d_fwd_with_add`) wins on large bandwidth-bound shapes; 2-D only, slightly lower precision at `hidden_size >= 1024` |
| **LayerNorm / Gemma RMSNorm** | ✅ | — | HIP | |
| **Sampling** | ✅ | — | HIP | Top-K / Top-P / Min-P / OnlineSoftmax / SamplingFromLogits |
| **Logits processor** | ✅ | — | HIP | Composable processor pipeline (cap, mask, temperature, …) |
| **Activation** | ✅ `native` | ✅ | **AITER** for `silu_and_mul` when `fp16` + 2-D + `>= 33M` elements; else **HIP `native`** | SiLU / GELU with fused gating. AITER path (`silu_and_mul` only) is opt-in via `backend="aiter"`; matches native precision in fp16, lower in bf16 |
| **Quantization** | ✅ | — | HIP | `packbits`, `segment_packbits` |
| **`torch.compile`** | ✅ (opt-in) | n/a | n/a | Set `FLASHINFER_USE_TORCH_CUSTOM_OPS=1` **before** importing `flashinfer`; requires PyTorch ≥ 2.4. Without it, `torch.compile` raises a clear error if it traces into a flashinfer op |

Every ✅ row above is exercised by a matching `tests/rocm_tests/test_*_hip.py`.
The full set of conditions that cause AITER auto-routing to fall back to
HIP is documented in [Known Limitations](#known-limitations) below.

## GPU, ROCm, and PyTorch Support

**Supported GPUs:** gfx942 (CDNA3 — MI300X, MI325X), gfx950 (CDNA4 — MI355X).

**Supported ROCm versions:** 7.0.2, 7.1.1, 7.2.

**Supported PyTorch+ROCm versions:** 2.8.0, 2.9.1.

Install the matching ROCm-enabled PyTorch wheel from
<https://repo.radeon.com>:

```bash
pip install torch==2.9.1 --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
```

Other versions may work but have not been tested. Replace `7.2` with the
ROCm version you need; refer to
<https://repo.radeon.com/rocm/manylinux/rocm-rel-{rocm-version}/> for
available wheels.

## Getting Started

### Option 1: Get a Pre-built Docker Image

AMD validates and publishes FlashInfer images with ROCm backends on
Docker Hub. The latest validated tag is:

| Docker image | ROCm | FlashInfer | PyTorch | Ubuntu | Python | GPU |
| ------------ | ---- | ---------- | ------- | ------ | ------ | --- |
| `rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1` | 7.2.0 | v0.5.3 | 2.9.1 | 24.04 | 3.12 | MI355X, MI325X, MI300X |

For older releases (earlier ROCm / PyTorch / FlashInfer combinations),
see the full tag list at
<https://hub.docker.com/r/rocm/flashinfer/tags>.

**Start a container:**

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 128G --name=flashinfer-rocm \
  rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
```

**Verify the installation:**

```bash
python -c "import flashinfer; print(flashinfer.__version__)"
```

Expected output: `0.5.3+amd.1` (with a possible JIT backend message).
The container's micromamba environment is activated automatically on
shell start — no manual `micromamba activate` is required.

### Option 2: Install from a Wheel Package

Install from AMD's package repository:

```bash
pip install amd-flashinfer --index-url https://pypi.amd.com/simple/
```

Install the matching ROCm-enabled torch package from <https://repo.radeon.com>:

```bash
pip install torch==2.9.1 --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
```

**NOTE:** Use `--index-url` (not `-f`) so pip cannot silently fall back
to a CPU-only PyPI wheel.

### Trying the Examples

Runnable scripts live in the [`examples/`](examples/) directory of this
repository (single/batch prefill, batch decode, plus an
`amd_flashinfer_rocm_tutorial.ipynb` Jupyter notebook). After cloning,
run any of them directly, for example:

```bash
python examples/single_prefill_example.py
```

## Install from Source

### Setting up a Development Environment

Build the development Docker image with the repository's Dockerfile:

```bash
docker build \
  --build-arg ROCM_VERSION=7.2 \
  --build-arg PY_VERSION=3.12 \
  --build-arg TORCH_VERSION=2.9.1 \
  --build-arg USERNAME=$USER \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1 \
  -f .devcontainer/rocm/Dockerfile .
```

<!-- markdownlint-disable MD033 -->
<details>
<summary>Build argument descriptions</summary>

* `ROCM_VERSION`: ROCm version (default: 7.2)
* `PY_VERSION`: Python version (default: 3.12)
* `TORCH_VERSION`: PyTorch version (default: 2.9.1)
* `USERNAME`: Username inside container (default: devuser)
* `USER_UID`: User ID for matching host permissions
* `USER_GID`: Group ID for matching host permissions

</details>
<!-- markdownlint-enable MD033 -->

**Run the development container:**

```bash
docker run -it \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --privileged --shm-size=128G --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v $PWD:/workspace \
  --name flashinfer-dev-container \
  flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
```

<!-- markdownlint-disable MD033 -->
<details>
<summary>Docker run argument descriptions</summary>

* `--cap-add=SYS_PTRACE`: Enables debugging
* `--security-opt seccomp=unconfined`: Relaxes security for development
* `--ipc=host`: Shares host IPC for better performance
* `--privileged`: Required for GPU access
* `--shm-size=128G`: Shared memory size (adjust as needed)
* `--network=host`: Uses host networking
* `--device=/dev/kfd --device=/dev/dri`: Exposes AMD GPU devices
* `--group-add video --group-add render`: GPU access groups
* `-v <host-path>:<container-path>`: Mounts source code

</details>
<!-- markdownlint-enable MD033 -->

### Building and Installing a Wheel Package

**Build with JIT (Just-in-Time) compilation only:**

```bash
python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
cd dist && pip install amd_flashinfer-*.whl
```

**Editable install for development:**

```bash
python -m pip install --no-build-isolation -ve .
```

**Note:** The `--no-deps` flag assumes dependencies are pre-installed.
Omit it to download dependencies during build. AOT builds take longer
and use more disk space but avoid JIT compilation at runtime.

### Running Tests

Run the Python test suite with pytest:

```bash
# Run default tests (configured in pyproject.toml)
pytest

# Run specific test file
pytest tests/rocm_tests/test_batch_decode_kernels_hip.py

# Run with pattern matching
pytest -k "test_batch_decode_kernels_hip"

# Verbose output
pytest -v

# Run tests in parallel across multiple GPUs
pytest -n auto  # Uses all available GPUs
pytest -n 2     # Use only two GPUs
```

The default test configuration is specified in [pyproject.toml](pyproject.toml)
under the `testpaths` setting.

#### Recommended invocation on AMD CPX systems

`pytest-rerunfailures` (declared in the `dev` extra — `pip install -e ".[dev]"`)
absorbs the residual transient HIP runtime crashes. Then for the full suite:

```bash
# Fast path — skips heavy 1M-trial sampling-frequency tests and 4 GB
# speculative-sampling cases (~7 min on a CPX 8-card host):
pytest -n auto --reruns 2 -m "not slow"

# Full coverage — including the slow tests (~20 min):
pytest -n auto --reruns 2

# Slow path only (~13 min):
pytest -n auto --reruns 2 -m "slow"
```

**Notes**

* **Worker count.** `pytest -n auto` for the `tests/rocm_tests/` suite
  spawns **half as many xdist workers as physical AMD cards** (e.g. 4
  workers on a CPX-mode 8-card MI308X / MI325X host) and pins each
  worker to its card via `HIP_VISIBLE_DEVICES`. One worker per physical
  card was tried first but produced sporadic failures across rope,
  single_prefill, and logits_cap under residual concurrent load.
  Pass an explicit `-n N` to override the halving.
* **Reruns.** `--reruns 2` (from `pytest-rerunfailures`) absorbs the
  residual ~0.01 % of transient HIP runtime crashes (HSA exceptions,
  HIPBLAS handle-pool exhaustion, intermittent generator
  non-determinism) that worker pinning cannot fully eliminate. Only
  failed tests are retried.
* **`slow` marker.** Registered in [pyproject.toml](pyproject.toml). It
  tags the 1M-trial sampling-frequency tests, the 4 GB-tensor
  speculative-sampling cases, and the entire `TestLogitsPipeCompilationHIP`
  class (each test runs the sampling kernel twice for compile=True/False).
* **HIPBLAS retry.** The reference attention helper in
  `tests/attention_reference.py` wraps `torch.matmul` in a
  `_hipblas_safe_matmul` retry that catches `HIPBLAS_STATUS_ALLOC_FAILED`
  and retries with a short back-off — needed under heavy concurrent
  xdist load.

## AITER Support

FlashInfer+ROCm can dispatch the `single_prefill`, `batch_prefill`
(paged and ragged), `batch_decode`, `append_paged_kv_cache`, `rmsnorm`,
`fused_add_rmsnorm`, `silu_and_mul`, and `MLA` paths to
[AITER](https://github.com/ROCm/aiter). MLA on ROCm
is **AITER-only** — there is no in-tree HIP MLA kernel yet, so
`backend="auto"` (the default for the MLA wrapper) resolves directly
to `"aiter"`.

On gfx942/gfx950, `backend="auto"` (the default) selects AITER when the
call is compatible (see [Known Limitations](#known-limitations) for the
full list) and otherwise falls back to the in-tree HIP kernel, emitting
a one-time `logger.warning`. Pass `backend="aiter"` to require AITER
explicitly, or pass the in-tree backend string to skip it:
`backend="fa2"` for the attention wrappers (single/batch
prefill/decode), `backend="native"` for non-attention ops
(`append_paged_kv_cache`, `rmsnorm`, `fused_add_rmsnorm`,
`silu_and_mul`). Four backend-specific exceptions to "auto picks AITER
when supported":

* `rmsnorm`: `backend="auto"` stays on the HIP `native` kernel; the
  AITER path is opt-in via `backend="aiter"`.
* `fused_add_rmsnorm`: `backend="auto"` is shape-gated — it picks AITER
  only for 2-D inputs with `>= 4M` elements (where the CK kernel is
  faster) and stays on the HIP `native` kernel otherwise.
* `silu_and_mul`: `backend="auto"` picks AITER only for `fp16` + 2-D +
  `>= 33M`-element inputs (where it is faster and matches native
  precision) and otherwise stays on HIP `native`; the AITER path is also
  available explicitly via `backend="aiter"`.
* `batch_decode`: `use_cuda_graph=True` or `use_tensor_cores=True`
  force `auto` back to `fa2` (AITER decode does not support either),
  and `pos_encoding_mode != "NONE"` raises under `backend="aiter"`.

Unless you are using the prebuilt Docker image, install AITER separately
via one of the options below.

### Install AITER from source

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

### Install AITER wheel package

Wheel packages are available from AMD's PyPI index: [pypi.amd.com/simple](https://pypi.amd.com/simple/).

```bash
pip install amd-aiter --index-url https://pypi.amd.com/simple/
```

### Known Limitations

AITER constraints fall into two groups: hard incompatibilities (the call
errors with `backend="aiter"` and triggers fallback under
`backend="auto"`), and silently-ignored kwargs (the call runs but the
flag has no effect on AITER — pass the in-tree backend explicitly if
you need any of them: `backend="fa2"` for attention wrappers, or
`backend="native"` for `append_paged_kv_cache` / `rmsnorm`).

**Conditions that fall back to the in-tree HIP kernel under
`backend="auto"`** (and raise under `backend="aiter"`):

* GPU is not gfx942 or gfx950
* `kv_layout` is not `NHD`
* a custom attention mask tensor is supplied
* `q_dtype` is not `float16` / `bfloat16` (no fp32, fp8, or int8)
* `q_dtype != kv_dtype` (mixed-precision Q/KV is unsupported)
* `head_dim_qk != head_dim_vo` (e.g. DeepSeek-style MLA with 192/128 head dims)
* `pos_encoding_mode != "NONE"` (AITER attention paths only support `"NONE"`)
* batch decode: `use_cuda_graph=True` or `use_tensor_cores=True`
* the `aiter` Python package is not importable

**Features silently ignored on the AITER path** (kwargs are accepted by
the FlashInfer wrapper but not forwarded to AITER, which can produce
wrong results):

* ALiBi slopes (`maybe_alibi_slopes`)
* RoPE scaling kwargs (`rope_scale`, `rope_theta`) — these are only
  consumed alongside `pos_encoding_mode != "NONE"`, which AITER
  attention rejects outright; the kwargs themselves pass through
  silently when the mode is `"NONE"`
* attention sinks (`sinks`)
* multi-modal / prefix-cache helpers (`maybe_prefix_len_ptr`,
  `maybe_token_pos_in_items_ptr`, `maybe_max_item_len_ptr`)
* FP8 dequant scales (`scale_q` / `scale_k` / `scale_v`)
* `use_fp16_qk_reduction`, `enable_pdl`

**Other notes:**

* Batch prefill: AITER's CK FMHA kernels natively support page sizes
  `{16, 1024}` (or `{128, 256, 1024}` on `amd-aiter==0.1.10`). Other page
  sizes still work but go through an extra GPU gather to flatten paged KV
  before the AITER call.
* Ragged (non-paged) batch prefill via AITER is supported through
  `BatchPrefillWithRaggedKVCacheWrapper`. The wrapper auto-routes to
  AITER under `backend="auto"` when the standard AITER compatibility
  conditions are met and falls back to `fa2` otherwise.
* MLA on ROCm currently supports only `bfloat16` and `page_size=1`
  through the AITER backend.

## Environment Variables

FlashInfer+ROCm reads the following environment variables at runtime
or import time. Build-time variables (`FLASHINFER_ROCM_ARCH_LIST`,
`FLASHINFER_JIT_VERBOSE`, `FLASHINFER_JIT_DEBUG`, `MAX_JOBS`, …) are
documented in [CLAUDE.md](CLAUDE.md).

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FLASHINFER_USE_TORCH_CUSTOM_OPS` | `0` | Set to `1` **before** importing `flashinfer` to wrap kernels in `torch.library.custom_op` so `torch.compile` / Dynamo can trace them. Requires PyTorch ≥ 2.4. Adds a small per-call dispatch overhead. |
| `FLASHINFER_HIP_FUSED_CASCADE` | `0` | Set to `1` to use a fused single-kernel HIP cascade attention path instead of the default two-level merge-based path. Experimental on ROCm. |
| `FLASHINFER_LOGGING_LEVEL` | `INFO` | Logger verbosity (e.g. `DEBUG`, `INFO`, `WARNING`). Affects AITER auto-fallback warnings and JIT build messages. |
| `FLASHINFER_DISABLE_JIT` | unset | Set to any non-empty value to skip JIT compilation. Useful when running an AOT-built wheel and you want to fail loudly on missing kernels rather than trigger a build. |
| `ROCM_PATH` / `ROCM_HOME` | `/opt/rocm` | Used by `flashinfer.hip_utils` to locate the ROCm install. Override only for non-standard ROCm layouts. |

## Runtime Helpers

`flashinfer` ships a few ROCm-specific helpers that are useful when
guarding code paths or diagnosing setup issues:

```python
import torch

from flashinfer.aiter_utils import is_aiter_supported
from flashinfer.hip_utils import check_torch_rocm_compatibility

# True on gfx942/gfx950 (a ROCm build + supported GPU arch). Does *not*
# verify the `aiter` Python package is importable — wrap the actual
# AITER call in a try/except ImportError if you need that guarantee.
if is_aiter_supported(torch.device("cuda")):
    ...

# Raises a clear error if PyTorch + ROCm versions are incompatible
# (e.g. a CPU-only torch wheel was picked up from PyPI).
check_torch_rocm_compatibility()
```

`flashinfer.hip_utils.validate_flashinfer_rocm_arch` is a related
build-time validator used by `setup.py` to cross-check
`FLASHINFER_ROCM_ARCH_LIST` against ROCm and PyTorch — not typically
called from application code.

## Basic Usage

```python
import torch
import flashinfer

# PyTorch+ROCm still uses device="cuda" for AMD GPUs.
q = torch.randn(1024, 32, 128, dtype=torch.float16, device="cuda")
k = torch.randn(1024,  8, 128, dtype=torch.float16, device="cuda")  # GQA 4:1
v = torch.randn(1024,  8, 128, dtype=torch.float16, device="cuda")

# backend="auto" (default) routes to AITER when supported on gfx942/gfx950
# and falls back to the in-tree fa2 HIP kernel otherwise.
output = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
```

See [`examples/`](examples/) for batch prefill, batch decode, and a
Jupyter tutorial that walks through the full public API on ROCm.

## License and Acknowledgements

FlashInfer+ROCm is released under the Apache-2.0 License — see
[LICENSE](LICENSE) and [NOTICE](NOTICE). Upstream project:
[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).

Contributions are welcome. Please run `pre-commit run -a` and the
relevant `pytest` selection before opening a PR.
