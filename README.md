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
`<upstream_version>+amd.<n>` — `0.6.18+amd.2` is the second AMD release based on
upstream `v0.6.18`. History is under
[Releases](https://github.com/AMD-Ecosystem/flashinfer/releases).

## Quick start

No wheel or image is published for this release — build from the repository.
The development image carries a matched ROCm, PyTorch, Python and AITER set,
so it is the shortest path to a working environment:

```bash
docker build -t flashinfer-dev:rocm10.1 -f docker/Dockerfile.rocm . \
  --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g)
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v "$PWD":/workspace -w /workspace flashinfer-dev:rocm10.1
```

The image build takes about 35 minutes, most of it prebuilding AITER. The image
carries no source: `-v "$PWD":/workspace` mounts your checkout, so run it from
the repository root. The `--build-arg` trio and the numeric `render` GID keep
file ownership and GPU access right; [CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md#setting-up-a-development-environment)
explains both. Then, inside the container:

```bash
python -m pip install --no-build-isolation -ve .
python -c "import flashinfer; print(flashinfer.__version__)"
```

Kernels JIT-compile on first use: about a minute for an in-tree HIP kernel,
several for a cold AITER variant. The image prebuilds every AITER prefill
variant; AITER batch decode compiles its kernel at `plan()`, and AITER MoE and
the opt-in shims build on first use, so a first call can sit silent for
minutes. Outside the image, take torch 2.12 from its base image,
`rocm/pytorch:rocm10.1.0_ubuntu24.04_py3.12_pytorch_release_2.12.0`
(`repo.radeon.com` has no ROCm 10.1 build), check it is a ROCm build with
`python -c "import torch; assert torch.version.hip"`, and build `amd-aiter`
from source at its tag as
[`backends.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md#installing-aiter) describes;
without it `auto` prefill and decode silently run on `fa2`, and MLA,
`aiter_fused_moe` and any explicit `backend="aiter"` call raise.
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md) covers wheel and ahead-of-time
builds, and
[`amd-flashinfer-jit-cache`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/amd-flashinfer-jit-cache/README.md)
builds the default in-tree kernel set into one wheel per architecture; none is
published.

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

The batch prefill and decode wrappers keep upstream's `plan()`/`run()` API;
see the [FlashInfer documentation](https://docs.flashinfer.ai), which tracks
upstream's latest release rather than v0.6.18 (this release's docstrings match
its own code), and the
[CUDA-only arguments](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md#cuda-only-arguments)
ROCm refuses.

## Supported hardware and toolchain

**One configuration is supported: the one `docker/Dockerfile.rocm` builds.**
At `v0.6.18+amd.2` the full test suite passes in that image on both
architectures:

| `pytest --reruns 2` | gfx942 (MI300X) | gfx950 (MI355X) |
| :--- | :--- | :--- |
| `-n auto -m "not slow"` | 33867 passed, 4024 skipped | 33875 passed, 4016 skipped |
| `-n 1 -m slow` | 72 passed | 72 passed |

| | Supported |
| :--- | :--- |
| GPUs | gfx942 (CDNA3 — MI300X, MI325X), gfx950 (CDNA4 — MI350X, MI355X) |
| ROCm | 10.1 |
| PyTorch+ROCm | 2.12.0 |
| Python | 3.12 |
| OS | Ubuntu 24.04 |
| `amd-aiter` | 0.1.21.post2 (source build) |

The suite needs a git checkout: four tests read the git index. Torch 2.12's
ROCm 10.1 build exists only in the base image named in the quick start. Other
combinations are not rejected at install time, but they are untested and
not what a bug report is reproduced against: the pins move together, since
`amd-aiter` is built from source against the image's ROCm and torch.

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

One row per (op, backend) pair, so ops with both backends appear twice. The
table is generated from
[`flashinfer/rocm/arch_caps.py`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/flashinfer/rocm/arch_caps.py),
the capability table `auto` routes on; the Notes column is hand-written there,
not derived from the gates. Regenerate it with
`python3 scripts/gen_arch_support_matrix.py` rather than editing it.

<!-- BEGIN GENERATED: arch support matrix -- scripts/gen_arch_support_matrix.py -->

| Op | Backend | gfx942 (CDNA3) | gfx950 (CDNA4) | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `batch_decode` | `aiter` -- auto picks this when compatible | ✅ | ✅ | MHA / GQA / MQA with sliding window; fp16/bf16 + NHD. Under graph capture `auto` needs a declared `max_seq_len`, else it stays on fa2. |
| `single_prefill` | `aiter` -- auto picks this when compatible | ✅ | ✅ | MHA / GQA / MQA with sliding window; fp16/bf16 + NHD, equal Q/KV dtypes and head dims, no custom mask. No fp8 query (`mha_fwd` has no fp8 kernel); an fp8 KV cache with a 2-byte query runs on fa2. On gfx950 an unwindowed, uncapped bf16 head_dim 128 call at `qo_len` >= 2048 takes AITER's asm kernel, outside HIP graph capture; everything else is CK Tile. |
| `batch_prefill` | `aiter` -- auto picks this when compatible | ✅ | ✅ | Paged and ragged, with sliding window. fp8 query/KV on the paged route at head_dim 128/256 (bf16 out, per-tensor descales, no LSE). Page sizes 1/16/1024 are native at head_dim 128/256 only, since AITER's paged kernel reads past the KV cache at 64/96 (others unmeasured); fp16/bf16 are routed there only at 1024 and flat-gather otherwise. |
| `mla` | `aiter` -- only backend | ✅ | ✅ | 512 compressed-KV + 64 rope (DeepSeek V2/V3); fp16/bf16. No HIP kernel exists, so `auto` resolves here. |
| `rope` | `aiter` -- opt-in | ✅ | ✅ | `apply_rope_with_cos_sin_cache` and its inplace variant, linked at the C++ level. Opt-in. |
| `append_paged_kv_cache` | `aiter` -- opt-in | ✅ | ✅ | fp16/bf16 + NHD. Bit-exact with the in-tree kernel but slower, so `auto` picks `native`. |
| `rmsnorm` | `aiter` -- opt-in | ✅ | ✅ | `aiter::rmsnorm`; 2-D fp16/bf16, hidden size even and <= 8192, weight dtype must match. Opt-in: level with native on speed and less accurate. |
| `fused_add_rmsnorm` | `aiter` -- opt-in | ✅ | ✅ | `aiter::add_rmsnorm`; 2-D, hidden size even and <= 8192, weight dtype must match. Opt-in: 1.6-1.8x slower, since correctness needs two staging buffers. |
| `silu_and_mul` | `aiter` -- opt-in | ✅ | ✅ | `aiter::silu_and_mul`, linked at the C++ level. Opt-in; matches native in fp16, lower in bf16. |
| `fused_moe` | `aiter` -- only backend | ✅ | ✅ | `aiter_fused_moe`; bf16/fp16. Weights must be pre-shuffled with `shuffle_moe_weight` or results are silently wrong. |
| `fused_moe_fp8` | `aiter` -- only backend | ✅ | ✅ | `aiter_fused_moe` with fp8 weights in `moe_fp8_dtype()` plus both scales; activations are quantized per token in the shim. |
| `single_decode` | `hip` -- only backend | ✅ | ✅ | MHA / GQA / MQA; fp8 KV-cache (E4M3FNUZ, E5M2FNUZ). No fp8 module is prebuilt, so first use pays a cold build. |
| `batch_decode` | `hip` -- fallback; auto tries `aiter` first | ✅ | ✅ | MHA / GQA / MQA; fp8 KV-cache (E4M3FNUZ, E5M2FNUZ) on both the plain and `use_tensor_cores=True` paths, and CUDA-graph capture. |
| `single_prefill` | `hip` -- fallback; auto tries `aiter` first | ✅ | ✅ | MHA / GQA / MQA, including custom attention masks. fp8 KV-cache (E4M3FNUZ, E5M2FNUZ) with a 2-byte query, JIT-only; the OCP spellings are refused, since they would be read under the fnuz exponent bias. |
| `batch_prefill` | `hip` -- fallback; auto tries `aiter` first | ✅ | ✅ | Paged and ragged; MHA / GQA / MQA, including custom attention masks. fp8 KV-cache (E4M3FNUZ, E5M2FNUZ) with a 2-byte query, JIT-only -- no fp8 prefill module is prebuilt, so first use pays a cold build. |
| `block_sparse` | `hip` -- only backend | ✅ | ✅ | `BlockSparseAttentionWrapper` and the variable-block variant. Native HIP FA2 only -- `determine_attention_backend` never returns `aiter` here. Both wrappers build through the batch-prefill module, so the fp8 KV-cache dtypes and refusals are the same as `batch_prefill`. |
| `cascade` | `hip` -- merge only; levels are auto-routed and can be `aiter` | ✅ | ✅ | Two-level shared-prefix attention. The `hip` backend is the merge kernels only: each level runs through the ordinary prefill/decode entry points at `backend="auto"`, so it can reach AITER, and no cascade wrapper exposes `backend=` to override that. `FLASHINFER_HIP_FUSED_CASCADE=1` threads partial state through the levels of `MultiLevelCascadeAttentionWrapper` only; AITER levels and both shared-prefix wrappers still merge post-hoc. |
| `pod` | `hip` -- only backend | ✅ | ✅ | `PODWithPagedKVCacheWrapper` and the batch variant. JIT-only, excluded from AOT as upstream. An fp8 KV-cache is refused: POD sizes its tiles independently of the prefill dispatchers and no geometry there has been tested. |
| `rope` | `hip` -- auto picks this | ✅ | ✅ | LLaMA and LLaMA 3.1 scaling; fused RoPE + fp8 quant + paged-KV append (E4M3FNUZ, E5M2FNUZ). |
| `append_paged_kv_cache` | `hip` -- auto picks this | ✅ | ✅ | fp8 KV-cache supported. Sustains 3.62 TB/s against AITER's 2.86 on gfx942, so `auto` picks this. |
| `rmsnorm` | `hip` -- auto picks this | ✅ | ✅ | What `auto` always picks: level with AITER on speed and more accurate. |
| `fused_add_rmsnorm` | `hip` -- auto picks this | ✅ | ✅ | What `auto` always picks: 1.6-1.8x faster than AITER on both arches. |
| `gemma_rmsnorm` | `hip` -- only backend | ✅ | ✅ | `gemma_rmsnorm` and `gemma_fused_add_rmsnorm`. No AITER path. |
| `sampling` | `hip` -- only backend | ✅ | ✅ | Top-K / Top-P / Min-P / OnlineSoftmax / SamplingFromLogits. |
| `logits_processor` | `hip` -- only backend | ✅ | ✅ | Composable processor pipeline (cap, mask, temperature, ...). |
| `silu_and_mul` | `hip` -- auto picks this | ✅ | ✅ | SiLU and GELU with fused gating; the default for `auto`. |
| `quantization` | `hip` -- only backend | ✅ | ✅ | `packbits` and `segment_packbits`. |

* ✅ **supported** — this op runs on this architecture and the test suite covers it.

<!-- END GENERATED: arch support matrix -->

Every row is covered by the default `pytest` selection — most by a matching
`tests/rocm/test_*.py`, `single_decode` from the batch-decode, sliding-window
and logits-cap files, and `quantization` by `tests/utils/test_quantization.py`.

**gfx942 and gfx950 route some cases differently** — soft-capped causal
prefill, long bf16 single prefill, and short-query paged and ragged prefill. The
[per-op notes](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md#per-op-notes) give the exact
conditions.

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

Two modules are simply unimportable here rather than unsupported:
`flashinfer.profiler` needs `tg4perfetto` and `flashinfer.artifacts` needs
`requests`, neither of which `docker/Dockerfile.rocm` ships.

[`docs/rocm/backends.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/backends.md)
has the complete lists and the evidence for each entry;
`tests/rocm/test_kernel_source_coverage.py` fails when a newly vendored op
names a kernel source `csrc/rocm` does not have, so the second group cannot
grow unnoticed.

## `torch.compile`

Set `FLASHINFER_USE_TORCH_CUSTOM_OPS=1` **before** importing `flashinfer` to
register the kernels as custom ops so TorchDynamo can trace them, and call
`plan()` outside the compiled region. Paged and ragged prefill, batch decode and
append compile, mostly under `fullgraph=True`; single prefill, MLA and
`aiter_fused_moe` compile with graph breaks. **Known issue:** PyTorch 2.12 can
silently miscompile an op that writes into a view of a larger tensor, including
the K and V halves append splits from a single `[pages, 2, ...]` cache: pass
append two separately allocated caches and give `out=` a whole tensor.
[`docs/rocm/torch-compile.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/torch-compile.md) has
the measured per-op table, the ops the flag does not register, the full known
issue and the `mode="reduce-overhead"` advice.

## Running the tests

```bash
pytest -n auto --reruns 2 -m "not slow"
```

**`-n auto` counts GPUs, not CPUs** — half the physical supported cards,
minimum one. Pass `-n N` to override.
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md)
covers worker pinning, the `slow` marker and the rerun policy.

## Benchmarking

The unified runner drives batch decode, paged/ragged batch prefill, and the
norm, rope and sampling routines from one testlist — not every op in the
support matrix:

```bash
cd benchmarks
python flashinfer_benchmark.py --testlist rocm/testlist_rocm.txt \
    --output_path run-$(date +%F).csv
```

Read `backend_resolved` in the output: `auto` is a request, not a result. Rows
where `fa2` cannot run (the Llama-3.1-405B lines) have no reference and are not
refchecked, so count rows per config.
[`benchmarks/rocm/README.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/benchmarks/rocm/README.md)
documents the columns and the per-op drivers.

## Environment variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FLASHINFER_USE_TORCH_CUSTOM_OPS` | `0` | Wrap kernels for `torch.compile`; set before importing `flashinfer`. |
| `FLASHINFER_WORKSPACE_BASE` | `$HOME` | Parent of the JIT cache (`.cache/flashinfer/`); point at fast local disk when `$HOME` is on NFS. |
| `FLASHINFER_AITER_STRICT` | `0` | `1` raises instead of degrading: when AITER cannot serve a page size natively, and when an AITER kernel bootstrap fails under `auto`. |
| `FLASHINFER_LOGGING_LEVEL` | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, …). Affects AITER fallback warnings and JIT build messages. |

[`docs/rocm/environment.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/docs/rocm/environment.md) has the full
runtime table and the runtime helpers;
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md) has the build-time ones.

## Contributing

Contributions are welcome — read
[CONTRIBUTING.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/CONTRIBUTING.md) first, and run
`pre-commit run -a` and the relevant `pytest` selection before opening a PR.

## Security

Do not report vulnerabilities in a public issue — follow
[SECURITY.md](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/SECURITY.md).

## License and acknowledgements

Apache-2.0 — see [LICENSE](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/LICENSE) and [NOTICE](https://github.com/AMD-Ecosystem/flashinfer/blob/amd-integration/NOTICE). Upstream
project: [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).
