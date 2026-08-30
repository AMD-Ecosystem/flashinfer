---
name: add-rocm-kernel
description: Step-by-step tutorial for adding new HIP kernels to FlashInfer+ROCm (amd-flashinfer)
---

# Adding a New Kernel to FlashInfer+ROCm

For a complete worked example to copy, read these together:
[`norm.cu`](../../../flashinfer/csrc/rocm/norm.cu) +
[`flashinfer_norm_binding.cu`](../../../flashinfer/csrc/rocm/flashinfer_norm_binding.cu) +
[`jit/norm.py`](../../../flashinfer/jit/norm.py) +
[`norm.py`](../../../flashinfer/norm.py). For plan-run / multi-backend / FP8 see
[`batch_prefill.cu`](../../../flashinfer/csrc/rocm/batch_prefill.cu) +
[`prefill_rocm.py`](../../../flashinfer/prefill_rocm.py).

## File touchpoints (every new op needs each row, in order)

| Step | File | Purpose |
| --- | --- | --- |
| 1 | `include/flashinfer/<op>.cuh` | Framework-agnostic kernel + launcher template. **No `<torch/...>` includes here.** |
| 2 | `flashinfer/csrc/rocm/<op>.cu` | PyTorch launcher: `at::Tensor` in, `at::hip::getCurrentHIPStream()`, `TORCH_CHECK`, `DISPATCH_PYTORCH_DTYPE_*`. |
| 3 | `flashinfer/csrc/rocm/flashinfer_<op>_binding.cu` | `TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) { m.def("<op>", <op>); }`. |
| 4 (opt) | `flashinfer/csrc/rocm/<op>_customize_config.jinja` | Compile-time type specialization. Skip if runtime dispatch is enough. |
| 5 | `flashinfer/jit/<op>.py` | `gen_<op>_module() -> JitSpec` via `gen_jit_spec(...)`. |
| 6 | `flashinfer/<op>.py` | Python API: `@functools.cache` module loader, destination-passing (`out=`). |
| 7 | `tests/rocm/test_<op>.py` | Correctness tests; FP32 reference math, loose BF16 tolerances. |
| 8 | `flashinfer/jit/__init__.py` (`IS_HIP` branch) | `from .<op> import gen_<op>_module as gen_<op>_module`. |
| 9 | `flashinfer/__init__.py` (`IS_HIP` branch) | `from .<op> import <op> as <op>`. |
| 10 (opt) | `flashinfer/aot_hip.py` | Register `gen_<op>_module` for pre-compiled wheels. |

**Forgetting steps 8 and 9 is the most common bug** — the module compiles but is invisible from `import flashinfer`.

## CUDA → ROCm porting cheat sheet

When porting an upstream kernel, mechanically rewrite:

| Upstream CUDA | This fork |
| --- | --- |
| `csrc/<op>.cu` | `flashinfer/csrc/rocm/<op>.cu` |
| `#include "tvm_ffi_utils.h"` | `#include "pytorch_extension_utils.h"` |
| `tvm::ffi::TensorView` | `at::Tensor` |
| `TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, op)` | `TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) { m.def("op", op); }` |
| `TVM_FFI_THROW(ValueError) << "..."` | `TORCH_CHECK(cond, "...")` |
| `DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16` | `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16` |
| `get_stream(tensor.device())` | `at::hip::getCurrentHIPStream()` |
| `c10::cuda::OptionalCUDAGuard` | `c10::hip::OptionalHIPGuardMasqueradingAsCUDA` |
| `nvcc` flags via `extra_cuda_cflags=[...]` | **Same kwarg name** (`extra_cuda_cflags`) — internally routed to `hipcc`. |
| `flashinfer/aot.py` registration | `flashinfer/aot_hip.py` |
| `tests/test_op.py` | `tests/rocm/test_op.py` |
| `supported_major_versions=[9, 10]` | No analogue. Guard at Python layer via `FLASHINFER_SUPPORTED_ROCM_ARCHS`. |
| `csrc/` (hardcoded) | `jit_env.FLASHINFER_CSRC_DIR` resolves to `flashinfer/csrc/rocm/` on HIP. **Never hardcode `csrc/`.** |
| `PYBIND11_MODULE(...)` | **Don't.** Use `TORCH_LIBRARY_FRAGMENT` (integrates with `torch.compile`). |

## Non-obvious gotchas

- **PyTorch's ROCm masquerade.** `input.device.type == "cuda"` even on AMD. Never check for `"hip"`. PyTorch's HIP namespaces are reachable via `at::hip::...` and `c10::hip::OptionalHIPGuardMasqueradingAsCUDA` (literally the type name).
- **Shared intrinsics over duplication.** If a primitive (MMA intrinsic, cross-lane shuffle, dtype container, warp reduction) needs a HIP-specific implementation, add it to the matching `_hip.h` in [`include/flashinfer/rocm/`](../../../include/flashinfer/rocm) rather than forking the kernel into `csrc/rocm/`. Existing ones: `mma_hip.h`, `memory_ops_hip.h`, `math_hip.h`, `vec_dtypes_hip.h`. Symbols go in `flashinfer::`, grouped by area (`flashinfer::math`, `flashinfer::memory`, `flashinfer::mma`), keeping the upstream namesake's names; each such header carries an `#error` tripwire on upstream's include guard.
- **`-ffast-math` adds `-ffinite-math-only` on clang/hipcc.** [`jit/core.py`](../../../flashinfer/jit/core.py) explicitly re-adds `-fno-finite-math-only` so kernels that use `-inf` as a sentinel (online-softmax Map+Reduce) keep working. CUDA's `-use_fast_math` does *not* enable finite-math-only — divergence to be aware of when porting.
- **`gen_jit_spec` auto-injects `--offload-arch=gfxNNN`** for every target arch plus `COMMON_HIPCC_FLAGS` (`-DFLASHINFER_ENABLE_HIP`, FP8 enables, etc.). Don't add `--offload-arch` by hand.
- **Validation macros** live in [`pytorch_extension_utils.h`](../../../flashinfer/csrc/rocm/pytorch_extension_utils.h): `CHECK_INPUT` (GPU + contiguous), `CHECK_LAST_DIM_CONTIGUOUS_INPUT`, `CHECK_EQ`, `CHECK_DIM`, `CHECK_GE`, `CHECK_SHAPE`. Dispatch macros: `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16` (FP16+BF16), `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8` (E4M3+E5M2, both `_fnuz` on CDNA3/4), and the unsuffixed `DISPATCH_PYTORCH_DTYPE_TO_CTYPE` (FP16+BF16+FP8 combined). There is **no** `_FP16_FP32` variant — if you need FP32, dispatch manually.
- **The `_jit_pybind.cu` naming pattern** (e.g. `batch_decode_jit_pybind.cu`) is used by newer AITER-integrated bindings; the older `flashinfer_<op>_binding.cu` pattern is used by everything else. Both work — match the neighbors.

## CDNA3 (`gfx942`) vs CDNA4 (`gfx950`)

- **Wavefront = 64 on both.** Anything ported from CUDA assuming warp = 32 is wrong. Use `warpSize` for portability.
- **FP8** is `__hip_fp8_e4m3_fnuz` / `__hip_fp8_e5m2_fnuz` on both. PyTorch dtype is `torch.float8_e4m3fnuz` (not `torch.float8_e4m3fn`, which is NVIDIA OCP FP8). Bit-exact parity with NVIDIA FP8 is not guaranteed — calibrate scale factors separately.
- **MFMA intrinsics:** CDNA4 has additional FP8 MFMA shapes not on CDNA3. Guard arch-specific intrinsics with `__gfx942__` / `__gfx950__` or compute-capability dispatch at the Python layer.
- **LDS / register / occupancy budgets differ.** Don't hard-code tile sizes — parameterize (Jinja) or query via `torch.cuda.get_device_properties(dev)` at plan time.

## Quick checklist before commit

- [ ] No `<torch/...>` under `include/`.
- [ ] Launcher uses `at::hip::getCurrentHIPStream()` + `OptionalHIPGuardMasqueradingAsCUDA`.
- [ ] Binding registered via `TORCH_LIBRARY_FRAGMENT`.
- [ ] JIT generator uses `jit_env.FLASHINFER_CSRC_DIR` (not hardcoded `csrc/`).
- [ ] Both `flashinfer/jit/__init__.py` and `flashinfer/__init__.py` IS_HIP branches updated.
- [ ] Test file under `tests/rocm/` named `test_*.py`.
- [ ] `pre-commit run -a` clean.
