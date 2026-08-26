---
name: debug-rocm-crash
description: Tutorial for debugging HIP kernel crashes in FlashInfer+ROCm using HIP/ROCm runtime tooling
---

# Debugging ROCm Crashes in FlashInfer+ROCm

> **Note:** earlier revisions of this skill (and CLAUDE.md) described a `@flashinfer_api`
> decorator with `FLASHINFER_LOGLEVEL` / `FLASHINFER_LOGDEST` env vars. **That machinery does
> not exist in this fork** — no matches in code (`git grep` under `flashinfer/` and `include/`
> returns nothing; the only hits are this disclaimer). Don't try to set those env vars —
> use the HIP/ROCm tooling below instead.

## The magic env-var combo

For an unknown HIP fault, set these **before** running so the traceback points at the actual faulting kernel:

```bash
export AMD_SERIALIZE_KERNEL=3   # pins fault to the actual faulting kernel
export HIP_LAUNCH_BLOCKING=1    # synchronous launches; tracebacks point at the right line
```

Both are near-zero-overhead and reasonable to leave on while iterating on a new kernel.

For an in-script view of what's being passed, wrap the suspect call with `print(input.shape, input.dtype, input.device, input.is_contiguous())` and `torch.cuda.synchronize()` immediately before the FlashInfer call — this gives you the same info `@flashinfer_api` would have, manually.

## Per-error recipe

| Symptom | First check |
| --- | --- |
| `Memory access fault by GPU node-N` / `hipErrorIllegalAddress` / "CUDA error: illegal memory access" (PyTorch's ROCm reports HIP errors as "CUDA" errors) | Run with the env combo above. Print tensor shapes/dtypes/strides just before the call. Verify: `is_contiguous()` where required, all tensors on the same `cuda:N`, `kv_indices` within `[0, num_pages)`, `head_dim_qk` matches between Q and KV. |
| `backend="aiter"` `ValueError` before launch | `kv_layout != "NHD"` (only NHD is allowed). Grep [`prefill_rocm.py`](../../../flashinfer/prefill_rocm.py) for `only supports kv_layout` — single prefill and both batch wrappers each raise it. |
| `backend="aiter"` `RuntimeError` | Non-gfx942/gfx950 GPU. |
| `backend="aiter"` `ImportError` | `amd-aiter` not installed — see the AITER wheel section in `README.md` for the pinned version and its index. |
| `backend="aiter"` hard GPU fault mid-kernel | `amd-aiter` version mismatch vs. ROCm. Reinstall matching your ROCm version. Try the default HIP backend to confirm the bug is in AITER, not our side. |
| NaN / Inf in outputs | Insert `torch.isnan(t).any()` / `torch.isinf(t).any()` checks around the call. On CDNA3/4: `_fnuz` FP8 has different representable range than NVIDIA OCP FP8 — scale factors calibrated against NVIDIA refs overflow. Or `-inf` from a previous op fed into `exp`. Or `torch.empty` vs `torch.zeros`. |
| `HIP out of memory` | `rocm-smi --showmeminfo vram --showpids` — kill zombies. JIT-compile spike → `MAX_JOBS=1`. Other tenant → `HIP_VISIBLE_DEVICES=N`. |
| `expected scalar type X but found Y` (FP8 callsites) | PyTorch dtype for `_fnuz` FP8 is `torch.float8_e4m3fnuz` / `torch.float8_e5m2fnuz`, **not** `torch.float8_e4m3fn` (which is NVIDIA OCP FP8). A callsite expecting `e4m3fn` mis-dispatches on ROCm. |

## ROCm-specific tooling

| Tool | Use |
| --- | --- |
| `rocgdb --args python my_script.py` | CUDA-GDB equivalent. Inside: `catch throw`, `run`, `bt`, `info agents`, `info wavefronts`. |
| `ROCM_DEBUG_WAIT_FOR_DEBUGGER=1` | Process blocks at first GPU API call until `rocgdb -p <pid>` attaches. |
| `AMD_LOG_LEVEL=3` (or `4`) | HIP API + stream trace. Linear under `HIP_LAUNCH_BLOCKING=1`, so each Python call correlates 1:1 with HIP launches. |
| `HSA_ENABLE_DEBUG=1` | HSA layer trace (one below HIP — queues, agents). |
| `sudo dmesg -T \| grep -iE 'amdgpu\|kfd\|vm_fault'` | `VM_CONTEXT1_PROTECTION_FAULT_STATUS` gives page-fault class, access type, offending address — useful when Python only says `hipErrorIllegalAddress`. |
| `watch -n 1 'rocm-smi --showuse --showmeminfo vram --showpids'` | Hang diagnosis: 100% GPU + no SQ activity = looping kernel; VRAM still pinned after exit = another process holds it. |

`compute-sanitizer` / `cuda-gdb` have **no direct ROCm equivalent.** Closest workflow is the env-var combo above plus `rocgdb`.

## AMD-specific gotchas

- **PyTorch's ROCm masquerade.** Device strings show `cuda:0` on AMD; "CUDA error" messages may be HIP errors. The unambiguous arch field is `torch.cuda.get_device_properties(0).gcnArchName`.
- **Wavefront = 64**, not 32. Any representative-thread `printf` ported from CUDA needs `threadIdx.x % 64 == 0` (or use the `warpSize` builtin).
- **`FLASHINFER_JIT_DEBUG=1` is wired on the CUDA path only.** On HIP it does nothing for debug build flags (no `-O0 -g`). Add `-g` via `extra_cuda_cflags` in the JIT generator for the op being debugged, clear `~/.cache/flashinfer/`, retry. See CLAUDE.md "Non-Obvious Gotchas".
- **HIP installs are stripped.** `rocgdb` exits with `no symbol table loaded` unless you rebuild with `-g` (see previous bullet).
- **Device `printf` flushes on `torch.cuda.synchronize()`** — works the same as CUDA.
- **`HIP_VISIBLE_DEVICES`** is the canonical AMD scoping env var (`ROCR_VISIBLE_DEVICES` works one layer deeper). `CUDA_VISIBLE_DEVICES` may also be honored by PyTorch.

## Quick recipes

```bash
# Hard GPU fault
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
python my_script.py
# Python traceback now points at the right call. Also: sudo dmesg -T | tail -50

# Step into a kernel
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
rocgdb --args python my_script.py
# (rocgdb) catch throw
# (rocgdb) run
# (rocgdb) bt

# HIP API trace
AMD_LOG_LEVEL=3 HIP_LAUNCH_BLOCKING=1 python my_script.py 2> hip.trace
# grep hipLaunchKernel / hipMemcpy / error in hip.trace
```
