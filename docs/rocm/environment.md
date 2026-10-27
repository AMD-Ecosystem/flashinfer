# Environment variables and runtime helpers

## Runtime variables

Read at runtime or import time:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `FLASHINFER_USE_TORCH_CUSTOM_OPS` | `0` | Wrap kernels for `torch.compile`; set before importing `flashinfer`. See [`torch-compile.md`](torch-compile.md). |
| `FLASHINFER_AITER_STRICT` | `0` | `1` raises instead of degrading: when AITER cannot serve a page size natively, and when an AITER kernel bootstrap fails under `auto`. Set in CI to catch coverage regressions rather than absorb them as a slowdown. |
| `FLASHINFER_AITER_ASM_PREFILL` | `1` | Exactly `0` keeps single prefill on AITER's CK Tile kernel; any other value leaves the asm arm on. Read once per process. AITER aborts the process on several asm failure modes rather than returning an error, so this is the supported way out. |
| `FLASHINFER_AITER_ASM_VERBOSE` | `0` | Any value except `0` logs to stderr which prefill arm ran. The two arms agree numerically, so this is the only way to tell asm from CK Tile. |
| `FLASHINFER_ARCH_ALLOW_KNOWN_BAD` | `0` | `1` runs an (op, backend, arch) combination the capability table marks known-broken on your toolchain. No row carries such a window in this release, and it does not lift the gfx950 soft-cap gate. |
| `FLASHINFER_HIP_FUSED_CASCADE` | `0` | In `MultiLevelCascadeAttentionWrapper` only, pass each level's partial state into the next prefill call instead of merging afterwards; AITER levels and the shared-prefix wrappers ignore it. Both paths tested. Read once at import, so set it first. |
| `FLASHINFER_WORKSPACE_BASE` | `$HOME` | Parent of the JIT cache (`.cache/flashinfer/`); point at fast local disk when `$HOME` is on NFS. Prefer an absolute path: the value is wrapped in `pathlib.Path` unvalidated, so a relative one resolves against the working directory and `~` becomes a literal `./~` directory. |
| `FLASHINFER_DISABLE_JIT` | unset | **Any non-empty value** — including `0` — loads only AOT or jit-cache kernels: it refuses to compile, and refuses modules already in `~/.cache/flashinfer` too. The AITER shims are never prebuilt, so an `auto` call that picks AITER raises `MissingJITCacheError` rather than falling back. |
| `FLASHINFER_DISABLE_VERSION_CHECK` | unset | Any non-empty value skips the JIT-cache package version check. |
| `FLASHINFER_LOGGING_LEVEL` | `INFO` | Logger verbosity (`DEBUG`, `INFO`, `WARNING`, …). Affects AITER fallback warnings and JIT build messages. |
| `FLASHINFER_LOGLEVEL` | `0` | API call logging, distinct from `FLASHINFER_LOGGING_LEVEL`: `1` logs function names, `3` adds inputs and outputs, `5` adds tensor statistics, `10` also dumps tensors to disk. Above `0`, each logged call breaks a `torch.compile` graph, so `fullgraph=True` fails. |
| `FLASHINFER_LOGDEST` | `stdout` | Where `FLASHINFER_LOGLEVEL` output goes: `stdout`, `stderr`, or a file path (`%i` becomes the process ID). |
| `FLASHINFER_TRACE_APPLY` | unset | `1` enables upstream's opt-in kernel substitution at import, from the mapping `FLASHINFER_TRACE_APPLY_PATH` names; any other value except `0` warns and leaves it off. A failure to enable it warns rather than breaking the import. |
| `FLASHINFER_DISABLE_AOT_ARCH_CHECK` | unset | Any non-empty value — including `0` — uses jit-cache kernels even when their architecture does not match the running GPU. By default a mismatch discards them, with a warning, and everything JIT-compiles. |
| `ROCM_PATH` / `ROCM_HOME` | `/opt/rocm` | Where `flashinfer.rocm.hip_utils` looks for ROCm. Override only for non-standard layouts. |
| `AITER_JIT_DIR` | AITER's own | Where the C++ shim `dlopen`s AITER's built `.so` files, overriding the path compiled in at build time. |
| `FLASHINFER_AITER_VARIANT_DIR` | FlashInfer's stores | `os.pathsep`-separated prebuilt AITER variant stores to load from instead of the cache and jit-cache stores; `AITER_JIT_DIR` is still tried first. |
| `AITER_ASM_DIR` | set by `import aiter` | Where AITER reads its asm code objects. Unset or empty, single prefill stays on CK Tile rather than let AITER abort. |
| `GPU_ARCHS` | autodetected | AITER's own JIT architecture. A shim build overrides it from `FLASHINFER_ROCM_ARCH_LIST` and restores your value afterwards, leaving the derived one only if you had not set it. |

Build-time variables — `FLASHINFER_ROCM_ARCH_LIST`, `PYTORCH_ROCM_ARCH`,
`FLASHINFER_JIT_VERBOSE`, `FLASHINFER_EXTRA_{LDFLAGS,CFLAGS,CUDAFLAGS}`,
`FLASHINFER_OWN_HEADERS_NON_SYSTEM`, `MAX_JOBS` — live in
[CONTRIBUTING.md](../../CONTRIBUTING.md#build--jit-gotchas),
as does `FLASHINFER_JIT_DEBUG`, the debug-build switch.

## Runtime helpers

```python
import torch

from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.hip_utils import check_torch_rocm_compatibility

# True on gfx942/gfx950 with a ROCm torch build. Does *not* verify the
# `aiter` package imports — wrap in try/except ImportError if you need that.
if is_aiter_supported(torch.device("cuda")):
    ...

# Warns when torch's ROCm version differs from the system's. A CPU-only torch
# never gets here: importing flashinfer already raises.
check_torch_rocm_compatibility()
```
