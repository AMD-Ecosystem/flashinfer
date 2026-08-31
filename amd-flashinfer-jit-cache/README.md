# AMD FlashInfer JIT Cache

This package contains pre-compiled HIP kernels for FlashInfer on AMD ROCm platforms.

## Purpose

The `amd-flashinfer-jit-cache` package provides ahead-of-time (AOT) compiled kernels to significantly reduce initialization time when using FlashInfer with AMD GPUs. Without this package, FlashInfer will compile kernels just-in-time (JIT) during runtime, which can take several minutes on first use.

## Installation

This package is intended to be installed alongside the main `amd-flashinfer` package:

```bash
pip install amd-flashinfer amd-flashinfer-jit-cache
```

## Contents

- FA2 attention: single/batch prefill and decode, across the head dims, dtypes,
  sliding-window and logits-soft-cap combinations in `get_default_config()`.
- The ops `backend="auto"` always resolves to a native HIP kernel: `rmsnorm`,
  rope, paged-KV append, sampling, quantization, cascade, and the gated
  activations. Build without them via `--add-misc false` / `--add-act false`.

The AITER shims are deliberately absent: they link against a per-user,
per-architecture AITER cache directory, so a prebuilt one cannot be relocated
to another machine.

## Architecture Support

This package is built specifically for the **AMD MI300 series (gfx942)** architecture.

The architecture it was built for is recorded in `jit_cache/aot_manifest.json`.
On a GPU it does not cover, FlashInfer ignores the prebuilt kernels, warns, and
compiles from source rather than loading the wrong ISA. The wheel tag carries no
architecture, so nothing prevents installing it on an unsupported GPU.

The check reads the running device, not `FLASHINFER_ROCM_ARCH_LIST` — that
variable is build intent and says nothing about the installed GPU. Set
`FLASHINFER_DISABLE_AOT_ARCH_CHECK=1` to use the kernels regardless.

## Development

To build this package from source:

```bash
cd amd-flashinfer-jit-cache
python -m build --wheel
```

The build process will:

1. Generate kernel specifications using `flashinfer.rocm.aot`
2. Compile kernels for the gfx942 architecture
3. Package compiled `.so` files into the wheel

## Environment Variables

- `FLASHINFER_ROCM_ARCH_LIST`: Target architecture (default: "gfx942")
- `HIP_PATH`: Path to ROCm/HIP installation (auto-detected if not set)

## License

Apache License 2.0
