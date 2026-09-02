# AMD FlashInfer JIT Cache

This package contains pre-compiled HIP kernels for FlashInfer on AMD ROCm platforms.

## Purpose

The `amd-flashinfer-jit-cache` package provides ahead-of-time (AOT) compiled kernels to significantly reduce initialization time when using FlashInfer with AMD GPUs. Without this package, FlashInfer will compile kernels just-in-time (JIT) during runtime, which can take several minutes on first use.

## Installation

Install alongside the main `amd-flashinfer` package, naming the architecture in full:

```bash
pip install amd-flashinfer "amd-flashinfer-jit-cache==0.6.18+amd.1.gfx942"
```

The architecture rides in the local version segment because a wheel tag cannot carry it. Pin it: an unqualified requirement resolves to whichever architecture sorts highest, which need not be yours.

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

One wheel per architecture: gfx942 (MI300/MI325) and gfx950 (MI350/MI355). Which
one a wheel holds is in its version's local segment and in
`jit_cache/aot_manifest.json`.

On a GPU it does not cover, FlashInfer ignores the prebuilt kernels, warns, and
compiles from source rather than loading the wrong ISA. Nothing at install time
enforces the match — the version is a label, not a platform tag.

The check reads the running device, not `FLASHINFER_ROCM_ARCH_LIST` — that
variable is build intent and says nothing about the installed GPU. Set
`FLASHINFER_DISABLE_AOT_ARCH_CHECK=1` to use the kernels regardless.

## Development

To build this package from source:

```bash
cd amd-flashinfer-jit-cache
for arch in gfx942 gfx950; do
  FLASHINFER_ROCM_ARCH_LIST=$arch python -m build --wheel --no-isolation
done
```

The build process will:

1. Generate kernel specifications using `flashinfer.rocm.aot`
2. Compile kernels for `FLASHINFER_ROCM_ARCH_LIST`, and append it to the version
3. Package compiled `.so` files into the wheel

Both wheels can be built on one host and land side by side in `dist/`;
cross-compiling does not need the target GPU. With `FLASHINFER_ROCM_ARCH_LIST`
unset the target is probed from the running GPU instead, and the build refuses
to finish if the version and the compiled kernels end up naming different
architectures.

## Environment Variables

- `FLASHINFER_ROCM_ARCH_LIST`: Target architecture (default: "gfx942")
- `HIP_PATH`: Path to ROCm/HIP installation (auto-detected if not set)

## License

Apache License 2.0
