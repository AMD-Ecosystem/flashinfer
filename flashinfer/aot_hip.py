# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import argparse
import contextlib
import json
import os
import shutil
from itertools import product
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch

# NOTE: jit modules are imported lazily below. jit/env.py freezes
# FLASHINFER_CACHE_DIR from FLASHINFER_WORKSPACE_BASE at import time, so an AOT
# build redirects the workspace paths in place instead -- see _redirected_jit_env.


def gen_fa2(
    dtype_qo: torch.dtype,
    dtype_kv: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> Iterator:
    # Import here to access gen_* functions
    from .jit.attention import (
        gen_batch_decode_module,
        gen_batch_prefill_module,
        gen_single_decode_module,
        gen_single_prefill_module,
    )

    if dtype_qo.itemsize == dtype_kv.itemsize and dtype_qo != dtype_kv:
        return
    if dtype_qo.itemsize == 1:
        return  # fp8 tensor cores not supported in fa2

    yield gen_single_prefill_module(
        backend="fa2",
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=False,
    )

    yield gen_batch_prefill_module(
        backend="fa2",
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        dtype_idx=torch.int32,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=False,
    )

    yield gen_single_decode_module(
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )

    yield gen_batch_decode_module(
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        dtype_idx=torch.int32,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )


def gen_attention(
    f16_dtype_: List[torch.dtype],
    fa2_head_dim_: List[Tuple[int, int]],
    use_sliding_window_: List[bool],
    use_logits_soft_cap_: List[bool],
) -> Iterator:
    # FA2 MHA / MQA / GQA
    for (
        (head_dim_qk, head_dim_vo),
        dtype_qo,
        dtype_kv,
        use_sliding_window,
        use_logits_soft_cap,
    ) in product(
        fa2_head_dim_,
        f16_dtype_,
        f16_dtype_,
        use_sliding_window_,
        use_logits_soft_cap_,
    ):
        yield from gen_fa2(
            dtype_qo=dtype_qo,
            dtype_kv=dtype_kv,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
        )


def gen_all_modules(
    f16_dtype_: List[torch.dtype],
    fa2_head_dim_: List[Tuple[int, int]],
    use_sliding_window_: List[bool],
    use_logits_soft_cap_: List[bool],
    add_act: bool = True,
    add_misc: bool = True,
) -> List:
    from .jit import JitSpec

    jit_specs: List[JitSpec] = []

    jit_specs += list(
        gen_attention(
            f16_dtype_,
            fa2_head_dim_,
            use_sliding_window_,
            use_logits_soft_cap_,
        )
    )

    if add_act:
        from .jit import gen_act_and_mul_module
        from .jit.activation import act_func_def_str

        for act_name in act_func_def_str:
            jit_specs.append(gen_act_and_mul_module(act_name))

    # Ops `backend="auto"` always resolves to a native HIP kernel: five
    # selectors return "native" unconditionally and sampling/quantization/
    # cascade have no AITER path, so each compiles on first use unless prebuilt.
    if add_misc:
        from .jit import (
            gen_norm_module,
            gen_page_module,
            gen_quantization_module,
            gen_rope_module,
            gen_sampling_module,
        )

        # Not re-exported on HIP (jit/rocm/api.py), so reach it by module path
        # -- flashinfer/cascade.py imports it the same way.
        from .jit.cascade import gen_cascade_module

        jit_specs += [
            gen_cascade_module(),
            gen_norm_module(),
            gen_page_module(),
            gen_quantization_module(),
            gen_rope_module(),
            gen_sampling_module(),
        ]

    # dedup
    names = set()
    ret: List[JitSpec] = []
    for jit_spec in jit_specs:
        if jit_spec.name not in names:
            names.add(jit_spec.name)
            ret.append(jit_spec)
    return ret


def copy_built_kernels(
    jit_specs: List,
    out_dir: Path,
    build_dir: Path,
    rocm_arch_list: Optional[str] = None,
) -> None:
    """Copy built kernel .so files from build_dir to out_dir.

    ``rocm_arch_list`` is recorded next to the kernels so a consumer can tell
    what they were compiled for; nothing in a ``.so``'s name or the wheel tag
    carries the architecture. Written here rather than by the caller so it
    cannot disagree with what was actually copied.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    jit_cache_dir = build_dir / "cached_ops"
    for jit_spec in jit_specs:
        src = jit_cache_dir / jit_spec.name / f"{jit_spec.name}.so"
        dst = out_dir / jit_spec.name / f"{jit_spec.name}.so"
        if not src.exists():
            raise FileNotFoundError(f"Built kernel not found: {src}")
        dst.parent.mkdir(exist_ok=False, parents=False)
        shutil.copy2(src, dst)

    # Falsy covers "" as well as None: an empty list would record that the
    # kernels target no architecture at all, and the reader would reject every
    # GPU. No manifest at least degrades to the documented unchecked path.
    if rocm_arch_list:
        from .jit.rocm.env import AOT_MANIFEST_NAME

        (out_dir / AOT_MANIFEST_NAME).write_text(
            json.dumps({"rocm_arch_list": rocm_arch_list}) + "\n"
        )


@contextlib.contextmanager
def _redirected_jit_env(build_dir: Path) -> Iterator[None]:
    """Point the JIT workspace at ``build_dir`` for the duration of the block.

    jit_env caches the workspace paths at import time, so an AOT build has to
    overwrite them in place; restoring on exit keeps that invisible to whatever
    runs next in the same process. FLASHINFER_CACHE_DIR is deliberately left
    alone -- it is frozen at import and nothing here can move it.
    """
    from .jit import env as jit_env

    saved_base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    saved = (
        jit_env.FLASHINFER_WORKSPACE_DIR,
        jit_env.FLASHINFER_JIT_DIR,
        jit_env.FLASHINFER_GEN_SRC_DIR,
    )

    # Inside the try: an mkdir that raises must still restore.
    try:
        os.environ["FLASHINFER_WORKSPACE_BASE"] = str(build_dir)
        jit_env.FLASHINFER_WORKSPACE_DIR = build_dir
        jit_env.FLASHINFER_JIT_DIR = build_dir / "cached_ops"
        jit_env.FLASHINFER_GEN_SRC_DIR = build_dir / "generated"
        jit_env.FLASHINFER_JIT_DIR.mkdir(parents=True, exist_ok=True)
        jit_env.FLASHINFER_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)
        yield
    finally:
        (
            jit_env.FLASHINFER_WORKSPACE_DIR,
            jit_env.FLASHINFER_JIT_DIR,
            jit_env.FLASHINFER_GEN_SRC_DIR,
        ) = saved
        if saved_base is None:
            os.environ.pop("FLASHINFER_WORKSPACE_BASE", None)
        else:
            os.environ["FLASHINFER_WORKSPACE_BASE"] = saved_base


def compile_and_package_modules(
    out_dir: Optional[Path],
    build_dir: Path,
    project_root: Path,
    config: dict = None,
    verbose: bool = False,
    skip_prebuilt: bool = True,
) -> None:
    """
    Compile and package modules based on the provided configuration.

    Args:
        out_dir: Output directory for packaged modules
        build_dir: Build directory for compilation
        project_root: Project root directory
        config: Configuration dictionary to override defaults (optional)
        verbose: Whether to print verbose build output
        skip_prebuilt: Whether to skip pre-built modules
    """
    with _redirected_jit_env(build_dir):
        _compile_and_package_modules(
            out_dir, build_dir, project_root, config, verbose, skip_prebuilt
        )


def _compile_and_package_modules(
    out_dir: Optional[Path],
    build_dir: Path,
    project_root: Path,
    config: Optional[dict],
    verbose: bool,
    skip_prebuilt: bool,
) -> None:
    from .jit import build_jit_specs

    # Start with default config and override with user config
    final_config = get_default_config()
    if config is not None:
        final_config.update(config)
    config = final_config

    # ROCm arch: resolve once, then validate.
    #
    # Publishing the result back into the environment is deliberate, not
    # incidental bookkeeping: the AITER shim resolves its own build architecture
    # from FLASHINFER_ROCM_ARCH_LIST (jit/aiter_source.py), and an AOT build has
    # no other channel to tell it what this build targets. Without this, a shim
    # built during an AOT run on a mixed or GPU-less host can disagree with the
    # kernels it is packaged alongside.
    #
    # It is a process-global side effect that outlives the call, which is worth
    # replacing with an explicit parameter threaded through the AOT -> JIT
    # boundary. That is a wider change than this one; leaving the lifetime
    # unchanged here keeps this commit to the resolution bug it is fixing.
    from .compilation_context_hip import CompilationContext

    # Publish the *validated* list, not the resolved one. Validation filters as
    # well as raises: `validate_flashinfer_rocm_arch` drops architectures this
    # ROCm or this PyTorch cannot build, warning rather than failing, so the
    # context's target set can be a strict subset of what the resolver returned.
    # Publishing the wider list recreates the disagreement this whole change
    # exists to remove -- on ROCm 6.4 with FLASHINFER_ROCM_ARCH_LIST=
    # "gfx950,gfx942", the kernels build for gfx942 while the shim reads gfx950
    # from the environment and is built for a card the kernels do not target.
    #
    # Constructing the context first also means a failed validation leaves
    # FLASHINFER_ROCM_ARCH_LIST as it found it, rather than exporting a list the
    # build then rejected.
    #
    # Read the order from `arch_flags`, not `TARGET_ROCM_ARCHS`. The latter is a
    # `set` (hip_utils: `arch_set = set(requested_archs)`), so the caller's order
    # is already gone by the time it gets here, and imposing `sorted()` is not
    # order-neutral: `resolve_aiter_build_arch()` takes `env_archs[0]` when no
    # device is visible, so republishing "gfx950,gfx942" as "gfx942,gfx950"
    # silently builds the shim for the architecture the caller listed *second*.
    # `arch_flags` is built by iterating `requested_archs` in order, so it still
    # carries the preference.
    compilation_context = CompilationContext()
    rocm_arch_list = ",".join(
        flag.removeprefix("--offload-arch=") for flag in compilation_context.arch_flags
    )
    os.environ["FLASHINFER_ROCM_ARCH_LIST"] = rocm_arch_list
    if verbose:
        print(f"Target ROCm architectures: {rocm_arch_list}")

    # Print summary
    if verbose:
        print("AOT build summary:")
        if out_dir is not None:
            print("  out_dir:", out_dir)
        print("  build_dir:", build_dir)
        print("  project_root:", project_root)
        print("  fa2_head_dim:", config["fa2_head_dim"])
        print("  f16_dtype:", config["f16_dtype"])
        print("  use_sliding_window:", config["use_sliding_window"])
        print("  use_logits_soft_cap:", config["use_logits_soft_cap"])
        print("  add_act:", config["add_act"])
        print("  add_misc:", config["add_misc"])
        print("  FLASHINFER_ROCM_ARCH_LIST:", rocm_arch_list)

    # Generate JIT specs
    if verbose:
        print("Generating JIT specs...")
    jit_specs = gen_all_modules(
        config["f16_dtype"],
        config["fa2_head_dim"],
        config["use_sliding_window"],
        config["use_logits_soft_cap"],
        config["add_act"],
        config["add_misc"],
    )
    if verbose:
        print("Total ops:", len(jit_specs))

    # Build
    build_jit_specs(jit_specs, verbose=verbose, skip_prebuilt=skip_prebuilt)

    # Copy built kernels
    if out_dir is not None:
        copy_built_kernels(jit_specs, out_dir, build_dir, rocm_arch_list)
        if verbose:
            print("AOT kernels saved to:", out_dir)


def parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1"):
        return True
    elif s.lower() in ("false", "0"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}")


def parse_head_dim(head_dim: str) -> Tuple[int, int]:
    qo, kv = map(int, head_dim.split(","))
    return qo, kv


def get_default_config():
    """Get default AOT configuration"""
    return {
        "fa2_head_dim": [(64, 64), (128, 128), (256, 256)],
        "fa3_head_dim": [(192, 128), (128, 128), (64, 64), (256, 256)],
        "f16_dtype": [torch.float16, torch.bfloat16],
        "use_sliding_window": [False, True],
        "use_logits_soft_cap": [False, True],
        "add_act": True,
        "add_misc": True,
    }


def register_default_modules() -> int:
    """Register the default set of modules (used by packaging system)"""
    config = get_default_config()

    jit_specs = gen_all_modules(
        config["f16_dtype"],
        config["fa2_head_dim"],
        config["use_sliding_window"],
        config["use_logits_soft_cap"],
        config["add_act"],
        config["add_misc"],
    )
    return len(jit_specs)


def main():
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build all modules"
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory")
    parser.add_argument(
        "--build-dir", type=Path, help="Build directory (default: current dir)"
    )
    parser.add_argument(
        "--fa2-head-dim",
        nargs="*",
        help="FA2 head dim pair of qk and vo, separated by comma",
    )
    parser.add_argument(
        "--f16-dtype",
        nargs="*",
        choices=["float16", "bfloat16"],
        help="16-bit data type",
    )
    parser.add_argument(
        "--f8-dtype",
        nargs="*",
        choices=["float8_e4m3fn", "float8_e5m2"],
        help="8-bit data type",
    )
    parser.add_argument(
        "--use-sliding-window", nargs="*", help="Use sliding window attention"
    )
    parser.add_argument("--use-logits-soft-cap", nargs="*", help="Use logits soft cap")
    # Scalar, unlike the axes above: these select whole groups rather than
    # values to iterate, so `type=` rather than `nargs="*"`.
    parser.add_argument(
        "--add-act", type=parse_bool, help="Build the gated-activation modules"
    )
    parser.add_argument(
        "--add-misc",
        type=parse_bool,
        help="Build cascade/norm/page/quantization/rope/sampling",
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    build_dir = Path(args.build_dir) if args.build_dir else Path.cwd()
    out_dir: Optional[Path] = Path(args.out_dir) if args.out_dir else None

    # Start with default configuration
    config = get_default_config()

    # Override with command line arguments
    if args.fa2_head_dim:
        config["fa2_head_dim"] = [parse_head_dim(dim) for dim in args.fa2_head_dim]
    if args.f16_dtype:
        config["f16_dtype"] = [getattr(torch, dtype) for dtype in args.f16_dtype]
    if args.use_sliding_window:
        config["use_sliding_window"] = [parse_bool(s) for s in args.use_sliding_window]
    if args.use_logits_soft_cap:
        config["use_logits_soft_cap"] = [
            parse_bool(s) for s in args.use_logits_soft_cap
        ]
    # `is not None`, not truthiness: `--add-act false` must survive.
    if args.add_act is not None:
        config["add_act"] = args.add_act
    if args.add_misc is not None:
        config["add_misc"] = args.add_misc

    # Use the reusable compile_and_package_modules function
    compile_and_package_modules(
        out_dir=out_dir,
        build_dir=build_dir,
        project_root=project_root,
        config=config,
        verbose=True,
        skip_prebuilt=False,
    )


if __name__ == "__main__":
    main()
