# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT spec for the AITER-backed fused MoE shim (ROCm).

Separate from ``flashinfer/jit/fused_moe.py``, which is the upstream CUTLASS /
TRT-LLM generator and is not touched by the ROCm port.
"""

from typing import Optional

import torch

from . import env as jit_env
from .aiter_source import AiterModule, aiter_jitspec_flags, refresh_aiter_jitspec
from .core import JitSpec, gen_jit_spec

# AITER's `dtype2str_dict` (aiter/ops/moe_op.py) spellings, for the codegen flags.
# The keys double as the supported-dtype set; flashinfer/fused_moe_rocm.py
# validates against SUPPORTED_DTYPES rather than keeping its own copy.
_DTYPE_TAG = {
    torch.bfloat16: "b16",
    torch.float16: "f16",
}
SUPPORTED_DTYPES = tuple(_DTYPE_TAG)

# Both fp8 flavours generate the same CK instances -- which one is legal is a
# property of the device, not of the codegen. The set itself lives with the
# per-arch table in flashinfer.fused_moe_rocm, so there is only one to maintain.
_FP8_TAG = "f8"

_SUPPORTED_ACTIVATIONS = frozenset({"silu", "gelu"})

# CK's routing-weight stage. 2 = apply topk weights in stage 2, which is what the
# shim does; the generated instances must agree or the lookup misses.
_MUL_WEIGHT_STAGE = 2


def _is_fp8(dtype: torch.dtype) -> bool:
    # Imported here, not at module scope: flashinfer.fused_moe_rocm imports this
    # module, so the reverse edge has to stay lazy.
    from ..fused_moe_rocm import FP8_DTYPES

    return dtype in FP8_DTYPES


def _quant_tag(weight_dtype: torch.dtype, dtype: torch.dtype) -> str:
    """The ``gen_instances.py -q`` value implied by the weight dtype."""
    return "no" if weight_dtype == dtype else "per_token"


def _ck2stages_module(
    dtype: torch.dtype, weight_dtype: torch.dtype, activation: str
) -> AiterModule:
    """A ``module_moe_ck2stages`` specialized to one configuration.

    Mirrors AITER's own ``get_moe_stage_module`` rather than importing it, so a
    rename there surfaces as a build error instead of a silent change.
    """
    # Imported lazily: this module is importable without aiter installed, and
    # only a real build needs the path.
    from aiter.jit.core import AITER_CSRC_DIR

    b = _FP8_TAG if _is_fp8(weight_dtype) else _DTYPE_TAG[weight_dtype]
    c = _DTYPE_TAG[dtype]
    quant = _quant_tag(weight_dtype, dtype)
    # No -a: gen_instances.py derives the activation dtype from -b and ignores
    # the flag entirely.
    md_name = "_".join(
        [
            "module_moe_ck2stages",
            b,
            b,
            "preshuffle_off",
            c,
            activation,
            quant,
            f"mulWeightStage{_MUL_WEIGHT_STAGE}",
        ]
    )
    # build_module runs `blob_gen_cmd.format(blob_dir)`, so the string must carry
    # exactly one `{}` and no other braces.
    blob_gen_cmd = (
        f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py "
        f"-b {b} -c {c} -q {quant} -act {activation} "
        f"-m {_MUL_WEIGHT_STAGE} --working_path {{}}"
    )
    return AiterModule(
        "module_moe_ck2stages", md_name=md_name, blob_gen_cmd=blob_gen_cmd
    )


def _spec_name(dtype: torch.dtype, activation: str, quant: str) -> str:
    """The JIT module name. Unsuffixed when unquantized, so shipped caches stay valid."""
    name = f"fused_moe_aiter_{_DTYPE_TAG[dtype]}_{activation}"
    return name if quant == "no" else f"{name}_{quant}"


def gen_fused_moe_aiter_module(
    dtype: torch.dtype,
    activation: str,
    weight_dtype: Optional[torch.dtype] = None,
) -> JitSpec:
    """Build the fused-MoE shim for one (dtype, activation, weight dtype).

    ``weight_dtype`` defaults to ``dtype`` (unquantized). An fp8 weight dtype
    selects CK's ``per_Token`` instances and pulls in AITER's quantization
    module, which the shim uses to quantize the activations.
    """
    if dtype not in _DTYPE_TAG:
        # torch.dtype is not orderable, so list in declaration order.
        raise ValueError(f"fused MoE (aiter) supports {list(_DTYPE_TAG)}, got {dtype}")
    if weight_dtype is None:
        weight_dtype = dtype
    if not _is_fp8(weight_dtype) and weight_dtype != dtype:
        from ..fused_moe_rocm import FP8_DTYPES

        raise ValueError(
            f"fused MoE (aiter) supports expert weights in {dtype} or "
            f"{sorted(FP8_DTYPES, key=str)}, got {weight_dtype}"
        )
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"fused MoE (aiter) supports activations "
            f"{sorted(_SUPPORTED_ACTIVATIONS)}, got {activation!r}"
        )

    quant = _quant_tag(weight_dtype, dtype)
    modules = [
        AiterModule("module_moe_sorting"),
        _ck2stages_module(dtype, weight_dtype, activation),
    ]
    extra_cuda_cflags = []
    if quant != "no":
        # aiter::dynamic_per_token_scaled_quant, which the shim uses to quantize
        # the activations. Linked and compiled in only for the quantized specs so
        # the shipped unquantized path gains no new way to fail to build.
        modules.append(AiterModule("module_quant"))
        extra_cuda_cflags.append("-DFLASHINFER_MOE_AITER_PER_TOKEN")

    extra_include_paths, extra_ldflags = aiter_jitspec_flags(*modules)
    return refresh_aiter_jitspec(
        gen_jit_spec(
            _spec_name(dtype, activation, quant),
            [
                jit_env.FLASHINFER_CSRC_DIR / "fused_moe_aiter.cu",
                jit_env.FLASHINFER_CSRC_DIR / "fused_moe_aiter_jit_pybind.cu",
            ],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
        )
    )
