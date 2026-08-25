# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""JIT spec for the AITER-backed fused MoE shim (ROCm).

Separate from ``flashinfer/jit/fused_moe.py``, which is the upstream CUTLASS /
TRT-LLM generator and is not touched by the ROCm port.
"""

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

_SUPPORTED_ACTIVATIONS = frozenset({"silu", "gelu"})

# CK's routing-weight stage. 2 = apply topk weights in stage 2, which is what the
# shim does; the generated instances must agree or the lookup misses.
_MUL_WEIGHT_STAGE = 2


def _ck2stages_module(dtype: torch.dtype, activation: str) -> AiterModule:
    """A ``module_moe_ck2stages`` specialized to one dtype/activation.

    Mirrors AITER's own ``get_moe_stage_module`` rather than importing it, so a
    rename there surfaces as a build error instead of a silent change.
    """
    # Imported lazily: this module is importable without aiter installed, and
    # only a real build needs the path.
    from aiter.jit.core import AITER_CSRC_DIR

    a = b = c = _DTYPE_TAG[dtype]
    act = activation
    quant = "no"
    md_name = "_".join(
        [
            "module_moe_ck2stages",
            a,
            b,
            "preshuffle_off",
            c,
            act,
            quant,
            f"mulWeightStage{_MUL_WEIGHT_STAGE}",
        ]
    )
    # build_module runs `blob_gen_cmd.format(blob_dir)`, so the string must carry
    # exactly one `{}` and no other braces.
    blob_gen_cmd = (
        f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py "
        f"-a {a} -b {b} -c {c} -q {quant} -act {act} "
        f"-m {_MUL_WEIGHT_STAGE} --working_path {{}}"
    )
    return AiterModule(
        "module_moe_ck2stages", md_name=md_name, blob_gen_cmd=blob_gen_cmd
    )


def gen_fused_moe_aiter_module(dtype: torch.dtype, activation: str) -> JitSpec:
    """Build the fused-MoE shim for one dtype/activation pair."""
    if dtype not in _DTYPE_TAG:
        # torch.dtype is not orderable, so list in declaration order.
        raise ValueError(f"fused MoE (aiter) supports {list(_DTYPE_TAG)}, got {dtype}")
    if activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"fused MoE (aiter) supports activations "
            f"{sorted(_SUPPORTED_ACTIVATIONS)}, got {activation!r}"
        )

    extra_include_paths, extra_ldflags = aiter_jitspec_flags(
        AiterModule("module_moe_sorting"),
        _ck2stages_module(dtype, activation),
    )
    return refresh_aiter_jitspec(
        gen_jit_spec(
            f"fused_moe_aiter_{_DTYPE_TAG[dtype]}_{activation}",
            [
                jit_env.FLASHINFER_CSRC_DIR / "fused_moe_aiter.cu",
                jit_env.FLASHINFER_CSRC_DIR / "fused_moe_aiter_jit_pybind.cu",
            ],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
        )
    )
