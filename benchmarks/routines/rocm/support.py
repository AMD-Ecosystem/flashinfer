# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Which backends the harness may run, on a ROCm device.

Kept out of ``flashinfer_benchmark_utils.py`` so the upstream file carries only
the branch that calls into here.
"""

import torch

from flashinfer.rocm.aiter_utils import is_aiter_available
from flashinfer.rocm.device_utils import IS_HIP
from flashinfer.rocm.arch_caps import capability_available, normalize_arch

# Benchmark routine -> (CAPABILITIES op key in flashinfer/rocm/arch_caps.py,
# the backend names this routine's own --backends accepts). Deriving the op from
# the arch-support matrix keeps the two from drifting; the names differ per
# routine because the upstream CLIs disagree on what to call the native kernel.
# MLA is absent until mla_rocm matches the CUDA wrapper.
_ATTENTION = ("fa2", "auto")
# Everything outside attention runs the library default, which upstream's CLI
# spells "cuda" -- the same legacy name torch uses for the HIP device.
_NATIVE = ("cuda",)

_ROCM_ROUTINE_TO_CAP_OP = {
    "BatchDecodeWithPagedKVCacheWrapper": ("batch_decode", _ATTENTION),
    "BatchPrefillWithPagedKVCacheWrapper": ("batch_prefill", _ATTENTION),
    "BatchPrefillWithRaggedKVCacheWrapper": ("batch_prefill", _ATTENTION),
    # norm
    "rmsnorm": ("rmsnorm", _NATIVE),
    "fused_add_rmsnorm": ("fused_add_rmsnorm", _NATIVE),
    "gemma_rmsnorm": ("layernorm", _NATIVE),
    "gemma_fused_add_rmsnorm": ("layernorm", _NATIVE),
    # rope. apply_rope_with_cos_sin_cache is absent: the routine builds
    # cos_sin_cache in --input_dtype, but the op requires float32 and
    # --input_dtype offers none, so it fails on CUDA too.
    "apply_rope": ("rope", _NATIVE),
    "apply_rope_pos_ids": ("rope", _NATIVE),
    "apply_llama31_rope": ("rope", _NATIVE),
    "apply_llama31_rope_pos_ids": ("rope", _NATIVE),
    "rope_quantize_fp8": ("rope", _NATIVE),
    "mla_rope_quantize_fp8": ("rope", _NATIVE),
    "rope_quantize_fp8_append_paged_kv_cache": ("rope", _NATIVE),
    # sampling. top_k, top_k_page_table_transform and top_k_ragged_transform are
    # absent: they call flashinfer.topk, which has no csrc/rocm kernel.
    "softmax": ("sampling", _NATIVE),
    "sampling_from_probs": ("sampling", _NATIVE),
    "sampling_from_logits": ("sampling", _NATIVE),
    "top_k_sampling_from_probs": ("sampling", _NATIVE),
    "top_p_sampling_from_probs": ("sampling", _NATIVE),
    "top_k_top_p_sampling_from_probs": ("sampling", _NATIVE),
    "top_k_top_p_sampling_from_logits": ("sampling", _NATIVE),
    "min_p_sampling_from_probs": ("sampling", _NATIVE),
    "top_k_renorm_probs": ("sampling", _NATIVE),
    "top_p_renorm_probs": ("sampling", _NATIVE),
    "top_k_mask_logits": ("sampling", _NATIVE),
    "chain_speculative_sampling": ("sampling", _NATIVE),
}

# GQA group sizes the HIP decode kernel instantiates; see DISPATCH_GQA_GROUP_SIZE
# in include/flashinfer/utils.cuh. Others raise "Unsupported group_size" from the
# kernel, which aborts the whole test case rather than skipping one backend.
HIP_DECODE_GQA_GROUP_SIZES = frozenset({1, 2, 3, 4, 8})


# is_float8_tensor (csrc/rocm/pytorch_extension_utils.h) accepts only the fnuz
# encodings, on both architectures -- unrelated to moe_fp8_dtype(), which is
# arch-dependent because it follows AITER's MFMA instructions. Without this the
# fp8 rope routines fail inside the kernel with "Output dtype must be float8",
# which names neither the dtype nor the argument that chose it.
_HIP_QUANT_DTYPES = {
    torch.float8_e4m3fn: torch.float8_e4m3fnuz,
    torch.float8_e5m2: torch.float8_e5m2fnuz,
}


def to_fnuz(dtype):
    """The fnuz counterpart of ``dtype``, or ``dtype`` if it has none.

    Pure, so the mapping stays testable on a machine that is not ROCm.
    """
    return _HIP_QUANT_DTYPES.get(dtype, dtype)


def _identity(dtype):
    return dtype


# Bound once rather than branching per call, so a CUDA run provably reaches the
# identity: rope.py calls this on both platforms, and inverting the mapping
# would otherwise turn every CUDA fp8 rope row fnuz with nothing to catch it.
#
# Deliberately not folded into ``dtype_str_to_torch_dtype``: that is shared with
# the attention routines, whose dtype accept-lists name the OCP spellings and
# would reject an fnuz tensor, dropping the row from the CSV.
hip_quant_dtype = to_fnuz if IS_HIP else _identity


def get_device_arch(device):
    """Normalized gfx architecture of ``device``, or ``"unknown"``."""
    try:
        return normalize_arch(torch.cuda.get_device_properties(device).gcnArchName)
    except Exception:
        return "unknown"


def aiter_serves(device, op):
    """Whether ``auto`` can actually reach AITER for ``op`` on ``device``.

    ``capability_available`` answers only the architecture and known-bad
    question; ``is_aiter_available`` also requires the package to import, which
    is what the selector really gates on.
    """
    return is_aiter_available(device, op)


def fa2_backed_backends(backends, device, op):
    """Requested backends that will execute the in-tree HIP kernel.

    "auto" belongs here whenever AITER cannot serve the call, since the selector
    then resolves it to fa2 -- so it inherits every fa2 constraint. Returns a
    list so callers may mutate ``backends`` while iterating.
    """
    names = [b for b in backends if b == "fa2"]
    if "auto" in backends and not aiter_serves(device, op):
        names.append("auto")
    return names


def rocm_supported_backends(routine, device):
    """Backends this harness can run for ``routine`` on a ROCm ``device``.

    "fa2" is the harness's name for the in-tree HIP kernel -- the same backend
    arch_caps calls "hip" and declares as the AITER rows' ``fallback``.
    """
    entry = _ROCM_ROUTINE_TO_CAP_OP.get(routine)
    if entry is None:
        return []
    op, backends = entry
    # AITER is reached through "auto", which records what it resolved to, so
    # "aiter" is deliberately not offered: no routine has a construction or
    # dispatch path for it, and advertising it here would let a backend through
    # the filter that argparse rejects and no wrapper builds.
    if capability_available(device, op, "hip"):
        return list(backends)
    return []


def filter_backends_by_arch(backends, routine, device):
    """ROCm counterpart of ``filter_backends_by_compute_capability``.

    gfx942/gfx950 report compute capability 9.4/9.5, which match no entry in the
    NVIDIA table -- routing them through it would strip every backend, fa2
    included. Mutates and returns ``backends``, as the upstream function does.
    """
    supported = rocm_supported_backends(routine, device)
    arch = get_device_arch(device)
    for backend in [b for b in backends if b not in supported]:
        backends.remove(backend)
        print(
            f"[WARNING] {backend} for routine {routine} is not supported on "
            f"architecture {arch}. Skipping."
        )
    return backends
