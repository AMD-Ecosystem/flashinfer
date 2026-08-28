# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Which backends the harness may run, on a ROCm device.

Kept out of ``flashinfer_benchmark_utils.py`` so the upstream file carries only
the branch that calls into here.
"""

import torch

from flashinfer.aiter_utils import is_aiter_available
from flashinfer.arch_caps import capability_available, normalize_arch

# Benchmark routine -> the CAPABILITIES op key in flashinfer/arch_caps.py, so
# the ROCm backend list derives from the arch-support matrix rather than
# restating it. MLA is absent until mla_rocm matches the CUDA wrapper.
_ROCM_ROUTINE_TO_CAP_OP = {
    "BatchDecodeWithPagedKVCacheWrapper": "batch_decode",
    "BatchPrefillWithPagedKVCacheWrapper": "batch_prefill",
    "BatchPrefillWithRaggedKVCacheWrapper": "batch_prefill",
}

# GQA group sizes the HIP decode kernel instantiates; see DISPATCH_GQA_GROUP_SIZE
# in include/flashinfer/utils.cuh. Others raise "Unsupported group_size" from the
# kernel, which aborts the whole test case rather than skipping one backend.
HIP_DECODE_GQA_GROUP_SIZES = frozenset({1, 2, 3, 4, 8})


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
    op = _ROCM_ROUTINE_TO_CAP_OP.get(routine)
    if op is None:
        return []
    # AITER is reached through "auto", which records what it resolved to, so
    # "aiter" is deliberately not offered: the attention routines have no
    # construction or dispatch path for it, and advertising it here would let a
    # backend through the filter that argparse rejects and no wrapper builds.
    if capability_available(device, op, "hip"):
        return ["fa2", "auto"]
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
