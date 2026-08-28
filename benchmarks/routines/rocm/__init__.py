# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The ROCm halves of the benchmark harness.

Everything the upstream benchmark files would otherwise carry inline lives
here, so their diff against flashinfer-ai/flashinfer stays a delegating call.
"""

from .harness import (
    PERF_COLUMNS,
    add_timing_budget_args,
    as_nhd_paged_kv_cache,
    bench_timing_kwargs,
    l2_flush_size_mb,
    load_routine_group,
    record_backend_resolution,
    use_cuda_graph_for,
)
from .support import (
    HIP_DECODE_GQA_GROUP_SIZES,
    aiter_serves,
    fa2_backed_backends,
    filter_backends_by_arch,
    get_device_arch,
    rocm_supported_backends,
)

__all__ = [
    "HIP_DECODE_GQA_GROUP_SIZES",
    "PERF_COLUMNS",
    "add_timing_budget_args",
    "aiter_serves",
    "as_nhd_paged_kv_cache",
    "bench_timing_kwargs",
    "fa2_backed_backends",
    "filter_backends_by_arch",
    "get_device_arch",
    "l2_flush_size_mb",
    "load_routine_group",
    "record_backend_resolution",
    "rocm_supported_backends",
    "use_cuda_graph_for",
]
