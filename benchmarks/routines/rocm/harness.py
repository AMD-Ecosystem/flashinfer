# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Harness helpers the ROCm port added to the benchmark runner.

Not all of these branch on the platform, but all of them exist to serve the
ROCm path; they live here so the upstream benchmark files carry only the call.
"""

import importlib

from flashinfer.device_utils import IS_HIP

# Result columns the ROCm path fills in. Appended to output_column_dict["perf"].
PERF_COLUMNS = ("backend_resolved", "backend_fallback_reason")


def l2_flush_size_mb():
    """Flush-buffer size large enough to evict the device's last-level cache.

    CDNA's 256 MB Infinity Cache exactly equals the upstream buffer, which would
    leave its own tail resident.
    """
    return 512 if IS_HIP else 256


def bench_timing_kwargs(args, device):
    """Timing arguments shared by every bench_gpu_time call site.

    `bench_gpu_time` honours a `*_time_ms` budget only when the matching
    `*_iters` is None, so the two are mutually exclusive per phase.
    """
    kwargs = {
        "dry_run_iters": args.dry_run_iters,
        "repeat_iters": args.num_iters,
        "l2_flush": True,
        "l2_flush_size_mb": l2_flush_size_mb(),
        "l2_flush_device": device,
    }
    if getattr(args, "dry_run_time_ms", None) is not None:
        kwargs["dry_run_iters"] = None
        kwargs["dry_run_time_ms"] = args.dry_run_time_ms
    if getattr(args, "repeat_time_ms", None) is not None:
        kwargs["repeat_iters"] = None
        kwargs["repeat_time_ms"] = args.repeat_time_ms
    return kwargs


def add_timing_budget_args(parser):
    """Register the wall-clock alternatives to --dry_run_iters / --num_iters."""
    parser.add_argument(
        "--dry_run_time_ms",
        type=int,
        required=False,
        default=None,
        help="Warmup budget in ms. Overrides --dry_run_iters. On ROCm prefer this "
        "over an iteration count: clock sampling intervals run to hundreds of ms, "
        "so a handful of iterations warms up well short of steady-state clocks.",
    )
    parser.add_argument(
        "--repeat_time_ms",
        type=int,
        required=False,
        default=None,
        help="Measurement budget in ms. Overrides --num_iters. Leaving this unset "
        "keeps the sample count fixed, which makes std_time comparable run over run.",
    )


def use_cuda_graph_for(backend, is_cuda_graph_compatible):
    """Whether ``backend`` may run under a graph capture.

    fa2 is upstream's exclusion; "auto" joins it because it may have resolved
    to AITER, whose launch grid is fixed at capture shapes.
    """
    return is_cuda_graph_compatible and backend not in ("fa2", "auto")


def as_nhd_paged_kv_cache(kv_cache):
    """View an HND-shaped ``[pages, 2, heads, page_size, dim]`` paged cache as NHD.

    ``result[p, i, s, h]`` is ``kv_cache[p, i, h, s]`` -- the same logical entry.
    Zero-copy, and contiguous when the caller built the cache NHD-ordered.
    """
    return kv_cache.transpose(2, 3)


def record_backend_resolution(cur_res, wrapper):
    """Record what ``auto`` resolved to, and why it declined AITER.

    Only meaningful for ``auto``: an explicit backend resolves to itself.
    """
    if wrapper is None or cur_res.get("backend") != "auto":
        return
    # `or ""` matters: the attribute exists and is None when auto did not decline,
    # and str(None) would write the literal "None" into the CSV.
    cur_res["backend_resolved"] = getattr(wrapper, "backend", "") or ""
    cur_res["backend_fallback_reason"] = (
        getattr(wrapper, "backend_fallback_reason", "") or ""
    )


class _UnavailableRoutineGroup:
    """Stand-in for a routine module that failed to import.

    Reaching for a routine re-raises with the original cause; falsy, so a
    caller can ask whether the group loaded without tripping that.
    """

    def __init__(self, name, exc):
        self._name = name
        self._exc = exc

    def __bool__(self):
        return False

    def __getattr__(self, attr):
        # Private and dunder lookups must answer AttributeError, or copy and
        # pickle recurse forever probing __setstate__ on an instance whose
        # _name is not set yet.
        if attr.startswith("_"):
            raise AttributeError(attr)
        raise RuntimeError(
            f"The '{self._name}' benchmark routines are unavailable on this "
            f"platform: {type(self._exc).__name__}: {self._exc}"
        ) from self._exc


def load_routine_group(name):
    """Import a sibling ``routines`` submodule, deferring failure to first use.

    The gemm and moe routines pull in CUDA-only modules; letting that raise
    here would take down the whole runner, attention routines included.
    """
    try:
        return importlib.import_module(f"..{name}", __package__)
    except Exception as exc:  # noqa: BLE001 - reported verbatim on first use
        return _UnavailableRoutineGroup(name, exc)
