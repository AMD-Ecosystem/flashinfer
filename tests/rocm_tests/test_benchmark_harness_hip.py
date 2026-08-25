# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Guards for the ROCm path of benchmarks/flashinfer_benchmark.py. Each of these
# covers a failure mode that is silent: a zero-row CSV, a mislabelled backend, or
# a row whose fields have shifted. None of them announce themselves as errors.

import csv
import io

import pytest
import torch

from flashinfer.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(not IS_HIP, reason="ROCm-only benchmark harness path")

ATTENTION_ROUTINES = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
]


def _utils():
    from routines import flashinfer_benchmark_utils as u

    return u


@pytest.mark.parametrize("routine", ATTENTION_ROUTINES)
def test_fa2_survives_the_backend_filter(routine):
    """gfx942/gfx950 report CC 9.4/9.5, absent from the NVIDIA table.

    Before the arch-keyed branch this stripped every backend including fa2, and
    the run ended with "No backends to test" and a zero-row CSV rather than a
    failure.
    """
    u = _utils()
    device = torch.device("cuda")
    assert "fa2" in u.filter_backends_by_compute_capability(["fa2"], routine, device)


@pytest.mark.parametrize("routine", ["bmm_fp8", "cutlass_fused_moe"])
def test_unported_routines_filter_to_empty(routine):
    """Routines with no ROCm port resolve to an empty list, not a KeyError."""
    u = _utils()
    device = torch.device("cuda")
    assert u.filter_backends_by_compute_capability(["cudnn"], routine, device) == []


def test_benchmark_module_imports():
    """routines.moe imports CUDA-only symbols; unguarded it broke every routine."""
    import flashinfer_benchmark

    assert callable(flashinfer_benchmark.run_test)


def test_unavailable_routine_group_reports_the_original_error():
    import flashinfer_benchmark

    if "moe" not in flashinfer_benchmark._ROUTINE_IMPORT_ERRORS:
        pytest.skip("routines.moe imported successfully on this build")
    with pytest.raises(RuntimeError, match="unavailable on this platform"):
        flashinfer_benchmark.require_routine_group("moe")


def test_nhd_view_is_an_exact_zero_copy_permutation():
    """The auto path feeds AITER this view instead of rebuilding the cache.

    A wrong permutation would still run and still produce plausible timings, so
    the exactness is the whole safety argument.
    """
    u = _utils()
    pages, heads, page_size, dim = 3, 4, 8, 16
    base = torch.randn(pages, 2, heads, page_size, dim)
    hnd = base.as_strided(
        base.shape,
        (
            2 * page_size * heads * dim,
            page_size * heads * dim,
            dim,
            heads * dim,
            1,
        ),
    )
    nhd = u.as_nhd_paged_kv_cache(hnd)

    assert nhd.shape == (pages, 2, page_size, heads, dim)
    assert nhd.is_contiguous()
    assert nhd.data_ptr() == hnd.data_ptr()
    for h in range(heads):
        for s in range(page_size):
            assert torch.equal(nhd[:, :, s, h], hnd[:, :, h, s])


def test_result_row_survives_a_comma_in_the_fallback_reason():
    """arch_caps' known-bad reason contains commas; a bare "," join splits it."""
    reason = (
        "causal=True is miscompiled on ROCm 7.2.x (not an error), 97.6% of "
        "elements off. Use ROCm 7.1, or backend='fa2'"
    )
    header = ["routine", "backend_fallback_reason", "tflops"]
    values = ["BatchPrefill", reason, "5.4"]

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerow(values)

    parsed = next(csv.DictReader(io.StringIO(buf.getvalue())))
    assert parsed["backend_fallback_reason"] == reason
    assert parsed["tflops"] == "5.4"
    # The naive join this replaced produced 5 fields for these 3 columns.
    assert len(",".join(values).split(",")) > len(header)


def test_hip_gqa_group_sizes_match_the_kernel_dispatch():
    """The harness constant must track DISPATCH_GQA_GROUP_SIZE, not just look right.

    A group size outside the macro raises from inside the kernel and takes the
    whole test case down, so the harness drops fa2 first. If someone extends the
    macro and not the constant, the benchmark silently stops measuring shapes
    that became supported -- read the header rather than restate its contents.
    """
    import re
    from pathlib import Path

    import flashinfer

    header = Path(flashinfer.get_include()) / "flashinfer" / "utils.cuh"
    if not header.is_file():
        pytest.skip(f"kernel header not present at {header}")
    body = header.read_text()
    macro = body[body.index("#define DISPATCH_GQA_GROUP_SIZE") :]
    macro = macro[: macro.index("#define", 1)]
    from_kernel = {int(n) for n in re.findall(r"group_size == (\d+)", macro)}

    assert from_kernel, "could not parse DISPATCH_GQA_GROUP_SIZE"
    assert frozenset(from_kernel) == _utils().HIP_DECODE_GQA_GROUP_SIZES
    # Llama-3.1-405B is 128 qo / 8 kv heads, i.e. group size 16.
    assert 128 // 8 not in from_kernel


def test_auto_inherits_fa2_constraints_when_aiter_cannot_serve():
    """`auto` resolves to fa2 when AITER is declined, so it inherits fa2's limits.

    Guarding only the literal "fa2" leaves `auto` to reach a kernel that aborts
    the whole test case -- exactly the crash the group-size guard prevents.
    """
    u = _utils()
    device = torch.device("cuda")
    for op in ("batch_decode", "batch_prefill"):
        backed = u.fa2_backed_backends(["fa2", "auto"], device, op)
        assert "fa2" in backed
        assert ("auto" in backed) is not u.aiter_serves(device, op)


def test_resolution_is_recorded_only_for_auto():
    """An explicit backend resolves to itself; filling the column for it would
    make every fa2 row match the README's "investigate this" signature."""
    u = _utils()

    class _Wrapper:
        backend = "fa2"
        backend_fallback_reason = None

    for requested, expected in (("fa2", ""), ("auto", "fa2")):
        row = {"backend": requested, "backend_resolved": "", "fallback": ""}
        u.record_backend_resolution(row, _Wrapper())
        assert row["backend_resolved"] == expected


def test_timing_kwargs_default_to_iteration_counts():
    """Passing no time budget must reproduce the previous behaviour exactly."""
    u = _utils()
    args = pytest.importorskip("argparse").Namespace(
        dry_run_iters=5, num_iters=30, dry_run_time_ms=None, repeat_time_ms=None
    )
    kwargs = u.bench_timing_kwargs(args, torch.device("cuda"))
    assert kwargs["dry_run_iters"] == 5
    assert kwargs["repeat_iters"] == 30
    assert "dry_run_time_ms" not in kwargs
    # 256 MB exactly equals CDNA's Infinity Cache, leaving its own tail resident.
    assert kwargs["l2_flush_size_mb"] > 256


def test_time_budget_overrides_the_matching_iteration_count():
    """bench_gpu_time honours *_time_ms only when the matching *_iters is None."""
    u = _utils()
    args = pytest.importorskip("argparse").Namespace(
        dry_run_iters=5, num_iters=30, dry_run_time_ms=250, repeat_time_ms=None
    )
    kwargs = u.bench_timing_kwargs(args, torch.device("cuda"))
    assert kwargs["dry_run_iters"] is None
    assert kwargs["dry_run_time_ms"] == 250
    assert kwargs["repeat_iters"] == 30
