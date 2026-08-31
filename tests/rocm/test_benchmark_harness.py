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

from flashinfer.rocm.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(not IS_HIP, reason="ROCm-only benchmark harness path")

ATTENTION_ROUTINES = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
]


def _utils():
    from routines import flashinfer_benchmark_utils as u

    return u


def _rocm():
    import routines.rocm as r

    return r


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


def test_unavailable_routine_group_defers_its_import_error():
    """A failed group must stay importable and only raise when actually used."""
    from routines.rocm import load_routine_group

    group = load_routine_group("no_such_routine_module")
    assert not group, "a failed group must be falsy so callers can test it"
    with pytest.raises(RuntimeError, match="unavailable on this platform"):
        _ = group.parse_args


def test_available_routine_group_is_the_real_module():
    """The loader must not mask a group that imports fine."""
    from routines.rocm import load_routine_group

    group = load_routine_group("attention")
    assert callable(group.run_attention_test)
    assert group


@pytest.mark.parametrize("group_name", ["gemm", "moe"])
def test_runner_routes_each_group_through_the_loader(group_name):
    """The runner's own bindings, not just the loader in isolation.

    Rebinding either to None passes every other test here while turning the
    actionable "unavailable on this platform" message into an AttributeError
    that the testlist loop swallows.
    """
    import flashinfer_benchmark

    group = getattr(flashinfer_benchmark, group_name)
    entry = f"run_{group_name}_test"
    if group:
        assert callable(getattr(group, entry))
    else:
        with pytest.raises(RuntimeError, match="unavailable on this platform"):
            getattr(group, entry)


def test_unavailable_group_survives_copy_and_pickle():
    """__getattr__ must answer AttributeError for private names.

    Returning the RuntimeError for them instead sends copy/pickle into
    infinite recursion probing __setstate__ before _name is bound.
    """
    import copy
    import pickle

    from routines.rocm.harness import _UnavailableRoutineGroup

    group = _UnavailableRoutineGroup("gemm", ImportError("boom"))
    assert not copy.copy(group)
    assert not pickle.loads(pickle.dumps(group))


def test_runner_registers_the_timing_budget_options():
    """Dropping the add_timing_budget_args call fails loudly here, not silently.

    argparse would reject --dry_run_time_ms outright, but a testlist line that
    carries it is the only place that shows up.
    """
    import flashinfer_benchmark

    args = flashinfer_benchmark.parse_args(
        [
            "--routine", "BatchDecodeWithPagedKVCacheWrapper",
            "--page_size", "16", "--batch_size", "4", "--s_qo", "1", "--s_kv", "128",
            "--num_qo_heads", "8", "--num_kv_heads", "8",
            "--head_dim_qk", "128", "--head_dim_vo", "128",
            "--dry_run_time_ms", "250", "--repeat_time_ms", "400",
        ]
    )  # fmt: skip
    assert args.dry_run_time_ms == 250
    assert args.repeat_time_ms == 400


def test_nhd_view_is_an_exact_zero_copy_permutation():
    """The auto path feeds AITER this view instead of rebuilding the cache.

    A wrong permutation would still run and still produce plausible timings, so
    the exactness is the whole safety argument.
    """
    r = _rocm()
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
    nhd = r.as_nhd_paged_kv_cache(hnd)

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
    assert frozenset(from_kernel) == _rocm().HIP_DECODE_GQA_GROUP_SIZES
    # Llama-3.1-405B is 128 qo / 8 kv heads, i.e. group size 16.
    assert 128 // 8 not in from_kernel


def _backend_choices():
    """The values attention.py's `--backends` will accept."""
    import argparse
    import contextlib

    import routines.attention as attention

    parser = argparse.ArgumentParser()
    # argparse may reject the stub argv; the action is registered either way.
    with contextlib.suppress(SystemExit):
        attention.parse_attention_args(
            ["--routine", "BatchDecodeWithPagedKVCacheWrapper"], parser
        )
    for action in parser._actions:
        if action.dest == "backends":
            return set(action.choices)
    raise AssertionError("--backends action not registered")


@pytest.mark.parametrize("routine", ATTENTION_ROUTINES)
def test_filter_only_offers_backends_the_cli_accepts(routine):
    """Whatever survives the filter must be a backend attention.py can dispatch.

    Offering one that argparse rejects, or that no wrapper branch constructs,
    turns a supported configuration into a silently missing CSV row.
    """
    r = _rocm()
    offered = r.rocm_supported_backends(routine, torch.device("cuda"))
    assert offered, f"{routine} offers no backend at all on this device"
    unknown = set(offered) - _backend_choices()
    assert not unknown, f"{unknown} survive the filter but --backends rejects them"


def test_cuda_path_still_matches_its_table(monkeypatch):
    """The CUDA table is keyed "10.0", so the lookup may not use the warning text.

    Forces the non-HIP branch on a ROCm box, since the defect is in the lookup
    rather than the device: reusing one variable for both the dict key and the
    "not supported on ..." message strips every backend on NVIDIA -- the exact
    failure this branch exists to fix for AMD.
    """
    u = _utils()
    monkeypatch.setattr(u, "IS_HIP", False)
    monkeypatch.setattr(u, "get_compute_capability", lambda device: (10, 0))
    kept = u.filter_backends_by_compute_capability(
        ["fa2", "cudnn"], "BatchDecodeWithPagedKVCacheWrapper", torch.device("cuda")
    )
    assert kept == ["fa2", "cudnn"]


def test_auto_inherits_fa2_constraints_when_aiter_cannot_serve():
    """`auto` resolves to fa2 when AITER is declined, so it inherits fa2's limits.

    Guarding only the literal "fa2" leaves `auto` to reach a kernel that aborts
    the whole test case -- exactly the crash the group-size guard prevents.
    """
    r = _rocm()
    device = torch.device("cuda")
    for op in ("batch_decode", "batch_prefill"):
        backed = r.fa2_backed_backends(["fa2", "auto"], device, op)
        assert "fa2" in backed
        assert ("auto" in backed) is not r.aiter_serves(device, op)


def test_resolution_is_recorded_only_for_auto():
    """An explicit backend resolves to itself; filling the column for it would
    make every fa2 row match the README's "investigate this" signature."""
    r = _rocm()

    class _Wrapper:
        backend = "fa2"
        backend_fallback_reason = None

    for requested, expected in (("fa2", ""), ("auto", "fa2")):
        row = {"backend": requested, "backend_resolved": "", "fallback": ""}
        r.record_backend_resolution(row, _Wrapper())
        assert row["backend_resolved"] == expected


@pytest.mark.parametrize(
    "backend,compatible,expected",
    [
        # "auto" may have resolved to AITER, whose launch grid is fixed at
        # capture shapes, so it must be timed eagerly exactly like fa2.
        ("auto", True, False),
        ("fa2", True, False),
        ("fa2_tc", True, True),
        ("trtllm-gen", True, True),
        ("trtllm-gen", False, False),
    ],
)
def test_only_fa2_and_auto_opt_out_of_graph_capture(backend, compatible, expected):
    assert _rocm().use_cuda_graph_for(backend, compatible) is expected


def test_timing_kwargs_default_to_iteration_counts():
    """Passing no time budget must reproduce the previous behaviour exactly."""
    r = _rocm()
    args = pytest.importorskip("argparse").Namespace(
        dry_run_iters=5, num_iters=30, dry_run_time_ms=None, repeat_time_ms=None
    )
    kwargs = r.bench_timing_kwargs(args, torch.device("cuda"))
    assert kwargs["dry_run_iters"] == 5
    assert kwargs["repeat_iters"] == 30
    assert "dry_run_time_ms" not in kwargs
    # 256 MB exactly equals CDNA's Infinity Cache, leaving its own tail resident.
    assert kwargs["l2_flush_size_mb"] > 256


def test_time_budget_overrides_the_matching_iteration_count():
    """bench_gpu_time honours *_time_ms only when the matching *_iters is None."""
    r = _rocm()
    args = pytest.importorskip("argparse").Namespace(
        dry_run_iters=5, num_iters=30, dry_run_time_ms=250, repeat_time_ms=None
    )
    kwargs = r.bench_timing_kwargs(args, torch.device("cuda"))
    assert kwargs["dry_run_iters"] is None
    assert kwargs["dry_run_time_ms"] == 250
    assert kwargs["repeat_iters"] == 30
