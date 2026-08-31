# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``backend="auto"`` must demote to fa2 when AITER cannot build a kernel.

The capability table and the aiter import probe both run before AITER's
per-variant .so is built, so a build that cannot compile — or cannot install
into a site-packages it does not own — fails only at plan()/call time. Each
test makes the relevant AITER bootstrap raise the failure observed on gfx950
(``ModuleNotFoundError`` after a successful compile) and requires ``auto`` to
fall back, while explicit ``backend="aiter"`` still raises.

Two harness notes, both of which silently void these tests if ignored:

- The probes are ``lru_cache``d, so every test clears them in a ``finally``.
  ``get_*_module`` memoize *successes*, so a test running after a passing AITER
  test would otherwise never reach the patched bootstrap at all.
- ``caplog`` does not capture flashinfer's logger. It is a
  ``FlashInferJITLogger`` built directly at ``jit/core.py``, bypassing
  ``logging.getLogger``, so ``.parent`` is None and records never reach the root
  handler caplog installs. Patch ``logger.warning`` instead.
"""

import logging

import pytest
import torch

import flashinfer
import flashinfer.rocm.decode
import flashinfer.rocm.prefill
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.arch_caps import ArchCapabilityError, capability_available
from flashinfer.rocm.decode import _aiter_pa_v1_available
from flashinfer.jit.core import MissingJITCacheError, logger
from flashinfer.rocm.prefill import (
    _aiter_batch_ragged_available,
    _aiter_native_paging_available,
    _aiter_ops_importable,
    _aiter_single_prefill_available,
)

logger.setLevel(logging.ERROR)

# The exact failure measured on gfx950: AITER's own JIT built the variant, then
# could not install it into a site-packages owned by another user.
_INSTALL_FAILURE = ModuleNotFoundError(
    "No module named 'aiter.jit.module_mha_fwd_bf16_causal'"
)

_ALL_PROBES = (
    _aiter_single_prefill_available,
    _aiter_batch_ragged_available,
    _aiter_pa_v1_available,
    _aiter_native_paging_available,
)


def _clear_probes():
    for probe in _ALL_PROBES:
        probe.cache_clear()


@pytest.fixture(autouse=True)
def clear_probe_caches():
    _clear_probes()
    yield
    _clear_probes()


@pytest.fixture
def device():
    dev = torch.device("cuda:0")
    if not is_aiter_supported(dev) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    return dev


def _raiser(exc):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


def _skip_if_op_gated(device, op):
    """Skip when the capability table already steers auto away from AITER.

    With the gate active the selector returns fa2 before any probe runs, so
    there is no demotion left to observe. Currently hits batch_prefill on
    gfx950/ROCm 7.2.x; single_prefill and batch_decode are ungated.
    """
    if not capability_available(device, op, "aiter"):
        pytest.skip(f"capability table gates AITER {op} on this toolchain")


# --------------------------------------------------------------------------
# Site 1 -- single prefill
# --------------------------------------------------------------------------


def _single_prefill_inputs(device, dtype=torch.bfloat16):
    torch.manual_seed(0)
    qo_len, kv_len, num_heads, head_dim = 64, 128, 8, 128
    q = torch.randn(qo_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(kv_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(kv_len, num_heads, head_dim, device=device, dtype=dtype)
    return q, k, v


@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_auto_demotes_to_fa2(device, monkeypatch, causal):
    """auto must produce fa2 numerics, not raise, when the mha_fwd build fails."""
    _skip_if_op_gated(device, "single_prefill")
    q, k, v = _single_prefill_inputs(device)

    # Record which backend actually built: single prefill is a free function with
    # no _backend attribute, so absence of a crash is not evidence of fa2.
    built = []
    real_get = flashinfer.rocm.prefill.get_single_prefill_module

    def _recording_get(backend, *args):
        built.append(backend)
        return real_get(backend, *args)

    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_single_prefill_mha_fwd",
        _raiser(_INSTALL_FAILURE),
    )
    monkeypatch.setattr(
        flashinfer.rocm.prefill, "get_single_prefill_module", _recording_get
    )

    out = flashinfer.rocm.prefill.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="auto"
    )

    assert built == ["fa2"], f"expected the fa2 module to be built, got {built}"

    monkeypatch.undo()
    ref = flashinfer.rocm.prefill.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa2"
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_single_prefill_explicit_aiter_still_raises(device, monkeypatch):
    """The demotion is auto-only; an explicit opt-in keeps its hard error."""
    _skip_if_op_gated(device, "single_prefill")
    q, k, v = _single_prefill_inputs(device)
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_single_prefill_mha_fwd",
        _raiser(_INSTALL_FAILURE),
    )
    with pytest.raises(ModuleNotFoundError):
        flashinfer.rocm.prefill.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="aiter"
        )


def test_single_prefill_strict_env_raises(device, monkeypatch):
    """FLASHINFER_AITER_STRICT=1 opts out of demotion entirely."""
    _skip_if_op_gated(device, "single_prefill")
    q, k, v = _single_prefill_inputs(device)
    monkeypatch.setenv("FLASHINFER_AITER_STRICT", "1")
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_single_prefill_mha_fwd",
        _raiser(_INSTALL_FAILURE),
    )
    with pytest.raises(ModuleNotFoundError):
        flashinfer.rocm.prefill.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="auto"
        )


def test_single_prefill_warns_once(device, monkeypatch):
    """A second call reuses the cached verdict: no rebuild, no second warning."""
    _skip_if_op_gated(device, "single_prefill")
    q, k, v = _single_prefill_inputs(device)

    attempts = []

    def _count_and_raise(*args, **kwargs):
        attempts.append(1)
        raise _INSTALL_FAILURE

    warnings = []
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_single_prefill_mha_fwd",
        _count_and_raise,
    )
    # caplog cannot see this logger; patch the method instead.
    monkeypatch.setattr(
        flashinfer.rocm.aiter_utils.logger,
        "warning",
        lambda *a, **kw: warnings.append(a),
    )

    for _ in range(2):
        flashinfer.rocm.prefill.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="auto"
        )

    assert len(attempts) == 1, (
        f"the failing AITER build was retried {len(attempts)} times; "
        "the probe verdict is not being memoized"
    )
    assert len(warnings) == 1, f"expected one warning, got {len(warnings)}"


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("caller contract violation"),
        torch.cuda.OutOfMemoryError("transient"),
        ArchCapabilityError("known-bad toolchain"),
        MissingJITCacheError("flashinfer AOT cache missing"),
    ],
    ids=["value", "oom", "arch_gate", "jit_cache"],
)
def test_non_demotable_failures_propagate(device, monkeypatch, exc):
    """These are not "AITER cannot build this variant" and must not become fa2.

    Also asserts the verdict was not cached: caching an OOM would downgrade the
    config for the rest of the process.
    """
    _skip_if_op_gated(device, "single_prefill")
    q, k, v = _single_prefill_inputs(device)
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_single_prefill_mha_fwd",
        _raiser(exc),
    )
    with pytest.raises(type(exc)):
        flashinfer.rocm.prefill.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="auto"
        )
    assert _aiter_single_prefill_available.cache_info().currsize == 0, (
        "a non-demotable failure was cached as a verdict"
    )


# --------------------------------------------------------------------------
# Sites 2 and 3 -- paged and ragged batch prefill
# --------------------------------------------------------------------------


def _paged_inputs(device, page_size=16, dtype=torch.bfloat16):
    torch.manual_seed(0)
    batch_size, qo_len, kv_len = 2, 16, 128
    num_qo_heads = num_kv_heads = 8
    head_dim = 128
    num_pages = (kv_len + page_size - 1) // page_size
    total_pages = num_pages * batch_size
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=dtype
    )
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages
    )
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return (
        q,
        kv_data,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        workspace,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )


def test_paged_prefill_auto_demotes_to_fa2(device, monkeypatch):
    _skip_if_op_gated(device, "batch_prefill")
    (
        q,
        kv_data,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        workspace,
        nqo,
        nkv,
        head_dim,
        page_size,
    ) = _paged_inputs(device)

    # Patch the native-paging bootstrap too. page_size=16 is not native on
    # amd-aiter 0.1.10, but _aiter_native_page_sizes() falls back to {16, 1024}
    # when the version cannot be read -- e.g. an AITER source build installed as
    # `aiter`. There the native probe would run first and win, and this test
    # would fail on an assertion that has nothing to do with what it covers.
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_batch_prefill",
        _raiser(_INSTALL_FAILURE),
    )
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_batch_ragged_prefill",
        _raiser(_INSTALL_FAILURE),
    )

    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        nqo,
        nkv,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    assert wrapper._backend == "fa2"
    assert wrapper.backend_fallback_reason.startswith(
        "aiter batch_prefill kernel bootstrap failed"
    )
    out = wrapper.run(q, kv_data)

    monkeypatch.undo()
    ref_wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    ref_wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        nqo,
        nkv,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    torch.testing.assert_close(out, ref_wrapper.run(q, kv_data), rtol=1e-3, atol=1e-3)


def test_ragged_prefill_auto_demotes_to_fa2(device, monkeypatch):
    _skip_if_op_gated(device, "batch_prefill")
    torch.manual_seed(0)
    batch_size, qo_len, kv_len = 2, 16, 128
    nqo = nkv = 8
    head_dim = 128
    dtype = torch.bfloat16
    q = torch.randn(batch_size * qo_len, nqo, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size * kv_len, nkv, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size * kv_len, nkv, head_dim, device=device, dtype=dtype)
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * kv_len
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Patch the native-paging bootstrap too. page_size=16 is not native on
    # amd-aiter 0.1.10, but _aiter_native_page_sizes() falls back to {16, 1024}
    # when the version cannot be read -- e.g. an AITER source build installed as
    # `aiter`. There the native probe would run first and win, and this test
    # would fail on an assertion that has nothing to do with what it covers.
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_batch_prefill",
        _raiser(_INSTALL_FAILURE),
    )
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_batch_ragged_prefill",
        _raiser(_INSTALL_FAILURE),
    )

    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        nqo,
        nkv,
        head_dim,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    assert wrapper._backend == "fa2"
    assert wrapper.backend_fallback_reason.startswith(
        "aiter batch_prefill kernel bootstrap failed"
    )
    out = wrapper.run(q, k, v)

    monkeypatch.undo()
    ref_wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    ref_wrapper.plan(
        qo_indptr,
        kv_indptr,
        nqo,
        nkv,
        head_dim,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    torch.testing.assert_close(out, ref_wrapper.run(q, k, v), rtol=1e-3, atol=1e-3)


def test_paged_prefill_explicit_aiter_still_raises(device, monkeypatch):
    _skip_if_op_gated(device, "batch_prefill")
    (
        _q,
        _kv,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        workspace,
        nqo,
        nkv,
        head_dim,
        page_size,
    ) = _paged_inputs(device)
    # Patch the native-paging bootstrap too. page_size=16 is not native on
    # amd-aiter 0.1.10, but _aiter_native_page_sizes() falls back to {16, 1024}
    # when the version cannot be read -- e.g. an AITER source build installed as
    # `aiter`. There the native probe would run first and win, and this test
    # would fail on an assertion that has nothing to do with what it covers.
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_batch_prefill",
        _raiser(_INSTALL_FAILURE),
    )
    monkeypatch.setattr(
        flashinfer.rocm.prefill,
        "_aiter_bootstrap_batch_ragged_prefill",
        _raiser(_INSTALL_FAILURE),
    )
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    with pytest.raises(ModuleNotFoundError):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            nqo,
            nkv,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )


# --------------------------------------------------------------------------
# Site 4 -- batch decode
# --------------------------------------------------------------------------


def _decode_inputs(device, page_size=16, dtype=torch.bfloat16):
    torch.manual_seed(0)
    batch_size, kv_len = 2, 128
    num_qo_heads = num_kv_heads = 8
    head_dim = 128
    num_pages = (kv_len + page_size - 1) // page_size
    total_pages = num_pages * batch_size
    q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * num_pages
    )
    indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return (
        q,
        kv_data,
        indptr,
        indices,
        last_page_len,
        workspace,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
    )


def _plan_decode(wrapper, args, dtype=torch.bfloat16):
    (_q, _kv, indptr, indices, last_page_len, _ws, nqo, nkv, head_dim, page_size) = args
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        nqo,
        nkv,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )


def test_decode_auto_demotes_to_fa2(device, monkeypatch):
    _skip_if_op_gated(device, "batch_decode")
    args = _decode_inputs(device)
    q, kv_data, workspace = args[0], args[1], args[5]

    monkeypatch.setattr(
        flashinfer.rocm.decode, "_aiter_pa_v1_resolve", _raiser(_INSTALL_FAILURE)
    )

    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto", use_tensor_cores=False
    )
    _plan_decode(wrapper, args)

    assert wrapper._backend == "fa2"
    assert wrapper.backend_fallback_reason.startswith(
        "aiter batch_decode kernel bootstrap failed"
    )
    # A demotion must leave no half-written AITER state. hasattr, not getattr --
    # the attribute is never defined at class level, so a `is None` check would
    # pass no matter what plan() did.
    assert not hasattr(wrapper, "_aiter_so_path")
    out = wrapper.run(q, kv_data)

    monkeypatch.undo()
    ref_wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", use_tensor_cores=False
    )
    _plan_decode(ref_wrapper, args)
    torch.testing.assert_close(out, ref_wrapper.run(q, kv_data), rtol=1e-3, atol=1e-3)


def test_decode_replan_demotes_instead_of_raising(device, monkeypatch):
    """A wrapper that planned once on AITER must still demote on a later plan().

    plan() overwrites _backend with the concrete choice, so deriving "did the
    caller ask for auto" from _backend makes every plan() after the first take
    the explicit path and raise. The probe key varies per plan (max_context_len
    picks the AITER variant), so this is reachable, not theoretical.
    """
    _skip_if_op_gated(device, "batch_decode")
    args = _decode_inputs(device)
    q, kv_data, workspace = args[0], args[1], args[5]

    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto", use_tensor_cores=False
    )
    _plan_decode(wrapper, args)
    first_backend = wrapper._backend

    # Only now does AITER become unable to build; re-plan must not propagate it.
    _clear_probes()
    monkeypatch.setattr(
        flashinfer.rocm.decode, "_aiter_pa_v1_resolve", _raiser(_INSTALL_FAILURE)
    )
    _plan_decode(wrapper, args)

    assert wrapper._backend == "fa2", (
        f"re-plan kept backend {wrapper._backend!r} after the AITER build failed "
        f"(first plan chose {first_backend!r})"
    )
    wrapper.run(q, kv_data)


def test_decode_explicit_aiter_still_raises(device, monkeypatch):
    _skip_if_op_gated(device, "batch_decode")
    args = _decode_inputs(device)
    workspace = args[5]
    monkeypatch.setattr(
        flashinfer.rocm.decode, "_aiter_pa_v1_resolve", _raiser(_INSTALL_FAILURE)
    )
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter", use_tensor_cores=False
    )
    with pytest.raises(ModuleNotFoundError):
        _plan_decode(wrapper, args)


def test_decode_strict_env_raises(device, monkeypatch):
    _skip_if_op_gated(device, "batch_decode")
    args = _decode_inputs(device)
    workspace = args[5]
    monkeypatch.setenv("FLASHINFER_AITER_STRICT", "1")
    monkeypatch.setattr(
        flashinfer.rocm.decode, "_aiter_pa_v1_resolve", _raiser(_INSTALL_FAILURE)
    )
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto", use_tensor_cores=False
    )
    with pytest.raises(ModuleNotFoundError):
        _plan_decode(wrapper, args)
