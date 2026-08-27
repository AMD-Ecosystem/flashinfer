# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dispatch and argument-validation branches in prefill_rocm and decode_rocm.

The kernel paths are covered by the per-op suites. What is exercised here is
everything that runs *before* a kernel: which backend ``auto`` picks and why it
declined the other one, the AITER version gate, and the constructor checks that
turn a malformed cudagraph setup into a message rather than a later crash.

A GPU is needed only to construct tensors and read the device architecture; no
attention kernel is launched.
"""

import logging

import pytest
import torch

from flashinfer import decode_rocm, prefill_rocm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm device"
)


@pytest.fixture
def device():
    return torch.device("cuda:0")


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def count(self, fragment):
        return sum(fragment in m for m in self.messages)


@pytest.fixture
def records():
    """Attach directly to flashinfer's logger.

    caplog cannot see these: the package installs its own handler and clears
    propagate, so nothing reaches the root logger pytest hooks.
    """
    logger = prefill_rocm.logger
    handler = _Records()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


class TestAiterNativePageSizes:
    """The page-size hint is version-gated; a wrong answer silently changes
    which page sizes are attempted without a flat-gather."""

    @pytest.fixture(autouse=True)
    def _uncached(self):
        prefill_rocm._aiter_native_page_sizes.cache_clear()
        yield
        prefill_rocm._aiter_native_page_sizes.cache_clear()

    def _with_version(self, monkeypatch, value):
        import importlib.metadata

        def _version(name):
            if name == "amd-aiter":
                if isinstance(value, Exception):
                    raise value
                return value
            return importlib.metadata.version(name)

        monkeypatch.setattr(importlib.metadata, "version", _version)

    @pytest.mark.parametrize("installed", ["0.1.10", "0.1.11", "1.0.0"])
    def test_at_or_after_the_native_paging_release(self, monkeypatch, installed):
        self._with_version(monkeypatch, installed)
        assert prefill_rocm._aiter_native_page_sizes() == frozenset({128, 256, 1024})

    def test_before_the_native_paging_release(self, monkeypatch):
        self._with_version(monkeypatch, "0.1.9")
        assert prefill_rocm._aiter_native_page_sizes() == frozenset({16, 1024})

    def test_newer_than_the_validated_pin_keeps_the_wide_set(self, monkeypatch):
        """Being past the pin is speculative, not disqualifying: the set stays
        wide and plan() probes for real support."""
        self._with_version(monkeypatch, "9.9.9")

        assert prefill_rocm._aiter_native_page_sizes() == frozenset({128, 256, 1024})

    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param("not-a-version", id="unparsable"),
            pytest.param(
                prefill_rocm.PackageNotFoundError("amd-aiter"), id="not-installed"
            ),
        ],
    )
    def test_an_unreadable_version_falls_back_to_the_narrow_set(
        self, monkeypatch, failure
    ):
        self._with_version(monkeypatch, failure)
        assert prefill_rocm._aiter_native_page_sizes() == frozenset({16, 1024})


class TestMakeHashableCache:
    def test_list_arguments_are_tupled_so_the_call_can_be_cached(self):
        calls = []

        @prefill_rocm.make_hashable_cache
        def f(a, b=None):
            calls.append((a, b))
            return len(calls)

        assert f([1, 2], b=[3]) == 1
        assert f([1, 2], b=[3]) == 1  # served from cache, not re-run
        assert calls == [((1, 2), (3,))]

    def test_non_list_arguments_are_passed_through_unchanged(self):
        @prefill_rocm.make_hashable_cache
        def f(a, b=None):
            return (a, b)

        assert f("x", b=7) == ("x", 7)

    def test_distinct_arguments_are_distinct_cache_entries(self):
        calls = []

        @prefill_rocm.make_hashable_cache
        def f(a):
            calls.append(a)
            return len(calls)

        assert (f([1]), f([2]), f([1])) == (1, 2, 1)
        assert calls == [(1,), (2,)]

    def test_the_wrapped_function_keeps_its_identity(self):
        def original(a):
            """docstring"""

        wrapped = prefill_rocm.make_hashable_cache(original)

        assert wrapped.__name__ == "original"
        assert wrapped.__doc__ == "docstring"


def _auto(device, **overrides):
    kwargs = dict(
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        kv_layout="NHD",
        has_custom_mask=False,
        head_dim_qk=128,
        head_dim_vo=128,
        pos_encoding_mode="NONE",
    )
    kwargs.update(overrides)
    return prefill_rocm._auto_select_prefill_backend(device, **kwargs)


class TestAutoBackendSelection:
    """Every decline must carry a reason: `auto` silently choosing fa2 is how a
    user loses the AITER path with nothing to explain it."""

    @pytest.fixture(autouse=True)
    def _unwarned(self):
        prefill_rocm._aiter_auto_warned.clear()
        yield
        prefill_rocm._aiter_auto_warned.clear()

    def test_a_conforming_call_selects_aiter_with_no_reason(self, device):
        backend, reason = _auto(device)
        if backend == "fa2":
            pytest.skip(f"AITER unavailable here: {reason}")
        assert (backend, reason) == ("aiter", None)

    @pytest.mark.parametrize(
        "overrides, fragment",
        [
            ({"kv_layout": "HND"}, "kv_layout"),
            ({"has_custom_mask": True}, "custom mask"),
            (
                {"dtype_q": torch.float32, "dtype_kv": torch.float32},
                "AITER requires fp16/bf16",
            ),
            ({"dtype_kv": torch.bfloat16}, "!= dtype_kv"),
            ({"head_dim_vo": 64}, "head_dim_qk=128 != head_dim_vo=64"),
            ({"pos_encoding_mode": "ROPE_LLAMA"}, "pos_encoding_mode"),
        ],
    )
    def test_each_unmet_constraint_falls_back_and_names_itself(
        self, device, overrides, fragment
    ):
        backend, reason = _auto(device, **overrides)

        assert backend == "fa2"
        assert fragment in reason

    def test_the_warning_fires_once_per_device_and_reason(self, device, records):
        first = _auto(device, kv_layout="HND")
        second = _auto(device, kv_layout="HND")

        assert first == second
        assert records.count("auto backend falling back to fa2") == 1

    def test_a_second_distinct_reason_warns_again(self, device, records):
        _auto(device, kv_layout="HND")
        _auto(device, has_custom_mask=True)

        assert records.count("auto backend falling back to fa2") == 2

    def test_the_reason_is_returned_on_every_call_not_only_the_first(self, device):
        """The log carries it once; a caller that needs it each time reads the
        return value."""
        _, first = _auto(device, kv_layout="HND")
        _, second = _auto(device, kv_layout="HND")

        assert first == second is not None

    def test_a_missing_aiter_package_is_its_own_reason(self, device, monkeypatch):
        prefill_rocm._aiter_ops_importable.cache_clear()
        monkeypatch.setattr(prefill_rocm, "_aiter_ops_importable", lambda: False)

        backend, reason = _auto(device)

        assert backend == "fa2"
        assert "aiter package not installed" in reason


class TestRequireAiterRuntime:
    def test_a_missing_package_names_the_install_command(self, device, monkeypatch):
        monkeypatch.setattr(prefill_rocm, "_aiter_ops_importable", lambda: False)

        with pytest.raises(ImportError, match="github.com/ROCm/aiter"):
            prefill_rocm._require_aiter_runtime(device)


@pytest.fixture
def workspace(device):
    return torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)


def _wrapper(workspace, **kwargs):
    return decode_rocm.BatchDecodeWithPagedKVCacheWrapper(workspace, **kwargs)


class TestCudagraphBufferValidation:
    """A batch size baked in at construction time; a malformed buffer set here
    otherwise surfaces much later as a shape error inside a captured graph."""

    @pytest.fixture
    def buffers(self, device):
        return {
            "paged_kv_indptr_buffer": torch.zeros(5, dtype=torch.int32, device=device),
            "paged_kv_indices_buffer": torch.zeros(
                64, dtype=torch.int32, device=device
            ),
            "paged_kv_last_page_len_buffer": torch.zeros(
                4, dtype=torch.int32, device=device
            ),
        }

    def test_a_well_formed_set_fixes_the_batch_size(self, workspace, buffers):
        wrapper = _wrapper(workspace, use_cuda_graph=True, **buffers)
        assert wrapper._fixed_batch_size == 4

    @pytest.mark.parametrize(
        "missing",
        [
            "paged_kv_indptr_buffer",
            "paged_kv_indices_buffer",
            "paged_kv_last_page_len_buffer",
        ],
    )
    def test_each_buffer_is_required_and_named(self, workspace, buffers, missing):
        buffers[missing] = None

        with pytest.raises(ValueError, match=missing):
            _wrapper(workspace, use_cuda_graph=True, **buffers)

    def test_indptr_must_be_one_longer_than_the_batch(self, workspace, buffers):
        buffers["paged_kv_indptr_buffer"] = torch.zeros(
            4, dtype=torch.int32, device=buffers["paged_kv_indptr_buffer"].device
        )

        with pytest.raises(ValueError, match="batch_size \\+ 1"):
            _wrapper(workspace, use_cuda_graph=True, **buffers)

    def test_without_cudagraph_the_batch_size_is_not_fixed(self, workspace):
        assert _wrapper(workspace)._fixed_batch_size == 0


class TestWrapperSurface:
    def test_backend_is_auto_before_plan(self, workspace):
        wrapper = _wrapper(workspace)

        assert wrapper.backend == "auto"
        assert wrapper.backend_fallback_reason is None

    def test_an_explicit_backend_is_reported_verbatim(self, workspace):
        assert _wrapper(workspace, backend="fa2").backend == "fa2"

    def test_an_unknown_kv_layout_is_rejected(self, workspace):
        with pytest.raises(KeyError, match="Invalid kv_layout"):
            _wrapper(workspace, kv_layout="XYZ")

    def test_reset_workspace_buffer_rebuilds_the_pinned_mirror(self, workspace, device):
        wrapper = _wrapper(workspace)
        new_float = torch.empty(1024, dtype=torch.uint8, device=device)
        new_int = torch.empty(2048, dtype=torch.uint8, device=device)

        wrapper.reset_workspace_buffer(new_float, new_int)

        assert wrapper._float_workspace_buffer is new_float
        assert wrapper._int_workspace_buffer is new_int
        # The pinned copy is reallocated to match, not left at the old size.
        assert wrapper._pin_memory_int_workspace_buffer.shape == new_int.shape
        assert wrapper._pin_memory_int_workspace_buffer.device.type == "cpu"
