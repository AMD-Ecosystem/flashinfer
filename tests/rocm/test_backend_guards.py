# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The per-op AITER routing guards in norm, page, MLA and aiter_utils.

Each ``maybe_*`` helper either runs the AITER path or returns a sentinel meaning
"fall through to native". The refusals in between are what stop an unsupported
shape reaching a shim that would read it as something else; none of them had a
test. All of these reject before the kernel, so the file costs no JIT.
"""

import sys

import pytest
import torch

from flashinfer.rocm import aiter_utils, norm, page
from flashinfer.rocm.aiter_utils import _aiter_importable, is_aiter_supported

_HIDDEN = 128


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("the shims take device tensors")
    return torch.device("cuda:0")


@pytest.fixture
def aiter_device(device):
    """Every `backend="aiter"` refusal sits behind `require_aiter`.

    Without AITER that raises first, with a different message, so the case
    would fail on a mismatched `match=` rather than testing its own guard.
    Both halves of require_aiter are needed: is_aiter_supported covers only
    HIP and the arch, never whether the package imports.
    """
    if not (is_aiter_supported(device) and _aiter_importable()):
        pytest.skip("require_aiter refuses before the guard under test")
    return device


class TestRmsnormRouting:
    def test_aiter_refuses_a_3d_input(self, aiter_device):
        """AITER's rmsnorm is 2-D only; native handles the 3-D case."""
        x = torch.randn(2, 4, _HIDDEN, dtype=torch.float16, device=aiter_device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device=aiter_device)
        with pytest.raises(ValueError, match="only supports 2D inputs"):
            norm.maybe_rmsnorm(None, x, w, 1e-6, "aiter")

    def test_aiter_refuses_a_weight_on_another_device(self, aiter_device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=aiter_device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device="cpu")
        with pytest.raises(ValueError, match="input and weight on the same device"):
            norm.maybe_rmsnorm(None, x, w, 1e-6, "aiter")

    def test_an_unknown_backend_is_named(self, device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device=device)
        with pytest.raises(ValueError, match="Unknown backend 'cudnn'"):
            norm.maybe_rmsnorm(None, x, w, 1e-6, "cudnn")

    def test_native_falls_through(self, device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device=device)
        assert norm.maybe_rmsnorm(None, x, w, 1e-6, "native") is None


class TestFusedAddRmsnormRouting:
    def test_aiter_refuses_tensors_on_different_devices(self, aiter_device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=aiter_device)
        residual = torch.randn(4, _HIDDEN, dtype=torch.float16, device=aiter_device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device="cpu")
        with pytest.raises(ValueError, match="input, residual and weight on the"):
            norm.maybe_fused_add_rmsnorm(x, residual, w, 1e-6, "aiter")

    def test_an_unknown_backend_is_named(self, device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
        residual = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device=device)
        with pytest.raises(ValueError, match="Unknown backend 'cudnn'"):
            norm.maybe_fused_add_rmsnorm(x, residual, w, 1e-6, "cudnn")

    def test_native_falls_through(self, device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
        residual = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device=device)
        assert norm.maybe_fused_add_rmsnorm(x, residual, w, 1e-6, "native") is False


def _append_args(device, batch=2, pages=4):
    ints = lambda v: torch.tensor(v, dtype=torch.int32, device=device)  # noqa: E731
    kv = torch.randn(pages, 16, 4, 64, dtype=torch.float16, device=device)
    return dict(
        append_key=torch.randn(batch, 4, 64, dtype=torch.float16, device=device),
        append_value=torch.randn(batch, 4, 64, dtype=torch.float16, device=device),
        batch_indices=ints(list(range(batch))),
        positions=ints([0] * batch),
        paged_k_cache=kv,
        paged_v_cache=kv.clone(),
        kv_indices=ints(list(range(pages))),
        kv_indptr=ints([0, 2, 4]),
        kv_last_page_len=ints([16] * batch),
        kv_layout="NHD",
    )


class TestPagedAppendRouting:
    def test_aiter_refuses_a_mismatched_indptr_length(self, aiter_device):
        """kv_last_page_len never reaches the shim, so this invariant has
        nowhere else to live."""
        args = _append_args(aiter_device)
        args["kv_last_page_len"] = torch.tensor(
            [16, 16, 16], dtype=torch.int32, device=aiter_device
        )
        with pytest.raises(ValueError, match="kv_last_page_len.numel\\(\\)\\+1"):
            page.maybe_append_paged_kv_cache(**args, backend="aiter")

    def test_an_unknown_backend_is_named(self, device):
        with pytest.raises(ValueError, match="Unknown backend 'cudnn'"):
            page.maybe_append_paged_kv_cache(**_append_args(device), backend="cudnn")

    def test_native_falls_through(self, device):
        assert (
            page.maybe_append_paged_kv_cache(**_append_args(device), backend="native")
            is None
        )

    def test_the_fake_append_op_is_a_no_op(self, device):
        """The meta arm returns nothing; it exists so tracing sees a signature."""
        args = _append_args(device)
        assert (
            page._fake_aiter_append_paged_kv_cache(
                args["append_key"],
                args["append_value"],
                args["batch_indices"],
                args["positions"],
                args["paged_k_cache"],
                args["paged_v_cache"],
                args["kv_indices"],
                args["kv_indptr"],
            )
            is None
        )


class TestAiterAvailabilityProbes:
    def test_an_unreadable_distribution_reads_as_absent(self, monkeypatch):
        import importlib.metadata as md

        def boom(_name):
            raise md.PackageNotFoundError("amd-aiter")

        monkeypatch.setattr(md, "version", boom)
        monkeypatch.setattr(  # the cached read may predate the patch
            aiter_utils,
            "_aiter_installed_version",
            aiter_utils._aiter_installed_version.__wrapped__,
        )
        assert aiter_utils._aiter_installed_version() is None
        assert aiter_utils._aiter_version_supported() is False

    def test_the_version_is_read_once_per_process(self, monkeypatch):
        """The auto fallback asks on every call, so the metadata is read once."""
        import importlib.metadata as md

        calls = []
        monkeypatch.setattr(md, "version", lambda name: calls.append(name) or "0.1.29")
        aiter_utils._aiter_installed_version.cache_clear()
        try:
            for _ in range(3):
                assert aiter_utils._aiter_installed_version() == "0.1.29"
                assert aiter_utils._aiter_version_supported() is True
            assert calls == ["amd-aiter"]
        finally:
            aiter_utils._aiter_installed_version.cache_clear()

    def test_an_unparsable_version_is_not_supported(self, monkeypatch):
        """A local build can carry a version string PEP 440 rejects; treating
        that as new enough would let the vendored struct layouts loose."""
        monkeypatch.setattr(
            aiter_utils, "_aiter_installed_version", lambda: "not-a-version"
        )
        assert aiter_utils._aiter_version_supported() is False

    def test_an_aiter_below_the_floor_says_so(self, device, monkeypatch):
        """require_capability runs first, so the arch must be one the table
        covers -- otherwise ArchCapabilityError matches a different message.
        Not `aiter_device`: this stubs importability itself, so requiring a real
        AITER install would skip a case that can run without one."""
        if not is_aiter_supported(device):
            pytest.skip("require_capability refuses before the version check")
        monkeypatch.setattr(aiter_utils, "_aiter_importable", lambda: False)
        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: "0.1.10")
        monkeypatch.setattr(aiter_utils, "_aiter_version_supported", lambda: False)

        with pytest.raises(ValueError, match="requires amd-aiter >="):
            aiter_utils.require_aiter(device, "rmsnorm")


class TestAiterImportFailure:
    """An installed amd-aiter that fails to import must say so, never read as absent."""

    @pytest.fixture(autouse=True)
    def _broken_import(self, monkeypatch):
        from flashinfer.rocm import prefill as prefill_rocm

        caches = (
            aiter_utils._aiter_import_error,
            aiter_utils._aiter_importable,
            prefill_rocm._aiter_ops_importable,
        )
        for cache in caches:
            cache.cache_clear()

        # A real failing import, stopped before any aiter code runs: a None
        # package entry makes `import aiter.ops` raise. GPU_ARCHS set keeps the
        # arch resolution off the device.
        monkeypatch.setitem(sys.modules, "aiter", None)
        monkeypatch.setenv("GPU_ARCHS", "gfx942")
        yield
        for cache in caches:
            cache.cache_clear()

    def test_an_installed_package_names_the_import_error(self, monkeypatch):
        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: "0.1.29")

        reason = aiter_utils.aiter_unavailable_reason()

        assert reason is not None
        assert reason.startswith("amd-aiter 0.1.29 is installed but failed to import: ")
        assert "aiter" in reason.split(": ", 1)[1]
        assert "\n" not in reason
        assert aiter_utils._aiter_importable() is False

    def test_an_absent_package_has_no_reason(self, monkeypatch):
        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: None)

        assert aiter_utils.aiter_unavailable_reason() is None
        assert aiter_utils._aiter_importable() is False

    def test_the_abi_floor_outranks_a_failed_import(self, monkeypatch):
        """Every caller reports the same root cause for one install."""
        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: "0.1.20")

        reason = aiter_utils.aiter_unavailable_reason()

        assert reason is not None and "below the 0.1.21 ABI floor" in reason
        # The probes never import a below-floor install, so it logs no second cause.
        assert aiter_utils._aiter_importable() is False
        assert aiter_utils._aiter_import_error.cache_info().currsize == 0

    def test_require_aiter_names_the_import_error(self, device, monkeypatch):
        if not is_aiter_supported(device):
            pytest.skip("require_capability refuses before the import check")
        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: "0.1.29")

        with pytest.raises(ValueError, match="0.1.29 is installed but failed"):
            aiter_utils.require_aiter(device, "rmsnorm")

    def test_require_aiter_runtime_names_the_import_error(self, device, monkeypatch):
        from flashinfer.rocm import prefill as prefill_rocm

        if not is_aiter_supported(device):
            pytest.skip("the capability gate answers before the import check")
        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: "0.1.29")
        monkeypatch.setattr(prefill_rocm, "require_capability", lambda *a, **k: None)

        with pytest.raises(ImportError, match="0.1.29 is installed but failed"):
            prefill_rocm._require_aiter_runtime(device)

    def test_explicit_aiter_decode_refuses_at_construction(self, device, monkeypatch):
        """Before plan() reaches AITER's codegen, which needs `import aiter` first."""
        from flashinfer.rocm import decode as decode_rocm
        from flashinfer.rocm import prefill as prefill_rocm

        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: "0.1.29")
        monkeypatch.setattr(prefill_rocm, "require_capability", lambda *a, **k: None)
        ws = torch.empty(1024, dtype=torch.uint8, device=device)

        with pytest.raises(ImportError, match="0.1.29 is installed but failed"):
            decode_rocm.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", backend="aiter")
