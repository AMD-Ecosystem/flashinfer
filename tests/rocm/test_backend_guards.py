# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The per-op AITER routing guards in norm, page, MLA and aiter_utils.

Each ``maybe_*`` helper either runs the AITER path or returns a sentinel meaning
"fall through to native". The refusals in between are what stop an unsupported
shape reaching a shim that would read it as something else; none of them had a
test. All of these reject before the kernel, so the file costs no JIT.
"""

import pytest
import torch

from flashinfer.rocm import aiter_utils, norm, page

_HIDDEN = 128


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("the shims take device tensors")
    return torch.device("cuda:0")


class TestRmsnormRouting:
    def test_aiter_refuses_a_3d_input(self, device):
        """AITER's rmsnorm is 2-D only; native handles the 3-D case."""
        x = torch.randn(2, 4, _HIDDEN, dtype=torch.float16, device=device)
        w = torch.randn(_HIDDEN, dtype=torch.float16, device=device)
        with pytest.raises(ValueError, match="only supports 2D inputs"):
            norm.maybe_rmsnorm(None, x, w, 1e-6, "aiter")

    def test_aiter_refuses_a_weight_on_another_device(self, device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
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
    def test_aiter_refuses_tensors_on_different_devices(self, device):
        x = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
        residual = torch.randn(4, _HIDDEN, dtype=torch.float16, device=device)
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
    def test_aiter_refuses_a_mismatched_indptr_length(self, device):
        """kv_last_page_len never reaches the shim, so this invariant has
        nowhere else to live."""
        args = _append_args(device)
        args["kv_last_page_len"] = torch.tensor(
            [16, 16, 16], dtype=torch.int32, device=device
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
        assert aiter_utils._aiter_installed_version() is None
        assert aiter_utils._aiter_version_supported() is False

    def test_an_unparsable_version_is_not_supported(self, monkeypatch):
        """A local build can carry a version string PEP 440 rejects; treating
        that as new enough would let the vendored struct layouts loose."""
        monkeypatch.setattr(
            aiter_utils, "_aiter_installed_version", lambda: "not-a-version"
        )
        assert aiter_utils._aiter_version_supported() is False

    def test_an_aiter_below_the_floor_says_so(self, device, monkeypatch):
        """The device check runs first, so this needs the real GPU."""
        monkeypatch.setattr(aiter_utils, "_aiter_importable", lambda: False)
        monkeypatch.setattr(aiter_utils, "_aiter_installed_version", lambda: "0.1.10")
        monkeypatch.setattr(aiter_utils, "_aiter_version_supported", lambda: False)

        with pytest.raises(ValueError, match="requires amd-aiter >="):
            aiter_utils.require_aiter(device, "rmsnorm")
