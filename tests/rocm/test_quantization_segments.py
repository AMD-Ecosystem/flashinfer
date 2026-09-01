# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Segmented bit packing, and the JIT workspace path that keys the cache by arch.

``segment_packbits`` packs each segment independently and rebuilds the indptr,
so it is not the same operation as packing the concatenation -- the padding
lands per segment. The reference below packs each segment on its own, which is
the property that actually has to hold.
"""

import pathlib

import pytest
import torch

from flashinfer import quantization
from flashinfer.jit.rocm import env as rocm_env

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm device"
)


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.mark.parametrize("bitorder", ["big", "little"])
def test_each_segment_is_packed_independently(device, bitorder):
    lengths = [1, 7, 8, 9, 17]
    x = torch.randint(0, 2, (sum(lengths),), dtype=torch.uint8, device=device)
    indptr = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device
    )

    packed, indptr_new = quantization.segment_packbits(x, indptr, bitorder)

    expected = torch.cat(
        [
            quantization.packbits(x[indptr[i] : indptr[i + 1]], bitorder)
            for i in range(len(lengths))
        ]
    )
    assert torch.equal(packed, expected)
    assert indptr_new.tolist() == [0, 1, 2, 3, 5, 8]


def test_the_new_indptr_is_in_packed_units_not_input_units(device):
    """Every segment rounds up to a whole byte, so the two indptrs diverge."""
    x = torch.ones(24, dtype=torch.uint8, device=device)
    indptr = torch.tensor([0, 9, 24], dtype=torch.int32, device=device)

    packed, indptr_new = quantization.segment_packbits(x, indptr, "big")

    assert indptr_new.tolist() == [0, 2, 4]
    assert packed.numel() == 4


def test_the_fake_op_matches_the_real_shape(device):
    """torch.compile traces the fake; a wrong shape here surfaces only there."""
    x = torch.randint(0, 2, (17,), dtype=torch.uint8, device=device)

    # v0.6.18 made quantization a package; the fake stayed private to packbits
    # and is no longer re-exported from the package root.
    from flashinfer.quantization.packbits import _fake_packbits

    assert _fake_packbits(x, "big").shape == (3,)
    assert quantization.packbits(x, "big").shape == (3,)


class TestWorkspaceDir:
    """The arch is part of the cache path because build.ninja is only written
    when absent, so one shared directory would serve two ISAs the same objects."""

    def test_a_live_device_keys_the_path_by_its_arch(self, tmp_path):
        path = rocm_env.get_workspace_dir(tmp_path)
        assert path.name.startswith("gfx")

    def test_no_visible_device_falls_back_to_the_configured_arch_list(
        self, tmp_path, monkeypatch
    ):
        """Importing flashinfer must not require a live GPU (#316), but the arch
        still has to appear in the path."""
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx950,gfx942")

        path = rocm_env.get_workspace_dir(tmp_path)

        assert path.name == "gfx942_gfx950"

    def test_an_unsupported_live_device_raises_rather_than_degrading(
        self, tmp_path, monkeypatch
    ):
        """A wrong-arch cache hit would be silent; this must not become noarch."""

        class _Props:
            gcnArchName = "gfx90a"

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda idx: _Props())

        with pytest.raises(RuntimeError, match="unsupported ROCm architecture"):
            rocm_env.get_workspace_dir(tmp_path)

    def test_an_unreadable_probe_degrades_to_noarch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            torch.cuda,
            "device_count",
            lambda: (_ for _ in ()).throw(ValueError("no runtime")),
        )

        assert rocm_env.get_workspace_dir(tmp_path).name == "noarch"

    def test_the_returned_path_is_under_the_given_cache_dir(self, tmp_path):
        assert tmp_path in pathlib.Path(rocm_env.get_workspace_dir(tmp_path)).parents
