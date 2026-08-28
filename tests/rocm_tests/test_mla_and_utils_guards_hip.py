# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-copy detection and argument guards in mla_rocm, plus two utils helpers.

``_combined_kv_view`` decides whether MLA can hand AITER a view or has to copy
the whole page pool on every call, so each rejection is a performance cliff
that nothing else reports. The guards around it turn a malformed plan into a
message rather than a kernel-side fault.
"""

import os

import pytest
import torch

from flashinfer import mla_rocm, utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a ROCm device"
)


@pytest.fixture
def device():
    return torch.device("cuda:0")


def _adjacent(device, dtype=torch.float16, pages=4, page_size=8, ckv=512, kpe=64):
    """One allocation split into the two halves AITER expects."""
    buf = torch.zeros(pages, page_size, ckv + kpe, dtype=dtype, device=device)
    return torch.split(buf, [ckv, kpe], dim=-1)


class TestCombinedKvView:
    def test_adjacent_halves_of_one_buffer_give_a_zero_copy_view(self, device):
        ckv, kpe = _adjacent(device)

        view = mla_rocm._combined_kv_view(ckv, kpe)

        assert view is not None
        assert view.untyped_storage().data_ptr() == ckv.untyped_storage().data_ptr()

    def test_separate_allocations_are_rejected(self, device):
        ckv = torch.zeros(4, 8, 512, dtype=torch.float16, device=device)
        kpe = torch.zeros(4, 8, 64, dtype=torch.float16, device=device)

        assert mla_rocm._combined_kv_view(ckv, kpe) is None

    def test_mismatched_dtypes_are_rejected(self, device):
        # Reinterpreted, not reallocated: storage stays shared so the dtype
        # check is the only one that can reject.
        ckv, kpe = _adjacent(device)

        assert mla_rocm._combined_kv_view(ckv, kpe.view(torch.bfloat16)) is None

    def test_a_wrong_rank_is_rejected(self, device):
        ckv, kpe = _adjacent(device)

        assert mla_rocm._combined_kv_view(ckv.reshape(-1), kpe) is None
        assert mla_rocm._combined_kv_view(ckv, kpe.reshape(-1)) is None

    def test_mismatched_page_geometry_is_rejected(self, device):
        ckv, kpe = _adjacent(device)
        # Shares storage so the data_ptr check cannot be what rejects it. The
        # shape guard still is not isolated -- removing it, the view assembly
        # below rejects instead -- so this pins the outcome, not the guard.
        regrouped = torch.as_strided(
            kpe, (2, 16, 64), (kpe.stride(0) * 2, kpe.stride(1), kpe.stride(2))
        )

        assert (
            regrouped.untyped_storage().data_ptr() == ckv.untyped_storage().data_ptr()
        )
        assert mla_rocm._combined_kv_view(ckv, regrouped) is None

    def test_tensors_on_different_devices_are_rejected(self, device):
        """Cannot be isolated the way the others are -- two devices cannot share
        storage, so the data_ptr check would reject this even without the
        device check. Kept because the pair is reachable from a caller."""
        ckv, _ = _adjacent(device)
        kpe = torch.zeros(4, 8, 64, dtype=torch.float16, device="cpu")

        assert mla_rocm._combined_kv_view(ckv, kpe) is None


class TestLastPageLenConversion:
    def test_a_batch_entry_with_no_pages_is_rejected_by_index(self):
        """Zero pages for a request is degenerate rather than empty work, and
        the message has to say which entry so a caller can find it."""
        kv_indptr = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
        kv_lens = torch.tensor([16, 0, 40], dtype=torch.int32)

        with pytest.raises(ValueError, match="no pages at batch idx 1"):
            mla_rocm._kv_lens_to_last_page_len_cpu(kv_indptr, kv_lens, page_size=8)

    def test_a_well_formed_plan_converts(self):
        kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
        kv_lens = torch.tensor([16, 12], dtype=torch.int32)

        out = mla_rocm._kv_lens_to_last_page_len_cpu(kv_indptr, kv_lens, page_size=8)

        assert out.numel() == 2


class TestRequireAiterMla:
    def test_a_missing_package_names_the_install_command(self, device, monkeypatch):
        monkeypatch.setattr(
            mla_rocm,
            "_aiter_mla",
            lambda: (_ for _ in ()).throw(ImportError("no aiter")),
        )
        monkeypatch.setattr(mla_rocm, "require_capability", lambda *a: None)

        with pytest.raises(ImportError, match="github.com/ROCm/aiter"):
            mla_rocm._require_aiter_mla(device)


class TestUtilsHelpers:
    def test_a_plan_info_sequence_becomes_a_cpu_int64_tensor(self):
        """The ROCm C++ ops read plan_info_vec on the host, so it must land on
        CPU whatever the workspace device is."""
        out = utils.plan_info_vec_as_tensor([1, 2, 3], device=torch.device("cuda:0"))

        assert out.device.type == "cpu"
        assert out.dtype == torch.int64
        assert out.tolist() == [1, 2, 3]

    def test_an_already_correct_tensor_is_passed_through(self):
        given = torch.tensor([1, 2], dtype=torch.int64, device="cpu")
        assert utils.plan_info_vec_as_tensor(given) is given

    def test_a_wrong_dtype_or_device_is_converted(self, device):
        out = utils.plan_info_vec_as_tensor(
            torch.tensor([1, 2], dtype=torch.int32, device=device)
        )

        assert out.device.type == "cpu"
        assert out.dtype == torch.int64

    def test_custom_op_mode_tracks_the_environment(self):
        expected = os.environ.get("FLASHINFER_USE_TORCH_CUSTOM_OPS", "0") == "1"
        assert utils.use_torch_custom_ops_enabled() is expected
