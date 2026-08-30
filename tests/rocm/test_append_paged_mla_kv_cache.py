# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Dtype coverage for the MLA paged-cache append on ROCm.
#
# AppendPagedKVMlaCache derives its access width from the dtype
# (vec_size = 16 / sizeof(DType)), so each dtype takes a different vec_t
# specialisation and a different block width. tests/attention/test_mla_page.py
# is upstream and pins float16, which leaves three of the four dispatched
# dtypes unexercised.

import math

import pytest
import torch

import flashinfer
from flashinfer.device_utils import IS_HIP

CKV_DIM = 512
KPE_DIM = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_HIP, reason="requires a ROCm GPU"
)

# vec_size = 16 / itemsize, bdx = 512 / vec_size. fp8 lands on 32, half a
# wavefront; 16-bit dtypes on 64, one wavefront.
_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
]


def _random(shape, dtype, device):
    """fp8 has no randn; round a small fp32 draw so values survive the cast."""
    x = torch.randn(*shape, device=device, dtype=torch.float32) * 0.1
    return x.to(dtype)


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("kv_len", [[45], [45, 8, 25, 22], [400]])
def test_append_mla_paged_kv_cache_dtypes(dtype, page_size, kv_len):
    """Appended values must land bit-exact, and padding must stay zero.

    The append is a pure copy, so `torch.equal` on the bit pattern is the right
    assertion -- a tolerance would hide a partial or misaligned vector write,
    which is the failure mode a dtype-dependent vec_size introduces.
    """
    dev = torch.device("cuda:0")
    nnz = sum(kv_len)

    ckv_append = _random((nnz, CKV_DIM), dtype, dev)
    kpe_append = _random((nnz, KPE_DIM), dtype, dev)

    num_pages_per_req = torch.tensor(
        [math.ceil(n / page_size) for n in kv_len], dtype=torch.int32, device=dev
    )
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=dev),
            torch.cumsum(torch.tensor(kv_len, dtype=torch.int32, device=dev), 0),
        ]
    ).int()

    max_num_pages = int(num_pages_per_req.sum())
    ckv_cache = torch.zeros(max_num_pages, page_size, CKV_DIM, dtype=dtype, device=dev)
    kpe_cache = torch.zeros(max_num_pages, page_size, KPE_DIM, dtype=dtype, device=dev)
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=dev),
            torch.cumsum(num_pages_per_req, 0),
        ]
    ).int()
    kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device=dev)
    kv_last_page_len = torch.tensor(
        [n % page_size if n % page_size else page_size for n in kv_len],
        dtype=torch.int32,
        device=dev,
    )

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz,
    )
    flashinfer.append_paged_mla_kv_cache(
        ckv_append,
        kpe_append,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )

    # Compare as raw bits: fp8 has no eq kernel on this torch build.
    flat_ckv = ckv_cache.view(-1, CKV_DIM).view(torch.uint8)
    flat_kpe = kpe_cache.view(-1, KPE_DIM).view(torch.uint8)
    src_ckv = ckv_append.view(torch.uint8)
    src_kpe = kpe_append.view(torch.uint8)

    acc, acc_pad = 0, 0
    for i, n in enumerate(kv_len):
        assert torch.equal(src_ckv[acc : acc + n], flat_ckv[acc_pad : acc_pad + n])
        assert torch.equal(src_kpe[acc : acc + n], flat_kpe[acc_pad : acc_pad + n])
        # Tail of the last page must be untouched -- catches an over-wide write.
        end = acc_pad + int(num_pages_per_req[i]) * page_size
        assert not flat_ckv[acc_pad + n : end].any()
        assert not flat_kpe[acc_pad + n : end].any()
        acc += n
        acc_pad = end
