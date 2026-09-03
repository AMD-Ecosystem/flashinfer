# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Block-sparse attention on ROCm.

Upstream's tests/attention/test_block_sparse.py needs scipy to build the BSR
structure; these build it with torch so they run in the ROCm image.
"""

import math

import pytest
import torch

import flashinfer


def _dense_reference(q, k, v, mask=None):
    """Full attention over (len, heads, dim) tensors, optionally masked."""
    qf, kf, vf = (t.float().transpose(0, 1) for t in (q, k, v))
    s = qf @ kf.transpose(-1, -2) / math.sqrt(q.size(-1))
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    return (torch.softmax(s, dim=-1) @ vf).transpose(0, 1)


def _bsr_from_block_mask(block_mask):
    """(MB, NB) bool -> the (indptr, indices) pair plan() expects."""
    nnz_per_row = block_mask.sum(dim=1)
    indptr = torch.zeros(block_mask.size(0) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(nnz_per_row, 0)
    indices = block_mask.nonzero()[:, 1].to(torch.int32)
    return indptr, indices


@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(4, 4), (8, 2)])
def test_a_fully_dense_block_mask_matches_full_attention(num_qo_heads, num_kv_heads):
    torch.manual_seed(0)
    M = N = 128
    R = C = 16
    head_dim = 128
    device = "cuda:0"

    block_mask = torch.ones(M // R, N // C, dtype=torch.bool)
    indptr, indices = _bsr_from_block_mask(block_mask)

    q = torch.randn(M, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device=device)

    wrapper = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    )
    wrapper.plan(
        indptr.to(device),
        indices.to(device),
        M,
        N,
        R,
        C,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=torch.float16,
    )
    out = wrapper.run(q, k, v)

    k_rep = k.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    v_rep = v.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    torch.testing.assert_close(
        out.float(), _dense_reference(q, k_rep, v_rep), atol=2e-2, rtol=2e-2
    )


def test_a_sparse_block_mask_is_honoured_not_ignored():
    """A dense-only test cannot tell "block-sparse ran" from "the mask was dropped"."""
    torch.manual_seed(0)
    M = N = 128
    R = C = 16
    heads = 4
    head_dim = 128
    device = "cuda:0"

    # Lower-triangular at block granularity: sparse, and every row keeps >=1 block.
    block_mask = torch.tril(torch.ones(M // R, N // C, dtype=torch.bool))
    indptr, indices = _bsr_from_block_mask(block_mask)

    q = torch.randn(M, heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(N, heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(N, heads, head_dim, dtype=torch.float16, device=device)

    wrapper = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    )
    wrapper.plan(
        indptr.to(device),
        indices.to(device),
        M,
        N,
        R,
        C,
        heads,
        heads,
        head_dim,
        q_data_type=torch.float16,
    )
    out = wrapper.run(q, k, v)

    token_mask = block_mask.repeat_interleave(R, 0).repeat_interleave(C, 1).to(device)
    torch.testing.assert_close(
        out.float(),
        _dense_reference(q, k, v, mask=token_mask),
        atol=2e-2,
        rtol=2e-2,
    )
    # And it must not agree with unmasked attention, or the mask did nothing.
    assert not torch.allclose(
        out.float(), _dense_reference(q, k, v), atol=2e-2, rtol=2e-2
    )


def test_variable_block_sizes_are_honoured():
    import einops  # noqa: F401  -- imported inside VariableBlockSparse.run()

    torch.manual_seed(0)
    heads = 1
    head_dim = 128
    seq_len = 6
    device = "cuda:0"

    block_mask_map = torch.tensor(
        [[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], dtype=torch.bool, device=device
    )
    block_row_sz = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=device)
    block_col_sz = torch.tensor([[3, 1, 2]], dtype=torch.int32, device=device)

    wrapper = flashinfer.VariableBlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    )
    wrapper.plan(
        block_mask_map,
        block_row_sz,
        block_col_sz,
        heads,
        heads,
        head_dim,
        q_data_type=torch.float16,
    )

    q = torch.randn(heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(heads, seq_len, head_dim, dtype=torch.float16, device=device)
    out = wrapper.run(q, k, v)

    rows = torch.repeat_interleave(
        torch.arange(3, device=device), block_row_sz[0].long()
    )
    cols = torch.repeat_interleave(
        torch.arange(3, device=device), block_col_sz[0].long()
    )
    token_mask = block_mask_map[0][rows][:, cols]
    s = (q[0].float() @ k[0].float().T) / math.sqrt(head_dim)
    ref = (
        torch.softmax(s.masked_fill(~token_mask, float("-inf")), dim=-1) @ v[0].float()
    )

    torch.testing.assert_close(
        out.float().reshape(seq_len, head_dim), ref, atol=2e-2, rtol=2e-2
    )


def test_convert_bsr_mask_layout_interleaves_blocks_by_row():
    """Two blocks in one row-block, so a dropped transpose changes the answer."""
    mask = torch.tensor(
        [[[1, 0], [1, 1]], [[0, 1], [0, 0]]], dtype=torch.bool, device="cuda:0"
    )
    indptr = torch.tensor([0, 2], dtype=torch.int32, device="cuda:0")

    converted = flashinfer.convert_bsr_mask_layout(mask, indptr)

    expected = torch.tensor([1, 0, 0, 1, 1, 1, 0, 0], dtype=torch.bool, device="cuda:0")
    torch.testing.assert_close(converted, expected)
