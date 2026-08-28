# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BatchPrefillWithPagedKVCacheWrapper with bfloat16 and custom masks.

These tests target two bugs that were found in the HIP FA2 kernels on CDNA3:

  Bug 1 (bf16 rowsum): m16k16_rowsum_f16f16f32() used the fp16 MFMA intrinsic
  unconditionally, misinterpreting bf16(1.0) as fp16(1.875) and inflating the
  softmax denominator.  Caught by test_batch_prefill_paged_kv_bf16_correctness
  which compares BatchPrefill bf16 output against a naive PyTorch reference.

  Bug 2 (custom mask MFMA row-mapping): logits_transform() and logits_mask()
  used an interleaved packed-index pattern instead of contiguous bands, causing
  the wrong GQA head to be masked for 3/4 of packed query rows.  Caught by
  test_batch_prefill_custom_mask_gqa_correctness which uses a custom mask with
  group_size > 1 and compares against a naive PyTorch reference, and by
  test_custom_mask_all_true_matches_causal which checks that an all-True custom
  mask produces the same output as the causal (kNone) code path.
"""

import math

import pytest
import torch

import flashinfer
from attention_reference import _hipblas_safe_matmul


def _plan_with_custom_mask(
    wrapper,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    custom_mask_flat,
):
    """Call wrapper.plan() with custom mask, adapting to the FlashInfer version.

    v0.2.5 uses ``custom_mask=`` (raw bool tensor, internally calls
    segment_packbits).  v0.5.3+ uses ``packed_custom_mask=`` +
    ``qk_indptr=`` (pre-packed uint8).  This helper tries the newer API
    first and falls back to the older one.
    """
    import inspect

    sig = inspect.signature(wrapper.plan)

    if "qk_indptr" in sig.parameters:
        packed = _packbits(custom_mask_flat, bitorder="little")
        batch_size = qo_indptr.shape[0] - 1
        qo_lens = qo_indptr[1:] - qo_indptr[:-1]
        kv_lens_per_seq = []
        pages_per = kv_indptr[1:] - kv_indptr[:-1]
        for i in range(batch_size):
            kv_lens_per_seq.append(
                (int(pages_per[i].item()) - 1) * page_size
                + int(kv_last_page_len[i].item())
            )
        qk_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=qo_indptr.device
        )
        for i in range(batch_size):
            qk_indptr[i + 1] = (
                qk_indptr[i] + int(qo_lens[i].item()) * kv_lens_per_seq[i]
            )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=False,
            q_data_type=dtype,
            kv_data_type=dtype,
            packed_custom_mask=packed,
            qk_indptr=qk_indptr,
        )
    else:
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            custom_mask=custom_mask_flat,
            q_data_type=dtype,
            kv_data_type=dtype,
        )


def _packbits(mask_flat, bitorder="little"):
    return flashinfer.quantization.packbits(mask_flat, bitorder=bitorder)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _naive_attention(q, k, v, causal=False, custom_mask=None):
    """Pure-PyTorch multi-head attention reference (supports GQA)."""
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape
    group_size = num_qo_heads // num_kv_heads

    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    scale = 1.0 / math.sqrt(head_dim)
    q_t = q.transpose(0, 1).float()
    k_t = k.transpose(0, 1).float()
    v_t = v.transpose(0, 1).float()

    scores = _hipblas_safe_matmul(q_t, k_t.transpose(1, 2)) * scale

    if custom_mask is not None:
        scores = scores.masked_fill(~custom_mask.unsqueeze(0), float("-inf"))
    elif causal:
        mask = torch.tril(
            torch.ones(qo_len, kv_len, device=q.device, dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = _hipblas_safe_matmul(attn, v_t)
    return out.transpose(0, 1).to(q.dtype)


def _build_paged_kv(batch_size, kv_len, page_size, num_kv_heads, head_dim, dtype):
    """Build a paged KV cache with random data and sequential page indices."""
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = num_pages_per_seq * batch_size
    k_cache = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, device="cuda:0", dtype=dtype
    )
    v_cache = torch.randn(
        total_pages, page_size, num_kv_heads, head_dim, device="cuda:0", dtype=dtype
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda:0")
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda:0",
    )
    return k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len


def _gather_kv_from_pages(
    k_cache, v_cache, kv_indptr, kv_last_page_len, batch_idx, page_size
):
    """Gather contiguous K/V for one sequence from paged cache."""
    start_page = kv_indptr[batch_idx].item()
    end_page = kv_indptr[batch_idx + 1].item()
    last_len = kv_last_page_len[batch_idx].item()
    k_parts = [k_cache[p] for p in range(start_page, end_page - 1)]
    k_parts.append(k_cache[end_page - 1][:last_len])
    v_parts = [v_cache[p] for p in range(start_page, end_page - 1)]
    v_parts.append(v_cache[end_page - 1][:last_len])
    return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


# ---------------------------------------------------------------------------
# Bug 1: bf16 BatchPrefill correctness (rowsum MMA type confusion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("kv_len", [128, 512])
@pytest.mark.parametrize("qo_len", [1, 24])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_prefill_paged_kv_bf16_correctness(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported with causal")

    dtype = torch.bfloat16
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=dtype
    )
    k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len = _build_paged_kv(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, dtype
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda:0") * qo_len
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o = wrapper.run(q, (k_cache, v_cache))

    for i in range(batch_size):
        qi = q[qo_indptr[i] : qo_indptr[i + 1]]
        ki, vi = _gather_kv_from_pages(
            k_cache, v_cache, kv_indptr, kv_last_page_len, i, page_size
        )
        o_ref = _naive_attention(qi, ki, vi, causal=True)
        o_i = o[qo_indptr[i] : qo_indptr[i + 1]]
        torch.testing.assert_close(o_i.float(), o_ref.float(), rtol=1e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Bug 2: custom mask with GQA (MFMA C/D row-mapping error)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("kv_len", [128, 512])
@pytest.mark.parametrize("qo_len", [8, 24])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 16, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
def test_batch_prefill_custom_mask_gqa_correctness(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported with causal")

    dtype = torch.bfloat16
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=dtype
    )
    k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len = _build_paged_kv(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, dtype
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda:0") * qo_len
    )

    # Build per-sequence causal custom masks
    custom_masks = []
    for _ in range(batch_size):
        mask = torch.tril(
            torch.ones(qo_len, kv_len, device="cuda:0", dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        custom_masks.append(mask)
    flat_mask = torch.cat([m.flatten() for m in custom_masks])

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    _plan_with_custom_mask(
        wrapper,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        flat_mask,
    )
    o = wrapper.run(q, (k_cache, v_cache))

    for i in range(batch_size):
        qi = q[qo_indptr[i] : qo_indptr[i + 1]]
        ki, vi = _gather_kv_from_pages(
            k_cache, v_cache, kv_indptr, kv_last_page_len, i, page_size
        )
        o_ref = _naive_attention(qi, ki, vi, custom_mask=custom_masks[i])
        o_i = o[qo_indptr[i] : qo_indptr[i + 1]]
        torch.testing.assert_close(o_i.float(), o_ref.float(), rtol=1e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Bug 2 (supplementary): all-True custom mask must match causal (kNone) path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 16, 32])
@pytest.mark.parametrize("head_dim", [128])
def test_custom_mask_all_true_matches_causal(
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")

    batch_size = 2
    kv_len = 256
    qo_len = 16
    page_size = 16
    dtype = torch.bfloat16

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=dtype
    )
    k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len = _build_paged_kv(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, dtype
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda:0") * qo_len
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

    # Run with causal=True (kNone path)
    wrapper_causal = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD"
    )
    wrapper_causal.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_causal = wrapper_causal.run(q, (k_cache, v_cache))

    # Run with custom mask = causal mask (kCustom path)
    custom_masks = []
    for _ in range(batch_size):
        mask = torch.tril(
            torch.ones(qo_len, kv_len, device="cuda:0", dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        custom_masks.append(mask)
    flat_mask = torch.cat([m.flatten() for m in custom_masks])

    wrapper_custom = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD"
    )
    _plan_with_custom_mask(
        wrapper_custom,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        flat_mask,
    )
    o_custom = wrapper_custom.run(q, (k_cache, v_cache))

    torch.testing.assert_close(o_custom.float(), o_causal.float(), rtol=1e-2, atol=5e-2)
