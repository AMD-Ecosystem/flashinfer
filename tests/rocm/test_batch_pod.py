# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.attention import gen_batch_pod_module

DEVICE = "cuda:0"
HEAD_DIM = 128
NUM_KV_HEADS = 8
PAGE_SIZE = 16
KV_LAYOUT = "NHD"
POS_ENCODING_MODE = "NONE"
REF_WORKSPACE_BYTES = 64 * 1024 * 1024
POD_WORKSPACE_BYTES = 128 * 1024 * 1024


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [torch.float16],
            [torch.float16],
            [HEAD_DIM],
            [0],
            [False],
            [False],
            [False],
        )
        + [
            gen_batch_pod_module(
                torch.float16,
                torch.float16,
                torch.float16,
                HEAD_DIM,
                0,
                False,
                False,
                False,
                torch.int32,
                0,
                False,
                False,
            )
        ],
        verbose=False,
    )
    yield


@pytest.fixture(scope="module")
def pod_workspace():
    return torch.empty(POD_WORKSPACE_BYTES, device=DEVICE, dtype=torch.uint8)


@pytest.fixture(scope="module")
def ref_workspace_p():
    return torch.empty(REF_WORKSPACE_BYTES, device=DEVICE, dtype=torch.uint8)


@pytest.fixture(scope="module")
def ref_workspace_d():
    return torch.empty(REF_WORKSPACE_BYTES, device=DEVICE, dtype=torch.uint8)


def _make_paged_kv(batch_size, kv_len, page_size, num_kv_heads, head_dim, dtype):
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = num_pages_per_seq * batch_size
    kv_data = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=dtype,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=DEVICE, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_pages, device=DEVICE, dtype=torch.int32)
    last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        device=DEVICE,
        dtype=torch.int32,
    )
    return kv_data, kv_indptr, kv_indices, last_page_len


def _make_prefill_inputs(batch_size, qo_len, kv_len, num_qo_heads, dtype):
    total_qo = batch_size * qo_len
    q = torch.randn(total_qo, num_qo_heads, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv_data, kv_indptr, kv_indices, last_page_len = _make_paged_kv(
        batch_size, kv_len, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, device=DEVICE, dtype=torch.int32) * qo_len
    )
    return q, kv_data, qo_indptr, kv_indptr, kv_indices, last_page_len


def _make_decode_inputs(batch_size, kv_len, num_qo_heads, dtype):
    q = torch.randn(batch_size, num_qo_heads, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv_data, kv_indptr, kv_indices, last_page_len = _make_paged_kv(
        batch_size, kv_len, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype
    )
    qo_indptr = torch.arange(0, batch_size + 1, device=DEVICE, dtype=torch.int32)
    return q, kv_data, qo_indptr, kv_indptr, kv_indices, last_page_len


def _plan_ref_prefill(
    workspace,
    qo_indptr,
    kv_indptr,
    kv_indices,
    last_page_len,
    num_qo_heads,
    dtype,
    causal,
    backend="auto",
):
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, KV_LAYOUT, backend=backend
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=causal,
        pos_encoding_mode=POS_ENCODING_MODE,
        kv_data_type=dtype,
        q_data_type=dtype,
    )
    return wrapper


def _run_batch_pod(
    workspace,
    qo_indptr_p,
    kv_indptr_p,
    kv_indices_p,
    last_page_len_p,
    qo_indptr_d,
    kv_indptr_d,
    kv_indices_d,
    last_page_len_d,
    q_p,
    kv_data_p,
    q_d,
    kv_data_d,
    num_qo_heads,
    dtype,
    causal_p,
):
    wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(workspace, KV_LAYOUT)
    wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads=num_qo_heads,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode=POS_ENCODING_MODE,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return wrapper.run(q_p, kv_data_p, q_d, kv_data_d, causal_p=causal_p)


@pytest.mark.parametrize("batch_size_p", [2, 4])
@pytest.mark.parametrize("qo_len_p", [64, 256])
@pytest.mark.parametrize("kv_len_p", [128, 512])
@pytest.mark.parametrize("batch_size_d", [4, 16])
@pytest.mark.parametrize("kv_len_d", [256, 1024])
@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("q_dtype", [torch.float16])
def test_batch_pod_with_paged_kv_cache(
    batch_size_p,
    qo_len_p,
    kv_len_p,
    batch_size_d,
    kv_len_d,
    num_qo_heads,
    q_dtype,
    pod_workspace,
    ref_workspace_p,
    ref_workspace_d,
):
    q_p, kv_data_p, qo_indptr_p, kv_indptr_p, kv_indices_p, last_page_len_p = (
        _make_prefill_inputs(batch_size_p, qo_len_p, kv_len_p, num_qo_heads, q_dtype)
    )
    q_d, kv_data_d, qo_indptr_d, kv_indptr_d, kv_indices_d, last_page_len_d = (
        _make_decode_inputs(batch_size_d, kv_len_d, num_qo_heads, q_dtype)
    )

    ref_p = _plan_ref_prefill(
        ref_workspace_p,
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        num_qo_heads,
        q_dtype,
        causal=False,
    )
    o_ref_p = ref_p.run(q_p, kv_data_p)

    ref_d = _plan_ref_prefill(
        ref_workspace_d,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads,
        q_dtype,
        causal=False,
    )
    o_ref_d = ref_d.run(q_d, kv_data_d)

    o_p, o_d = _run_batch_pod(
        pod_workspace,
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        q_p,
        kv_data_p,
        q_d,
        kv_data_d,
        num_qo_heads,
        q_dtype,
        causal_p=False,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="BatchPOD prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="BatchPOD decode mismatch"
    )


def test_batch_pod_bf16(pod_workspace, ref_workspace_p, ref_workspace_d):
    dtype = torch.bfloat16
    batch_size_p, qo_len_p, kv_len_p = 2, 64, 128
    batch_size_d, kv_len_d = 4, 256
    num_qo_heads = 8

    q_p, kv_data_p, qo_indptr_p, kv_indptr_p, kv_indices_p, last_page_len_p = (
        _make_prefill_inputs(batch_size_p, qo_len_p, kv_len_p, num_qo_heads, dtype)
    )
    q_d, kv_data_d, qo_indptr_d, kv_indptr_d, kv_indices_d, last_page_len_d = (
        _make_decode_inputs(batch_size_d, kv_len_d, num_qo_heads, dtype)
    )

    # backend="fa2" avoids routing through AITER, whose bf16 .so requires a separate JIT build.
    ref_p = _plan_ref_prefill(
        ref_workspace_p,
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        num_qo_heads,
        dtype,
        causal=False,
        backend="fa2",
    )
    o_ref_p = ref_p.run(q_p, kv_data_p)

    ref_d = _plan_ref_prefill(
        ref_workspace_d,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads,
        dtype,
        causal=False,
        backend="fa2",
    )
    o_ref_d = ref_d.run(q_d, kv_data_d)

    o_p, o_d = _run_batch_pod(
        pod_workspace,
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        q_p,
        kv_data_p,
        q_d,
        kv_data_d,
        num_qo_heads,
        dtype,
        causal_p=False,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="BatchPOD bf16 prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="BatchPOD bf16 decode mismatch"
    )


def test_batch_pod_causal_p(pod_workspace, ref_workspace_p, ref_workspace_d):
    """Exercises MaskMode::kCausal on the prefill arm of BatchPOD."""
    dtype = torch.float16
    batch_size_p, qo_len_p, kv_len_p = 2, 64, 128
    batch_size_d, kv_len_d = 4, 256
    num_qo_heads = 8

    q_p, kv_data_p, qo_indptr_p, kv_indptr_p, kv_indices_p, last_page_len_p = (
        _make_prefill_inputs(batch_size_p, qo_len_p, kv_len_p, num_qo_heads, dtype)
    )
    q_d, kv_data_d, qo_indptr_d, kv_indptr_d, kv_indices_d, last_page_len_d = (
        _make_decode_inputs(batch_size_d, kv_len_d, num_qo_heads, dtype)
    )

    ref_p = _plan_ref_prefill(
        ref_workspace_p,
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        num_qo_heads,
        dtype,
        causal=True,
    )
    o_ref_p = ref_p.run(q_p, kv_data_p)

    ref_d = _plan_ref_prefill(
        ref_workspace_d,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads,
        dtype,
        causal=False,
    )
    o_ref_d = ref_d.run(q_d, kv_data_d)

    o_p, o_d = _run_batch_pod(
        pod_workspace,
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        q_p,
        kv_data_p,
        q_d,
        kv_data_d,
        num_qo_heads,
        dtype,
        causal_p=True,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="BatchPOD causal prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="BatchPOD causal decode mismatch"
    )
