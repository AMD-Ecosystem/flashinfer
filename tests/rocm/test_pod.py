# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.attention import gen_pod_module

DEVICE = "cuda:0"
HEAD_DIM = 128
NUM_KV_HEADS = 8
PAGE_SIZE = 16
KV_LAYOUT = "NHD"
POS_ENCODING_MODE = "NONE"
WORKSPACE_BYTES = 32 * 1024 * 1024


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
            gen_pod_module(
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
def workspace():
    return torch.empty(WORKSPACE_BYTES, device=DEVICE, dtype=torch.int8)


@pytest.fixture(scope="module")
def decode_workspace():
    return torch.empty(WORKSPACE_BYTES, device=DEVICE, dtype=torch.int8)


def _make_paged_kv(batch_size, kv_len, page_size, num_kv_heads, head_dim, dtype):
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = torch.randn(
        total_num_pages,
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
    kv_indices = torch.arange(0, total_num_pages, device=DEVICE, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        device=DEVICE,
        dtype=torch.int32,
    )
    return kv_data, kv_indptr, kv_indices, kv_last_page_len


@pytest.mark.parametrize("kv_len_p", [127, 4096])
@pytest.mark.parametrize("qo_len_p", [127, 512])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch_size_d", [1, 17])
@pytest.mark.parametrize("kv_len_d", [127, 2048])
@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("q_dtype", [torch.float16])
def test_pod_with_paged_kv_cache(
    kv_len_p,
    qo_len_p,
    causal,
    batch_size_d,
    kv_len_d,
    num_qo_heads,
    q_dtype,
    workspace,
    decode_workspace,
):
    if causal and qo_len_p > kv_len_p:
        pytest.skip("Causal prefill with qo_len_p > kv_len_p is not supported")

    kv_dtype = q_dtype

    q_p = torch.randn(qo_len_p, num_qo_heads, HEAD_DIM, device=DEVICE, dtype=q_dtype)
    k_p = torch.randn(kv_len_p, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=kv_dtype)
    v_p = torch.randn(kv_len_p, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=kv_dtype)

    o_ref_p = flashinfer.single_prefill_with_kv_cache(
        q_p,
        k_p,
        v_p,
        causal=causal,
        pos_encoding_mode=POS_ENCODING_MODE,
    )

    q_d = torch.randn(
        batch_size_d, num_qo_heads, HEAD_DIM, device=DEVICE, dtype=q_dtype
    )
    kv_data, kv_indptr_d, kv_indices_d, kv_last_page_len = _make_paged_kv(
        batch_size_d, kv_len_d, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, kv_dtype
    )

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        decode_workspace, KV_LAYOUT
    )
    decode_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        pos_encoding_mode=POS_ENCODING_MODE,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_d = decode_wrapper.run(q_d, kv_data)

    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(workspace, KV_LAYOUT)
    pod_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        pos_encoding_mode=POS_ENCODING_MODE,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )

    o_p, o_d = pod_wrapper.run(
        q_p,
        k_p,
        v_p,
        q_d,
        kv_data,
        pos_encoding_mode_p=POS_ENCODING_MODE,
        causal_p=causal,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="Prefill output mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="Decode output mismatch"
    )


def test_pod_bf16(workspace, decode_workspace):
    dtype = torch.bfloat16
    qo_len_p, kv_len_p = 64, 128
    batch_size_d, kv_len_d = 4, 256
    num_qo_heads = 8

    q_p = torch.randn(qo_len_p, num_qo_heads, HEAD_DIM, device=DEVICE, dtype=dtype)
    k_p = torch.randn(kv_len_p, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)
    v_p = torch.randn(kv_len_p, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=dtype)

    # backend="fa2" avoids routing through AITER, whose bf16 .so requires a separate JIT build.
    o_ref_p = flashinfer.single_prefill_with_kv_cache(
        q_p,
        k_p,
        v_p,
        causal=False,
        pos_encoding_mode=POS_ENCODING_MODE,
        backend="fa2",
    )

    q_d = torch.randn(batch_size_d, num_qo_heads, HEAD_DIM, device=DEVICE, dtype=dtype)
    kv_data, kv_indptr_d, kv_indices_d, kv_last_page_len = _make_paged_kv(
        batch_size_d, kv_len_d, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype
    )

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        decode_workspace, KV_LAYOUT
    )
    decode_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        pos_encoding_mode=POS_ENCODING_MODE,
        data_type=dtype,
        q_data_type=dtype,
    )
    o_ref_d = decode_wrapper.run(q_d, kv_data)

    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(workspace, KV_LAYOUT)
    pod_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        pos_encoding_mode=POS_ENCODING_MODE,
        data_type=dtype,
        q_data_type=dtype,
    )
    o_p, o_d = pod_wrapper.run(
        q_p,
        k_p,
        v_p,
        q_d,
        kv_data,
        pos_encoding_mode_p=POS_ENCODING_MODE,
        causal_p=False,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="bf16 prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="bf16 decode mismatch"
    )
