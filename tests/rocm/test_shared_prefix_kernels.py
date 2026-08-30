# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Ported from tests/attention/test_shared_prefix_kernels.py

import pytest
import torch
from jit_utils import gen_decode_attention_modules, gen_prefill_attention_modules

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],
            [torch.float16],
            [128, 256],
            [0],
            [False],
            [False],
        )
        + gen_prefill_attention_modules(
            [torch.float16],
            [torch.float16],
            [128, 256],
            [0],
            [False],
            [False],
            [False],
        ),
        verbose=False,
    )
    yield


def ceil_div(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("stage", ["decode", "append"])
@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("unique_kv_len", [37, 17])
@pytest.mark.parametrize("shared_kv_len", [128, 512, 2048])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_attention_with_shared_prefix_paged_kv_cache(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    causal,
    head_dim,
    page_size,
):
    if stage == "decode" and causal:
        pytest.skip("Causal attention is not required in decode stage")
    assert shared_kv_len % page_size == 0
    kv_layout = "NHD"
    if stage == "append":
        q = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()
        q_indptr = torch.arange(0, batch_size + 1).to(0).int() * unique_kv_len
    else:
        q = torch.randn(batch_size, num_heads, head_dim).to(0).half()
        q_indptr = torch.arange(0, batch_size + 1).to(0).int()
    k_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    v_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    k_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()
    v_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()

    kv_data = (
        torch.zeros(
            ceil_div(shared_kv_len, page_size)
            + batch_size * ceil_div(unique_kv_len, page_size),
            2,
            page_size,
            num_heads,
            head_dim,
        )
        .to(0)
        .half()
    )
    shared_kv_indices = torch.arange(0, ceil_div(shared_kv_len, page_size)).to(0).int()
    shared_append_indptr = torch.arange(0, 2).to(0).int() * shared_kv_len
    shared_kv_indptr = torch.arange(0, 2).to(0).int() * ceil_div(
        shared_kv_len, page_size
    )
    shared_last_page_len = torch.full(
        (1,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_shared,
        v_shared,
        *flashinfer.get_batch_indices_positions(
            shared_append_indptr,
            flashinfer.get_seq_lens(shared_kv_indptr, shared_last_page_len, page_size),
            k_shared.shape[0],
        ),
        kv_data,
        shared_kv_indices,
        shared_kv_indptr,
        shared_last_page_len,
        kv_layout,
    )
    unique_kv_indices = torch.arange(
        0, batch_size * ceil_div(unique_kv_len, page_size)
    ).to(0).int() + ceil_div(shared_kv_len, page_size)
    unique_append_indptr = torch.arange(0, batch_size + 1).to(0).int() * unique_kv_len
    unique_kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * ceil_div(
        unique_kv_len, page_size
    )
    unique_last_page_len = torch.full(
        (batch_size,), (unique_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_unique,
        v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            k_unique.shape[0],
        ),
        kv_data,
        unique_kv_indices,
        unique_kv_indptr,
        unique_last_page_len,
        kv_layout,
    )

    multi_level_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
    )
    if stage == "decode":
        shared_prefix_decode_wrapper = (
            flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
            )
        )
    else:
        shared_prefix_prefill_wrapper = (
            flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
            )
        )

    qo_indptr_top = torch.tensor([0, q.shape[0]], dtype=torch.int32).to(0)
    if stage == "decode":
        qo_indptr_bottom = torch.arange(0, batch_size + 1, dtype=torch.int32).to(0)
        multi_level_wrapper.plan(
            [qo_indptr_top, qo_indptr_bottom],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_multi_level = multi_level_wrapper.run(q, kv_data)
    else:
        qo_indptr_bottom = (
            torch.arange(0, batch_size + 1, dtype=torch.int32).to(0) * unique_kv_len
        )
        multi_level_wrapper.plan(
            [qo_indptr_top, qo_indptr_bottom],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_heads,
            num_heads,
            head_dim,
            page_size,
            causal=causal,
        )
        o_multi_level = multi_level_wrapper.run(q, kv_data)

    if stage == "decode":
        shared_prefix_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_decode_wrapper.forward(
            q, k_shared, v_shared, kv_data
        )
    else:
        shared_prefix_prefill_wrapper.begin_forward(
            q_indptr,
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_prefill_wrapper.forward(
            q, k_shared, v_shared, kv_data, causal=causal
        )

    torch.testing.assert_close(o_multi_level, o_two_level, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("stage", ["decode", "append"])
@pytest.mark.parametrize("batch_size", [4, 12])
@pytest.mark.parametrize("unique_kv_len", [17, 37])
@pytest.mark.parametrize("shared_kv_len", [128, 512, 2048])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("page_size", [16])
def test_multilevel_cascade_fused_vs_unfused(
    dtype,
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    head_dim,
    page_size,
):
    """Fused epilogue path must agree with unfused merge_state_in_place path."""
    import flashinfer.cascade as cascade_mod

    torch.manual_seed(42)
    atol = 5e-3 if dtype == torch.float16 else 1e-2
    assert shared_kv_len % page_size == 0
    kv_layout = "NHD"

    if stage == "append":
        q = torch.randn(
            batch_size * unique_kv_len, num_heads, head_dim, dtype=dtype, device="cuda"
        )
    else:
        q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    k_shared = torch.randn(
        shared_kv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v_shared = torch.randn(
        shared_kv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    k_unique = torch.randn(
        batch_size * unique_kv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    v_unique = torch.randn(
        batch_size * unique_kv_len, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    kv_data = torch.zeros(
        ceil_div(shared_kv_len, page_size)
        + batch_size * ceil_div(unique_kv_len, page_size),
        2,
        page_size,
        num_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    shared_kv_indices = torch.arange(
        0, ceil_div(shared_kv_len, page_size), dtype=torch.int32, device="cuda"
    )
    shared_append_indptr = (
        torch.arange(0, 2, dtype=torch.int32, device="cuda") * shared_kv_len
    )
    shared_kv_indptr = torch.arange(0, 2, dtype=torch.int32, device="cuda") * ceil_div(
        shared_kv_len, page_size
    )
    shared_last_page_len = torch.full(
        (1,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda"
    )
    flashinfer.append_paged_kv_cache(
        k_shared,
        v_shared,
        *flashinfer.get_batch_indices_positions(
            shared_append_indptr,
            flashinfer.get_seq_lens(shared_kv_indptr, shared_last_page_len, page_size),
            k_shared.shape[0],
        ),
        kv_data,
        shared_kv_indices,
        shared_kv_indptr,
        shared_last_page_len,
        kv_layout,
    )

    unique_kv_indices = torch.arange(
        0,
        batch_size * ceil_div(unique_kv_len, page_size),
        dtype=torch.int32,
        device="cuda",
    ) + ceil_div(shared_kv_len, page_size)
    unique_append_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
        * unique_kv_len
    )
    unique_kv_indptr = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device="cuda"
    ) * ceil_div(unique_kv_len, page_size)
    unique_last_page_len = torch.full(
        (batch_size,),
        (unique_kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    flashinfer.append_paged_kv_cache(
        k_unique,
        v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            k_unique.shape[0],
        ),
        kv_data,
        unique_kv_indices,
        unique_kv_indptr,
        unique_last_page_len,
        kv_layout,
    )

    def make_wrapper_and_plan():
        w = flashinfer.MultiLevelCascadeAttentionWrapper(
            2,
            torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda"),
            kv_layout,
        )
        qo_indptr_top = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda")
        if stage == "decode":
            qo_indptr_bottom = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device="cuda"
            )
        else:
            qo_indptr_bottom = (
                torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
                * unique_kv_len
            )
        w.plan(
            [qo_indptr_top, qo_indptr_bottom],
            [shared_kv_indptr, unique_kv_indptr],
            [shared_kv_indices, unique_kv_indices],
            [shared_last_page_len, unique_last_page_len],
            num_heads,
            num_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
        )
        return w

    # Unfused reference
    orig_flag = cascade_mod._HIP_FUSED_CASCADE
    cascade_mod._HIP_FUSED_CASCADE = False
    try:
        w_unfused = make_wrapper_and_plan()
        out_unfused = w_unfused.run(q, kv_data)
    finally:
        cascade_mod._HIP_FUSED_CASCADE = orig_flag

    # Fused path
    cascade_mod._HIP_FUSED_CASCADE = True
    try:
        w_fused = make_wrapper_and_plan()
        out_fused = w_fused.run(q, kv_data)
    finally:
        cascade_mod._HIP_FUSED_CASCADE = orig_flag

    torch.testing.assert_close(
        out_fused.float(), out_unfused.float(), rtol=1e-3, atol=atol
    )


if __name__ == "__main__":
    test_batch_attention_with_shared_prefix_paged_kv_cache(
        "decode", 12, 37, 128, 8, False, 128, 16
    )
    test_batch_attention_with_shared_prefix_paged_kv_cache(
        "append", 12, 37, 128, 8, False, 128, 16
    )
    test_multilevel_cascade_fused_vs_unfused(
        torch.float16, "append", 4, 17, 128, 8, 128, 16
    )
