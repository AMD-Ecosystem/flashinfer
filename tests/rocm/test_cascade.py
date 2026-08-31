# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flashinfer


def merge_state_ref(v_a, s_a, v_b, s_b):
    """Float32 reference: merge two attention states (s values are log2-based LSE)."""
    s_a = s_a.float()
    s_b = s_b.float()
    v_a = v_a.float()
    v_b = v_b.float()
    # s values are logsumexp in base 2: s = log2(sum(2^scores))
    m = torch.maximum(s_a, s_b)
    scale_a = torch.exp2(s_a - m)  # 2^(s_a - m)
    scale_b = torch.exp2(s_b - m)  # 2^(s_b - m)
    denom = scale_a + scale_b
    v_merged = (
        v_a * scale_a.unsqueeze(-1) + v_b * scale_b.unsqueeze(-1)
    ) / denom.unsqueeze(-1)
    s_merged = m + torch.log2(denom)  # log2(2^s_a + 2^s_b)
    return v_merged, s_merged


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 64, 512])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_merge_state(dtype, seq_len, num_heads, head_dim):
    torch.manual_seed(42)
    atol = 5e-3 if dtype == torch.float16 else 1e-2

    v_a = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    s_a = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")
    v_b = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    s_b = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")

    v_ref, s_ref = merge_state_ref(v_a, s_a, v_b, s_b)
    v_merged, s_merged = flashinfer.merge_state(v_a, s_a, v_b, s_b)

    torch.testing.assert_close(v_merged.float(), v_ref, rtol=1e-3, atol=atol)
    torch.testing.assert_close(s_merged, s_ref, rtol=1e-3, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("seq_len", [1, 64, 512])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_merge_state_in_place(dtype, seq_len, num_heads, head_dim):
    torch.manual_seed(42)
    atol = 5e-3 if dtype == torch.float16 else 1e-2

    v_a = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    s_a = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")
    v_b = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    s_b = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")

    v_ref, s_ref = merge_state_ref(v_a, s_a, v_b, s_b)
    v_a_copy = v_a.clone()
    s_a_copy = s_a.clone()
    flashinfer.merge_state_in_place(v_a_copy, s_a_copy, v_b, s_b)

    torch.testing.assert_close(v_a_copy.float(), v_ref, rtol=1e-3, atol=atol)
    torch.testing.assert_close(s_a_copy, s_ref, rtol=1e-3, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_index_sets", [1, 4, 16, 64, 256])
@pytest.mark.parametrize("seq_len", [1, 32])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_merge_states(dtype, num_index_sets, seq_len, num_heads, head_dim):
    torch.manual_seed(42)
    atol = 5e-3 if dtype == torch.float16 else 1e-2

    v = torch.randn(
        seq_len, num_index_sets, num_heads, head_dim, dtype=dtype, device="cuda"
    )
    s = torch.randn(
        seq_len, num_index_sets, num_heads, dtype=torch.float32, device="cuda"
    )

    # Reference: merge all states iteratively in float32 to avoid accumulation error
    v_ref = v[:, 0, :, :].float()
    s_ref = s[:, 0, :]
    for i in range(1, num_index_sets):
        v_ref, s_ref = merge_state_ref(v_ref, s_ref, v[:, i, :, :].float(), s[:, i, :])

    v_merged, s_merged = flashinfer.merge_states(v, s)

    torch.testing.assert_close(v_merged.float(), v_ref.float(), rtol=1e-3, atol=atol)
    torch.testing.assert_close(s_merged, s_ref, rtol=1e-3, atol=atol)


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("num_tries", [20])
def test_merge_state_in_place_with_mask(seed, num_tries):
    seq_len = 512
    num_heads = 8
    head_dim = 128
    va = torch.randn(seq_len, num_heads, head_dim).half().to("cuda")
    sa = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda")
    vb = torch.randn(seq_len, num_heads, head_dim).half().to("cuda")
    sb = torch.randn(seq_len, num_heads, dtype=torch.float32).to("cuda")
    va_original = va.clone()
    sa_original = sa.clone()

    # No mask
    flashinfer.merge_state_in_place(va, sa, vb, sb)
    va_merged_ref = va.clone()
    sa_merged_ref = sa.clone()
    assert not torch.allclose(va_merged_ref, va_original)
    assert not torch.allclose(sa_merged_ref, sa_original)

    # Mask all-ones: identical to no mask
    mask = torch.ones(seq_len, dtype=torch.bool, device="cuda")
    va2 = va_original.clone()
    sa2 = sa_original.clone()
    flashinfer.merge_state_in_place(va2, sa2, vb, sb, mask=mask)
    torch.testing.assert_close(va2, va_merged_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(sa2, sa_merged_ref, rtol=1e-3, atol=1e-3)

    # Mask all-zeros: output unchanged
    mask = torch.zeros(seq_len, dtype=torch.bool, device="cuda")
    va2 = va_original.clone()
    sa2 = sa_original.clone()
    flashinfer.merge_state_in_place(va2, sa2, vb, sb, mask=mask)
    torch.testing.assert_close(va2, va_original, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(sa2, sa_original, rtol=1e-3, atol=1e-3)

    # Random masks
    randgen = torch.Generator(device="cuda")
    randgen.manual_seed(seed)
    for _ in range(num_tries):
        rand_mask = (
            torch.rand(seq_len, generator=randgen, dtype=torch.float32, device="cuda")
            > 0.5
        ).to(dtype=torch.bool)
        true_indices = rand_mask.nonzero()
        false_indices = (rand_mask == 0).nonzero()
        va2 = va_original.clone()
        sa2 = sa_original.clone()
        flashinfer.merge_state_in_place(va2, sa2, vb, sb, mask=rand_mask)
        torch.testing.assert_close(
            va2[false_indices], va_original[false_indices], rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            sa2[false_indices], sa_original[false_indices], rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            va2[true_indices], va_merged_ref[true_indices], rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            sa2[true_indices], sa_merged_ref[true_indices], rtol=1e-3, atol=1e-3
        )


def _ceil_div(a, b):
    return (a + b - 1) // b


def _build_cascade_fixture(
    batch_size, num_heads, head_dim, shared_kv_len, unique_kv_len, dtype
):
    """Build paged-KV fixture for a two-level cascade (shared + unique KV)."""
    page_size = 16
    assert shared_kv_len % page_size == 0
    kv_layout = "NHD"

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
        _ceil_div(shared_kv_len, page_size)
        + batch_size * _ceil_div(unique_kv_len, page_size),
        2,
        page_size,
        num_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )

    shared_kv_indices = torch.arange(
        0, _ceil_div(shared_kv_len, page_size), dtype=torch.int32, device="cuda"
    )
    shared_append_indptr = (
        torch.arange(0, 2, dtype=torch.int32, device="cuda") * shared_kv_len
    )
    shared_kv_indptr = torch.arange(0, 2, dtype=torch.int32, device="cuda") * _ceil_div(
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
        batch_size * _ceil_div(unique_kv_len, page_size),
        dtype=torch.int32,
        device="cuda",
    ) + _ceil_div(shared_kv_len, page_size)
    unique_append_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
        * unique_kv_len
    )
    unique_kv_indptr = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device="cuda"
    ) * _ceil_div(unique_kv_len, page_size)
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

    return (
        q,
        kv_data,
        kv_layout,
        shared_kv_indptr,
        shared_kv_indices,
        shared_last_page_len,
        unique_kv_indptr,
        unique_kv_indices,
        unique_last_page_len,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_heads", [4, 16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_cascade_epilogue(dtype, batch_size, num_heads, head_dim):
    """Fused in-kernel cascade merge must agree with standalone merge_state_in_place path."""
    import flashinfer.cascade as cascade_mod

    torch.manual_seed(42)
    atol = 5e-3 if dtype == torch.float16 else 1e-2
    shared_kv_len = 128
    unique_kv_len = 17
    page_size = 16

    (
        q,
        kv_data,
        kv_layout,
        shared_kv_indptr,
        shared_kv_indices,
        shared_last_page_len,
        unique_kv_indptr,
        unique_kv_indices,
        unique_last_page_len,
    ) = _build_cascade_fixture(
        batch_size, num_heads, head_dim, shared_kv_len, unique_kv_len, dtype
    )

    def make_wrapper_and_plan():
        w = flashinfer.MultiLevelCascadeAttentionWrapper(
            2, torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda"), kv_layout
        )
        qo_indptr_top = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda")
        qo_indptr_bottom = torch.arange(
            0, batch_size + 1, dtype=torch.int32, device="cuda"
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

    orig_flag = cascade_mod._HIP_FUSED_CASCADE

    cascade_mod._HIP_FUSED_CASCADE = False
    try:
        w_unfused = make_wrapper_and_plan()
        out_unfused = w_unfused.run(q, kv_data)
    finally:
        cascade_mod._HIP_FUSED_CASCADE = orig_flag

    cascade_mod._HIP_FUSED_CASCADE = True
    try:
        w_fused = make_wrapper_and_plan()
        out_fused = w_fused.run(q, kv_data)
    finally:
        cascade_mod._HIP_FUSED_CASCADE = orig_flag

    torch.testing.assert_close(
        out_fused.float(), out_unfused.float(), rtol=1e-3, atol=atol
    )


def _segment_sum_ref(v, counts, seq_len):
    """Float32 reference: sum each row's contiguous run of index sets."""
    seg = torch.repeat_interleave(
        torch.arange(seq_len, device=v.device), counts.to(v.device)
    )
    out = torch.zeros(
        seq_len, v.size(1), v.size(2), dtype=torch.float32, device=v.device
    )
    return out.index_add_(0, seg, v.float())


def _sum_tolerance(dtype):
    # threadblock_sum rounds each partial to the storage dtype before the
    # cross-warp reduction, so the reference cannot be matched exactly.
    return (1e-3, 1e-3) if dtype == torch.float16 else (8e-3, 8e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_variable_length_attention_sum_multi_iteration(dtype, head_dim):
    """Persistent sum kernel, sized so each CTA runs several loop iterations.

    head_dim is parametrized because it sets bdy (8/4/8), the width of the
    cross-warp reduction that a stale accumulator is amplified through.
    """
    import flashinfer.cascade as cascade_mod

    torch.manual_seed(42)
    num_heads = 4
    # Grid is num_sms * min(occupancy, ...), and occupancy approaches the
    # 32-blocks/CU wave limit for this small kernel; oversubscribe well past it
    # or every CTA runs a single iteration and the carry goes uncovered.
    num_cus = torch.cuda.get_device_properties(0).multi_processor_count
    seq_len = num_cus * 32 * 4 // num_heads

    # Upper bound exceeds bdy (8 at head_dim 64/256, 4 at 128) so every head_dim
    # also runs the inner smem-staging loop more than once.
    counts = torch.randint(2, 11, (seq_len,), dtype=torch.int32)
    # An empty and a single-set row adjacent to accumulating ones, so an
    # early-out iteration is followed by one that accumulates.
    counts[0] = 0
    counts[1] = 1
    counts[2] = 10
    indptr = torch.zeros(seq_len + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(counts, dim=0)
    nnz = int(indptr[-1])

    v = (torch.randn(nnz, num_heads, head_dim, device="cuda") * 0.1).to(dtype)
    v_sum = torch.empty(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    cascade_mod.get_cascade_module().variable_length_attention_sum(
        v, indptr.cuda(), v_sum
    )

    assert torch.isfinite(v_sum.float()).all(), "kernel produced a non-finite value"

    ref = _segment_sum_ref(v, counts, seq_len)
    rtol, atol = _sum_tolerance(dtype)
    torch.testing.assert_close(v_sum.float(), ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_index_sets", [2, 7])
def test_attention_sum(dtype, num_index_sets):
    """Non-persistent sum kernel (the fixed-count op)."""
    import flashinfer.cascade as cascade_mod

    torch.manual_seed(42)
    seq_len, num_heads, head_dim = 256, 4, 128

    v = (
        torch.randn(seq_len, num_index_sets, num_heads, head_dim, device="cuda") * 0.1
    ).to(dtype)
    v_sum = torch.empty(seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    cascade_mod.get_cascade_module().attention_sum(v, v_sum, num_index_sets)

    assert torch.isfinite(v_sum.float()).all()
    ref = v.float().sum(dim=1)
    rtol, atol = _sum_tolerance(dtype)
    torch.testing.assert_close(v_sum.float(), ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_merge_state(torch.float16, 64, 8, 128)
    test_merge_state_in_place(torch.float16, 64, 8, 128)
    test_merge_states(torch.float16, 16, 32, 8, 128)
    test_merge_state_in_place_with_mask(0, 20)
    test_fused_cascade_epilogue(torch.float16, 4, 16, 128)
    test_fused_cascade_epilogue(torch.bfloat16, 4, 16, 128)
    test_variable_length_attention_sum_multi_iteration(torch.float16, 128)
    test_attention_sum(torch.float16, 7)
