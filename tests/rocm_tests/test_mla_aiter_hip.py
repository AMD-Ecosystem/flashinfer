# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Correctness tests for BatchMLAPagedAttentionWrapper on ROCm (AITER backend).
# Reference: pure-PyTorch paged MLA attention.

import math

import pytest
import torch

from tests.test_helpers.test_helpers import requires_aiter


def _paged_mla_ref(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Pure-PyTorch paged MLA decode reference.

    q_nope: [batch, num_heads, head_dim_ckv]
    q_pe:   [batch, num_heads, head_dim_kpe]
    ckv_cache: [num_pages, page_size, head_dim_ckv]
    kpe_cache: [num_pages, page_size, head_dim_kpe]
    """
    batch = q_nope.shape[0]

    out = torch.zeros_like(q_nope)
    for b in range(batch):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        last_len = int(kv_last_page_len[b].item())

        keys_nope = []
        keys_pe = []
        for p_idx in range(start, end - 1):
            pg = int(kv_indices[p_idx].item())
            keys_nope.append(ckv_cache[pg])  # [page_size, ckv]
            keys_pe.append(kpe_cache[pg])
        if start < end:
            pg = int(kv_indices[end - 1].item())
            keys_nope.append(ckv_cache[pg, :last_len])
            keys_pe.append(kpe_cache[pg, :last_len])

        k_nope = torch.cat(keys_nope, dim=0).float()  # [kv_len, ckv]
        k_pe = torch.cat(keys_pe, dim=0).float()  # [kv_len, kpe]
        k = torch.cat([k_nope, k_pe], dim=-1)  # [kv_len, ckv+kpe]

        q = torch.cat(
            [q_nope[b].float(), q_pe[b].float()], dim=-1
        )  # [num_heads, ckv+kpe]

        # [num_heads, kv_len]
        scores = torch.einsum("hd,ld->hl", q, k) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        # [num_heads, head_dim_ckv]  —  value = k_nope (matrix-absorbed)
        out[b] = torch.einsum("hl,lc->hc", attn, k_nope).to(out.dtype)
    return out


def _build_paged_kv(
    batch_size: int,
    kv_lens: list,
    page_size: int,
    head_dim_ckv: int,
    head_dim_kpe: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
):
    """Build consistent paged KV caches and indptr/indices/last_page_len tensors."""
    torch.manual_seed(seed)
    num_pages_per = [(L + page_size - 1) // page_size for L in kv_lens]
    total_pages = sum(num_pages_per) + 4  # slack pages

    ckv_cache = (
        torch.randn(total_pages, page_size, head_dim_ckv, dtype=dtype, device=device)
        * 0.1
    )
    kpe_cache = (
        torch.randn(total_pages, page_size, head_dim_kpe, dtype=dtype, device=device)
        * 0.1
    )

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, n in enumerate(num_pages_per):
        kv_indptr[i + 1] = kv_indptr[i] + n

    kv_indices = torch.arange(sum(num_pages_per), dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [L - (n - 1) * page_size for L, n in zip(kv_lens, num_pages_per, strict=True)],
        dtype=torch.int32,
        device=device,
    )
    return ckv_cache, kpe_cache, kv_indptr, kv_indices, kv_last_page_len


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("num_heads,head_dim_ckv,head_dim_kpe", [(16, 512, 64)])
@pytest.mark.parametrize(
    "kv_lens",
    [
        [1],
        [64],
        [127],
        [1, 64, 127, 32],
    ],
)
def test_mla_decode_vs_ref(
    dtype, page_size, num_heads, head_dim_ckv, head_dim_kpe, kv_lens
):
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    batch_size = len(kv_lens)
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    ckv_cache, kpe_cache, kv_indptr, kv_indices, kv_last_page_len = _build_paged_kv(
        batch_size, kv_lens, page_size, head_dim_ckv, head_dim_kpe, dtype, device
    )
    kv_len_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    torch.manual_seed(42)
    q_nope = (
        torch.randn(batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device)
        * 0.1
    )
    q_pe = (
        torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
        * 0.1
    )

    # decode: one token per request → qo_indptr = [0,1,2,...,batch]
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)

    ws = torch.empty(1, dtype=torch.float32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(ws, backend="aiter")
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_tensor,
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # q_nope/q_pe for AITER: [total_q, num_heads, head_dim] where total_q == batch_size for decode
    got = wrapper.run(
        q_nope=q_nope.view(batch_size, num_heads, head_dim_ckv),
        q_pe=q_pe.view(batch_size, num_heads, head_dim_kpe),
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
    )

    ref = _paged_mla_ref(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        sm_scale,
    )

    # MLA decode: fp16 atol ~5e-2 due to large head_dim softmax, bf16 slightly wider
    rtol, atol = (1e-1, 1e-1) if dtype == torch.bfloat16 else (5e-2, 5e-2)
    torch.testing.assert_close(got.float(), ref.float(), rtol=rtol, atol=atol)


@requires_aiter
def test_mla_decode_out_tensor():
    """run() respects a pre-allocated out= tensor."""
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper
    import math

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    batch_size, num_heads, head_dim_ckv, head_dim_kpe, page_size = 2, 16, 512, 64, 1
    kv_lens = [32, 64]
    sm_scale = 1.0 / math.sqrt(head_dim_ckv + head_dim_kpe)

    ckv_cache, kpe_cache, kv_indptr, kv_indices, kv_last_page_len = _build_paged_kv(
        batch_size, kv_lens, page_size, head_dim_ckv, head_dim_kpe, dtype, device
    )
    kv_len_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    q_nope = (
        torch.randn(batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device)
        * 0.1
    )
    q_pe = (
        torch.randn(batch_size, num_heads, head_dim_kpe, dtype=dtype, device=device)
        * 0.1
    )
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)

    ws = torch.empty(1, dtype=torch.float32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(ws)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_tensor,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal=False,
        sm_scale=sm_scale,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    out = torch.zeros(batch_size, num_heads, head_dim_ckv, dtype=dtype, device=device)
    ret = wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, out=out)
    assert ret.data_ptr() == out.data_ptr()
    assert not torch.all(out == 0)


@requires_aiter
def test_mla_plan_validation():
    """plan() raises on invalid arguments."""
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    ws = torch.empty(1, dtype=torch.float32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(ws)

    base = dict(
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
        kv_indices=torch.tensor([0], dtype=torch.int32, device=device),
        kv_len_arr=torch.tensor([8], dtype=torch.int32, device=device),
        num_heads=16,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=16,
        causal=False,
        sm_scale=0.1,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )

    with pytest.raises(ValueError, match="use_profiler"):
        wrapper.plan(**{**base, "use_profiler": True})

    with pytest.raises(ValueError, match="fp16|bf16"):
        wrapper.plan(
            **{**base, "q_data_type": torch.float32, "kv_data_type": torch.float32}
        )

    with pytest.raises(ValueError, match="q_data_type == kv_data_type"):
        wrapper.plan(
            **{**base, "q_data_type": torch.float16, "kv_data_type": torch.bfloat16}
        )


@requires_aiter
def test_mla_plan_kv_len_inconsistent_with_paging():
    """Passing last-page counts as kv_len_arr must fail (was accepted pre-conversion)."""
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    ws = torch.empty(1, dtype=torch.float32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(ws)
    # One full page (16) + one partial last page: true kv_len=24, last_page_len=8.
    # Passing 8 as if it were total length is inconsistent with num_pages=2.
    with pytest.raises(ValueError, match="inconsistent with paging"):
        wrapper.plan(
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
            kv_indptr=torch.tensor([0, 2], dtype=torch.int32, device=device),
            kv_indices=torch.tensor([0, 1], dtype=torch.int32, device=device),
            kv_len_arr=torch.tensor([8], dtype=torch.int32, device=device),
            num_heads=16,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=16,
            causal=False,
            sm_scale=0.1,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )


@requires_aiter
def test_mla_run_before_plan_raises():
    """run() before plan() raises RuntimeError."""
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    ws = torch.empty(1, dtype=torch.float32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(ws)
    with pytest.raises(RuntimeError, match="plan\\(\\)"):
        wrapper.run(
            torch.zeros(1, 16, 512, dtype=torch.float16, device=device),
            torch.zeros(1, 16, 64, dtype=torch.float16, device=device),
            torch.zeros(4, 16, 512, dtype=torch.float16, device=device),
            torch.zeros(4, 16, 64, dtype=torch.float16, device=device),
        )


@requires_aiter
@pytest.mark.parametrize("backend", ["auto", "aiter"])
def test_mla_backend_accepts_auto_and_aiter(backend):
    """The ROCm MLA wrapper accepts both 'auto' (default) and 'aiter'.

    'auto' resolves to 'aiter' since there is no HIP MLA kernel.
    """
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    ws = torch.empty(1, dtype=torch.float32, device=device)
    BatchMLAPagedAttentionWrapper(ws, backend=backend)


def test_mla_backend_rejects_unsupported():
    """Any backend other than 'auto'/'aiter' raises ValueError.

    The check fires before the AITER-availability probe, so this test
    runs on any host (no GPU / no AITER required).
    """
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    ws = torch.empty(1, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="aiter.*auto"):
        BatchMLAPagedAttentionWrapper(ws, backend="fa2")
