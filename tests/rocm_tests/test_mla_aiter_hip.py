# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Correctness tests for BatchMLAPagedAttentionWrapper on ROCm (AITER backend).
# Reference: pure-PyTorch paged MLA attention.

import math
import warnings

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


def test_combined_kv_view_detects_split_buffer():
    """_combined_kv_view returns a zero-copy view for adjacent halves, else None.

    Pure pointer/stride logic, so CPU tensors exercise it exactly as GPU ones do
    and no AITER install is needed.
    """
    from flashinfer.mla_rocm import _combined_kv_view

    ckv_dim, kpe_dim = 512, 64
    width = ckv_dim + kpe_dim

    # The layout production uses: one buffer, split along the last dim.
    cache = torch.empty(8, 4, width, dtype=torch.bfloat16)
    ckv, kpe = cache.split([ckv_dim, kpe_dim], dim=-1)
    view = _combined_kv_view(ckv, kpe)
    assert view is not None
    assert view.shape == (8, 4, 1, width)
    assert view.is_contiguous()
    # A view, not a copy: same storage, no allocation.
    assert view.data_ptr() == cache.data_ptr()
    assert view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()

    # A layer-indexed cache still yields adjacent halves.
    layered = torch.empty(3, 8, 4, width, dtype=torch.bfloat16)
    lckv, lkpe = layered[1].split([ckv_dim, kpe_dim], dim=-1)
    assert _combined_kv_view(lckv, lkpe) is not None

    # Everything else must fall back rather than alias the wrong memory.
    sep_ckv = torch.empty(8, 4, ckv_dim, dtype=torch.bfloat16)
    sep_kpe = torch.empty(8, 4, kpe_dim, dtype=torch.bfloat16)
    assert _combined_kv_view(sep_ckv, sep_kpe) is None, "separate allocations"
    assert _combined_kv_view(kpe, ckv) is None, "halves in the wrong order"

    padded = torch.empty(8, 4, width + 8, dtype=torch.bfloat16)
    assert (
        _combined_kv_view(padded[..., :ckv_dim], padded[..., ckv_dim:width]) is None
    ), "padding between rows"

    assert _combined_kv_view(ckv, kpe.to(torch.float16)) is None, "mismatched dtypes"


@requires_aiter
@pytest.mark.parametrize("kv_lens", [[64], [1, 64, 127, 32]])
def test_mla_combined_kv_buffer_matches_separate(kv_lens):
    """The zero-copy path and the concatenating fallback agree bitwise.

    Same bytes reach the kernel either way, so this is exact equality, not a
    tolerance check.
    """
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    page_size, num_heads, ckv_dim, kpe_dim = 1, 16, 512, 64
    batch_size = len(kv_lens)
    sm_scale = 1.0 / math.sqrt(ckv_dim + kpe_dim)

    ckv_cache, kpe_cache, kv_indptr, kv_indices, _ = _build_paged_kv(
        batch_size, kv_lens, page_size, ckv_dim, kpe_dim, dtype, device
    )
    kv_len_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # Same values in the combined layout, so any difference is the code path.
    combined = torch.cat([ckv_cache, kpe_cache], dim=-1).contiguous()
    split_ckv, split_kpe = combined.split([ckv_dim, kpe_dim], dim=-1)

    torch.manual_seed(42)
    q_nope = torch.randn(batch_size, num_heads, ckv_dim, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size, num_heads, kpe_dim, dtype=dtype, device=device)
    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)

    results = []
    for ckv, kpe in ((split_ckv, split_kpe), (ckv_cache, kpe_cache)):
        wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_tensor,
            num_heads,
            ckv_dim,
            kpe_dim,
            page_size,
            False,
            sm_scale,
            dtype,
            dtype,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(wrapper.run(q_nope, q_pe, ckv, kpe).clone())

    assert torch.equal(results[0], results[1])
    assert torch.isfinite(results[0]).all()


@requires_aiter
def test_mla_warns_only_for_separate_kv_allocations():
    """Separate ckv/kpe allocations warn; the combined layout stays silent.

    The warning is the only signal a caller gets that every decode step is
    copying the whole page pool, so it must not fire on the good path.
    """
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    page_size, num_heads, ckv_dim, kpe_dim = 1, 16, 512, 64
    kv_lens = [64]
    batch_size = len(kv_lens)

    ckv_cache, kpe_cache, kv_indptr, kv_indices, _ = _build_paged_kv(
        batch_size, kv_lens, page_size, ckv_dim, kpe_dim, dtype, device
    )
    combined = torch.cat([ckv_cache, kpe_cache], dim=-1).contiguous()
    split_ckv, split_kpe = combined.split([ckv_dim, kpe_dim], dim=-1)

    q_nope = torch.randn(batch_size, num_heads, ckv_dim, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size, num_heads, kpe_dim, dtype=dtype, device=device)

    def _run(ckv, kpe, runs=1):
        wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        )
        wrapper.plan(
            torch.arange(batch_size + 1, dtype=torch.int32, device=device),
            kv_indptr,
            kv_indices,
            torch.tensor(kv_lens, dtype=torch.int32, device=device),
            num_heads,
            ckv_dim,
            kpe_dim,
            page_size,
            False,
            1.0 / math.sqrt(ckv_dim + kpe_dim),
            dtype,
            dtype,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(runs):
                wrapper.run(q_nope, q_pe, ckv, kpe)
        return [w for w in caught if issubclass(w.category, UserWarning)]

    assert _run(split_ckv, split_kpe) == [], "combined layout must not warn"

    separate = _run(ckv_cache, kpe_cache)
    assert len(separate) == 1
    assert "adjacent halves" in str(separate[0].message)

    # "always" defeats warnings' own (message, category, module, lineno) dedup,
    # so without the per-wrapper guard this would be one warning per decode step.
    assert len(_run(ckv_cache, kpe_cache, runs=5)) == 1


@requires_aiter
@pytest.mark.parametrize("index_device", ["cuda", "cpu", "mixed"])
def test_mla_plan_accepts_host_and_mixed_device_indices(index_device):
    """plan() batches its index tensors into one host copy; placement must not matter.

    The batched path uses torch.cat, which rejects tensors on different devices,
    so the mixed case exercises the per-tensor fallback.
    """
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    page_size, num_heads, ckv_dim, kpe_dim = 1, 16, 512, 64
    kv_lens = [1, 64, 127, 32]
    batch_size = len(kv_lens)

    ckv_cache, kpe_cache, kv_indptr, kv_indices, _ = _build_paged_kv(
        batch_size, kv_lens, page_size, ckv_dim, kpe_dim, dtype, device
    )
    combined = torch.cat([ckv_cache, kpe_cache], dim=-1).contiguous()
    split_ckv, split_kpe = combined.split([ckv_dim, kpe_dim], dim=-1)

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    kv_len_tensor = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    if index_device == "cpu":
        qo_indptr = qo_indptr.cpu()
        kv_indptr = kv_indptr.cpu()
        kv_len_tensor = kv_len_tensor.cpu()
    elif index_device == "mixed":
        # qo on host, the rest on device -> torch.cat would raise; fallback path.
        qo_indptr = qo_indptr.cpu()
        kv_len_tensor = kv_len_tensor.cpu()

    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_tensor,
        num_heads,
        ckv_dim,
        kpe_dim,
        page_size,
        False,
        1.0 / math.sqrt(ckv_dim + kpe_dim),
        dtype,
        dtype,
    )
    assert wrapper._max_seqlen_q == 1
    assert wrapper._kv_last_page_len.device.type == "cuda"
    assert torch.equal(
        wrapper._kv_last_page_len,
        torch.ones(batch_size, dtype=torch.int32, device=device),
    )

    q_nope = torch.randn(batch_size, num_heads, ckv_dim, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size, num_heads, kpe_dim, dtype=dtype, device=device)
    out = wrapper.run(q_nope, q_pe, split_ckv, split_kpe)
    assert torch.isfinite(out).all()


@requires_aiter
def test_mla_run_rejects_4d_caches_with_actionable_error():
    """4-D ckv/kpe must raise, not warn that an adjacent buffer is non-adjacent.

    An earlier docstring told callers to allocate
    [num_pages, page_size, 1, ckv+kpe] and pass 4-D slices. That shape never
    worked (the fallback unsqueezed it to 5-D, which AITER rejects), so the error
    has to name the 3-D form rather than blame the caller's allocation.
    """
    from flashinfer.mla_rocm import BatchMLAPagedAttentionWrapper

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    num_heads, ckv_dim, kpe_dim, page_size = 16, 512, 64, 1
    kv_lens = [8]
    batch_size = len(kv_lens)

    _, _, kv_indptr, kv_indices, _ = _build_paged_kv(
        batch_size, kv_lens, page_size, ckv_dim, kpe_dim, dtype, device
    )
    num_pages = int(kv_indices.numel()) + 2
    combined = torch.zeros(
        num_pages, page_size, 1, ckv_dim + kpe_dim, dtype=dtype, device=device
    )
    ckv_4d, kpe_4d = combined.split([ckv_dim, kpe_dim], dim=-1)

    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    )
    wrapper.plan(
        torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        kv_indptr,
        kv_indices,
        torch.tensor(kv_lens, dtype=torch.int32, device=device),
        num_heads,
        ckv_dim,
        kpe_dim,
        page_size,
        False,
        1.0 / math.sqrt(ckv_dim + kpe_dim),
        dtype,
        dtype,
    )
    q_nope = torch.randn(batch_size, num_heads, ckv_dim, dtype=dtype, device=device)
    q_pe = torch.randn(batch_size, num_heads, kpe_dim, dtype=dtype, device=device)

    with pytest.raises(ValueError, match="must be 3-D"):
        wrapper.run(q_nope, q_pe, ckv_4d, kpe_4d)


def test_gather_plan_inputs_passes_through_host_tensors():
    """Host-resident inputs must not be concatenated — there is no sync to save.

    Batching exists to collapse device->host round-trips; on CPU inputs a cat is
    pure added allocation, which the previous docstring wrongly called a no-op.
    """
    from flashinfer.mla_rocm import _gather_plan_inputs_on_host

    qo = torch.arange(5, dtype=torch.int32)
    kvp = torch.arange(5, dtype=torch.int32) * 4
    kvl = torch.full((4,), 4, dtype=torch.int32)

    qo_h, kvp_h, kvl_h = _gather_plan_inputs_on_host(qo, kvp, kvl)

    # Same objects, i.e. no copy was made at all.
    assert qo_h is qo and kvp_h is kvp and kvl_h is kvl
