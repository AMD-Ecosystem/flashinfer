# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""fp8 KV cache on the in-tree (fa2) prefill kernel, and the refusals around it.

The oracle is that fp8 -> fp16 is lossless, so feeding the kernel an fp8 cache
must equal feeding the trusted 2-byte kernel the same values dequantized by
torch. Both runs select the same tile geometry -- the LDS tile is 2-byte either
way -- so the accumulation order matches and the results are bit-identical.

pos_encoding_mode is NONE throughout: the fa2 RoPE path returns a different
answer on every call for *any* KV dtype (measured on an unmodified kernel), so
it cannot serve as a reference here.
"""

import logging
import math

import pytest
import torch
from attention_reference import naive_attention
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)

FNUZ_DTYPES = [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
OCP_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
WORKSPACE = 128 * 1024 * 1024


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    common = ([0], [False, True], [False], [False])  # posenc, swa, softcap, f16qk
    specs = gen_prefill_attention_modules(
        [torch.float16], [torch.float16] + FNUZ_DTYPES, [64, 128], *common
    ) + gen_prefill_attention_modules(
        [torch.bfloat16], [torch.bfloat16] + FNUZ_DTYPES, [128], *common
    )
    # Both calls emit the shared helper specs; ninja rejects duplicate rules.
    flashinfer.jit.build_jit_specs(
        list({spec.name: spec for spec in specs}.values()), verbose=False
    )
    yield


def _quantized_pair(shape, fp8_dtype, wide_dtype=torch.float16, device="cuda:0"):
    """An fp8 tensor and its exact dequantization into wide_dtype."""
    src = torch.randn(shape, dtype=torch.float16, device=device)
    q8 = src.to(fp8_dtype)
    return q8, q8.to(wide_dtype)


def _assert_matches(got, ref):
    # Bit-equality, not a tolerance: both runs use a 2-byte LDS tile and so the
    # same geometry and accumulation order. A tolerance here would swallow a
    # tile-selection change, which is the one thing this oracle exists to catch.
    assert torch.equal(got, ref), (
        f"not bit-identical: max |delta| "
        f"{(got.float() - ref.float()).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("qo_len,kv_len", [(1, 64), (37, 128), (77, 396), (577, 2048)])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_fp8_kv_matches_dequantized(
    q_dtype, fp8_dtype, head_dim, qo_len, kv_len, causal
):
    if q_dtype is torch.bfloat16 and head_dim != 128:
        pytest.skip("bf16 queries are warmed up at head_dim 128 only")
    torch.manual_seed(0)
    num_qo_heads, num_kv_heads = 32, 8
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype, device="cuda:0")
    k8, k16 = _quantized_pair((kv_len, num_kv_heads, head_dim), fp8_dtype, q_dtype)
    v8, v16 = _quantized_pair((kv_len, num_kv_heads, head_dim), fp8_dtype, q_dtype)

    kwargs = dict(causal=causal, backend="fa2", pos_encoding_mode="NONE")
    got = flashinfer.single_prefill_with_kv_cache(q, k8, v8, **kwargs)
    ref = flashinfer.single_prefill_with_kv_cache(q, k16, v16, **kwargs)
    _assert_matches(got, ref)


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("page_size", [1, 16])
def test_paged_prefill_fp8_kv_matches_dequantized(fp8_dtype, head_dim, page_size):
    torch.manual_seed(0)
    bs, qo_len, num_qo_heads, num_kv_heads = 3, 41, 32, 8
    # kv_len must exceed qo_len or the bottom-right causal mask blanks most
    # query rows and both sides trivially agree on zeros.
    pages_per_seq = max(5, -(-2 * qo_len // page_size))
    num_pages = bs * pages_per_seq
    kv8, kv16 = _quantized_pair(
        (num_pages, 2, page_size, num_kv_heads, head_dim), fp8_dtype
    )
    dev = "cuda:0"
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * qo_len
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * pages_per_seq
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device=dev)
    last_page_len = torch.full((bs,), page_size, dtype=torch.int32, device=dev)
    q = torch.randn(
        bs * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=dev
    )

    outs = []
    for cache, kv_dtype in ((kv8, fp8_dtype), (kv16, torch.float16)):
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(WORKSPACE, dtype=torch.uint8, device=dev), "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.float16,
            kv_data_type=kv_dtype,
        )
        outs.append(wrapper.run(q, cache))
    _assert_matches(*outs)


def _decode(
    kv_indptr,
    kv_indices,
    last_page_len,
    q,
    cache,
    kv_dtype,
    *,
    tc,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
):
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(WORKSPACE, dtype=torch.uint8, device=q.device),
        "NHD",
        use_tensor_cores=tc,
        backend="fa2",
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=kv_dtype,
    )
    return wrapper.run(q, cache)


def _decode_inputs(
    head_dim,
    fp8_dtype,
    bs=4,
    num_qo_heads=32,
    num_kv_heads=8,
    page_size=16,
    pages_per_seq=4,
):
    torch.manual_seed(0)
    dev = "cuda:0"
    num_pages = bs * pages_per_seq
    cache8, cache16 = _quantized_pair(
        (num_pages, 2, page_size, num_kv_heads, head_dim), fp8_dtype
    )
    shapes = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    idx = (
        torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * pages_per_seq,
        torch.arange(0, num_pages, dtype=torch.int32, device=dev),
        torch.full((bs,), page_size, dtype=torch.int32, device=dev),
    )
    q = torch.randn(bs, num_qo_heads, head_dim, dtype=torch.float16, device=dev)
    return idx, q, cache8, cache16, shapes


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_tensor_core_decode_fp8_kv(fp8_dtype, head_dim):
    """The reported defect: use_tensor_cores=True routes decode at the prefill kernel."""
    idx, q, cache8, cache16, shapes = _decode_inputs(head_dim, fp8_dtype)
    got = _decode(*idx, q, cache8, fp8_dtype, tc=True, **shapes)
    ref = _decode(*idx, q, cache16, torch.float16, tc=True, **shapes)
    _assert_matches(got, ref)


@pytest.mark.parametrize("kv_dtype", [torch.float16] + FNUZ_DTYPES)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_tensor_core_decode_matches_plain_decode(kv_dtype, head_dim):
    """Cross-check the two decode kernels against each other.

    Nothing covered the tensor-core path with any dtype, which is why an fp8
    cache there reached a user as a ninja log. This also exercises plain decode
    with e5m2fnuz, newly reachable now that the dtype maps name it.
    """
    fp8 = kv_dtype if kv_dtype in FNUZ_DTYPES else torch.float8_e4m3fnuz
    idx, q, cache8, cache16, shapes = _decode_inputs(head_dim, fp8)
    cache = cache16 if kv_dtype == torch.float16 else cache8
    tc = _decode(*idx, q, cache, kv_dtype, tc=True, **shapes)
    plain = _decode(*idx, q, cache, kv_dtype, tc=False, **shapes)
    # Different kernels and reduction orders, so agreement is numerical.
    torch.testing.assert_close(tc.float(), plain.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
def test_ragged_prefill_fp8_kv_matches_dequantized(fp8_dtype):
    torch.manual_seed(0)
    dev = "cuda:0"
    bs, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = 3, 37, 128, 32, 8, 128
    q = torch.randn(
        bs * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=dev
    )
    k8, k16 = _quantized_pair((bs * kv_len, num_kv_heads, head_dim), fp8_dtype)
    v8, v16 = _quantized_pair((bs * kv_len, num_kv_heads, head_dim), fp8_dtype)
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * qo_len
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * kv_len

    outs = []
    for k, v, kv_dtype in ((k8, v8, fp8_dtype), (k16, v16, torch.float16)):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(WORKSPACE, dtype=torch.uint8, device=dev), "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            q_data_type=torch.float16,
            kv_data_type=kv_dtype,
        )
        outs.append(wrapper.run(q, k, v))
    _assert_matches(*outs)


def test_ragged_prefill_fp8_query_is_refused_at_plan():
    """run() used to silently cast q, k AND v to f16 on an fp8 query, which would
    now discard a caller's fp8 KV cache. The refusal moved to plan(), so that
    branch is unreachable and has been removed."""
    dev = "cuda:0"
    bs, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = 2, 16, 64, 32, 8, 128
    qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * qo_len
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * kv_len
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(WORKSPACE, dtype=torch.uint8, device=dev), "NHD", backend="fa2"
    )
    with pytest.raises((NotImplementedError, ValueError), match="(?i)fp8"):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=True,
            q_data_type=torch.float8_e4m3fnuz,
            kv_data_type=torch.float8_e4m3fnuz,
        )


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
def test_mismatched_k_v_dtypes_are_refused(fp8_dtype):
    """An fp8 k with a 2-byte v used to die in the 8-bit static_assert; with the
    kernel serving fp8 it returned NaN instead."""
    dev = "cuda:0"
    q = torch.randn(32, 32, 128, dtype=torch.float16, device=dev)
    k = torch.randn(64, 8, 128, dtype=torch.float16, device=dev).to(fp8_dtype)
    v = torch.randn(64, 8, 128, dtype=torch.float16, device=dev)
    with pytest.raises(ValueError, match="single KV dtype"):
        flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="fa2")


@pytest.mark.parametrize("int_dtype", [torch.int8, torch.uint8])
def test_integer_kv_is_refused(int_dtype):
    """1-byte integers have no float interpretation; widening them would read
    0..255 as values. The dtype allowlist refuses them before any build."""
    dev = "cuda:0"
    q = torch.randn(32, 32, 128, dtype=torch.float16, device=dev)
    k = torch.randint(0, 8, (64, 8, 128), dtype=int_dtype, device=dev)
    with pytest.raises(NotImplementedError, match="(?i)kv dtype"):
        flashinfer.single_prefill_with_kv_cache(q, k, k, causal=True, backend="fa2")


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES + OCP_DTYPES)
def test_fp8_query_is_refused(fp8_dtype):
    dev = "cuda:0"
    q = torch.randn(32, 8, 128, dtype=torch.float16, device=dev).to(fp8_dtype)
    k = torch.randn(64, 8, 128, dtype=torch.float16, device=dev).to(fp8_dtype)
    with pytest.raises((NotImplementedError, ValueError), match="(?i)fp8"):
        flashinfer.single_prefill_with_kv_cache(q, k, k, causal=True, backend="fa2")


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES + OCP_DTYPES)
def test_fp8_query_with_wide_kv_is_refused(fp8_dtype):
    """A typed refusal, never a bare assert.

    The fnuz spellings are already refused earlier; the OCP ones reached the
    fp8-query branch before any backend was chosen and failed its
    `assert q.dtype == k.dtype == v.dtype` with no message.
    """
    dev = "cuda:0"
    q = torch.randn(32, 8, 128, dtype=torch.float16, device=dev).to(fp8_dtype)
    k = torch.randn(64, 8, 128, dtype=torch.float16, device=dev)
    with pytest.raises(NotImplementedError, match="(?i)fp8|dtype"):
        flashinfer.single_prefill_with_kv_cache(q, k, k, causal=True)


@pytest.mark.parametrize("ocp_dtype", OCP_DTYPES)
def test_ocp_fp8_kv_is_refused(ocp_dtype):
    """dtype_map_hip sends OCP to the fnuz type, whose bias is one greater."""
    dev = "cuda:0"
    q = torch.randn(32, 32, 128, dtype=torch.float16, device=dev)
    k = torch.randn(64, 8, 128, dtype=torch.float16, device=dev).to(ocp_dtype)
    with pytest.raises(NotImplementedError, match="(?i)ocp|bias"):
        flashinfer.single_prefill_with_kv_cache(q, k, k, causal=True, backend="fa2")


@pytest.mark.parametrize(
    "dtype_q,dtype_kv,dtype_o,allowed",
    [
        (torch.int8, torch.int8, torch.int8, False),
        (torch.uint8, torch.uint8, torch.uint8, False),
        (torch.float32, torch.float32, torch.float32, False),
        (torch.float8_e5m2fnuz, torch.float8_e5m2fnuz, torch.bfloat16, False),
        (torch.float16, torch.bfloat16, torch.float16, False),
        (torch.float16, torch.float16, torch.int8, False),
        (torch.float16, torch.float16, torch.float16, True),
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, True),
        # AITER's own fp8 prefill, both arch spellings of e4m3: must stay open.
        (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, torch.bfloat16, True),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16, True),
    ],
)
def test_aiter_generator_dtype_allowlist(dtype_q, dtype_kv, dtype_o, allowed):
    """Equality alone let int8/uint8 through: both are in dtype_map_hip, so they
    built a URI and reached ninja with no kernel behind them."""
    from flashinfer.jit.rocm.modules import (
        gen_batch_prefill_module,
        gen_single_prefill_module,
    )

    calls = (
        lambda: gen_single_prefill_module(
            "aiter", dtype_q, dtype_kv, dtype_o, 128, 128, 0, False, False, False
        ),
        lambda: gen_batch_prefill_module(
            "aiter",
            dtype_q,
            dtype_kv,
            dtype_o,
            torch.int32,
            128,
            128,
            0,
            False,
            False,
            False,
        ),
    )
    for call in calls:
        if allowed:
            call()
        else:
            with pytest.raises(NotImplementedError, match="(?i)aiter"):
                call()


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
def test_pod_refuses_fp8_kv(fp8_dtype):
    """POD sizes its tiles with upstream's CUDA register constant, so it can ask
    for a NUM_MMA_KV the prefill dispatchers never produce. Guarded at the
    generator, which is the single seam both POD wrappers build through."""
    from flashinfer.jit.rocm.modules import gen_batch_pod_module, gen_pod_module

    common = dict(
        head_dim=128,
        pos_encoding_mode_p=0,
        use_sliding_window_p=False,
        use_logits_soft_cap_p=False,
        use_fp16_qk_reduction=False,
        dtype_idx=torch.int32,
        pos_encoding_mode_d=0,
        use_sliding_window_d=False,
        use_logits_soft_cap_d=False,
    )
    for gen in (gen_pod_module, gen_batch_pod_module):
        with pytest.raises(NotImplementedError, match="(?i)fp8"):
            gen(torch.float16, fp8_dtype, torch.float16, **common)


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
def test_decode_mismatched_k_v_dtypes_are_refused(fp8_dtype):
    """Decode specializes on k.dtype and casts both pointers to it. Measured
    before the guard: an fp8 k with an fp16 v returned NaN and no error."""
    dev = "cuda:0"
    bs, nq, nkv, hd, page_size, pages = 2, 32, 8, 128, 16, 4
    k8 = torch.randn(
        bs * pages, page_size, nkv, hd, dtype=torch.float16, device=dev
    ).to(fp8_dtype)
    v16 = torch.randn(bs * pages, page_size, nkv, hd, dtype=torch.float16, device=dev)
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device=dev) * pages
    kv_indices = torch.arange(0, bs * pages, dtype=torch.int32, device=dev)
    last = torch.full((bs,), page_size, dtype=torch.int32, device=dev)
    q = torch.randn(bs, nq, hd, dtype=torch.float16, device=dev)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(WORKSPACE, dtype=torch.uint8, device=dev), "NHD", backend="fa2"
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last,
        nq,
        nkv,
        hd,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=fp8_dtype,
    )
    # RuntimeError, not ValueError: plan() already built the module, so the C++
    # entry point is what refuses this rather than a Python pre-check.
    with pytest.raises(RuntimeError, match="single KV dtype"):
        wrapper.run(q, (k8, v16))


def test_mixed_wide_kv_dtype_is_refused():
    """The MFMA is instantiated from DTypeQ, so a bf16 cache under an fp16 query
    is read as fp16. aot.py already excludes the pair from the AOT matrix."""
    dev = "cuda:0"
    q = torch.randn(32, 32, 128, dtype=torch.float16, device=dev)
    k = torch.randn(64, 8, 128, dtype=torch.bfloat16, device=dev)
    with pytest.raises(NotImplementedError, match="(?i)differs from query dtype"):
        flashinfer.single_prefill_with_kv_cache(q, k, k, causal=True, backend="fa2")


@pytest.mark.parametrize("ocp_dtype", OCP_DTYPES)
def test_customize_decode_generators_refuse_ocp(ocp_dtype):
    """gen_{single,batch}_decode_module delegate here, and both customize
    generators are publicly re-exported, so guarding only the wrappers left the
    public JIT API able to compile OCP fp8 as fnuz."""
    from flashinfer.jit.rocm.modules import (
        gen_customize_batch_decode_module,
        gen_customize_single_decode_module,
    )

    common = dict(
        dtype_q=torch.float16,
        dtype_kv=ocp_dtype,
        dtype_o=torch.float16,
        head_dim_qk=128,
        head_dim_vo=128,
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=[],
        additional_scalar_dtypes=[],
        variant_name="v",
        variant_decl="",
    )
    with pytest.raises(NotImplementedError, match="(?i)ocp"):
        gen_customize_single_decode_module(uri="u", **common)
    with pytest.raises(NotImplementedError, match="(?i)ocp"):
        gen_customize_batch_decode_module(uri="u", idtype=torch.int32, **common)


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
@pytest.mark.parametrize("window_left", [0, 63])
def test_single_prefill_fp8_kv_sliding_window(fp8_dtype, window_left):
    """Windowing changes which KV tiles are produced and predicated, which is
    the kFillZero/kNoFill split the fp8 produce arm rewrites."""
    torch.manual_seed(0)
    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = 77, 396, 32, 8, 128
    q = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    k8, k16 = _quantized_pair((kv_len, num_kv_heads, head_dim), fp8_dtype)
    v8, v16 = _quantized_pair((kv_len, num_kv_heads, head_dim), fp8_dtype)

    kwargs = dict(
        causal=True, backend="fa2", pos_encoding_mode="NONE", window_left=window_left
    )
    got = flashinfer.single_prefill_with_kv_cache(q, k8, v8, **kwargs)
    ref = flashinfer.single_prefill_with_kv_cache(q, k16, v16, **kwargs)
    _assert_matches(got, ref)


def test_unmapped_dtype_refuses_before_the_uri():
    """The URI builder indexes filename_safe_dtype_map, so a dtype absent from it
    raised KeyError before the allowlist ran. float32 is the reachable case."""
    dev = "cuda:0"
    q = torch.randn(32, 32, 128, dtype=torch.float32, device=dev)
    with pytest.raises(NotImplementedError, match="(?i)query dtype"):
        flashinfer.single_prefill_with_kv_cache(q, q, q, causal=True, backend="fa2")


def _block_sparse_out(k, v, kv_dtype, q, n, nh, head_dim, dev):
    block = torch.ones(n // 16, n // 16, dtype=torch.bool)
    indptr = torch.zeros(block.size(0) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(block.sum(dim=1), 0)
    indices = block.nonzero()[:, 1].to(torch.int32)
    wrapper = flashinfer.BlockSparseAttentionWrapper(
        torch.empty(WORKSPACE, dtype=torch.uint8, device=dev)
    )
    wrapper.plan(
        indptr.to(dev),
        indices.to(dev),
        n,
        n,
        16,
        16,
        nh,
        nh,
        head_dim,
        q_data_type=torch.float16,
        kv_data_type=kv_dtype,
    )
    return wrapper.run(q, k, v)


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
def test_block_sparse_fp8_kv_matches_dequantized(fp8_dtype):
    """Block-sparse builds through the batch-prefill module, so deleting the
    blanket static_assert made fp8 reachable here too."""
    torch.manual_seed(0)
    n, nh, head_dim, dev = 128, 4, 128, "cuda:0"
    q = torch.randn(n, nh, head_dim, dtype=torch.float16, device=dev)
    k8, k16 = _quantized_pair((n, nh, head_dim), fp8_dtype)
    v8, v16 = _quantized_pair((n, nh, head_dim), fp8_dtype)
    args = (q, n, nh, head_dim, dev)
    got = _block_sparse_out(k8, v8, fp8_dtype, *args)
    ref = _block_sparse_out(k16, v16, torch.float16, *args)
    _assert_matches(got, ref)


def test_block_sparse_mismatched_k_v_dtypes_are_refused():
    """sparse.py is upstream and unshadowed, so the C++ entry point is the only
    seam that can refuse this without carrying an edit into every sync."""
    torch.manual_seed(0)
    n, nh, head_dim, dev = 128, 4, 128, "cuda:0"
    q = torch.randn(n, nh, head_dim, dtype=torch.float16, device=dev)
    k16 = torch.randn(n, nh, head_dim, dtype=torch.float16, device=dev)
    v_bf = torch.randn(n, nh, head_dim, dtype=torch.bfloat16, device=dev)
    with pytest.raises(RuntimeError, match="single KV dtype"):
        _block_sparse_out(k16, v_bf, torch.float16, q, n, nh, head_dim, dev)


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
@pytest.mark.parametrize("causal", [False, True])
def test_fp8_kv_scales_fold_as_documented(fp8_dtype, causal):
    """k_scale/v_scale are how a real cache is read, and they are folded outside
    the kernel: k_scale into sm_scale, v_scale onto the output.

    Asserting those two identities rather than comparing against a rescaled
    cache -- that comparison is dominated by rounding the output to fp16 while
    it is still in the un-scaled domain, not by whether the scales are right.
    """
    torch.manual_seed(0)
    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim = 77, 396, 32, 8, 128
    dev = "cuda:0"
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=dev)
    k8, _ = _quantized_pair((kv_len, num_kv_heads, head_dim), fp8_dtype)
    v8, _ = _quantized_pair((kv_len, num_kv_heads, head_dim), fp8_dtype)
    base_sm_scale = 1.0 / math.sqrt(head_dim)
    common = dict(causal=causal, backend="fa2", pos_encoding_mode="NONE")

    def run(**kw):
        return flashinfer.single_prefill_with_kv_cache(q, k8, v8, **common, **kw)

    # k_scale folds into sm_scale, so naming it is the same call.
    k_scale = 0.37
    torch.testing.assert_close(
        run(sm_scale=base_sm_scale, k_scale=k_scale).float(),
        run(sm_scale=base_sm_scale * k_scale).float(),
        rtol=0,
        atol=0,
    )

    # v_scale is a post-multiply on the output, so it must be exactly linear.
    # atol is one fp16 subnormal: v_scale multiplies the fp16 output in place,
    # and only results that land in the subnormal range round differently here.
    v_scale = 0.25  # a power of two, so nothing else can round
    torch.testing.assert_close(
        run(sm_scale=base_sm_scale, v_scale=v_scale).float(),
        run(sm_scale=base_sm_scale).float() * v_scale,
        rtol=0,
        atol=6e-8,
    )


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
def test_fp8_kv_quantization_quality(fp8_dtype):
    """What the fp8 cache costs in accuracy, against the *unquantized* values.

    The reference must be the pre-quantization tensor: dequantizing the same
    fp8 cache the kernel reads makes the quantization error zero on both sides
    and measures nothing. Bound is relative L2, which the per-element max is
    too outlier-driven to give.
    """
    torch.manual_seed(0)
    qo_len, kv_len, num_heads, head_dim = 64, 256, 8, 128
    dev = "cuda:0"
    q = torch.randn(qo_len, num_heads, head_dim, dtype=torch.float16, device=dev)
    k_src = torch.randn(kv_len, num_heads, head_dim, dtype=torch.float16, device=dev)
    v_src = torch.randn(kv_len, num_heads, head_dim, dtype=torch.float16, device=dev)
    k8, v8 = k_src.to(fp8_dtype), v_src.to(fp8_dtype)

    got = flashinfer.single_prefill_with_kv_cache(q, k8, v8, causal=True, backend="fa2")
    ref, _ = naive_attention(q.float(), k_src.float(), v_src.float(), causal=True)
    rel_l2 = ((got.float() - ref).norm() / ref.norm()).item()
    # e5m2fnuz keeps 2 mantissa bits against e4m3fnuz's 3, so it earns ~3x.
    bound = 0.05 if fp8_dtype is torch.float8_e4m3fnuz else 0.16
    assert rel_l2 < bound, f"{fp8_dtype} relative L2 {rel_l2:.4f} exceeds {bound}"


@pytest.mark.parametrize("fp8_dtype", FNUZ_DTYPES)
def test_fp8_kv_matches_its_own_dequantization_in_fp32(fp8_dtype):
    """The fp32 twin of the fp16 oracle: same quantized values both sides, so
    any gap is the kernel's accumulation rather than the cache's coarseness."""
    torch.manual_seed(0)
    qo_len, kv_len, num_heads, head_dim = 64, 256, 8, 128
    dev = "cuda:0"
    q = torch.randn(qo_len, num_heads, head_dim, dtype=torch.float16, device=dev)
    k8, _ = _quantized_pair((kv_len, num_heads, head_dim), fp8_dtype)
    v8, _ = _quantized_pair((kv_len, num_heads, head_dim), fp8_dtype)

    got = flashinfer.single_prefill_with_kv_cache(q, k8, v8, causal=True, backend="fa2")
    # naive_attention, not F.scaled_dot_product_attention: torch's is_causal
    # anchors the mask top-left, FlashInfer bottom-right, and qo_len != kv_len here.
    ref, _ = naive_attention(
        q.float(), k8.to(torch.float32), v8.to(torch.float32), causal=True
    )
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
