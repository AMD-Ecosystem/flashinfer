# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A q/k/v whose last dimension is strided is refused, not read as if packed.

The kernels honour token and head strides but load head_dim, and write o and
lse, packed. ``single_decode_with_kv_cache`` copies its one-token q instead.
"""

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter

H, D, PAGE = 8, 128, 16
DTYPE = torch.float16
MSG = "must be contiguous at last dimension"


@pytest.fixture
def device():
    return torch.device("cuda:0")


@pytest.fixture
def workspace(device):
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)


def _i32(device, *v):
    return torch.tensor(v, dtype=torch.int32, device=device)


def _strided(shape, device):
    """Random values laid out with stride(-1) == 2."""
    return torch.randn(*shape[:-1], shape[-1] * 2, dtype=DTYPE, device=device)[..., ::2]


def _dense(shape, device):
    return torch.randn(*shape, dtype=DTYPE, device=device)


BACKENDS = ["fa2", pytest.param("aiter", marks=requires_aiter)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strided", ["q", "k", "v"])
def test_single_prefill(device, backend, strided):
    t = {
        "q": _dense((16, H, D), device),
        "k": _dense((64, H, D), device),
        "v": _dense((64, H, D), device),
    }
    t[strided] = _strided(t[strided].shape, device)
    with pytest.raises(RuntimeError, match=f"{strided} {MSG}"):
        flashinfer.single_prefill_with_kv_cache(
            t["q"], t["k"], t["v"], causal=True, backend=backend
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("strided", ["q", "k", "v"])
def test_ragged_prefill(device, workspace, backend, strided):
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    w.plan(
        _i32(device, 0, 16),
        _i32(device, 0, 64),
        H,
        H,
        D,
        causal=True,
        q_data_type=DTYPE,
    )
    t = {
        "q": _dense((16, H, D), device),
        "k": _dense((64, H, D), device),
        "v": _dense((64, H, D), device),
    }
    t[strided] = _strided(t[strided].shape, device)
    with pytest.raises(RuntimeError, match=f"{strided} {MSG}"):
        w.run(t["q"], t["k"], t["v"])


def _paged_meta(device):
    """One request over four full pages: indptr, indices, last_page_len."""
    indices = torch.arange(4, dtype=torch.int32, device=device)
    return _i32(device, 0, 4), indices, _i32(device, PAGE)


@pytest.mark.parametrize(
    "backend,strided",
    [
        ("fa2", "q"),
        ("fa2", "cache"),
        pytest.param("aiter", "q", marks=requires_aiter),
    ],
)
def test_paged_prefill(device, workspace, backend, strided):
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend=backend
    )
    w.plan(
        _i32(device, 0, 16),
        *_paged_meta(device),
        H,
        H,
        D,
        PAGE,
        causal=True,
        q_data_type=DTYPE,
    )
    q = _strided((16, H, D), device) if strided == "q" else _dense((16, H, D), device)
    kv_shape = (4, 2, PAGE, H, D)
    kv = _strided(kv_shape, device) if strided == "cache" else _dense(kv_shape, device)
    with pytest.raises(RuntimeError, match=MSG):
        w.run(q, kv)


@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("strided", ["q", "cache"])
def test_batch_decode_fa2(device, workspace, use_tensor_cores, strided):
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2", use_tensor_cores=use_tensor_cores
    )
    w.plan(*_paged_meta(device), H, H, D, PAGE, q_data_type=DTYPE)
    q = _strided((1, H, D), device) if strided == "q" else _dense((1, H, D), device)
    kv_shape = (4, 2, PAGE, H, D)
    kv = _strided(kv_shape, device) if strided == "cache" else _dense(kv_shape, device)
    with pytest.raises(RuntimeError, match=MSG):
        w.run(q, kv)


@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("layout", ["head_sliced", "last_dim_strided", "transposed"])
def test_single_decode_accepts_any_q_layout(device, use_tensor_cores, layout):
    k = _dense((64, H, D), device)
    v = _dense((64, H, D), device)
    if layout == "head_sliced":
        q = _dense((H, 2 * D), device)[:, :D]
    elif layout == "last_dim_strided":
        q = _strided((H, D), device)
    else:
        q = _dense((D, H), device).t()
    out = flashinfer.single_decode_with_kv_cache(
        q, k, v, use_tensor_cores=use_tensor_cores
    )
    ref = flashinfer.single_decode_with_kv_cache(
        q.contiguous(), k, v, use_tensor_cores=use_tensor_cores
    )
    assert out.is_contiguous()
    torch.testing.assert_close(out, ref)


@requires_aiter
def test_paged_prefill_aiter_native_route_refuses_a_strided_cache(device, workspace):
    """Page size 1024 at head_dim 128 pages natively, which reads head_dim packed."""
    page = 1024
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    w.plan(
        _i32(device, 0, 16),
        _i32(device, 0, 1),
        _i32(device, 0),
        _i32(device, page),
        H,
        H,
        D,
        page,
        causal=True,
        q_data_type=torch.bfloat16,
    )
    if w._aiter_flat_gather_idx is not None:
        pytest.skip("this AITER build does not page natively here")
    q = torch.randn(16, H, D, dtype=torch.bfloat16, device=device)
    kv = torch.randn(1, 2, page, H, 2 * D, dtype=torch.bfloat16, device=device)[
        ..., ::2
    ]
    with pytest.raises(RuntimeError, match=f"paged_k_cache {MSG}"):
        w.run(q, kv)


@pytest.mark.parametrize("route", ["decode", "decode-tc", "paged", "ragged"])
def test_fa2_refuses_a_non_contiguous_out(device, workspace, route):
    """The fa2 kernels write o packed, so a strided out= would be filled wrongly."""
    out = _dense((16, H, 2 * D), device)[:, :, :D]
    kv = _dense((4, 2, PAGE, H, D), device)
    if route.startswith("decode"):
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2", use_tensor_cores=route == "decode-tc"
        )
        w.plan(*_paged_meta(device), H, H, D, PAGE, q_data_type=DTYPE)
        call = lambda: w.run(_dense((1, H, D), device), kv, out=out[:1])  # noqa: E731
    elif route == "paged":
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        w.plan(
            _i32(device, 0, 16),
            *_paged_meta(device),
            H,
            H,
            D,
            PAGE,
            causal=True,
            q_data_type=DTYPE,
        )
        call = lambda: w.run(_dense((16, H, D), device), kv, out=out)  # noqa: E731
    else:
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        w.plan(
            _i32(device, 0, 16),
            _i32(device, 0, 64),
            H,
            H,
            D,
            causal=True,
            q_data_type=DTYPE,
        )
        k, v = _dense((64, H, D), device), _dense((64, H, D), device)
        call = lambda: w.run(_dense((16, H, D), device), k, v, out=out)  # noqa: E731
    with pytest.raises(RuntimeError, match="o must be contiguous"):
        call()


@pytest.mark.parametrize("strided", ["q_p", "k_p", "q_d", "cache"])
def test_pod_refuses_a_strided_head_dim(device, workspace, strided):
    w = flashinfer.PODWithPagedKVCacheWrapper(workspace, "NHD")
    w.plan(*_paged_meta(device), H, H, D, PAGE, q_data_type=DTYPE, data_type=DTYPE)
    t = {
        "q_p": _dense((16, H, D), device),
        "k_p": _dense((64, H, D), device),
        "q_d": _dense((1, H, D), device),
        "cache": _dense((4, 2, PAGE, H, D), device),
    }
    t[strided] = _strided(t[strided].shape, device)
    with pytest.raises(RuntimeError, match=MSG):
        w.run(t["q_p"], t["k_p"], _dense((64, H, D), device), t["q_d"], t["cache"])


@pytest.mark.parametrize("strided", ["q_p", "cache_p", "q_d", "cache_d"])
def test_batch_pod_refuses_a_strided_head_dim(device, workspace, strided):
    w = flashinfer.BatchPODWithPagedKVCacheWrapper(workspace, "NHD")
    meta = _paged_meta(device)
    w.plan(
        _i32(device, 0, 16),
        *meta,
        _i32(device, 0, 1),
        *meta,
        H,
        H,
        D,
        PAGE,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    t = {
        "q_p": _dense((16, H, D), device),
        "cache_p": _dense((4, 2, PAGE, H, D), device),
        "q_d": _dense((1, H, D), device),
        "cache_d": _dense((4, 2, PAGE, H, D), device),
    }
    t[strided] = _strided(t[strided].shape, device)
    with pytest.raises(RuntimeError, match=MSG):
        w.run(t["q_p"], t["cache_p"], t["q_d"], t["cache_d"])


@pytest.mark.parametrize("bad", [None, "strided_o", "short_o", "strided_lse"])
def test_paged_prefill_partial_state_layout(device, workspace, bad):
    """partial_state is merged into o with o's packed layout."""
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    w.plan(
        _i32(device, 0, 16),
        *_paged_meta(device),
        H,
        H,
        D,
        PAGE,
        causal=True,
        q_data_type=DTYPE,
    )
    q, kv = _dense((16, H, D), device), _dense((4, 2, PAGE, H, D), device)
    po = _dense((16, H, D), device)
    plse = torch.zeros(16, H, dtype=torch.float32, device=device)
    if bad == "strided_o":
        po = _dense((16, H, 2 * D), device)[:, :, :D]
    elif bad == "short_o":
        po = po[:8]
    elif bad == "strided_lse":
        plse = torch.zeros(16, 2 * H, dtype=torch.float32, device=device)[:, ::2]
    if bad is None:
        w.run(q, kv, partial_state=(po, plse))
    else:
        with pytest.raises(RuntimeError, match="partial_(o|lse) must be contiguous"):
            w.run(q, kv, partial_state=(po, plse))
