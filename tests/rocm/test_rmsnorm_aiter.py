# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the AITER CK rmsnorm2d backend exposed via flashinfer.rmsnorm(backend="aiter").
#
# Note on tolerances: AITER's CK rmsnorm2d uses lower-precision reductions than the
# flashinfer native JIT kernel. For production inference these differences are
# negligible, but they exceed the native kernel's test tolerance (fp16 atol=1e-3,
# bf16 atol=1.6e-2). The tolerances below reflect AITER's actual precision.

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter


def _rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Float32 reference matching the test in test_norm.py."""
    orig = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * w.float()).to(orig)


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 512, 1024, 4096])
@pytest.mark.parametrize("batch_size", [1, 32, 256])
def test_rmsnorm_aiter_vs_ref(dtype, hidden_size, batch_size):
    torch.manual_seed(0xA17E2)
    device = torch.device("cuda:0")
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    w = torch.randn(hidden_size, dtype=dtype, device=device)

    ref = _rms_norm_ref(x, w)
    got = flashinfer.rmsnorm(x, w, backend="aiter")

    # AITER precision: fp16 ≤ 4e-3, bf16 ≤ 7e-2 observed across shapes.
    rtol, atol = (7e-2, 7e-2) if dtype == torch.bfloat16 else (4e-3, 4e-3)
    torch.testing.assert_close(got.float(), ref.float(), rtol=rtol, atol=atol)


def test_rmsnorm_auto_backend_is_native(monkeypatch):
    """auto is native for every shape and dtype, including the 2-D fp16 case
    that used to route to AITER: the two are level on speed and native is the
    more accurate kernel (benchmarks/rocm/bench_norm.py).

    AITER is forced available so a revert is caught even on a box without the
    wheel, where "native" would otherwise be right for the wrong reason. Both
    the wrapper and the probe it is built from are stubbed.
    """
    import flashinfer.aiter_utils as aiter_utils
    from flashinfer.rocm.norm import _auto_select_norm_backend

    monkeypatch.setattr(aiter_utils, "is_aiter_available", lambda device, op: True)
    monkeypatch.setattr(aiter_utils, "_aiter_importable", lambda: True)

    device = torch.device("cuda:0")
    x2d = torch.randn(8, 128, dtype=torch.float16, device=device)
    w = torch.randn(128, dtype=torch.float16, device=device)
    x3d = torch.randn(8, 4, 128, dtype=torch.float16, device=device)
    w_fp32 = torch.randn(128, dtype=torch.float32, device=device)
    assert _auto_select_norm_backend(x2d, w) == "native"
    assert _auto_select_norm_backend(x3d, w) == "native"
    assert _auto_select_norm_backend(x2d, w_fp32) == "native"


@requires_aiter
def test_rmsnorm_aiter_rejects_fp32():
    """backend='aiter' with fp32 raises a clear FlashInfer error, not the deep CK one.

    fp32 is unsupported by both ROCm rmsnorm kernels; the Python-level guard
    rejects it before reaching AITER so the message is actionable.
    """
    device = torch.device("cuda:0")
    x = torch.randn(8, 128, dtype=torch.float32, device=device)
    w = torch.randn(128, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="float16/bfloat16"):
        flashinfer.rmsnorm(x, w, backend="aiter")


@requires_aiter
def test_rmsnorm_aiter_rejects_weight_dtype_mismatch():
    """backend='aiter' with weight.dtype != input.dtype raises rather than silently
    producing NaN/garbage (CK rmsnorm2d reads weight bytes with the input dtype)."""
    device = torch.device("cuda:0")
    x = torch.randn(8, 128, dtype=torch.float16, device=device)
    w = torch.randn(128, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="weight.dtype == input.dtype"):
        flashinfer.rmsnorm(x, w, backend="aiter")


@requires_aiter
def test_rmsnorm_aiter_rejects_strided_weight():
    """The shim's reshape({1, -1}) keeps the stride rather than packing, so CK
    would read a strided weight as contiguous: 7.1 abs error, silently. Native
    rejects it, so this keeps the two backends on one contract."""
    device = torch.device("cuda:0")
    x = torch.randn(8, 128, dtype=torch.float16, device=device)
    w = torch.randn(256, dtype=torch.float16, device=device)[::2]
    with pytest.raises(ValueError, match="contiguous weight"):
        flashinfer.rmsnorm(x, w, backend="aiter")


@requires_aiter
@pytest.mark.parametrize("shape", [(1, 128), (64,)])
def test_rmsnorm_aiter_rejects_wrong_weight_shape(shape):
    """Native rejects both; AITER accepted them, and the short one is an
    out-of-bounds read inside CK rather than an error."""
    device = torch.device("cuda:0")
    x = torch.randn(8, 128, dtype=torch.float16, device=device)
    w = torch.randn(*shape, dtype=torch.float16, device=device)
    with pytest.raises(ValueError, match="1-D weight of length 128"):
        flashinfer.rmsnorm(x, w, backend="aiter")


@requires_aiter
def test_rmsnorm_aiter_with_out_tensor():
    """backend='aiter' respects the out= argument."""
    device = torch.device("cuda:0")
    x = torch.randn(8, 128, dtype=torch.float16, device=device)
    w = torch.ones(128, dtype=torch.float16, device=device)
    out = torch.empty_like(x)
    ret = flashinfer.rmsnorm(x, w, out=out, backend="aiter")
    assert ret.data_ptr() == out.data_ptr()
    assert not torch.all(out == 0)


@requires_aiter
@pytest.mark.parametrize("hidden_size", [64, 128, 1024])
@pytest.mark.parametrize("batch_size", [19, 989])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_aiter_in_place_out_matches_reference(hidden_size, batch_size, dtype):
    """out=x is a documented idiom, and CK cannot alias its output onto its
    input: at small n it packs several rows per block and corrupts them."""
    eps = 1e-6
    device = torch.device("cuda:0")
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    w = torch.randn(hidden_size, dtype=dtype, device=device)

    expected = _rms_norm_ref(x, w, eps)

    aliased = x.clone()
    flashinfer.rmsnorm(aliased, w, eps, out=aliased, backend="aiter")

    rtol, atol = (7e-2, 7e-2) if dtype == torch.bfloat16 else (4e-3, 4e-3)
    torch.testing.assert_close(aliased, expected, rtol=rtol, atol=atol)
