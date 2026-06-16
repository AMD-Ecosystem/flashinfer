# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the AITER rms_norm backend exposed via flashinfer.rmsnorm(backend="aiter").
#
# Note on tolerances: AITER rms_norm uses lower-precision reductions than the
# flashinfer native JIT kernel. For production inference these differences are
# negligible, but they exceed the native kernel's test tolerance (fp16 atol=1e-3,
# bf16 atol=1.6e-2). The tolerances below reflect AITER's actual precision.

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter


def _rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Float32 reference matching the test in test_norm_hip.py."""
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


@requires_aiter
def test_rmsnorm_auto_backend_stays_native():
    """auto backend on gfx942/950 should stay on native kernel (precision parity with tests)."""
    from flashinfer.norm import _auto_select_norm_backend

    device = torch.device("cuda:0")
    assert _auto_select_norm_backend(device) == "native"


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
