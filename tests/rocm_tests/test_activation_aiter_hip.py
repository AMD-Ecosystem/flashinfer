# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the AITER silu_and_mul backend exposed via
# flashinfer.activation.silu_and_mul(backend="aiter").
#
# Note on tolerances: AITER silu_and_mul matches the native JIT kernel exactly in
# fp16, but uses lower-precision arithmetic in bf16 (max err ~6e-2 vs the native
# kernel's ~4e-3). The tolerances below reflect AITER's actual precision.

import pytest
import torch

import flashinfer
from flashinfer.aiter_utils import is_aiter_supported
from tests.test_helpers.test_helpers import requires_aiter


def _silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x_f32 = x.float()
    gate, up = x_f32[..., :d], x_f32[..., d:]
    return (gate / (1.0 + torch.exp(-gate)) * up).to(x.dtype)


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d", [128, 512, 4096, 8192, 14336])
@pytest.mark.parametrize("num_tokens", [1, 8, 256])
def test_silu_and_mul_aiter_vs_ref(dtype, d, num_tokens):
    torch.manual_seed(0xA17E2)
    device = torch.device("cuda:0")
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device=device)

    ref = _silu_and_mul_ref(x)
    got = flashinfer.activation.silu_and_mul(x, backend="aiter")

    # AITER precision: fp16 matches native; bf16 ~6e-2 observed across shapes.
    rtol, atol = (7e-2, 7e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(got.float(), ref.float(), rtol=rtol, atol=atol)


@requires_aiter
def test_silu_and_mul_auto_backend_selection():
    """auto stays native for small/bf16 inputs and picks AITER for large fp16 2D inputs."""
    from flashinfer.activation import (
        _AITER_SILU_AND_MUL_MIN_ELEMS,
        _auto_select_silu_and_mul_backend,
    )

    device = torch.device("cuda:0")

    # Small inputs: native regardless of dtype.
    for dtype in (torch.float16, torch.bfloat16):
        small = torch.empty(8, 256, dtype=dtype, device=device)
        assert _auto_select_silu_and_mul_backend(small) == "native"

    # Large enough fp16 2D input (>= cutoff). cols is a multiple of 8 so the
    # shape also clears silu_and_mul's 16-byte alignment guard, i.e. it is a
    # shape that could actually flow through the public function to AITER.
    rows = 8192
    cols = -(-_AITER_SILU_AND_MUL_MIN_ELEMS // (rows * 8)) * 8  # ceil to mult of 8
    large_fp16 = torch.empty(rows, cols, dtype=torch.float16, device=device)
    assert large_fp16.numel() >= _AITER_SILU_AND_MUL_MIN_ELEMS
    assert _auto_select_silu_and_mul_backend(large_fp16) == "aiter"

    # Same large shape in bf16: native (precision guard).
    large_bf16 = torch.empty(rows, cols, dtype=torch.bfloat16, device=device)
    assert _auto_select_silu_and_mul_backend(large_bf16) == "native"

    # Large fp16 but 3D: native (2D guard).
    large_3d = torch.empty(2, rows, cols, dtype=torch.float16, device=device)
    assert _auto_select_silu_and_mul_backend(large_3d) == "native"


@requires_aiter
def test_silu_and_mul_aiter_with_out_tensor():
    """backend='aiter' writes the correct result into the supplied out= tensor."""
    device = torch.device("cuda:0")
    x = torch.randn(8, 256, dtype=torch.float16, device=device)
    # Seed out with a sentinel the kernel must overwrite, so a no-op write fails.
    out = torch.full((8, 128), float("nan"), dtype=torch.float16, device=device)
    ret = flashinfer.activation.silu_and_mul(x, out=out, backend="aiter")
    assert ret.data_ptr() == out.data_ptr()
    ref = _silu_and_mul_ref(x)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-3, atol=1e-3)


def test_silu_and_mul_unknown_backend_raises():
    # Backend validation is platform-independent, so this needs no aiter device.
    x = torch.randn(8, 256, dtype=torch.float16)
    with pytest.raises(ValueError, match="Unknown backend"):
        flashinfer.activation.silu_and_mul(x, backend="nope")


def test_silu_and_mul_aiter_backend_rejected_when_unsupported():
    """Explicit backend='aiter' raises (not silently falls back) on an unsupported device."""
    cpu_x = torch.randn(8, 256, dtype=torch.float16)
    if not is_aiter_supported(cpu_x.device):
        with pytest.raises(ValueError, match="requires a ROCm"):
            flashinfer.activation.silu_and_mul(cpu_x, backend="aiter")
