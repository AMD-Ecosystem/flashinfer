"""
Copyright (c) 2024 by FlashInfer team.
Copyright (c) 2025-2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer
from tests.test_helpers.test_helpers import requires_aiter


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def gemma_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    # Add in float32 to match the kernel, which promotes both operands before adding.
    x_f32 = x.float() + residual.float()
    residual = x_f32.to(orig_dtype)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x = x_f32 * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    # Pin to native: this test checks the native kernel against a tight float32
    # reference. On ROCm, backend="auto" routes 2D inputs to AITER's lower-precision
    # rms_norm, whose accuracy is validated separately in test_rmsnorm_aiter_hip.py.
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.rmsnorm(x, w, out=y, enable_pdl=enable_pdl, backend="native")
    else:
        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl, backend="native")

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(y_ref, y, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_pdl", [False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype, enable_pdl, contiguous):
    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(x_fused, x_native, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_fused, residual_native, rtol=rtol, atol=atol)


@requires_aiter
@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_aiter(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.fused_add_rmsnorm(x_fused, residual_fused, weight, eps, backend="aiter")

    # AITER CK reductions are lower precision than the native kernel.
    rtol, atol = (7e-2, 7e-2) if dtype == torch.bfloat16 else (4e-3, 4e-3)
    torch.testing.assert_close(x_fused, x_native, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_fused, residual_native, rtol=rtol, atol=atol)


@requires_aiter
def test_fused_add_rmsnorm_auto_selection():
    """auto routes fused_add_rmsnorm to the C++ AITER kernel on supported devices."""
    from flashinfer.norm import _auto_select_fused_add_rmsnorm_backend

    x = torch.empty(512, 8192, dtype=torch.float16, device="cuda")
    assert _auto_select_fused_add_rmsnorm_backend(x) == "aiter"


@pytest.mark.parametrize("backend", ["auto", "native", "aiter"])
def test_fused_add_rmsnorm_rejects_non_2d(backend):
    # Both native and AITER kernels are 2D-only; any backend must raise a clear
    # ValueError up front rather than failing deeper in the kernel.
    x = torch.randn(4, 8, 128, dtype=torch.float16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="2D"):
        flashinfer.fused_add_rmsnorm(x, residual, weight, backend=backend)


@requires_aiter
@pytest.mark.parametrize("batch_size", [128, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_auto_correct(batch_size, hidden_size, dtype):
    # The 2048-row shapes land above the 4M cutoff, so this grid exercises the
    # AITER path under backend="auto" (not just native), confirming auto stays
    # correct there.
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_auto = x.clone()
    residual_auto = residual.clone()
    flashinfer.fused_add_rmsnorm(x_auto, residual_auto, weight, eps, backend="auto")

    # auto may pick AITER (CK lower precision) for large shapes, so use relaxed tol.
    rtol, atol = (7e-2, 7e-2) if dtype == torch.bfloat16 else (4e-3, 4e-3)
    torch.testing.assert_close(x_auto, x_native, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_auto, residual_native, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_norm(
    batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous
):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.gemma_rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.gemma_rmsnorm(x, w, enable_pdl=enable_pdl)

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(y_ref, y, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("enable_pdl", [False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_fused_add_rmsnorm(
    batch_size, hidden_size, dtype, enable_pdl, contiguous
):
    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.gemma_fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(x_fused, x_native, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_fused, residual_native, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_norm(1, 1024, torch.float16, False, True)
    test_fused_add_rmsnorm(1, 4096, torch.float16, True, True)
