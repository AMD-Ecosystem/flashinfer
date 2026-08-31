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
@pytest.mark.parametrize("hidden_size", [111, 128, 500, 1024, 3072, 3584, 4096, 8192])
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
    # Explicit, though `auto` now resolves here too: this test asserts the
    # native kernel against a tight float32 reference.
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.rmsnorm(x, w, out=y, enable_pdl=enable_pdl, backend="native")
    else:
        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl, backend="native")

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(y_ref, y, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 128, 500, 1024, 3072, 3584, 4096, 8192])
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
    # Explicit, though `auto` now resolves here too: this test asserts the
    # native kernel specifically. test_fused_add_rmsnorm_aiter covers AITER.
    flashinfer.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl, backend="native"
    )

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(x_fused, x_native, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_fused, residual_native, rtol=rtol, atol=atol)


# 16384 crosses the gfx942 LDS threshold and 40960 the gfx950 one, so the
# re-read-from-global fallback is exercised on both arches.
@pytest.mark.parametrize("hidden_size", [16352, 16384, 40960])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gemma", [False, True])
def test_fused_add_rmsnorm_large_hidden_size(hidden_size, dtype, gemma):
    eps = 1e-6
    batch_size = 2
    torch.manual_seed(0)
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda") * 0.1
    residual = torch.randn_like(x) * 0.1
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda") * 0.1

    if gemma:
        x_ref, residual_ref = gemma_fused_add_rms_norm(
            x.clone(), residual.clone(), weight, eps
        )
    else:
        x_ref, residual_ref = fused_add_rms_norm(
            x.clone(), residual.clone(), weight, eps
        )

    x_out, residual_out = x.clone(), residual.clone()
    if gemma:
        flashinfer.gemma_fused_add_rmsnorm(x_out, residual_out, weight, eps)
    else:
        flashinfer.fused_add_rmsnorm(x_out, residual_out, weight, eps, backend="native")

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(x_out, x_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_out, residual_ref, rtol=rtol, atol=atol)


# 64 and 128 guard the CK aliasing bug: CK packs several rows per block at small
# n, so an aliased output is silently wrong there and correct at larger sizes.
@requires_aiter
@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize(
    "hidden_size", [64, 111, 128, 500, 1024, 3072, 3584, 4096, 8192]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_aiter(batch_size, hidden_size, dtype):
    eps = 1e-6
    if hidden_size % 2 != 0:
        pytest.skip("AITER rmsnorm mis-handles odd hidden sizes (amd-aiter 0.1.20)")

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


def test_fused_add_rmsnorm_auto_selection(monkeypatch):
    """auto is native even where AITER is fully available -- AITER is 1.6-1.8x
    slower here (benchmarks/rocm/bench_norm.py).

    AITER is forced available rather than skipped, so a revert is caught even
    on a box without the wheel, where "native" would otherwise be the right
    answer for the wrong reason. Both the wrapper and the probe it is built
    from are stubbed, since a revert could consult either.
    """
    import flashinfer.aiter_utils as aiter_utils
    from flashinfer.rocm.norm import _auto_select_fused_add_rmsnorm_backend

    monkeypatch.setattr(aiter_utils, "is_aiter_available", lambda device, op: True)
    monkeypatch.setattr(aiter_utils, "_aiter_importable", lambda: True)

    x = torch.empty(512, 8192, dtype=torch.float16, device="cuda")
    assert _auto_select_fused_add_rmsnorm_backend(x) == "native"


@requires_aiter
@pytest.mark.parametrize("hidden_size", [8448, 16384])
def test_oversized_hidden_explicit_aiter_raises(hidden_size):
    """AITER's kernel gives up past 8192; raise before it does, with the size.

    The CK kernel this replaced had no ceiling, so without the guard a caller
    that pinned backend="aiter" gets a bare "not support n:" from inside AITER.
    """
    x = torch.randn(8, hidden_size, dtype=torch.float16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="hidden sizes up to"):
        flashinfer.rmsnorm(x, weight, backend="aiter")
    with pytest.raises(ValueError, match="hidden sizes up to"):
        flashinfer.fused_add_rmsnorm(x, residual, weight, backend="aiter")


@requires_aiter
@pytest.mark.parametrize("hidden_size", [111, 63])
def test_odd_hidden_explicit_aiter_raises(hidden_size):
    x = torch.randn(32, hidden_size, dtype=torch.float16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="odd hidden"):
        flashinfer.rmsnorm(x, weight, backend="aiter")
    with pytest.raises(ValueError, match="odd hidden"):
        flashinfer.fused_add_rmsnorm(x, residual, weight, backend="aiter")


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
class TestAiterFusedAddArgChecks:
    """CK derives one dtype from `input` and reads the other buffers with it, so
    a mismatch is silent garbage rather than an error. The native kernel rejects
    all of these in C++; the AITER path used to accept them."""

    @pytest.fixture
    def base(self):
        x = torch.randn(8, 128, dtype=torch.float16, device="cuda")
        return (
            x,
            torch.randn_like(x),
            torch.randn(128, dtype=torch.float16, device="cuda"),
        )

    def test_fp32_input_is_rejected(self):
        x = torch.randn(8, 128, dtype=torch.float32, device="cuda")
        w = torch.randn(128, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="float16/bfloat16"):
            flashinfer.fused_add_rmsnorm(x, torch.randn_like(x), w, backend="aiter")

    def test_mismatched_weight_dtype_is_rejected(self, base):
        x, residual, _ = base
        w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="weight.dtype == input.dtype"):
            flashinfer.fused_add_rmsnorm(x, residual, w, backend="aiter")

    def test_mismatched_residual_dtype_is_rejected(self, base):
        x, _, w = base
        residual = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="residual.dtype == input.dtype"):
            flashinfer.fused_add_rmsnorm(x, residual, w, backend="aiter")

    def test_mismatched_residual_shape_is_rejected(self, base):
        x, _, w = base
        residual = torch.randn(4, 128, dtype=torch.float16, device="cuda")
        with pytest.raises(ValueError, match="residual.shape == input.shape"):
            flashinfer.fused_add_rmsnorm(x, residual, w, backend="aiter")

    def test_wrong_weight_length_is_rejected(self, base):
        x, residual, _ = base
        w = torch.randn(64, dtype=torch.float16, device="cuda")
        with pytest.raises(ValueError, match="1-D weight of length 128"):
            flashinfer.fused_add_rmsnorm(x, residual, w, backend="aiter")

    def test_non_1d_weight_is_rejected(self, base):
        """Native raises "weight must be a 1D tensor"; AITER accepted a (1, n)
        weight of the right numel, so the two backends disagreed on validity."""
        x, residual, _ = base
        w = torch.randn(1, 128, dtype=torch.float16, device="cuda")
        with pytest.raises(ValueError, match="1-D weight of length 128"):
            flashinfer.fused_add_rmsnorm(x, residual, w, backend="aiter")

    def test_non_contiguous_last_dim_is_rejected(self, base):
        _, _, w = base
        wide = torch.randn(8, 256, dtype=torch.float16, device="cuda")
        x = wide[:, ::2]
        with pytest.raises(ValueError, match="contiguous last dimension"):
            flashinfer.fused_add_rmsnorm(x, torch.randn_like(x), w, backend="aiter")

    def test_strided_weight_is_rejected(self, base):
        """The shim's reshape({1, -1}) keeps the stride rather than packing, so
        CK would read a strided weight as contiguous: 9.6 abs error, silently."""
        x, residual, _ = base
        w = torch.randn(256, dtype=torch.float16, device="cuda")[::2]
        with pytest.raises(ValueError, match="contiguous weight"):
            flashinfer.fused_add_rmsnorm(x, residual, w, backend="aiter")


# The default path end to end through the public API. Not a routing guard --
# AITER also passes the native tolerance at these shapes, so this stays green
# either way; test_fused_add_rmsnorm_auto_selection is what pins the policy.
@pytest.mark.parametrize("batch_size", [128, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_auto_correct(batch_size, hidden_size, dtype):
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

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(x_auto, x_native, rtol=rtol, atol=atol)
    torch.testing.assert_close(residual_auto, residual_native, rtol=rtol, atol=atol)


# rmsnorm had no public-API auto test at all; same scope as above.
@pytest.mark.parametrize("batch_size", [128, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_auto_correct(batch_size, hidden_size, dtype):
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    y_ref = llama_rms_norm(x, weight)
    y_auto = flashinfer.rmsnorm(x, weight, backend="auto")

    rtol, atol = (1.6e-2, 1.6e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(y_auto, y_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 128, 500, 1024, 3072, 3584, 4096, 8192])
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
@pytest.mark.parametrize("hidden_size", [111, 128, 500, 1024, 3072, 3584, 4096, 8192])
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
