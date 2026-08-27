# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flashinfer


def _silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x_f32 = x.float()
    gate, up = x_f32[..., :d], x_f32[..., d:]
    return (gate / (1.0 + torch.exp(-gate)) * up).to(x.dtype)


def _gelu_tanh_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x_f32 = x.float()
    gate, up = x_f32[..., :d], x_f32[..., d:]
    y = torch.nn.functional.gelu(gate, approximate="tanh") * up
    return y.to(x.dtype)


def _gelu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x_f32 = x.float()
    gate, up = x_f32[..., :d], x_f32[..., d:]
    y = torch.nn.functional.gelu(gate, approximate="none") * up
    return y.to(x.dtype)


# (num_tokens, d) — input tensor is (num_tokens, 2*d)
# d values chosen to cover: < blockDim threshold, wave64-unaligned (5504),
# typical LLM dims, blockDim-cap boundary (8192→1024 threads).
_SHAPES = [
    (1, 64),
    (1, 128),
    (4, 256),
    (8, 512),
    (1, 2752),  # 5504 // 2 — from LLaMA-2 intermediate with GQA
    (4, 4096),
    (8, 5504),  # LLaMA-2 ffn_dim // 2  (non-multiple-of-64-threads)
    (16, 7168),  # Mistral-7B ffn_dim // 2
    (1, 8192),  # hits 1024-thread cap
    (4, 14336),  # Llama-3-70B ffn_dim // 2, multiple stride iterations
    (2, 14336),  # few tokens, large d → blocks_per_row > 1 (row split across blocks)
    (4096, 4096),  # prefill-scale token count → blocks_per_row == 1 (no row split)
]


@pytest.mark.parametrize("num_tokens,d", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul(num_tokens, d, dtype):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * d, device="cuda", dtype=dtype)
    ref = _silu_and_mul_ref(x)
    out = flashinfer.activation.silu_and_mul(x)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_tokens,d", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_tanh_and_mul(num_tokens, d, dtype):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * d, device="cuda", dtype=dtype)
    ref = _gelu_tanh_and_mul_ref(x)
    out = flashinfer.activation.gelu_tanh_and_mul(x)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_tokens,d", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_and_mul(num_tokens, d, dtype):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * d, device="cuda", dtype=dtype)
    ref = _gelu_and_mul_ref(x)
    out = flashinfer.activation.gelu_and_mul(x)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_silu_and_mul_auto_backend_selection():
    """auto resolves to the in-tree native kernel; AITER is opt-in via backend='aiter'.

    Independent of the aiter package, so it lives in the native test module to run
    on any HIP build (the contract must hold even when aiter is not installed).
    """
    from flashinfer.rocm.activation import _auto_select_silu_and_mul_backend

    x = torch.empty(8, 256, dtype=torch.float16, device="cuda")
    assert _auto_select_silu_and_mul_backend(x) == "native"
