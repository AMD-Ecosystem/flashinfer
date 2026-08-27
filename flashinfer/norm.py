"""
Copyright (c) 2024 by FlashInfer team.

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

import functools
from typing import Optional

import torch

from .device_utils import IS_HIP

if IS_HIP:
    from ._rocm.norm import maybe_fused_add_rmsnorm, maybe_rmsnorm
from .jit.norm import gen_norm_module
from .utils import device_support_pdl, register_custom_op, register_fake_op


@functools.cache
def get_norm_module():
    return gen_norm_module().build_and_load()


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    backend: str
        Kernel backend to use. ``"auto"`` (default) selects the best available backend:
        the AITER C++ kernel for 2D fp16/bf16 inputs on supported ROCm devices, else native.
        ``"native"`` uses the FlashInfer JIT kernel on all platforms.
        ``"aiter"`` uses AMD AITER's CK ``rmsnorm2d`` C++ kernel — ROCm (gfx942/gfx950)
        only, 2D fp16/bf16 inputs only. Precision is slightly lower than ``"native"`` at
        ``hidden_size >= 1024`` (fp16 atol ~4e-3, bf16 ~7e-2).

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    """
    if IS_HIP:
        if (result := maybe_rmsnorm(out, input, weight, eps, backend)) is not None:
            return result
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
def _rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().rmsnorm(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::rmsnorm")
def _rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
    backend: str = "auto",
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    backend: str
        Kernel backend to use. ``"auto"`` (default) routes to the C++ AITER CK
        kernel on ROCm (gfx942/gfx950) and to the native FlashInfer JIT kernel
        everywhere else.
        ``"native"`` uses the FlashInfer JIT kernel on all platforms.
        ``"aiter"`` uses AMD AITER's CK ``rmsnorm2d_with_add`` C++ kernel — ROCm
        (gfx942/gfx950) only, 2D inputs only. Precision is slightly lower than
        ``"native"`` at ``hidden_size >= 1024``.
    """
    # Both the native and AITER kernels are 2D-only (the native kernel enforces
    # CHECK_DIM(2) on input/residual), so reject other ranks up front with a clear
    # Python error rather than letting it fail deeper in the kernel.
    if input.ndim != 2:
        raise ValueError(
            f"fused_add_rmsnorm only supports 2D inputs; got {input.ndim}D."
        )
    if IS_HIP:
        if maybe_fused_add_rmsnorm(input, residual, weight, eps, backend):
            return
    _fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)


@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"))
def _fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Gemma-style root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if out is None:
        out = torch.empty_like(input)
    _gemma_rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::gemma_rmsnorm", mutates_args=("out",))
def _gemma_rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().gemma_rmsnorm(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_rmsnorm")
def _gemma_rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@register_custom_op(
    "flashinfer::gemma_fused_add_rmsnorm", mutates_args=("input", "residual")
)
def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Gemma-style fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().gemma_fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_fused_add_rmsnorm")
def _gemma_fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@register_custom_op("flashinfer::layernorm", mutates_args=())
def layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Layer normalization.
    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size). Need to be bfloat16.
    gemma: torch.Tensor
        Gemma tensor, shape (hidden_size,). Need to be float32.
    beta: torch.Tensor
        Beta tensor, shape (hidden_size,). Need to be float32.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    output: torch.Tensor
        Layer Normalized tensor, shape (batch_size, hidden_size). Same dtype as input.
    """
    out = torch.empty_like(input)
    get_norm_module().layernorm(out, input, gemma, beta, eps)
    return out


@register_fake_op("flashinfer::layernorm")
def _layernorm_fake(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    b, k = input.shape
    return input.new_empty([b, k])
