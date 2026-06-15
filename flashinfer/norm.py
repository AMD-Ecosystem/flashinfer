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
from .jit.norm import gen_norm_module
from .utils import device_support_pdl, register_custom_op, register_fake_op


@functools.cache
def get_norm_module():
    return gen_norm_module().build_and_load()


if IS_HIP:

    @functools.cache
    def _aiter_norm_ops():
        import aiter as _aiter

        return _aiter

    def _auto_select_norm_backend(device: torch.device, dtype: torch.dtype) -> str:
        # AITER rms_norm uses lower-precision reductions that exceed the flashinfer
        # test tolerance (fp16 atol=1e-3, bf16 atol=1.6e-2) at hidden_size >= 1024.
        # Keep auto on the native JIT kernel; pass backend="aiter" to opt in explicitly.
        return "native"

    # AITER's CK fused_add_rmsnorm only overtakes the native kernel on large,
    # bandwidth-bound shapes (~10-14% faster at >=2048 rows x 8192 cols); below
    # that the native kernel is faster and more accurate. Route auto to AITER only
    # past this element-count cutoff, safely above the ~4M-element tie point.
    _AITER_FUSED_ADD_RMSNORM_MIN_ELEMS = 8 * 1024 * 1024

    def _auto_select_fused_add_rmsnorm_backend(input: torch.Tensor) -> str:
        # Cheapest guards first so the common small/medium case exits early.
        if input.ndim != 2:
            return "native"
        if input.numel() < _AITER_FUSED_ADD_RMSNORM_MIN_ELEMS:
            return "native"
        from .aiter_utils import is_aiter_supported

        if not is_aiter_supported(input.device):
            return "native"
        try:
            # Best-effort probe: a supported arch can still lack a usable aiter
            # (not installed, or its compiled extension fails to load). auto must
            # always be able to fall back to native, so catch any import failure.
            _aiter_norm_ops()
        except Exception:
            return "native"
        return "aiter"


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
        Kernel backend to use. ``"auto"`` (default) selects the best available backend.
        ``"native"`` uses the FlashInfer JIT kernel on all platforms.
        ``"aiter"`` uses AMD AITER's rms_norm — ROCm (gfx942/gfx950) only; requires the
        ``aiter`` package and only supports 2D inputs. Precision is slightly lower than
        ``"native"`` at ``hidden_size >= 1024`` (fp16 atol ~4e-3, bf16 ~7e-2).

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    """
    if IS_HIP:
        _backend = (
            backend
            if backend != "auto"
            else _auto_select_norm_backend(input.device, input.dtype)
        )
        if _backend == "aiter":
            if input.ndim != 2:
                raise ValueError(
                    f"AITER rmsnorm only supports 2D inputs; got {input.ndim}D. "
                    "Use backend='native' for 3D inputs."
                )
            result = _aiter_norm_ops().rms_norm(input, weight, eps)
            if out is not None:
                out.copy_(result)
                return out
            return result
        if _backend not in ("native",):
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
            )
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
        Kernel backend to use. ``"auto"`` (default) uses the native kernel for small
        and medium inputs, and switches to AITER on ROCm for large (>= 8M element) 2D
        inputs where its CK kernel is faster; it falls back to native whenever AITER
        is unavailable.
        ``"native"`` uses the FlashInfer JIT kernel on all platforms.
        ``"aiter"`` uses AMD AITER's CK ``rmsnorm2d_fwd_with_add`` — ROCm
        (gfx942/gfx950) only; requires the ``aiter`` package and only supports 2D
        inputs. Precision is slightly lower than ``"native"`` at ``hidden_size >= 1024``.
    """
    if IS_HIP:
        _backend = (
            backend
            if backend != "auto"
            else _auto_select_fused_add_rmsnorm_backend(input)
        )
        if _backend == "aiter":
            if input.ndim != 2:
                raise ValueError(
                    f"AITER fused_add_rmsnorm only supports 2D inputs; got {input.ndim}D. "
                    "Use backend='native' for 3D inputs."
                )
            # CK kernel writes out/residual_out separately; alias them onto the
            # input/residual tensors to match FlashInfer's in-place semantics.
            _aiter_norm_ops().rmsnorm2d_fwd_with_add(
                input, input, residual, residual, weight, eps
            )
            return
        if _backend != "native":
            raise ValueError(
                f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
            )
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
