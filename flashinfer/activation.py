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
from types import SimpleNamespace
from typing import Optional

import torch

from .device_utils import IS_CUDA, IS_HIP
from .jit import gen_act_and_mul_module
from .utils import (
    device_support_pdl,
    register_custom_op,
    register_fake_op,
    get_compute_capability,
)

if IS_CUDA:
    from .fp4_quantization import get_fp4_quantization_module


if IS_HIP:

    @functools.cache
    def _aiter_act_ops():
        import aiter as _aiter

        return _aiter

    # AITER's silu_and_mul only overtakes the native kernel on large,
    # bandwidth-bound shapes (~5-10% faster); below that a fixed ~0.7us launch
    # overhead makes it slower. It also matches the native kernel's precision
    # only in fp16 (bf16 is ~6e-2 vs ~4e-3 max err). The cutoff counts elements
    # of the full input (rows x 2*hidden); the measured break-even is ~33M input
    # elements (e.g. 2048 x 16384), so 64M is a safe 2x margin. fp16 only.
    _AITER_SILU_AND_MUL_MIN_ELEMS = 64 * 1024 * 1024

    def _auto_select_silu_and_mul_backend(input: torch.Tensor) -> str:
        # Cheapest guards first so the common small/medium case exits early.
        if input.dtype != torch.float16:
            return "native"
        if input.ndim != 2:
            return "native"
        if input.numel() < _AITER_SILU_AND_MUL_MIN_ELEMS:
            return "native"
        from .aiter_utils import is_aiter_supported

        if not is_aiter_supported(input.device):
            return "native"
        try:
            # Best-effort probe: a supported arch can still lack a usable aiter
            # (not installed, or its compiled extension fails to load). auto must
            # always be able to fall back to native, so catch any import failure.
            _aiter_act_ops()
        except Exception:
            return "native"
        return "aiter"


@functools.cache
def get_act_and_mul_module(act_func_name: str):
    module = gen_act_and_mul_module(act_func_name).build_and_load()

    # torch library for act_and_mul
    fname = f"{act_func_name}_and_mul"
    fn = getattr(module, fname)

    @register_custom_op(f"flashinfer::{fname}", mutates_args=("out",))
    def _act_and_mul(
        out: torch.Tensor, input: torch.Tensor, enable_pdl: Optional[bool] = None
    ) -> None:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)
        fn(out, input, enable_pdl)

    @register_fake_op(f"flashinfer::{fname}")
    def _fake_act_and_mul(
        out: torch.Tensor, input: torch.Tensor, enable_pdl: Optional[bool] = None
    ) -> None:
        pass

    # Register the module
    return SimpleNamespace(**{fname: _act_and_mul})


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert input.shape[:-1] == output.shape[:-1], (
        f"{input.shape[:-1]} != {output.shape[:-1]}"
    )
    assert input.shape[-1] == 2 * output.shape[-1], (
        f"{input.shape[-1]} != {2 * output.shape[-1]}"
    )


def silu_and_mul(
    input: torch.Tensor,
    out: torch.Tensor = None,
    enable_pdl: Optional[bool] = None,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Fused SiLU and Mul operation.

    ``silu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    backend: str
        Kernel backend to use. ``"auto"`` (default) uses the native kernel for small
        and medium inputs, and switches to AITER on ROCm for large (>= 64M element)
        2D fp16 inputs where its kernel is faster and matches native precision; it
        falls back to native whenever AITER is unavailable.
        ``"native"`` uses the FlashInfer JIT kernel on all platforms.
        ``"aiter"`` uses AMD AITER's ``silu_and_mul`` — ROCm (gfx942/gfx950) only;
        requires the ``aiter`` package, and raises ``ValueError`` on any other
        platform. Precision matches ``"native"`` in fp16 but is lower in bf16
        (max err ~6e-2 vs ~4e-3), which is why ``"auto"`` restricts the AITER path
        to fp16.

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if backend not in ("auto", "native", "aiter"):
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of 'auto', 'native', 'aiter'."
        )
    if backend == "aiter":
        # Validate the explicit opt-in on every platform so a misconfiguration
        # surfaces here instead of silently running native off ROCm.
        from .aiter_utils import is_aiter_supported

        if not (IS_HIP and is_aiter_supported(input.device)):
            raise ValueError(
                "backend='aiter' requires a ROCm gfx942/gfx950 device with the "
                "aiter package installed."
            )
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    if IS_HIP:
        _backend = (
            backend if backend != "auto" else _auto_select_silu_and_mul_backend(input)
        )
        if _backend == "aiter":
            _aiter_act_ops().silu_and_mul(out, input)
            return out
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_act_and_mul_module("silu").silu_and_mul(
        out,
        input,
        enable_pdl,
    )
    return out


def gelu_tanh_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None
) -> torch.Tensor:
    r"""Fused GeLU Tanh and Mul operation.

    ``gelu(tanh(input[..., :hidden_size])) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("gelu_tanh").gelu_tanh_and_mul(out, input, enable_pdl)
    return out


def gelu_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, enable_pdl: Optional[bool] = None
) -> torch.Tensor:
    r"""Fused GeLU and Mul operation.

    ``gelu(input[..., :hidden_size]) * input[..., hidden_size:]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (..., 2 * hidden_size).

    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.

    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Output tensor, shape (..., hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    get_act_and_mul_module("gelu").gelu_and_mul(out, input, enable_pdl)
    return out


def silu_and_mul_scaled_nvfp4_experts_quantize(
    a,
    mask,
    a_global_sf,
):
    """
    Silu and multiply and quantize batched input tensor to NVFP4 format with mask.
    Parameters:
        a (torch.Tensor): Input tensor of shape [B, M, K] with dtype fp16/bf16.
        a_global_sf (torch.Tensor): Global scale factor of shape [1] with dtype float32.
        mask (torch.Tensor): Mask tensor to apply before quantization.
        sf_vec_size (int, optional): Scale factor vector size. Defaults to 16.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Quantized tensor of shape [B, M, K/2] with dtype FLOAT4_E2M1X2
            - Scale factors tensor with shape determined by layout and sf_vec_size
    """
    major, minor = get_compute_capability(a.device)
    device_arch = f"{major * 10 + minor}"
    a_fp4, a_sf = get_fp4_quantization_module(
        device_arch
    ).silu_and_mul_scaled_nvfp4_experts_quantize_sm100(
        a,
        mask,
        a_global_sf,
    )
    return a_fp4, a_sf
