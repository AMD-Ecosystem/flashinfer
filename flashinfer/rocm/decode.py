"""
Copyright (c) 2023 by FlashInfer team.

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
import math
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch

from ..jit import (
    gen_batch_decode_aiter_module,
    gen_batch_decode_module,
    gen_customize_batch_decode_module,
    gen_customize_batch_prefill_module,
    gen_single_decode_module,
    get_batch_decode_aiter_uri,
    get_batch_decode_uri,
    get_single_decode_uri,
)
from .aiter_utils import handle_aiter_probe_failure
from ..jit.core import logger
from ..page import get_seq_lens
from .prefill import (
    _aiter_bootstrap_lock,
    _auto_select_prefill_backend,
    get_batch_prefill_jit_module,
    get_batch_prefill_module,
    get_single_prefill_module,
)
from ..utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _get_range_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    ceil_div,
    device_support_pdl,
    is_float8,
    plan_info_vec_as_tensor,
    register_custom_op,
    register_fake_op,
)


@functools.cache
def get_single_decode_module(*args):
    uri = get_single_decode_uri(*args)
    module = gen_single_decode_module(*args).build_and_load()
    run_func = module.run.default

    # torch library for single_decode_with_kv_cache

    @register_custom_op(f"flashinfer::{uri}_run", mutates_args=("tmp", "o"))
    def run_single_decode(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tmp: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        run_func(
            q,
            k,
            v,
            tmp,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            alibi_slopes,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
        )

    @register_fake_op(f"flashinfer::{uri}_run")
    def _fake_run_single_decode(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tmp: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        pass

    # Register the module.
    return SimpleNamespace(run=run_single_decode)


@functools.cache
def get_batch_decode_jit_module(module_name: str, jit_module: Any):
    plan_func = jit_module.plan.default
    run_func = jit_module.run.default

    @register_custom_op(
        f"flashinfer::{module_name}_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        *args,
    ) -> None:
        run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            # enable_pdl,  # Not supported by HIP kernels, skipped
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_run")
    def _fake_run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        *args,
    ) -> None:
        pass

    return SimpleNamespace(
        plan=plan_func,
        run=run_batch_decode,
    )


@functools.cache
def get_batch_decode_module(*args):
    uri = get_batch_decode_uri(*args)
    mod = gen_batch_decode_module(*args).build_and_load()
    plan_func = mod.plan.default
    run_func = mod.run.default

    # torch library for batch_decode_with_paged_kv_cache_run

    @register_custom_op(
        f"flashinfer::{uri}_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            # enable_pdl,  # ROCm kernel does not support this parameter yet
            alibi_slopes,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
        )

    @register_fake_op(f"flashinfer::{uri}_run")
    def _fake_run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=plan_func,
        run=run_batch_decode,
    )


@functools.cache
def get_batch_decode_aiter_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
):
    """Build & load the AITER PA v1 C++ harness extension for (dtype, head_dim).

    The returned namespace has a single ``run`` callable mirroring the C++ entry's
    signature. The per-variant AITER .so is loaded lazily inside C++ via dlopen
    on the first run() call using (so_path, func_name) resolved at plan() time.
    """
    uri = get_batch_decode_aiter_uri(
        dtype_q, dtype_kv, dtype_o, head_dim_qk, head_dim_vo
    )
    mod = gen_batch_decode_aiter_module(
        dtype_q, dtype_kv, dtype_o, head_dim_qk, head_dim_vo
    ).build_and_load()
    run_func = mod.run.default

    @register_custom_op(
        f"flashinfer::{uri}_run",
        mutates_args=("paged_k_cache", "paged_v_cache", "o"),
    )
    def run_batch_decode_aiter(
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        so_path: str,
        func_name: str,
        max_context_len: int,
        partition_size: int,
        sliding_window: int,
        logits_soft_cap: float,
        sm_scale: float,
        max_blocks_per_seq: int,
    ) -> None:
        run_func(
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            so_path,
            func_name,
            max_context_len,
            partition_size,
            sliding_window,
            logits_soft_cap,
            sm_scale,
            max_blocks_per_seq,
        )

    @register_fake_op(f"flashinfer::{uri}_run")
    def _fake_run_batch_decode_aiter(
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        so_path: str,
        func_name: str,
        max_context_len: int,
        partition_size: int,
        sliding_window: int,
        logits_soft_cap: float,
        sm_scale: float,
        max_blocks_per_seq: int,
    ) -> None:
        pass

    return SimpleNamespace(run=run_batch_decode_aiter)


# Mapping from torch dtype to AITER's pa_v1 C++ template `dtype` / `kv_dtype` strings.
_AITER_DTYPE_STR = {
    torch.float16: "_Float16",
    torch.bfloat16: "__hip_bfloat16",
}

# PA v1 splits the KV sequence into partitions of this many tokens before reduction.
_AITER_PA_V1_PARTITION_SIZE = 256

# One-time-per-device warning that any return_lse=True call against an AITER-planned
# wrapper will be dispatched through the FA2 shadow plan (AITER PA v1 does not output LSE).
_aiter_lse_fallback_warned: set[torch.device] = set()

# One-time-per-device warning about AITER decode's capture-at-max-seq-len contract
# under CUDA-graph capture (opt-in via explicit backend="aiter").
_aiter_graph_capture_warned: set[torch.device] = set()


def _aiter_graph_capacity_overrun(
    indptr_host: torch.Tensor,
    last_page_len_host: torch.Tensor,
    page_size: int,
    capacity: int,
) -> Optional[str]:
    """Describe how this batch exceeds a declared graph capacity, or None if it fits.

    Derives lengths from indptr/last_page_len the way the PA v1 block-table kernel
    does — a caller-supplied ``seq_lens`` can understate them, and the block table
    is sized from the capacity, so trusting it would let the kernel read past a row.
    """
    kv_lens = get_seq_lens(indptr_host, last_page_len_host, page_size)
    max_kv_len = int(kv_lens.max().item()) if kv_lens.numel() else 0
    npages = indptr_host[1:].to(torch.int64) - indptr_host[:-1].to(torch.int64)
    max_blocks = int(npages.max().item()) if npages.numel() else 0
    width = ceil_div(capacity, page_size)
    if max_kv_len > capacity:
        return (
            f"batch has kv_len={max_kv_len} exceeding the declared "
            f"max_seq_len={capacity}"
        )
    if max_blocks > width:
        # Reachable with last_page_len=0, where npages is one more than kv_len implies.
        return (
            f"batch has {max_blocks} pages/seq exceeding the {width} the declared "
            f"max_seq_len={capacity} allows at page_size={page_size}"
        )
    return None


def _aiter_pa_v1_resolve(
    *,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
    page_size: int,
    max_context_len: int,
    logits_soft_cap: float,
    sliding_window: int,
    partition_size: int = _AITER_PA_V1_PARTITION_SIZE,
    warp_size: int = 64,
):
    """Invoke aiter's pa_v1.compile() to produce the variant .so for this problem,
    then return (so_path, func_name) extracted from the returned ctypes function.

    Importing aiter.csrc.cpp_itfs.pa.pa_v1 is deferred so that flashinfer can be
    imported on machines without AITER installed.
    """
    from csrc.cpp_itfs.pa.pa_v1 import compile as aiter_pa_v1_compile  # type: ignore
    from csrc.cpp_itfs.utils import BUILD_DIR  # type: ignore

    if dtype_q not in _AITER_DTYPE_STR:
        raise ValueError(f"AITER PA v1 supports fp16/bf16 only; got dtype_q={dtype_q}")
    if dtype_kv not in _AITER_DTYPE_STR:
        raise ValueError(
            f"AITER PA v1 supports fp16/bf16 only; got dtype_kv={dtype_kv}"
        )

    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )
    gqa_ratio = num_qo_heads // num_kv_heads
    max_num_partitions = (max_context_len + partition_size - 1) // partition_size
    npar_loops = (max_num_partitions + warp_size - 1) // warp_size

    func = aiter_pa_v1_compile(
        gqa_ratio=gqa_ratio,
        head_size=head_dim,
        npar_loops=npar_loops,
        dtype=_AITER_DTYPE_STR[dtype_q],
        kv_dtype=_AITER_DTYPE_STR[dtype_kv],
        fp8_kv_dtype="auto",
        out_dtype=_AITER_DTYPE_STR[dtype_o],
        block_size=page_size,
        alibi_enabled=False,
        logits_soft_cap_enabled=(logits_soft_cap > 0),
        partition_size=partition_size,
        mtp=1,
        sliding_window_enabled=(sliding_window > 0),
    )
    # func is a ctypes._FuncPtr with .__name__ = entry symbol.
    # AITER stores the variant .so at BUILD_DIR/<folder>/lib.so where folder == func_name
    # (compile_template_op default). This convention is the authoritative way to recover
    # the .so path post-compile, since ctypes._FuncPtr doesn't expose the CDLL path
    # consistently across CPython versions.
    func_name = func.__name__
    so_path = f"{BUILD_DIR}/{func_name}/lib.so"
    return so_path, func_name


# maxsize bounds a key that carries per-request max_context_len; only
# ceil(max_context_len / (partition_size * warp_size)) reaches the compiler, so
# the entries are near-duplicates. Caching the success also spares every later
# plan() a round trip through AITER's compile() machinery.
@functools.lru_cache(maxsize=128)
def _aiter_pa_v1_available(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
    page_size: int,
    max_context_len: int,
    logits_soft_cap: float,
    sliding_window: int,
    partition_size: int,
    device_idx: int,
) -> Tuple[Optional[Tuple[str, str]], Optional[str]]:
    """Resolve AITER PA v1 for this problem; returns ((so_path, func_name), None)
    or (None, reason) when this AITER install cannot build the variant.

    device_idx is in the key but unused: the resolved .so is arch-specific, and
    this host can hold gfx942 and gfx950 at once. Positional-only by convention,
    since lru_cache keys kwargs by insertion order.
    """
    del device_idx
    try:
        resolved = _aiter_pa_v1_resolve(
            dtype_q=dtype_q,
            dtype_kv=dtype_kv,
            dtype_o=dtype_o,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_context_len=max_context_len,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            partition_size=partition_size,
        )
    except Exception as e:
        return None, handle_aiter_probe_failure(e, op="batch_decode")
    return resolved, None


def single_decode_with_kv_cache_with_jit_module(
    jit_module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    kv_layout: str = "NHD",
    window_left: int = -1,
    return_lse: bool = False,
):
    device = q.device
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, device)
    o = torch.empty_like(q)
    if return_lse:
        lse = torch.empty((q.size(0)), dtype=torch.float32, device=device)
    else:
        lse = None
    jit_module.run.default(
        q,
        k,
        v,
        tmp,
        o,
        lse,
        TensorLayout[kv_layout].value,
        window_left,
        *args,
    )
    return o


@overload
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Decode attention with KV Cache for single request, return attention output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[num_qo_heads, head_dim]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The value tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if
        :attr:`kv_layout` is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Defaults to ``NONE``.
    use_tensor_cores: bool
        Whether to use tensor cores for the computation. Will be faster for large group
        size in grouped query attention. Defaults to ``False``.
    q_scale : Optional[float]
        The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
    k_scale : Optional[float]
        The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
    v_scale : Optional[float]
        The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
    window_left : int
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.
    logits_soft_cap : Optional[float]
        The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
        provided, will be set to ``0``. If greater than 0, the logits will be capped according to
        formula:
        :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
        where :math:`x` is the input logits.
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[num_qo_heads, head_dim_vo]``.
        * The log sum exp value, shape: ``[num_qo_heads]``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.single_decode_with_kv_cache(q, k, v)
    >>> o.shape
    torch.Size([32, 128])

    Note
    ----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    _check_pos_encoding_mode(pos_encoding_mode)
    _check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, q.device)
    head_dim = q.shape[-1]
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    if q_scale is not None:
        sm_scale *= q_scale
    if k_scale is not None:
        sm_scale *= k_scale
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    num_qo_heads = q.shape[0]

    lse = None
    if return_lse:
        lse = torch.empty((num_qo_heads,), dtype=torch.float32, device=q.device)

    if use_tensor_cores:
        out = torch.empty_like(q.unsqueeze(0))
        get_single_prefill_module(
            "fa2",
            q.dtype,
            k.dtype,
            q.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            False,  # use_fp16_qk_reduction
        ).run(
            q.unsqueeze(0),
            k,
            v,
            tmp,
            out,
            lse.unsqueeze(0) if lse is not None else None,
            MaskMode.NON_CAUSAL.value,
            TensorLayout[kv_layout].value,
            window_left,
            None,  # packed_custom_mask
            _get_cache_alibi_slopes_buf(num_qo_heads, q.device),
            logits_soft_cap,
            sm_scale,
            None,  # scale_q, not supported yet
            None,  # scale_k
            None,  # scale_v
            rope_scale,
            rope_theta,
        )
        out = out.squeeze(0)
        if return_lse:
            lse = lse.squeeze(0)
    else:
        out = torch.empty_like(q)
        get_single_decode_module(
            q.dtype,
            k.dtype,
            q.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
        ).run(
            q,
            k,
            v,
            tmp,
            out,
            lse,
            _get_cache_alibi_slopes_buf(num_qo_heads, q.device),
            TensorLayout[kv_layout].value,
            window_left,
            logits_soft_cap,
            sm_scale,
            rope_scale,
            rope_theta,
        )

    if v_scale is not None:
        # TODO(Zihao): fused into kernel
        if out.itemsize == 1:
            out = (out.to(float) * v_scale).to(out.dtype)
        else:
            out *= v_scale
    if return_lse:
        return out, lse
    else:
        return out


class BatchDecodeWithPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with paged kv-cache (first proposed in
    `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> decode_wrapper.plan(
    ...     kv_page_indptr,
    ...     kv_page_indices,
    ...     kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     pos_encoding_mode="NONE",
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = decode_wrapper.run(q, kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's batch decode attention creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
        max_seq_len: Optional[int] = None,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor. Must be initialized to 0 for its first use.
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        paged_kv_indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        backend : str
            The implementation backend, could be ``auto``/``fa2``/``aiter``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability. ``aiter`` selects AMD AITER's
            paged_attention_v1 kernel and requires gfx942/gfx950 plus NHD layout, fp16/bf16,
            no positional encoding, and ``use_tensor_cores=False``.

            Notes on AITER-specific behavior:

            * ``use_cuda_graph=True`` IS supported with ``backend="aiter"``, but the
              launch grid and ``.so`` variant are fixed at capture time: without
              ``max_seq_len`` you must capture at your maximum sequence length, and
              ``backend="auto"`` stays on ``fa2`` under capture.
            * ``run(..., return_lse=True)`` raises under CUDA-graph capture on this
              backend — the FA2 shadow plan below is not capture-safe.
            * Sliding-window attention (``window_left >= 0``) IS supported by AITER PA v1.
              The wrapper handles the convention difference internally
              (AITER ``sliding_window = window_left + 1``).
            * ``run(..., return_lse=True)`` is supported: AITER PA v1 does not output
              log-sum-exp, so the wrapper transparently dispatches the call through a
              parallel FA2 decode plan that is built lazily on the first such call (so
              AITER-only workloads pay no JIT/plan cost). A one-time-per-device warning
              is emitted on that first call so the backend switch is not silent.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.

        max_seq_len : Optional[int]
            Graph capacity for the AITER decode backend: the largest per-sequence KV
            length that will ever be replayed. Sizing from it rather than the planned
            batch decouples capture from replay, and lets ``auto`` pick AITER under
            capture. Requires ``use_cuda_graph=True``. Note the PA v1 partition
            workspace scales with this value, so an over-generous capacity costs memory.

            ``plan()`` only sees batches routed through it — a replay that writes the
            persistent buffers directly is not checked against the capacity, and an
            over-length replay is silently clamped to the declared max_seq_len (truncated attention).
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            if use_tensor_cores:
                self._jit_module = get_batch_prefill_jit_module(
                    jit_args[0],
                    gen_customize_batch_prefill_module(
                        "fa2", *jit_args
                    ).build_and_load(),
                )
            else:
                self._jit_module = get_batch_decode_jit_module(
                    jit_args[0],
                    gen_customize_batch_decode_module(*jit_args).build_and_load(),
                )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._kv_lens_buffer: Optional[torch.Tensor] = None

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._use_tensor_cores = use_tensor_cores
        self._use_cuda_graph = use_cuda_graph

        if use_tensor_cores:
            if use_cuda_graph:
                # NOTE(Zihao): if once created, no need to update it in plan/run
                self._qo_indptr_buf = torch.arange(
                    self._fixed_batch_size + 1,
                    dtype=torch.int32,
                    device=float_workspace_buffer.device,
                )
        self._backend = backend
        # plan() overwrites _backend with the concrete choice; a later plan()
        # with a longer max_context_len needs a different AITER variant, so it
        # still has to know the caller asked for auto.
        self._backend_requested = backend
        self._backend_fallback_reason: Optional[str] = None
        # Set when the last plan() demoted off AITER purely because that batch exceeded
        # the declared graph capacity — a per-batch reason, so it is re-evaluated.
        self._backend_capacity_demoted = False
        # AITER-under-CUDA-graph capacity: when set, size AITER's grid/.so/workspace
        # from this max per-seq KV length (capture-order-independent) instead of the
        # capture-time plan. Only meaningful with use_cuda_graph=True on the AITER path.
        if max_seq_len is not None:
            if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool):
                raise TypeError(f"max_seq_len must be an int, got {type(max_seq_len)}")
            if max_seq_len <= 0:
                raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
            if not use_cuda_graph:
                raise ValueError(
                    "max_seq_len is a CUDA-graph capacity hint and requires "
                    "use_cuda_graph=True; drop it for eager decode."
                )
        self._aiter_graph_max_seq_len = max_seq_len
        # ROCm never supports PDL; cache once to avoid per-call device property lookup.
        self._pdl_supported = device_support_pdl(float_workspace_buffer.device)

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    @property
    def backend(self) -> str:
        """The backend in use -- concrete after :meth:`plan`, ``"auto"`` before it."""
        return self._backend

    @property
    def backend_fallback_reason(self) -> Optional[str]:
        """Why ``auto`` declined AITER, or ``None`` if it did not.

        Set only by the ``auto`` constraint check; an explicit ``backend=`` and
        the ``use_tensor_cores`` / graph-capture short-circuits leave it ``None``.
        """
        return self._backend_fallback_reason

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> None:
        r"""Plan batch decode for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to
            ``q_data_type``. Defaults to ``None``.
        data_type: Optional[Union[str, torch.dtype]]
            The data type of both the query and key/value tensors. Defaults to torch.float16.
            data_type is deprecated, please use q_data_type and kv_data_type instead.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        seq_lens: Optional[torch.Tensor]
            A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``.
        block_tables: Optional[torch.Tensor]
            A uint32 2D tensor indicating the block table of each prompt. shape: ``[batch_size, max_num_blocks_per_seq]``.
        fixed_split_size: Optional[int]
            Not used on ROCm/HIP (accepted for API compatibility with CUDA).
        disable_split_kv: bool
            Not used on ROCm/HIP (accepted for API compatibility with CUDA).

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        self._workspace_size = (
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size()
        )

        batch_size = len(last_page_len)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        indptr_host = indptr.to("cpu")
        last_page_len_host = last_page_len.to("cpu")

        # An over-capacity demotion is a property of one batch, not of the device, so
        # unlike the capability-driven resolution it must not stick: re-resolve.
        if self._backend_capacity_demoted:
            self._backend = self._backend_requested
            self._backend_capacity_demoted = False
            self._backend_fallback_reason = None

        # Validate a declared graph capacity before touching any persistent buffer:
        # the caller may keep replaying its captured graph, so a rejected batch has to
        # leave the wrapper exactly as it was.
        over_capacity_reason = None
        if (
            self._aiter_graph_max_seq_len is not None
            and not self.use_tensor_cores
            and self._backend in ("auto", "aiter")
        ):
            over = _aiter_graph_capacity_overrun(
                indptr_host,
                last_page_len_host,
                page_size,
                int(self._aiter_graph_max_seq_len),
            )
            if over is not None:
                if self._backend_requested == "auto":
                    over_capacity_reason = over
                else:
                    raise ValueError(
                        f"AITER graph decode: {over}; increase max_seq_len."
                    )

        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(indices)].copy_(
                indices, non_blocking=(indices.device == self.device) and non_blocking
            )
        else:
            self._paged_kv_indptr_buf = indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        self._batch_size = batch_size
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._block_tables: Optional[torch.Tensor] = block_tables
        self._max_kv_len: Optional[int] = None
        self._page_size: int = page_size

        if seq_lens is None:
            kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)
        else:
            kv_lens_arr_host = seq_lens.cpu()

        # Resolve auto → concrete backend. AITER decode requires use_tensor_cores=False
        # (the AITER PA v1 kernel handles its own dispatch internally). Under CUDA-graph
        # capture, AITER's launch grid and .so variant are fixed at the shapes seen when
        # the graph is captured, whereas fa2's graph path is capacity-based. So `auto`
        # only selects AITER under capture when a `max_seq_len` capacity was declared
        # (letting us size AITER from that capacity → capture-order-independent, like
        # fa2); without it, `auto` stays on fa2.
        resolved_from_auto = self._backend_requested == "auto"
        if over_capacity_reason is not None:
            # auto never fails a plan for a backend-capability reason; fa2's graph
            # path is length-agnostic, so serve the over-long batch there. Flagged so
            # the next in-capacity plan() resolves back to AITER.
            self._backend = "fa2"
            self._backend_fallback_reason = over_capacity_reason
            self._backend_capacity_demoted = True
        elif self._backend == "auto":
            graph_without_capacity = (
                self.is_cuda_graph_enabled and self._aiter_graph_max_seq_len is None
            )
            if self.use_tensor_cores or graph_without_capacity:
                self._backend = "fa2"
            else:
                self._backend, self._backend_fallback_reason = (
                    _auto_select_prefill_backend(
                        self.device,
                        dtype_q=q_data_type,
                        dtype_kv=kv_data_type,
                        kv_layout=self._kv_layout,
                        has_custom_mask=False,
                        head_dim_qk=head_dim,
                        head_dim_vo=head_dim,
                        pos_encoding_mode=pos_encoding_mode,
                        # Decode borrows the prefill selector, but it is a distinct
                        # capability row -- the gates are not the same op.
                        op="batch_decode",
                    )
                )
        if self._backend == "aiter":
            if self.use_tensor_cores:
                raise ValueError("AITER decode backend requires use_tensor_cores=False")
            if self._kv_layout != "NHD":
                raise ValueError(
                    f"AITER decode backend requires NHD kv_layout, got {self._kv_layout!r}"
                )
            if pos_encoding_mode != "NONE":
                raise ValueError(
                    f"AITER decode backend requires pos_encoding_mode='NONE', "
                    f"got {pos_encoding_mode!r}"
                )
            if self._aiter_graph_max_seq_len is not None:
                # Size the grid / .so variant / partition workspace from the declared
                # capacity rather than this batch, so capture and replay decouple.
                # Validated before the buffer writes above. Round up to a whole page so
                # this matches the bound the build kernel clamps context_lens to.
                aiter_max_blocks_per_seq = ceil_div(
                    int(self._aiter_graph_max_seq_len), page_size
                )
                max_kv_len = aiter_max_blocks_per_seq * page_size
            else:
                if (
                    self.is_cuda_graph_enabled
                    and self.device not in _aiter_graph_capture_warned
                ):
                    _aiter_graph_capture_warned.add(self.device)
                    logger.warning(
                        "AITER decode under CUDA-graph capture without max_seq_len: the "
                        "launch grid and kernel variant are fixed at the shapes seen when "
                        "the graph is captured. Capture at your maximum sequence length — "
                        "replays with longer sequences will be incorrect. Pass max_seq_len "
                        "to the wrapper for capture-order-independent capture, or use "
                        "backend='fa2'."
                    )
                max_kv_len = int(max(kv_lens_arr_host).item())
                # max blocks per seq across the batch — sizes the dense block_tables.
                npages_arr = indptr_host[1:].to(torch.int64) - indptr_host[:-1].to(
                    torch.int64
                )
                aiter_max_blocks_per_seq = (
                    int(npages_arr.max().item()) if batch_size > 0 else 0
                )
            # Convention mapping: flashinfer's window_left = W means the query at
            # position kv_len-1 sees kv positions [kv_len-1-W, kv_len-1] (W+1 tokens).
            # AITER's sliding_window = S masks positions where local_token_idx + i <
            # context_len - S, so it admits S tokens. Therefore S = W + 1. The sentinel
            # window_left == -1 (disabled) maps to S = 0, which is also AITER's compile-
            # time "disabled" flag (sliding_window_enabled = (S > 0)).
            aiter_sliding_window = 0 if window_left == -1 else window_left + 1

            # Bootstrap & resolve (so_path, func_name) — this triggers AITER's own JIT
            # build of the variant .so on first call for this template-param combination.
            # Held in locals and committed only on success: a demotion to fa2 below must
            # leave no half-written _aiter_* state, since run() gates on _backend alone.
            resolved = None
            reason = None
            with _aiter_bootstrap_lock:
                if resolved_from_auto:
                    resolved, reason = _aiter_pa_v1_available(
                        q_data_type,
                        kv_data_type,
                        q_data_type,
                        head_dim,
                        num_qo_heads,
                        num_kv_heads,
                        page_size,
                        max_kv_len,
                        logits_soft_cap,
                        aiter_sliding_window,
                        _AITER_PA_V1_PARTITION_SIZE,
                        self.device.index if self.device.index is not None else 0,
                    )
                else:
                    resolved = _aiter_pa_v1_resolve(
                        dtype_q=q_data_type,
                        dtype_kv=kv_data_type,
                        dtype_o=q_data_type,
                        head_dim=head_dim,
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        page_size=page_size,
                        max_context_len=max_kv_len,
                        logits_soft_cap=logits_soft_cap,
                        sliding_window=aiter_sliding_window,
                        partition_size=_AITER_PA_V1_PARTITION_SIZE,
                    )

            if reason is None:
                self._max_kv_len = max_kv_len
                self._aiter_max_blocks_per_seq = aiter_max_blocks_per_seq
                self._aiter_partition_size = _AITER_PA_V1_PARTITION_SIZE
                self._aiter_sliding_window = aiter_sliding_window
                self._cached_module = get_batch_decode_aiter_module(
                    q_data_type, kv_data_type, q_data_type, head_dim, head_dim
                )
                self._aiter_so_path, self._aiter_func_name = resolved
                # Skip FA2-style plan_info; AITER kernel doesn't use it.
                self._plan_info = plan_info_vec_as_tensor(
                    [], device=self._float_workspace_buffer.device
                )

                # FA2 shadow plan for return_lse=True; built lazily (AITER PA v1 has no LSE).
                self._fa2_lse_module: Optional[Any] = None
                self._fa2_lse_plan_info: Optional[torch.Tensor] = None
                self._fa2_lse_build_args = (
                    q_data_type,
                    kv_data_type,
                    indptr.dtype,
                    head_dim,
                    pos_encoding_mode,
                    window_left,
                    logits_soft_cap,
                    indptr_host,
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    page_size,
                )

                self._pos_encoding_mode = pos_encoding_mode
                self._window_left = window_left
                self._logits_soft_cap = logits_soft_cap
                self._sm_scale = sm_scale
                self._rope_scale = rope_scale
                self._rope_theta = rope_theta
                return

            # auto promised fa2 when AITER cannot serve; fall through to it below.
            # Reachable only from the auto path, which excludes use_tensor_cores, so
            # the landing site is the plain decode branch. It may be graph-enabled
            # now that a declared max_seq_len lets auto pick AITER under capture —
            # fa2's plain branch is capacity-based, so it serves that case too.
            self._backend = "fa2"
            self._backend_fallback_reason = reason

        if self.use_tensor_cores:
            self._max_kv_len = max(kv_lens_arr_host).item()
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_prefill_module(
                    "fa2",
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                    False,  # use_fp16_qk_reduction
                )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                indptr_host,
                kv_lens_arr_host,
                batch_size,  # total_num_rows
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                head_dim,
                head_dim,
                False,  # causal
            )
            self._plan_info = plan_info_vec_as_tensor(
                self._plan_info, device=self._float_workspace_buffer.device
            )
        else:
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_decode_module(
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                indptr_host,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                window_left,
                logits_soft_cap,
                head_dim,
                head_dim,
                torch.empty(0, dtype=q_data_type),
                torch.empty(0, dtype=kv_data_type),
            )
            self._plan_info = plan_info_vec_as_tensor(
                self._plan_info, device=self._float_workspace_buffer.device
            )

        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def _ensure_fa2_lse_plan(self) -> None:
        if self._fa2_lse_plan_info is not None:
            return
        (
            q_data_type,
            kv_data_type,
            indptr_dtype,
            head_dim,
            pos_encoding_mode,
            window_left,
            logits_soft_cap,
            indptr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
        ) = self._fa2_lse_build_args

        if self.device not in _aiter_lse_fallback_warned:
            _aiter_lse_fallback_warned.add(self.device)
            logger.warning(
                "AITER decode wrapper on device %s received a return_lse=True call; "
                "dispatching through an FA2 decode shadow plan (AITER PA v1 does not "
                "output log-sum-exp). Expect a per-call performance cliff vs. "
                "return_lse=False on the same wrapper.",
                self.device,
            )

        self._fa2_lse_module = get_batch_decode_module(
            q_data_type,
            kv_data_type,
            q_data_type,
            indptr_dtype,
            head_dim,
            head_dim,
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,
            logits_soft_cap > 0,
        )
        fa2_lse_plan = self._fa2_lse_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            indptr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            window_left,
            logits_soft_cap,
            head_dim,
            head_dim,
            torch.empty(0, dtype=q_data_type),
            torch.empty(0, dtype=kv_data_type),
        )
        self._fa2_lse_plan_info = plan_info_vec_as_tensor(
            fa2_lse_plan, device=self._float_workspace_buffer.device
        )

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: this function is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q, paged_kv_cache, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
        )

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        q_len_per_req: Optional[int] = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch decode attention between query and paged kv cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.
        *args
            Additional arguments for the custom kernel.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention scores, defaults to ``False``.
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        q_len_per_req : int
            The number of query tokens per request, if not provided, will be set to ``1``.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * attention output, shape: ``[batch_size, num_qo_heads, head_dim]``
            * logsumexp of attention scores, shape: ``[batch_size, num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = self._pdl_supported
        if enable_pdl:
            import warnings

            warnings.warn(
                "enable_pdl is not supported in the HIP/ROCm backend and will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            enable_pdl = False
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        if self._kv_layout == "NHD":
            page_size = k_cache.shape[1]
        else:
            page_size = k_cache.shape[2]
        _check_cached_qkv_data_type(
            q, k_cache, self._cached_q_data_type, self._cached_kv_data_type
        )

        pos_encoding_mode = self._pos_encoding_mode
        window_left = self._window_left if window_left is None else window_left
        # NOTE(Siyuan): since window_left is appeared in the plan function, we
        # need to make sure it is the same as the one in the plan function.
        # Remove this check if the backend supports dynamic window_left.
        assert window_left == self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q)
        else:
            check_shape_dtype_device(out, q.shape, q.dtype, q.device, "out")

        if self._backend == "aiter":
            if return_lse:
                if torch.cuda.is_current_stream_capturing():
                    # The shadow plan is built lazily and re-nulled by every plan(), so
                    # capturing it would JIT-load inside the graph and later replay a
                    # freed plan_info. Eager calls are fine, hence the capture check
                    # rather than is_cuda_graph_enabled.
                    raise ValueError(
                        "return_lse=True is not supported for AITER decode under "
                        "CUDA-graph capture (PA v1 emits no LSE and its FA2 shadow "
                        "plan is not capture-safe); use backend='fa2'."
                    )
                self._ensure_fa2_lse_plan()
                self._fa2_lse_module.run(
                    self._float_workspace_buffer,
                    self._int_workspace_buffer,
                    self._fa2_lse_plan_info,
                    q,
                    k_cache,
                    v_cache,
                    self._paged_kv_indptr_buf,
                    self._paged_kv_indices_buf,
                    self._paged_kv_last_page_len_buf,
                    out,
                    lse,
                    TensorLayout[self._kv_layout].value,
                    window_left,
                    enable_pdl,
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    logits_soft_cap,
                    sm_scale,
                    rope_scale,
                    rope_theta,
                )
                if v_scale is not None:
                    if is_float8(out):
                        out = (out.to(torch.float32) * v_scale).to(out.dtype)
                    else:
                        out *= v_scale
                return (out, lse)
            self._cached_module.run(
                q,
                k_cache,
                v_cache,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                self._aiter_so_path,
                self._aiter_func_name,
                self._max_kv_len or 0,
                self._aiter_partition_size,
                self._aiter_sliding_window,
                logits_soft_cap,
                sm_scale,
                self._aiter_max_blocks_per_seq,
            )
            if v_scale is not None:
                if is_float8(out):
                    out = (out.to(torch.float32) * v_scale).to(out.dtype)
                else:
                    out *= v_scale
            return (out, lse) if return_lse else out

        if self.use_tensor_cores:
            assert self._plan_info is not None, (
                "plan info is not initialized; call plan() first"
            )
            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k_cache,
                v_cache,
                self._qo_indptr_buf,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                MaskMode.NON_CAUSAL.value,
                TensorLayout[self._kv_layout].value,
                window_left,
                enable_pdl,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    None,  # packed_custom_mask
                    None,  # mask_indptr_buf
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    None,  # maybe_prefix_len_ptr
                    None,  # maybe_token_pos_in_items_ptr
                    None,  # maybe_max_item_len_ptr
                    logits_soft_cap,
                    sm_scale,
                    None,  # scale_q, not supported yet
                    None,  # scale_k
                    None,  # scale_v
                    rope_scale,
                    rope_theta,
                    0,  # token_pos_in_items_len
                    self._workspace_size,
                    self._num_qo_heads,
                    self._num_kv_heads,
                    self._block_tables,
                    self._kv_lens_buffer,
                    page_size,
                    None,  # max_q_len (decode: single token)
                    self._max_kv_len,
                    None,  # batch_size
                    None,  # cum_seq_lens_q
                    None,  # cum_seq_lens_kv
                    sinks,
                ]

            self._cached_module.paged_run(*run_args)
        else:
            plan_info = self._plan_info
            assert plan_info is not None, "plan info is not initialized"

            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                plan_info,
                q,
                k_cache,
                v_cache,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                window_left,
                enable_pdl,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    logits_soft_cap,
                    sm_scale,
                    rope_scale,
                    rope_theta,
                ]

            self._cached_module.run(*run_args)
        if v_scale is not None:
            # TODO(Zihao): fused into kernel
            if is_float8(out):
                out = (out.to(torch.float32) * v_scale).to(out.dtype)
            else:
                out *= v_scale

        return (out, lse) if return_lse else out

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: this function is deprecated, please use :meth:`run_return_lse` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q,
            paged_kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


class CUDAGraphBatchDecodeWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    r"""CUDAGraph-compatible Wrapper class for decode attention with paged kv-cache (first
    proposed in `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Note that this wrapper may not be as efficient as :class:`BatchDecodeWithPagedKVCacheWrapper`
    because we won't dispatch to different kernels for different batch sizes/sequence lengths/etc
    to accommodate the CUDAGraph requirement.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Note
    ----
    The :meth:`plan` method could not be captured by CUDAGraph.

    See Also
    --------
    :class:`BatchDecodeWithPagedKVCacheWrapper`
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        indptr_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        last_page_len_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_tensor_cores: bool = False,
        backend: str = "auto",
        max_seq_len: Optional[int] = None,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer on GPU used to store auxiliary data structures,
            recommended size is 128MB, the device of the workspace buffer should be the
            same as the device of the input tensors.

        indptr_buffer : torch.Tensor
            The user reserved buffer on GPU to store the indptr of the paged kv cache, should
            be large enough to store the indptr of maximum batch size (``[max_batch_size + 1]``)
            during the lifecycle of this wrapper.

        indices_buffer : torch.Tensor
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.

        last_page_len_buffer : torch.Tensor
            The user reserved buffer on GPU to store the number of entries in the last page,
            should be large enough to store the maximum batch size (``[max_batch_size]``)
            during the lifecycle of this wrapper.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        backend : str
            Decode backend (``auto``/``fa2``/``aiter``). Defaults to ``auto``. With
            ``max_seq_len`` set, ``auto`` may select AITER under graph capture.

        max_seq_len : Optional[int]
            Capacity hint (max per-seq KV length) enabling capture-order-independent
            AITER decode under CUDA graph. See
            :class:`BatchDecodeWithPagedKVCacheWrapper` for details.
        """
        super().__init__(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=indptr_buffer,
            paged_kv_indices_buffer=indices_buffer,
            paged_kv_last_page_len_buffer=last_page_len_buffer,
            backend=backend,
            max_seq_len=max_seq_len,
        )
