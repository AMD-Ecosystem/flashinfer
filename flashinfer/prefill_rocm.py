"""
Copyright (c) 2023 by FlashInfer team.
Copyright (c) 2026 by Advance Micro Devices Inc.

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
import logging
import math
import os
import threading
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import torch
from .aiter_utils import handle_aiter_probe_failure
from .arch_caps import capability_reason, require_capability
from .jit.core import logger
from .jit import (
    gen_batch_prefill_module,
    gen_customize_batch_prefill_module,
    gen_single_prefill_module,
    get_batch_prefill_uri,
    get_single_prefill_uri,
)
from .page import get_seq_lens
from .quantization import packbits, segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    device_support_pdl,
    is_float8,
    plan_info_vec_as_tensor,
    register_custom_op,
    register_fake_op,
)


# Two independent versions — do not merge them. The first is the release that
# widened native paged-prefill to {128, 256, 1024}; changing it changes which page
# sizes we try. The second is the newest release we have actually validated against;
# bumping it must not silently move the support boundary.
_AITER_NATIVE_PAGING_SINCE = "0.1.10"
_AITER_LAST_VALIDATED = "0.1.10"


@functools.cache
def _aiter_native_page_sizes() -> frozenset:
    """Page sizes AITER is *expected* to serve without a flat-gather.

    Only a hint — the caller must confirm with _aiter_native_paging_available()
    before relying on it. Every page size works via flat-gather regardless.
    """
    try:
        from importlib.metadata import version
        from packaging.version import Version

        installed = Version(version("amd-aiter"))
        if installed > Version(_AITER_LAST_VALIDATED):
            # Debug, not a warning: this fires for every build newer than the pin,
            # which is the common case going forward, and it is speculative — the
            # probe warns with the real error if support is actually missing.
            logger.debug(
                "amd-aiter %s is newer than the last validated version (%s); "
                "native paged-prefill support will be probed at plan() time.",
                installed,
                _AITER_LAST_VALIDATED,
            )
        if installed >= Version(_AITER_NATIVE_PAGING_SINCE):
            return frozenset({128, 256, 1024})
        return frozenset({16, 1024})
    except (PackageNotFoundError, ValueError):
        return frozenset({16, 1024})


def make_hashable_cache(func):
    """
    Decorator that converts unhashable arguments (like lists) to hashable ones (tuples)
    before applying functools.cache.
    """

    @functools.cache
    def cached_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert unhashable arguments to hashable ones
        hashable_args = []
        for arg in args:
            if isinstance(arg, list):
                hashable_args.append(tuple(arg))
            else:
                hashable_args.append(arg)

        hashable_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                hashable_kwargs[key] = tuple(value)
            else:
                hashable_kwargs[key] = value

        return cached_wrapper(*hashable_args, **hashable_kwargs)

    return wrapper


@make_hashable_cache
def get_customize_batch_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    fp8_enabled: bool = False,
):
    return gen_customize_batch_prefill_module(
        backend,
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim_qk,
        head_dim_vo,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name,
        variant_decl,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
        fp8_enabled,
    ).build_and_load()


@functools.cache
def get_single_prefill_module(backend, *args):
    uri = get_single_prefill_uri(backend, *args)
    module = gen_single_prefill_module(backend, *args).build_and_load()
    run_func = module.run.default

    if backend == "aiter":
        # Skip torch custom-op dispatch for AITER (saves ~3 µs per call).
        # AITER is inference-only on ROCm; torch.compile support not required.
        # AITER ignores custom_mask, alibi, rope, and FP8 scale params.
        def run_single_prefill(
            q,
            k,
            v,
            tmp,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            maybe_packed_custom_mask,
            maybe_alibi_slopes,
            logits_soft_cap,
            sm_scale,
            scale_q,
            scale_k,
            scale_v,
            rope_scale,
            rope_theta,
        ):
            run_func(
                q,
                k,
                v,
                tmp,
                o,
                maybe_lse,
                mask_mode,
                layout,
                window_left,
                None,
                None,
                logits_soft_cap,
                sm_scale,
                1.0,
                1.0,  # rope_rcp_scale / rope_rcp_theta ignored by AITER
            )

        return SimpleNamespace(run=run_single_prefill)

    @register_custom_op(
        f"flashinfer::{uri}_run", mutates_args=("tmp", "o", "maybe_lse")
    )
    def run_single_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tmp: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        maybe_packed_custom_mask: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        scale_q: Optional[torch.Tensor],
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
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
            mask_mode,
            layout,
            window_left,
            maybe_packed_custom_mask,
            maybe_alibi_slopes,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
        )

    @register_fake_op(f"flashinfer::{uri}_run")
    def _fake_run_single_prefill(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tmp: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        maybe_packed_custom_mask: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        scale_q: Optional[torch.Tensor],
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        pass

    return SimpleNamespace(run=run_single_prefill)


def _aiter_noop_plan(*args, **kwargs):
    """No-op plan function for the AITER backend.

    AITER does not require a separate planning / workspace-preparation step,
    so we simply return an empty list that is accepted as ``plan_info_vec``
    by the wrapper.
    """
    return []


@functools.cache
def _aiter_ops_importable() -> bool:
    try:
        import aiter.ops  # noqa: F401

        return True
    except Exception:
        return False


def _require_aiter_runtime(device: torch.device, op: str = "batch_prefill") -> None:
    """Raise a clear error when AITER is requested on an unsupported GPU or without the package.

    ``op`` selects the capability row, so a gate that applies to batch prefill
    (such as the ROCm 7.2 causal miscompile on gfx950) does not also block single
    prefill. ``ArchCapabilityError`` subclasses ``RuntimeError``, so the previous
    contract is unchanged for callers.
    """
    require_capability(device, op, "aiter")
    if not _aiter_ops_importable():
        raise ImportError(
            "The 'aiter' package is required for the AITER backend. "
            "Install it via:\n"
            "  git clone --recursive https://github.com/ROCm/aiter.git\n"
            "  cd aiter && python3 setup.py develop"
        )


_aiter_auto_warned: set[tuple[torch.device, str]] = set()


def _auto_select_prefill_backend(
    device: torch.device,
    *,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    kv_layout: str,
    has_custom_mask: bool,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: str = "NONE",
    op: str = "batch_prefill",
) -> Tuple[str, Optional[str]]:
    """Return ``(backend, reason)``: 'aiter' when the GPU and call parameters satisfy
    AITER's constraints, else 'fa2' plus the reason AITER was declined.

    On gfx942/gfx950, checks NHD layout, no custom mask, fp16/bf16, equal dtypes and head dims.
    Falls back to 'fa2' with a one-time warning for each distinct skip reason.

    ``op`` selects the capability row. It matters because the gates are not
    uniform across ops: the ROCm 7.2 causal miscompile on gfx950 applies to batch
    prefill only, so checking a single shared "is AITER supported here" would
    either under- or over-block.

    The warning fires once per ``(device, reason)`` for the process lifetime, so a
    caller that needs the reason on *every* call must read it from the return
    value -- the log carries it only the first time.
    """
    # The capability gate is just one more reason, so an architecture or
    # toolchain that cannot serve this op warns once and falls back like any
    # other unmet constraint. Previously an unsupported architecture returned
    # "fa2" silently, which on CDNA4 would mean a user quietly losing the AITER
    # path with nothing to explain it.
    reason: Optional[str] = capability_reason(device, op, "aiter")
    if reason is None:
        if kv_layout != "NHD":
            reason = f"kv_layout={kv_layout!r} (AITER requires NHD)"
        elif has_custom_mask:
            reason = "custom mask (not supported by AITER)"
        elif dtype_q not in (torch.float16, torch.bfloat16):
            reason = f"dtype={dtype_q} (AITER requires fp16/bf16)"
        elif dtype_q != dtype_kv:
            reason = f"dtype_q={dtype_q} != dtype_kv={dtype_kv} (AITER requires equal dtypes)"
        elif head_dim_qk != head_dim_vo:
            reason = f"head_dim_qk={head_dim_qk} != head_dim_vo={head_dim_vo} (AITER requires equal head dims)"
        elif pos_encoding_mode != "NONE":
            reason = (
                f"pos_encoding_mode={pos_encoding_mode!r} (AITER only supports NONE)"
            )

    if reason is not None:
        key = (device, reason)
        if key not in _aiter_auto_warned:
            _aiter_auto_warned.add(key)
            logger.warning("auto backend falling back to fa2: %s", reason)
        return "fa2", reason

    if not _aiter_ops_importable():
        reason = (
            "aiter package not installed "
            "(see https://github.com/ROCm/aiter for install instructions)"
        )
        key = (device, "import_failed")
        if key not in _aiter_auto_warned:
            _aiter_auto_warned.add(key)
            logger.warning("auto backend falling back to fa2: %s", reason)
        return "fa2", reason

    return "aiter", None


_aiter_bootstrap_lock = threading.Lock()


@functools.lru_cache(maxsize=None)
def _aiter_bootstrap_single_prefill_varlen(
    dtype: torch.dtype,
    causal: bool,
    head_dim: int,
    device_idx: int,
) -> None:
    """Force AITER's lazy JIT to compile mha_varlen_fwd_*.so logits variants.

    Non-logits variants are pre-shipped with the AITER package; the logits
    variants are missing from the pre-built set and must be bootstrapped on
    first use. Used by single-prefill when logits_soft_cap > 0 (which forces the
    varlen .so because mha_fwd has no _logits arm). The .so is split by causal
    (mask vs nmask), so build the variant matching the actual request.
    """
    device = torch.device("cuda", device_idx)
    q = torch.zeros(2, 2, head_dim, dtype=dtype, device=device)
    k = torch.zeros(4, 2, head_dim, dtype=dtype, device=device)
    v = torch.zeros(4, 2, head_dim, dtype=dtype, device=device)
    cu_q = torch.tensor([0, 2], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 4], dtype=torch.int32, device=device)
    scale = head_dim**-0.5
    _common = dict(
        max_seqlen_q=2,
        max_seqlen_k=4,
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=scale,
        logits_soft_cap=0.5,
        zero_tensors=False,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        sink_size=0,
        return_dropout_randval=False,
    )
    for return_lse in (True, False):
        torch.ops.aiter.mha_varlen_fwd(
            q, k, v, cu_q, cu_k, return_softmax_lse=return_lse, **_common
        )


@functools.lru_cache(maxsize=None)
def _aiter_bootstrap_single_prefill_mha_fwd(
    dtype: torch.dtype,
    causal: bool,
    has_lse: bool,
    head_dim: int,
    device_idx: int,
) -> None:
    """Force AITER's lazy JIT to compile mha_fwd_*.so for this (dtype, causal, has_lse) variant.

    Unlike mha_varlen_fwd, mha_fwd ships no prebuilt .so files in the aiter
    package; every (dtype, causal, has_lse) combination is JIT-built on first
    use, which takes 20+ minutes per variant. Bootstrapping here surfaces the
    build at plan time rather than as a dlopen failure inside the C++ path.
    """
    from aiter.ops.mha import mha_fwd

    device = torch.device("cuda", device_idx)
    # mha_fwd expects (batch, seqlen, nhead, hdim) — non-varlen layout.
    q = torch.zeros(1, 2, 2, head_dim, dtype=dtype, device=device)
    k = torch.zeros(1, 4, 2, head_dim, dtype=dtype, device=device)
    v = torch.zeros(1, 4, 2, head_dim, dtype=dtype, device=device)
    mha_fwd(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=head_dim**-0.5,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        sink_size=0,
        return_softmax_lse=has_lse,
        return_dropout_randval=False,
    )


@functools.lru_cache(maxsize=None)
def _aiter_bootstrap_batch_ragged_prefill(
    dtype: torch.dtype,
    has_logits_cap: bool,
    causal: bool,
    head_dim: int,
    device_idx: int,
) -> None:
    """Trigger AITER's lazy JIT for mha_varlen_fwd_*.so variants used by batch prefill.

    Same .so family as single-prefill (varlen group-mode), but we parameterize causal
    and logits-cap so we can cover non-shipped combos for any (causal, logits) the
    user requests. AITER pre-ships only a subset of the
    (dtype, causal, has_lse, has_logits_cap) family — which subset is not contractual
    and varies by amd-aiter build — so any combination may need a lazy build, including
    no-logits ones. On amd-aiter 0.1.10, for example, bf16 ships only nmask_lse and
    mask_nlse, so both remaining nlogits arms are built here on first use.

    Used by both batch-prefill wrappers that reach this .so family: the ragged wrapper,
    and the paged wrapper's flat-gather path (page sizes outside
    ``_aiter_native_page_sizes()``). Both loop over return_lse here because the .so is
    also split on lse, and plan() cannot know which run() will request.
    """
    device = torch.device("cuda", device_idx)
    q = torch.zeros(2, 2, head_dim, dtype=dtype, device=device)
    k = torch.zeros(4, 2, head_dim, dtype=dtype, device=device)
    v = torch.zeros(4, 2, head_dim, dtype=dtype, device=device)
    cu_q = torch.tensor([0, 2], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 4], dtype=torch.int32, device=device)
    scale = head_dim**-0.5
    _common = dict(
        max_seqlen_q=2,
        max_seqlen_k=4,
        min_seqlen_q=0,
        dropout_p=0.0,
        softmax_scale=scale,
        logits_soft_cap=0.5 if has_logits_cap else 0.0,
        zero_tensors=False,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        sink_size=0,
        return_dropout_randval=False,
    )
    for return_lse in (True, False):
        torch.ops.aiter.mha_varlen_fwd(
            q, k, v, cu_q, cu_k, return_softmax_lse=return_lse, **_common
        )


@functools.lru_cache(maxsize=None)
def _aiter_bootstrap_batch_prefill(
    dtype: torch.dtype,
    has_logits_cap: bool,
    causal: bool,
    has_lse: bool,
    page_size: int,
    head_dim: int,
    device_idx: int,
) -> None:
    """Force AITER's lazy JIT to compile mha_batch_prefill_*.so for this variant."""
    from aiter.ops.mha import mha_batch_prefill_func

    device = torch.device("cuda", device_idx)
    nhead_q, nhead_k, seq_q, seq_k = 2, 2, 2, page_size
    q = torch.zeros(seq_q, nhead_q, head_dim, dtype=dtype, device=device)
    k = torch.zeros(1, page_size, nhead_k, head_dim, dtype=dtype, device=device)
    v = torch.zeros(1, page_size, nhead_k, head_dim, dtype=dtype, device=device)
    cu_seqlens_q = torch.tensor([0, seq_q], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_page_indices = torch.tensor([0], dtype=torch.int32, device=device)
    kv_last_page_lens = torch.tensor([seq_k], dtype=torch.int32, device=device)
    softmax_scale = head_dim**-0.5
    mha_batch_prefill_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        max_seqlen_q=seq_q,
        max_seqlen_k=seq_k,
        softmax_scale=softmax_scale,
        logits_soft_cap=0.5 if has_logits_cap else 0.0,
        causal=causal,
        return_lse=has_lse,
        kv_last_page_lens=kv_last_page_lens,
    )


@functools.lru_cache(maxsize=None)
def _aiter_native_paging_available(
    dtype: torch.dtype,
    has_logits_cap: bool,
    causal: bool,
    page_size: int,
    head_dim: int,
    device_idx: int,
) -> bool:
    """Probe whether AITER really dispatches a native paged kernel for this config.

    _aiter_native_page_sizes() can only report what the *validated* amd-aiter
    release supported. Serving stacks pin an AITER source commit instead, and such
    a build can reject a page size the version predicate claims is native, e.g.
    ``no matching kernel found. page_size=128, num_pages=1, dtype=bf16``. Rather
    than trust the predicate, run the bootstrap — which launches the real kernel —
    and treat a failure as "not available".

    The bootstrap is more than a proxy for what run() does: it is what *produces*
    the mha_batch_prefill_*.so that the C++ path later dlopens. If it cannot build,
    the native path definitionally cannot work — forcing it anyway just trades this
    error for ``AITER .so not found`` inside run().

    Falling back is always correct: the flat-gather path serves every page size.
    It is not free, though — it materializes a contiguous copy of K and V on each
    run() — so the fallback is warned about rather than taken silently. Set
    FLASHINFER_AITER_STRICT=1 to re-raise instead of degrading.

    Bootstraps both has_lse variants, since plan() cannot know which run() needs.
    This is a separate cached function rather than a try/except inlined into plan()
    because functools.lru_cache does not memoize exceptions: inlining would re-launch
    a known-failing kernel, and re-warn, on every plan() call.
    """
    # Drain any pending async error from earlier work *outside* the try, so it is
    # not mistaken for an AITER capability failure and cached as "unsupported".
    torch.cuda.synchronize(device_idx)
    try:
        for has_lse in (True, False):
            _aiter_bootstrap_batch_prefill(
                dtype, has_logits_cap, causal, has_lse, page_size, head_dim, device_idx
            )
        # HIP launches are async, so a device-side failure would otherwise land
        # somewhere unrelated and leave us wrongly committed to the native path.
        torch.cuda.synchronize(device_idx)
    except torch.cuda.OutOfMemoryError:
        # Transient and unrelated to kernel availability — never cache it as
        # "unsupported", or one OOM would downgrade this config for the process.
        raise
    except Exception as e:
        if os.environ.get("FLASHINFER_AITER_STRICT", "0") == "1":
            raise
        logger.warning(
            "AITER has no native paged-prefill kernel for page_size=%d "
            "(dtype=%s, causal=%s, logits_cap=%s): %s. Falling back to the "
            "flat-gather path, which copies K/V per run(). Set "
            "FLASHINFER_AITER_STRICT=1 to raise instead.",
            page_size,
            dtype,
            causal,
            has_logits_cap,
            e,
        )
        return False
    return True


# The two probes below answer a different question from _aiter_native_paging_available:
# not "which AITER kernel", but "can this AITER install produce any kernel for this
# variant". They return the fallback reason instead of a bool because it feeds
# backend_fallback_reason, and it has to be memoized with the outcome or the second
# plan() reports a demotion it cannot explain. Same lru_cache rationale as above:
# exceptions are not memoized, so a failing build must not be retried per plan().


@functools.lru_cache(maxsize=None)
def _aiter_single_prefill_available(
    dtype: torch.dtype,
    causal: bool,
    has_lse: bool,
    has_logits_cap: bool,
    head_dim: int,
    device_idx: int,
) -> Optional[str]:
    """Probe whether AITER can serve single prefill here; None means it can.

    amd-aiter ships no prebuilt mha_fwd .so, so this variant is JIT-built on
    first use and can fail for reasons that are the install's, not the caller's
    — e.g. a site-packages AITER cannot write its built module into.
    """
    torch.cuda.synchronize(device_idx)
    try:
        if has_logits_cap:
            _aiter_bootstrap_single_prefill_varlen(dtype, causal, head_dim, device_idx)
        else:
            _aiter_bootstrap_single_prefill_mha_fwd(
                dtype, causal, has_lse, head_dim, device_idx
            )
        torch.cuda.synchronize(device_idx)
    except Exception as e:
        return handle_aiter_probe_failure(e, op="single_prefill")
    return None


@functools.lru_cache(maxsize=None)
def _aiter_batch_ragged_available(
    dtype: torch.dtype,
    has_logits_cap: bool,
    causal: bool,
    head_dim: int,
    device_idx: int,
) -> Optional[str]:
    """Probe whether AITER can serve the varlen batch-prefill path; None means it can.

    Keyed exactly like _aiter_bootstrap_batch_ragged_prefill, so the paged and
    ragged wrappers share one cache entry per variant.
    """
    torch.cuda.synchronize(device_idx)
    try:
        _aiter_bootstrap_batch_ragged_prefill(
            dtype, has_logits_cap, causal, head_dim, device_idx
        )
        torch.cuda.synchronize(device_idx)
    except Exception as e:
        return handle_aiter_probe_failure(e, op="batch_prefill")
    return None


@functools.cache
def get_batch_prefill_module(backend, *args):
    if backend == "aiter":
        module = gen_batch_prefill_module(backend, *args).build_and_load()
        _c_paged_run = module.paged_run.default
        _c_ragged_run = module.ragged_run.default

        def aiter_paged_run(
            float_workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
            plan_info_vec: List[int],
            q: torch.Tensor,
            paged_k_cache: torch.Tensor,
            paged_v_cache: torch.Tensor,
            qo_indptr: torch.Tensor,
            paged_kv_indptr: torch.Tensor,
            paged_kv_indices: torch.Tensor,
            paged_kv_last_page_len: torch.Tensor,
            o: torch.Tensor,
            maybe_lse: Optional[torch.Tensor],
            mask_mode: int,
            layout: int,
            window_left: int,
            enable_pdl: bool,
            maybe_custom_mask: Optional[torch.Tensor],
            maybe_mask_indptr: Optional[torch.Tensor],
            maybe_alibi_slopes: Optional[torch.Tensor],
            maybe_prefix_len_ptr: Optional[torch.Tensor],
            maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
            maybe_max_item_len_ptr: Optional[torch.Tensor],
            logits_soft_cap: float,
            sm_scale: float,
            scale_q: Optional[torch.Tensor],
            scale_k: Optional[torch.Tensor],
            scale_v: Optional[torch.Tensor],
            rope_scale: float,
            rope_theta: float,
            token_pos_in_items_len: int,
            workspace_size: int,
            num_qo_heads: Optional[int] = None,
            num_kv_heads: Optional[int] = None,
            block_tables: Optional[torch.Tensor] = None,
            kv_lens_buffer: Optional[torch.Tensor] = None,
            page_size: Optional[int] = None,
            max_q_len: Optional[int] = None,
            max_kv_len: Optional[int] = None,
            batch_size: Optional[int] = None,
            cum_seq_lens_q: Optional[torch.Tensor] = None,
            cum_seq_lens_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            aiter_flat_gather_idx: Optional[torch.Tensor] = None,
            aiter_flat_kv_indptr: Optional[torch.Tensor] = None,
        ) -> None:
            _c_paged_run(
                q,
                paged_k_cache,
                paged_v_cache,
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                o,
                maybe_lse,
                mask_mode,
                window_left,
                logits_soft_cap,
                sm_scale,
                page_size,
                max_q_len,
                max_kv_len,
                aiter_flat_gather_idx,
                aiter_flat_kv_indptr,
            )

        def aiter_ragged_run(
            float_workspace_buffer: torch.Tensor,
            int_workspace_buffer: torch.Tensor,
            plan_info_vec: List[int],
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            qo_indptr: torch.Tensor,
            kv_indptr: torch.Tensor,
            o: torch.Tensor,
            maybe_lse: Optional[torch.Tensor],
            mask_mode: int,
            layout: int,
            window_left: int,
            enable_pdl: bool,
            maybe_custom_mask: Optional[torch.Tensor],
            maybe_mask_indptr: Optional[torch.Tensor],
            maybe_alibi_slopes: Optional[torch.Tensor],
            maybe_prefix_len_ptr: Optional[torch.Tensor],
            maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
            maybe_max_item_len_ptr: Optional[torch.Tensor],
            logits_soft_cap: float,
            sm_scale: float,
            rope_scale: float,
            rope_theta: float,
            token_pos_in_items_len: int,
            max_q_len: Optional[int] = None,
            max_kv_len: Optional[int] = None,
        ) -> None:
            if max_q_len is None or max_kv_len is None:
                raise ValueError(
                    "AITER ragged backend requires max_q_len/max_kv_len from plan()"
                )
            _c_ragged_run(
                q,
                k,
                v,
                qo_indptr,
                kv_indptr,
                o,
                maybe_lse,
                mask_mode,
                layout,
                window_left,
                logits_soft_cap,
                sm_scale,
                max_q_len,
                max_kv_len,
            )

        return SimpleNamespace(
            plan=_aiter_noop_plan,
            ragged_run=aiter_ragged_run,
            paged_run=aiter_paged_run,
        )

    uri = get_batch_prefill_uri(backend, *args)
    module = gen_batch_prefill_module(backend, *args).build_and_load()
    plan_func = module.plan.default
    ragged_run_func = module.ragged_run.default
    paged_run_func = module.paged_run.default

    # torch library for ragged_run

    @register_custom_op(
        f"flashinfer::{uri}_ragged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "o",
            "maybe_lse",
        ),
    )
    def ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
    ) -> None:
        ragged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            # enable_pdl,  # Not supported by HIP kernels
            maybe_custom_mask,
            maybe_mask_indptr,
            maybe_alibi_slopes,
            # maybe_prefix_len_ptr,  # Not supported by HIP FA2 kernels
            # maybe_token_pos_in_items_ptr,  # Not supported by HIP FA2 kernels
            # maybe_max_item_len_ptr,  # Not supported by HIP FA2 kernels
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
            # token_pos_in_items_len,  # Not supported by HIP FA2 kernels
        )

    @register_fake_op(f"flashinfer::{uri}_ragged_run")
    def _fake_ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
    ) -> None:
        pass

    # torch library for paged_run

    @register_custom_op(
        f"flashinfer::{uri}_paged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        scale_q: Optional[torch.Tensor],
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
        workspace_size: int,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_q_len: Optional[int] = None,
        max_kv_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        cum_seq_lens_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        maybe_partial_o: Optional[torch.Tensor] = None,
        maybe_partial_lse: Optional[torch.Tensor] = None,
    ) -> None:
        assert not is_float8(q)
        paged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            # enable_pdl,  # Not supported by HIP kernels
            maybe_custom_mask,
            maybe_mask_indptr,
            maybe_alibi_slopes,
            # maybe_prefix_len_ptr,  # Not supported by HIP FA2 kernels
            # maybe_token_pos_in_items_ptr,  # Not supported by HIP FA2 kernels
            # maybe_max_item_len_ptr,  # Not supported by HIP FA2 kernels
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
            # token_pos_in_items_len,  # Not supported by HIP FA2 kernels
            maybe_partial_o,
            maybe_partial_lse,
        )

    @register_fake_op(f"flashinfer::{uri}_paged_run")
    def _fake_paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
        workspace_size: int,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_q_len: Optional[int] = None,
        max_kv_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        cum_seq_lens_kv: Optional[torch.Tensor] = None,
        maybe_partial_o: Optional[torch.Tensor] = None,
        maybe_partial_lse: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=plan_func,
        ragged_run=ragged_run,
        paged_run=paged_run,
    )


@functools.cache
def get_batch_prefill_jit_module(module_name: str, jit_module: Any):
    plan_func = jit_module.plan.default
    ragged_run_func = jit_module.ragged_run.default
    paged_run_func = jit_module.paged_run.default

    # torch library for ragged_run
    @register_custom_op(
        f"flashinfer::{module_name}_ragged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "o",
            "maybe_lse",
        ),
    )
    def ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        ragged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_ragged_run")
    def _fake_ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    # torch library for paged_run
    @register_custom_op(
        f"flashinfer::{module_name}_paged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        paged_run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            mask_mode,
            layout,
            window_left,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_paged_run")
    def _fake_paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: torch.Tensor,
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=plan_func,
        ragged_run=ragged_run,
        paged_run=paged_run,
    )


def single_prefill_with_kv_cache_with_jit_module(
    jit_module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    kv_layout: str = "NHD",
    mask_mode: int = MaskMode.NON_CAUSAL.value,
    window_left: int = -1,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    device = q.device
    tmp = _get_cache_buf(
        "single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, device=device
    )
    o = torch.empty(q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=device)
    lse = None
    if return_lse:
        lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=device)
    jit_module.run.default(
        q,
        k,
        v,
        tmp,
        o,
        lse,
        mask_mode,
        TensorLayout[kv_layout].value,
        window_left,
        *args,
    )
    return (o, lse) if return_lse else o


@overload
def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    backend: str = "auto",
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Prefill/Append attention with KV cache for single request, return the attention
    output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[qo_len, num_qo_heads, head_dim_qk]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_qk]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim_qk]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_vo]`` if :attr:`kv_layout`
        is ``NHD``, ``[num_kv_heads, kv_len, head_dim_vo]`` if :attr:`kv_layout` is
        ``HND``.
    scale_q : Optional[torch.Tensor]
        The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_k : Optional[torch.Tensor]
        The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_v : Optional[torch.Tensor]
        The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    o_dtype : Optional[torch.dtype]
        The output tensor data type, if not provided, will be set to the same as the q.
        This is necessary as output dtype cannot be automatically inferred in quant.
    custom_mask : Optional[torch.Tensor]
        The custom boolean mask tensor, shape: ``[qo_len, kv_len]``.
        The elements in the mask tensor should be either ``True`` or ``False``,
        where ``False`` means the corresponding element in the attention matrix will be
        masked out.

        When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
        function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
        additional overhead.
    packed_custom_mask : Optional[torch.Tensor]
        The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
        The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
    causal : bool
        Whether to apply causal mask to the attention matrix.
        This is only effective when :attr:`custom_mask` is not provided.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Default is ``NONE``.
    use_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
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
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim_qk)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    backend : str
        The implementation backend, could be ``auto``/``fa2``/``aiter``. Defaults to ``auto``.
        On ROCm gfx942/gfx950, ``auto`` selects the AITER backend when the call parameters
        satisfy its constraints (NHD layout, fp16/bf16, no custom mask, equal head dims);
        otherwise falls back to FA2.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        * The log sum exp value, shape: ``[qo_len, num_qo_heads]``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> qo_len = 128
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 4
    >>> head_dim = 128
    >>> q = torch.randn(qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True,
            use_fp16_qk_reduction=True)
    >>> o.shape
    torch.Size([128, 32, 128])
    >>> mask = torch.tril(
    >>>     torch.full((qo_len, kv_len), True, device="cuda:0"),
    >>>     diagonal=(kv_len - qo_len),
    >>> )
    >>> mask
    tensor([[ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            ...,
            [ True,  True,  True,  ...,  True, False, False],
            [ True,  True,  True,  ...,  True,  True, False],
            [ True,  True,  True,  ...,  True,  True,  True]], device='cuda:0')
    >>> o_custom = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_custom, rtol=1e-3, atol=1e-3)
    True

    Note
    ----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    _check_pos_encoding_mode(pos_encoding_mode)
    _check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, q.device)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    if custom_mask is not None and packed_custom_mask is None:
        # create packed custom mask from custom mask
        packed_custom_mask = packbits(
            custom_mask.contiguous().view(-1), bitorder="little"
        )

    if packed_custom_mask is not None:
        mask_mode = MaskMode.CUSTOM.value
    else:
        if causal:
            mask_mode = MaskMode.CAUSAL.value
        else:
            mask_mode = MaskMode.NON_CAUSAL.value

    lse = None
    if return_lse:
        lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=q.device)

    if is_float8(q):
        # FP8 quant enabled, do sanity check:
        #   1. unsupported feature
        #   2. dtype check
        assert window_left == -1
        assert q.dtype == k.dtype == v.dtype
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        if scale_q is None:
            scale_q = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
        if scale_k is None:
            scale_k = torch.ones(k.shape[1], dtype=torch.float32, device=q.device)
        if scale_v is None:
            scale_v = torch.ones(v.shape[1], dtype=torch.float32, device=q.device)

    resolved_from_auto = backend == "auto"
    if backend == "auto":
        backend, _ = _auto_select_prefill_backend(
            q.device,
            dtype_q=q.dtype,
            dtype_kv=k.dtype,
            kv_layout=kv_layout,
            has_custom_mask=packed_custom_mask is not None,
            head_dim_qk=q.shape[-1],
            head_dim_vo=v.shape[-1],
            pos_encoding_mode=pos_encoding_mode,
            op="single_prefill",
        )

    if backend == "aiter":
        # Outside the probe on purpose: this raises ArchCapabilityError, which
        # gates known-bad toolchains and must never be demoted to a silent fa2.
        _require_aiter_runtime(q.device, "single_prefill")
        if pos_encoding_mode != "NONE":
            raise ValueError(
                f"AITER backend does not support pos_encoding_mode={pos_encoding_mode!r}; "
                "use backend='fa2' or backend='auto' instead."
            )
        # logits_soft_cap > 0 forces the varlen .so (mha_fwd template has no _logits
        # arm); the logits .so is split by causality (mask vs nmask) and neither is
        # pre-shipped by AITER, so bootstrap the variant matching the request.
        # logits_soft_cap == 0 takes the mha_fwd .so, none of whose variants are
        # pre-shipped — JIT-build the exact (dtype, causal, has_lse) we need.
        with _aiter_bootstrap_lock:
            if resolved_from_auto:
                # auto promised "AITER when possible, otherwise fa2", and whether
                # AITER can build this variant is only knowable here.
                reason = _aiter_single_prefill_available(
                    q.dtype,
                    causal,
                    return_lse,
                    logits_soft_cap > 0,
                    q.shape[-1],
                    q.device.index or 0,
                )
            else:
                reason = None
                if logits_soft_cap > 0:
                    _aiter_bootstrap_single_prefill_varlen(
                        q.dtype, causal, q.shape[-1], q.device.index or 0
                    )
                else:
                    _aiter_bootstrap_single_prefill_mha_fwd(
                        q.dtype, causal, return_lse, q.shape[-1], q.device.index or 0
                    )
        if reason is not None:
            backend = "fa2"

    # o_dtype should be provided for FP8 attention
    if o_dtype is None:
        o_dtype = q.dtype
    out = torch.empty(q.shape[:-1] + v.shape[-1:], dtype=o_dtype, device=q.device)

    module = get_single_prefill_module(
        backend,
        q.dtype,
        k.dtype,
        out.dtype,
        q.shape[-1],  # head_dim_qk
        v.shape[-1],  # head_dim_vo
        PosEncodingMode[pos_encoding_mode].value,
        window_left >= 0,  # use_sliding_window
        logits_soft_cap > 0,  # use_logits_soft_cap
        use_fp16_qk_reduction,
    )

    module.run(
        q,
        k,
        v,
        tmp,
        out,
        lse,
        mask_mode,
        TensorLayout[kv_layout].value,
        window_left,
        packed_custom_mask,
        _get_cache_alibi_slopes_buf(q.shape[1], q.device),
        logits_soft_cap,
        sm_scale,
        scale_q,
        scale_k,
        scale_v,
        rope_scale,
        rope_theta,
    )

    return (out, lse) if return_lse else out


single_prefill_with_kv_cache_return_lse = functools.partial(
    single_prefill_with_kv_cache, return_lse=True
)


def _compute_page_mask_indptr(
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    if len(qo_indptr) != len(paged_kv_indptr):
        raise ValueError(
            "The length of qo_indptr and paged_kv_indptr should be the same."
        )
    mask_indptr = torch.empty_like(qo_indptr)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1])
        * (
            (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) * page_size
            + paged_kv_last_page_len
        ),
        0,
    )
    return mask_indptr


class BatchPrefillWithPagedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with paged kv-cache for batch of
    requests.

    Check :ref:`our tutorial <kv-layout>` for page table layout.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> paged_kv_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= paged_kv_last_page_len <= page_size
    >>> paged_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> kv_cache_at_layer = torch.randn(
    ...     num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ... )
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     causal=True,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.run(q, kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.tril(
    ...         torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i]),
    ...     )
    ...     mask_arr.append(mask_i.flatten())
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     custom_mask=mask,
    ... )
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.run(q, kv_cache)
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...



    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    create some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
        jit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchPrefillWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        paged_kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indptr`` array, the size of this
            buffer should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indices`` array, should be large
            enough to store the maximum possible size of the ``paged_kv_indices`` array during
            the lifetime of the wrapper. This argument is only effective when ``use_cuda_graph``
            is ``True``.

        paged_kv_last_page_len_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_last_page_len`` array, the size of
            the buffer should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved buffer to store the custom mask tensor, should be large enough to
            store the maximum possible size of the packed custom mask tensor during the lifetime of
            the wrapper. This argument is only effective when ``use_cuda_graph`` is set to ``True``
            and the custom mask will be used in attention computation.

        mask_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``mask_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True`` and the custom
            mask will be used in attention computation.

        backend : str
            The implementation backend, could be ``auto``/``fa2``/``aiter``. Defaults to ``auto``.
            On ROCm gfx942/gfx950, ``auto`` selects the AITER backend when constraints are met
            (NHD layout, fp16/bf16, no custom mask, equal head dims); otherwise falls back to FA2.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.

        jit_kwargs : Optional[Dict[str, Any]]
            The keyword arguments to create the JIT module, defaults to None.
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            if jit_kwargs is None:
                jit_kwargs = {}
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                get_customize_batch_prefill_module(backend, *jit_args, **jit_kwargs),
            )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        if backend not in ("fa2", "aiter", "auto"):
            logger.warning(
                f"{backend} backend not supported on ROCm. Selecting FA2 as the backend."
            )
            backend = "fa2"
        elif backend == "aiter":
            _require_aiter_runtime(float_workspace_buffer.device, "batch_prefill")

        self._float_workspace_buffer = float_workspace_buffer
        self._workspace_size = (
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size()
        )
        self.device = float_workspace_buffer.device
        self._vector_sparse_indptr_buffer: Optional[torch.Tensor] = None
        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indptr_buf):
                raise ValueError(
                    "paged_kv_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buf):
                raise ValueError(
                    "paged_kv_indices_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buf):
                raise ValueError(
                    "paged_kv_last_page_len_buf should be a torch.Tensor in CUDA graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(paged_kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of paged_kv_indptr_buf should be batch_size + 1."
                )
            if len(paged_kv_last_page_len_buf) != self._fixed_batch_size:
                raise ValueError(
                    "The length of paged_kv_last_page_len_buf should be batch_size."
                )
            # NOTE(Zihao): do not check custom_mask_buf and mask_indptr_buf here, as they are optional
        else:
            self._fixed_batch_size = 0

        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._custom_mask_buf = custom_mask_buf
        self._mask_indptr_buf = mask_indptr_buf
        self._max_total_num_rows = None
        self._backend = backend
        # plan() overwrites _backend with the concrete choice, so it cannot say
        # whether the *caller* asked for auto. Later plan() calls need that: the
        # probe key varies per call (causal, dtype), so a wrapper that planned
        # once on AITER can still meet an unbuildable variant later.
        self._backend_requested = backend
        self._backend_fallback_reason: Optional[str] = None
        self._plan_info: Optional[torch.Tensor] = None
        self._cached_module = None
        self._seq_lens_kv = None
        self._seq_lens_q = None
        self._block_tables = None
        # Pre-computed flat-KV buffers for the AITER backend when the page
        # size is not natively supported (see _aiter_native_page_sizes()).
        self._aiter_flat_gather_idx: Optional[torch.Tensor] = None
        self._aiter_flat_kv_indptr: Optional[torch.Tensor] = None

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

        Only set by the ``auto`` constraint check; an explicit ``backend=``
        leaves it ``None``.
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
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_q: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
    ) -> None:
        r"""Plan batch prefill/append attention on Paged KV-Cache for given problem specification.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        paged_kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        paged_kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[qo_indptr[-1]]``.
        paged_kv_last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged
            kv-cache, shape: ``[batch_size]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim_qk : int
            The dimension of the query/key heads.
        page_size : int
            The size of each page in the paged kv-cache.
        head_dim_vo : Optional[int]
            The dimension of the value/output heads, if not provided, will be set to
            ``head_dim_qk``.
        custom_mask : Optional[torch.Tensor]
            The flattened boolean mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            The elements in the mask tensor should be either ``True`` or ``False``,
            where ``False`` means the corresponding element in the attention matrix will be
            masked out.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

            When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
            function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
            additional overhead.
        packed_custom_mask : Optional[torch.Tensor]
            The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
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
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Union[str, torch.dtype]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        prefix_len_ptr :Optional[torch.Tensor]
            prefix length. A uint32 1D tensor indicating the prefix length of each prompt. The tensor size is equal to the batch size.
        token_pos_in_items_ptr : Optional[float]
            A uint16 1D tensor (it will be converted to uint16 in flashinfer) indicating the token position of each item and started from 0 (delimiter)
            for each item. E.g., if we have 3 items of length 3, 2, 4 respectively for this member. This vector will be looking like
            `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]` with 4 delimiters indexed as 0. For batch size > 1,
            we will concat them as 1D with zero paddings to make sure each has the same length, the padding length is defined by
            `token_pos_in_items_len` - length of the raw `token_pos_in_items_ptr` for each prompt.
        token_pos_in_items_len : int
            zero padding length for `token_pos_in_items_ptr` to better handle the bsz > 1 case. Still using the above 3,2,4 example.
            If we set `token_pos_in_items_len` to be 20, it will be  `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]`
            with 7 padded zeros. (note there're 8 zeros in the end where the first one is the delimiter token 0 in the end of the prompt)
        max_item_len_ptr : Optional[float]
            a uint16 vector contains the max token length of all items for each prompt
        seq_lens: Optional[torch.Tensor]
            A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``.
        seq_lens_q: Optional[torch.Tensor]
            A uint32 1D tensor indicating the q sequence length of each prompt. shape: ``[batch_size]``.
            If not provided, will be set to the same value as ``seq_lens``.
        block_tables: Optional[torch.Tensor]
            A uint32 2D tensor indicating the block table of each prompt. shape: ``[batch_size, max_num_blocks_per_seq]``.
        max_token_per_sequence: Optional[int],
            Required for cudnn backend. This is the scalar max token length of each sequence.
        max_sequence_kv: Optional[int],
            Required for cudnn backend. This is the scalar max sequence length of each sequence in kv cache.
        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk

        batch_size = len(qo_indptr) - 1
        self._batch_size = batch_size
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        if custom_mask is not None or packed_custom_mask is not None:
            mask_indptr = _compute_page_mask_indptr(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                page_size,
            )
        if packed_custom_mask is None and custom_mask is not None:
            # create packed custom mask from custom mask
            packed_custom_mask, mask_indptr = segment_packbits(
                custom_mask.contiguous().view(-1),
                mask_indptr,
                bitorder="little",
            )

        self._prefix_len_ptr = prefix_len_ptr
        self._token_pos_in_items_ptr = token_pos_in_items_ptr
        self._token_pos_in_items_len = token_pos_in_items_len
        self._max_item_len_ptr = max_item_len_ptr

        # NOTE(Zihao): only required if qo_indptr/paged_kv_indptr are device tensors
        if max_token_per_sequence is not None:
            self._max_q_len = max_token_per_sequence
        else:
            qo_indptr_host = qo_indptr.to("cpu")
            total_num_rows = qo_indptr_host[-1]
            self._max_q_len = int(
                (qo_indptr_host[1:] - qo_indptr_host[:-1]).max().item()
            )

        if max_sequence_kv is not None:
            self._max_kv_len = max_sequence_kv
        else:
            paged_kv_indptr_host = paged_kv_indptr.to("cpu")
            paged_kv_last_page_len_host = paged_kv_last_page_len.to("cpu")
            if seq_lens is None:
                kv_lens_arr_host = get_seq_lens(
                    paged_kv_indptr_host, paged_kv_last_page_len_host, page_size
                )
            else:
                kv_lens_arr_host = seq_lens.cpu().flatten()
            self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
                kv_lens_arr_host, non_blocking=non_blocking
            )
            self._max_kv_len = max(kv_lens_arr_host).item()

        if self.is_cuda_graph_enabled:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_num_rows
            elif total_num_rows > self._max_total_num_rows:
                raise ValueError(
                    "The total number of rows in qo_indptr {} in cuda graph mode cannot "
                    "exceed the number of rows set during initialization {}.".format(
                        total_num_rows, self._max_total_num_rows
                    )
                )

            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed during the lifecycle of the wrapper in "
                    "cuda graph mode, the runtime batch size {} mismatches the batch size {} "
                    " set during initialization.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(paged_kv_indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The length of paged_kv_indices exceeds the allocated buffer size."
                )

            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
            self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                paged_kv_last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
                paged_kv_indices,
                non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
            )

            if packed_custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._mask_indptr_buf):
                    raise ValueError(
                        "mask_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                self._custom_mask_buf[: len(packed_custom_mask)].copy_(
                    packed_custom_mask,
                    non_blocking=(packed_custom_mask.device == self.device)
                    and non_blocking,
                )
                # NOTE(Zihao): mask_indptr has the same length as qo_indptr
                self._mask_indptr_buf.copy_(mask_indptr, non_blocking=non_blocking)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=non_blocking)
            self._paged_kv_indptr_buf = paged_kv_indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = paged_kv_indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = paged_kv_last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            if packed_custom_mask is not None:
                self._custom_mask_buf = packed_custom_mask.to(
                    self.device, non_blocking=non_blocking
                )
                self._mask_indptr_buf = mask_indptr.to(
                    self.device, non_blocking=non_blocking
                )
            else:
                self._custom_mask_buf = None
                self._mask_indptr_buf = None

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type

        # Hoisted above the jit-module split: backend-independent, and the fa2
        # re-get on an AITER demotion below needs it too.
        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            paged_kv_indptr.dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode[pos_encoding_mode].value,
            window_left >= 0,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            use_fp16_qk_reduction,
        )

        resolved_from_auto = self._backend_requested == "auto"
        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            if self._backend == "auto":
                self._backend, self._backend_fallback_reason = (
                    _auto_select_prefill_backend(
                        self.device,
                        dtype_q=q_data_type,
                        dtype_kv=kv_data_type,
                        kv_layout=self._kv_layout,
                        has_custom_mask=self._custom_mask_buf is not None,
                        head_dim_qk=head_dim_qk,
                        head_dim_vo=head_dim_vo,
                        pos_encoding_mode=pos_encoding_mode,
                        op="batch_prefill",
                    )
                )
            if self._backend == "aiter" and pos_encoding_mode != "NONE":
                raise ValueError(
                    f"AITER backend does not support pos_encoding_mode={pos_encoding_mode!r}; "
                    "use backend='fa2' or backend='auto' instead."
                )
            if self._backend == "aiter" and self._kv_layout != "NHD":
                raise ValueError(
                    f"AITER backend only supports kv_layout='NHD'; got {self._kv_layout!r}. "
                    "use backend='fa2' or backend='auto' instead."
                )
            if self._backend != "cudnn":
                self._cached_module = get_batch_prefill_module(
                    self._backend, *get_module_args
                )

        # Decide native-paged vs flat-gather ONCE. run() dispatches on whether
        # self._aiter_flat_gather_idx is set, so a single decision here is what
        # keeps the bootstrapped .so and the kernel run() reaches in agreement.
        # Bootstrapping AITER's lazy JIT is also the probe: it launches the real
        # kernel, so a config the installed AITER cannot serve fails here, at
        # plan() time, and degrades to flat-gather instead of killing run().
        # Runs before _plan_info below, so a demotion to fa2 cannot strand plan
        # bookkeeping built against the AITER module.
        use_native_paging = False
        if self._backend == "aiter":
            dev_idx = self.device.index if self.device.index is not None else 0
            has_logits = logits_soft_cap > 0
            reason = None
            with _aiter_bootstrap_lock:
                if page_size in _aiter_native_page_sizes():
                    use_native_paging = _aiter_native_paging_available(
                        q_data_type,
                        has_logits,
                        causal,
                        page_size,
                        head_dim_qk,
                        dev_idx,
                    )
                if not use_native_paging:
                    # The flat-gather route dispatches through
                    # get_aiter_mha_varlen_fwd_handle, whose .so variant is keyed on
                    # (dtype, causal, has_lse, has_logits_cap).  AITER pre-ships only
                    # a subset of that family, so bootstrap the rest here rather than
                    # let the C++ dlopen fail inside run().  The helper compiles both
                    # has_lse variants because plan() can't know which one run() will
                    # request; causal is fixed per plan() call.
                    # Demoting after a graph capture would null the flat-gather
                    # buffers the captured graph still points at, so once they
                    # exist under capture the failure has to stay an exception.
                    if resolved_from_auto and not (
                        self.is_cuda_graph_enabled
                        and self._aiter_flat_gather_idx is not None
                    ):
                        reason = _aiter_batch_ragged_available(
                            q_data_type, has_logits, causal, head_dim_qk, dev_idx
                        )
                    else:
                        _aiter_bootstrap_batch_ragged_prefill(
                            q_data_type,
                            has_logits,
                            causal,
                            head_dim_qk,
                            dev_idx,
                        )
            if reason is not None:
                self._backend = "fa2"
                self._backend_fallback_reason = reason
                self._cached_module = get_batch_prefill_module("fa2", *get_module_args)

        self._block_tables = block_tables

        if self._cached_module is not None:
            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                paged_kv_indptr_host,
                kv_lens_arr_host,
                self._max_total_num_rows or total_num_rows,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                head_dim_qk,
                head_dim_vo,
                causal,
            )
            self._plan_info = plan_info_vec_as_tensor(
                self._plan_info, device=self._float_workspace_buffer.device
            )

        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._seq_lens_kv = seq_lens
        self._seq_lens_q = seq_lens_q if seq_lens_q is not None else seq_lens

        # Pre-compute flat-KV gather indices whenever AITER is not serving this
        # page size natively.  These are stored as GPU tensors so that the
        # ``run()`` path (which may be inside a CUDA graph capture) can use them
        # without any host-side operations.
        if self._backend == "aiter" and not use_native_paging:
            kv_indptr_h = paged_kv_indptr.cpu()
            kv_indices_h = paged_kv_indices.cpu()
            kv_lpl_h = paged_kv_last_page_len.cpu()
            bs = kv_indptr_h.size(0) - 1

            gather_indices: list[int] = []
            token_counts: list[int] = []
            for i in range(bs):
                start = int(kv_indptr_h[i].item())
                end = int(kv_indptr_h[i + 1].item())
                lpl = int(kv_lpl_h[i].item())
                # Full pages
                for p_local in range(start, end - 1):
                    page_global = int(kv_indices_h[p_local].item())
                    for t in range(page_size):
                        gather_indices.append(page_global * page_size + t)
                # Last page (trimmed)
                last_page = int(kv_indices_h[end - 1].item())
                for t in range(lpl):
                    gather_indices.append(last_page * page_size + t)
                token_counts.append((end - start - 1) * page_size + lpl)

            total_tokens = len(gather_indices)
            gather_t = torch.tensor(
                gather_indices, dtype=torch.long, device=self.device
            )
            flat_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
            flat_indptr[1:] = torch.tensor(
                token_counts, dtype=torch.int32, device=self.device
            ).cumsum(0)
            if self.is_cuda_graph_enabled:
                # In CUDA-graph mode the tensors used inside the captured
                # graph must keep the **same addresses and sizes** across
                # plan() calls.  We therefore allocate with the *maximum*
                # possible size on the first call and pad unused positions
                # with safe values (index 0).  Subsequent plan() calls
                # overwrite the used portion via copy_().
                max_flat_tokens = self._paged_kv_indices_buf.numel() * page_size
                if self._aiter_flat_gather_idx is None:
                    # First plan() – allocate max-size buffers.
                    self._aiter_flat_gather_idx = torch.zeros(
                        max_flat_tokens, dtype=torch.long, device=self.device
                    )
                    self._aiter_flat_kv_indptr = torch.zeros(
                        bs + 1, dtype=torch.int32, device=self.device
                    )
                # Fill the used portion, leave the rest as zero-padding.
                self._aiter_flat_gather_idx[:total_tokens].copy_(gather_t)
                self._aiter_flat_kv_indptr.copy_(flat_indptr)
            else:
                self._aiter_flat_gather_idx = gather_t
                self._aiter_flat_kv_indptr = flat_indptr
        else:
            self._aiter_flat_gather_idx = None
            self._aiter_flat_kv_indptr = None

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This function is deprecated, please use :meth:`run` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
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
        partial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch prefill/append attention between query and paged kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
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
            Additional arguments for custom kernels.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention output
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)
        if enable_pdl:
            logger.warning(
                "enable_pdl is not supported in the HIP/ROCm backend and will be ignored. "
                "This parameter is only effective on CUDA devices with sm_90+."
            )
        if self._prefix_len_ptr is not None or self._token_pos_in_items_ptr is not None:
            logger.warning(
                "Token position tracking features (prefix_len_ptr, token_pos_in_items_ptr) "
                "are not supported in the HIP/ROCm FA2 backend and will be ignored. "
                "These features are only available in CUDA implementation."
            )
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        _check_cached_qkv_data_type(
            q, k_cache, self._cached_q_data_type, self._cached_kv_data_type
        )
        if self._kv_layout == "NHD":
            page_size = k_cache.shape[1]
        else:
            page_size = k_cache.shape[2]
        window_left = self._window_left if window_left is None else window_left
        # NOTE(Siyuan): since window_left is appeared in the plan function, we
        # need to make sure it is the same as the one in the plan function.
        # Remove this check if the backend supports dynamic window_left.
        assert window_left == self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
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
            out = torch.empty(
                q.shape[:-1] + v_cache.shape[-1:], dtype=q.dtype, device=q.device
            )
        else:
            check_shape_dtype_device(
                out, q.shape[:-1] + v_cache.shape[-1:], q.dtype, q.device, "out"
            )

        if self._custom_mask_buf is not None:
            mask_mode = MaskMode.CUSTOM.value
        else:
            if self._causal:
                mask_mode = MaskMode.CAUSAL.value
            else:
                mask_mode = MaskMode.NON_CAUSAL.value

        if self._prefix_len_ptr is not None:
            mask_mode = MaskMode.MULTIITEMSCORING.value

        sparse_indices = self._paged_kv_indices_buf
        sparse_indptr = self._paged_kv_indptr_buf

        assert self._plan_info is not None, "plan info is not initialized"
        if partial_state is not None:
            if partial_state[0].dtype != out.dtype:
                raise ValueError(
                    f"partial_state dtype {partial_state[0].dtype} must match output dtype {out.dtype}"
                )
            if partial_state[0].device != out.device:
                raise ValueError(
                    f"partial_state device {partial_state[0].device} must match output device {out.device}"
                )
            if partial_state[1].dtype != torch.float32:
                raise ValueError(
                    f"partial_state lse must be float32, got {partial_state[1].dtype}"
                )
            if partial_state[1].device != out.device:
                raise ValueError(
                    f"partial_state lse device {partial_state[1].device} must match output device {out.device}"
                )
            # Ensure lse is allocated so the kernel can write the merged LSE output.
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
        run_args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._qo_indptr_buf,
            sparse_indptr,
            sparse_indices,
            self._paged_kv_last_page_len_buf,
            out,
            lse,
            mask_mode,
            TensorLayout[self._kv_layout].value,
            window_left,
            enable_pdl,
        ]
        if self._jit_module is not None:
            run_args.extend(list(args))
        else:
            run_args += [
                self._custom_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                self._prefix_len_ptr,
                self._token_pos_in_items_ptr,
                self._max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                None,  # scale_q, not supported yet
                None,  # scale_k
                None,  # scale_v
                rope_scale,
                rope_theta,
                self._token_pos_in_items_len,
                self._workspace_size,
                self._num_qo_heads,
                self._num_kv_heads,
                self._block_tables,
                self._kv_lens_buffer,
                page_size,
                self._max_q_len,
                self._max_kv_len,
                self._batch_size,
                self._qo_indptr_buf,
                self._vector_sparse_indptr_buffer,
                sinks,
            ]
            if self._backend == "aiter":
                # Pre-computed flat-KV gather info for AITER (None for
                # natively-supported page sizes).
                run_args += [
                    self._aiter_flat_gather_idx,
                    self._aiter_flat_kv_indptr,
                ]
            else:
                po, plse = partial_state if partial_state is not None else (None, None)
                run_args += [po, plse]

        assert self._cached_module is not None, "cached module is not initialized"
        self._cached_module.paged_run(*run_args)
        if self._backend == "aiter" and partial_state is not None:
            # AITER kernel doesn't accept partial_state; merge post-hoc.
            from .cascade import merge_state_in_place

            merge_state_in_place(partial_state[0], partial_state[1], out, lse)
            out, lse = partial_state
        if v_scale is not None:
            # TODO(Zihao): fused into kernel
            if is_float8(out):
                out = (out.to(torch.float32) * v_scale).to(out.dtype)
            else:
                out *= v_scale
        return (out, lse) if return_lse else out

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: This function is deprecated, please use :meth:`run_return_lse` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run_return_lse(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


def _compute_mask_indptr(
    qo_indptr: torch.Tensor, kv_indptr: torch.Tensor
) -> torch.Tensor:
    if len(qo_indptr) != len(kv_indptr):
        raise ValueError("The length of qo_indptr and kv_indptr should be the same.")
    mask_indptr = torch.empty_like(qo_indptr)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1]) * (kv_indptr[1:] - kv_indptr[:-1]),
        0,
    )
    return mask_indptr


class BatchPrefillWithRaggedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with ragged (tensor) kv-cache for
    batch of requests.

    Check :ref:`our tutorial <kv-layout>` for ragged kv-cache layout.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_kv = 100
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_indptr = qo_indptr.clone()
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     causal=True,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.run(q, k, v)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.tril(
    ...         torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i]),
    ...     )
    ...     mask_arr.append(mask_i.flatten())
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     custom_mask=mask
    ... )
    >>> outputs_custom_mask = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.run(q, k, v)
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...
    >>> outputs_custom_mask[0].shape
    torch.Size([100, 64, 128])


    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    create some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
        jit_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchPrefillWithRaggedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored as the provided buffers.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the custom mask tensor, should be large
            enough to store the maximum possible size of the packed custom mask tensor during the
            lifetime of the wrapper. This argument is only effective when ``use_cuda_graph``
            is ``True`` and custom mask will be used in attention computation.

        mask_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``mask_indptr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True`` and custom mask
            will be used in attention computation.

        backend : str
            The implementation backend, could be ``auto``/``fa2``/``aiter``.
            Defaults to ``auto``. ``auto`` routes to AITER on gfx942/gfx950 when
            constraints (NHD layout, fp16/bf16, no custom mask) are met, else FA2.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.

        jit_kwargs : Optional[Dict[str, Any]]
            The keyword arguments to create the JIT module, defaults to None.
        """
        _check_kv_layout(kv_layout)
        if jit_args is not None:
            if jit_kwargs is None:
                jit_kwargs = {}
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                get_customize_batch_prefill_module("fa2", *jit_args, **jit_kwargs),
            )
        else:
            self._jit_module = None
        if backend not in ("fa2", "aiter", "auto"):
            raise ValueError(
                f"backend must be one of 'fa2', 'aiter', 'auto'; got {backend!r}"
            )
        if backend == "aiter":
            _require_aiter_runtime(float_workspace_buffer.device, "batch_prefill")
        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            if not torch.is_tensor(kv_indptr_buf):
                raise ValueError(
                    "kv_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of kv_indptr_buf ({}) should be the same as qo_indptr_buf ({}).".format(
                        len(kv_indptr_buf), self._fixed_batch_size
                    )
                )
            # NOTE(Zihao): do not check custom_mask_buf and mask_indptr_buf here,
            # as they may not be used.

        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._custom_mask_buf = custom_mask_buf
        self._mask_indptr_buf = mask_indptr_buf
        self._max_total_num_rows = None
        self._backend = backend
        # See the paged wrapper: _backend goes concrete after the first plan(),
        # so the caller's original request has to be kept separately.
        self._backend_requested = backend
        self._backend_fallback_reason: Optional[str] = None
        self._plan_info: Optional[torch.Tensor] = None
        self._cached_module = None

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

        Only set by the ``auto`` constraint check; an explicit ``backend=``
        leaves it ``None``.
        """
        return self._backend_fallback_reason

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer
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
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Plan batch prefill/append attention on Ragged KV-Cache for given problem specification.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        kv_indptr : torch.Tensor
            The indptr of the key/value tensor, shape: ``[batch_size + 1]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim_qk : int
            The dimension of the heads on query/key tensor.
        head_dim_vo : Optional[int]
            The dimension of the heads on value/output tensor.
            If not provided, will be set to ``head_dim_vo``.
        custom_mask : Optional[torch.Tensor]
            The flattened boolean mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            The elements in the mask tensor should be either ``True`` or ``False``,
            where ``False`` means the corresponding element in the attention matrix will be
            masked out.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

            When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
            function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
            additional overhead.
        packed_custom_mask : Optional[torch.Tensor]
            The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.

            If provided, the custom mask will be added to the attention matrix before softmax
            and after scaling. The mask tensor should be in the same device as the input tensors.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This argument is ignored if ``mask`` is provided in :meth:`plan`.
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
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
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim_qk)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Union[str, torch.dtype]
            The data type of the query tensor, defaults to torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        prefix_len_ptr :Optional[torch.Tensor]
            prefix length. A uint32 1D tensor indicating the prefix length of each prompt. The tensor size is equal to the batch size.
        token_pos_in_items_ptr : Optional[float]
            A uint16 1D tensor (it will be converted to uint16 in flashinfer) indicating the token position of each item and started from 0 (delimiter)
            for each item. E.g., if we have 3 items of length 3, 2, 4 respectively for this member. This vector will be looking like
            `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]` with 4 delimiters indexed as 0. For batch size > 1,
            we will concat them as 1D with zero paddings to make sure each has the same length, the padding length is defined by
            `token_pos_in_items_len` - length of the raw `token_pos_in_items_ptr` for each prompt.
        token_pos_in_items_len : int
            zero padding length for `token_pos_in_items_ptr` to better handle the bsz > 1 case. Still using the above 3,2,4 example.
            If we set `token_pos_in_items_len` to be 20, it will be  `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]`
            with 7 padded zeros. (note there're 8 zeros in the end where the first one is the delimiter token 0 in the end of the prompt)
        max_item_len_ptr : Optional[float]
            a uint16 vector contains the max token length of all items for each prompt

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this plan call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        batch_size = len(qo_indptr) - 1
        if len(kv_indptr) != batch_size + 1:
            raise ValueError(
                "The kv_indptr length should be equal to mask_indptr length."
            )
        if custom_mask is not None or packed_custom_mask is not None:
            mask_indptr = _compute_mask_indptr(qo_indptr, kv_indptr)
        if packed_custom_mask is None and custom_mask is not None:
            # create packed custom mask from custom mask
            packed_custom_mask, mask_indptr = segment_packbits(
                custom_mask.contiguous().view(-1),
                mask_indptr,
                bitorder="little",
            )

        # NOTE(Zihao): only required if qo_indptr/paged_kv_indptr are device tensors
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")

        total_num_rows = qo_indptr_host[-1]

        if self.is_cuda_graph_enabled:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_num_rows
            elif total_num_rows > self._max_total_num_rows:
                raise ValueError(
                    "The total number of rows in qo_indptr {} in cuda graph mode cannot "
                    "exceed the number of rows set during initialization {}.".format(
                        total_num_rows, self._max_total_num_rows
                    )
                )

            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
            self._kv_indptr_buf.copy_(kv_indptr, non_blocking=non_blocking)
            if packed_custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._mask_indptr_buf):
                    raise ValueError(
                        "mask_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in the attention computation."
                    )
                self._custom_mask_buf[: len(packed_custom_mask)] = packed_custom_mask
                self._mask_indptr_buf.copy_(mask_indptr, non_blocking=non_blocking)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=non_blocking)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
            if packed_custom_mask is not None:
                self._custom_mask_buf = packed_custom_mask.to(
                    self.device, non_blocking=non_blocking
                )
                self._mask_indptr_buf = mask_indptr.to(
                    self.device, non_blocking=non_blocking
                )

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        kv_len_arr = kv_indptr_host[1:] - kv_indptr_host[:-1]
        qo_len_arr = qo_indptr_host[1:] - qo_indptr_host[:-1]
        self._max_q_len = int(qo_len_arr.max().item()) if batch_size > 0 else 0
        self._max_kv_len = int(kv_len_arr.max().item()) if batch_size > 0 else 0

        self._prefix_len_ptr = prefix_len_ptr
        self._token_pos_in_items_ptr = token_pos_in_items_ptr
        self._token_pos_in_items_len = token_pos_in_items_len
        self._max_item_len_ptr = max_item_len_ptr

        # Hoisted above the jit-module split: backend-independent, and the fa2
        # re-get on an AITER demotion below needs it too.
        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            kv_indptr.dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode[pos_encoding_mode].value,
            window_left >= 0,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            use_fp16_qk_reduction,
        )

        resolved_from_auto = self._backend_requested == "auto"
        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            if self._backend == "auto":
                self._backend, self._backend_fallback_reason = (
                    _auto_select_prefill_backend(
                        self.device,
                        dtype_q=q_data_type,
                        dtype_kv=kv_data_type,
                        kv_layout=self._kv_layout,
                        has_custom_mask=packed_custom_mask is not None,
                        head_dim_qk=head_dim_qk,
                        head_dim_vo=head_dim_vo,
                        pos_encoding_mode=pos_encoding_mode,
                        op="batch_prefill",
                    )
                )
            if self._backend == "aiter" and pos_encoding_mode != "NONE":
                raise ValueError(
                    f"AITER backend does not support pos_encoding_mode={pos_encoding_mode!r}; "
                    "use backend='fa2' or backend='auto' instead."
                )
            if self._backend == "aiter" and self._kv_layout != "NHD":
                raise ValueError(
                    f"AITER backend only supports kv_layout='NHD'; got {self._kv_layout!r}. "
                    "use backend='fa2' or backend='auto' instead."
                )
            self._cached_module = get_batch_prefill_module(
                self._backend, *get_module_args
            )

        # Bootstrap AITER's lazy JIT so the C++ dlopen finds mha_varlen_fwd_*.so
        # for the (dtype, causal, has_logits) combo this plan() call will use.
        # Stays outside the jit-module split above -- jit_args with an explicit
        # backend="aiter" still needs the .so -- and runs before _plan_info below
        # so a demotion to fa2 cannot strand plan bookkeeping built for AITER.
        if self._backend == "aiter":
            dev_idx = self.device.index if self.device.index is not None else 0
            reason = None
            with _aiter_bootstrap_lock:
                if resolved_from_auto:
                    reason = _aiter_batch_ragged_available(
                        q_data_type,
                        logits_soft_cap > 0,
                        causal,
                        head_dim_qk,
                        dev_idx,
                    )
                else:
                    _aiter_bootstrap_batch_ragged_prefill(
                        q_data_type,
                        logits_soft_cap > 0,
                        causal,
                        head_dim_qk,
                        dev_idx,
                    )
            if reason is not None:
                self._backend = "fa2"
                self._backend_fallback_reason = reason
                self._cached_module = get_batch_prefill_module("fa2", *get_module_args)

        assert self._cached_module is not None, "cached module is not initialized"
        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr,
            self._max_total_num_rows or total_num_rows,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            1,  # page_size
            self.is_cuda_graph_enabled,
            head_dim_qk,
            head_dim_vo,
            causal,
        )
        self._plan_info = plan_info_vec_as_tensor(
            self._plan_info, device=self._float_workspace_buffer.device
        )

        self._causal: bool = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left: int = window_left
        self._logits_soft_cap: float = logits_soft_cap
        self._sm_scale: float = sm_scale
        self._rope_scale: float = rope_scale
        self._rope_theta: float = rope_theta

    begin_forward = plan

    @overload
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch prefill/append attention between query and kv-cache stored as
        ragged tensor.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_qk]``
        k : torch.Tensor
            The key tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim_qk]``
        v : torch.Tensor
            The value tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim_vo]``
        *args
            Additional arguments for the custom kernel.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention output
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_vo]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_vo]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)
        _check_cached_qkv_data_type(
            q, k, self._cached_q_data_type, self._cached_kv_data_type
        )

        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
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
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=q.device
            )
        else:
            check_shape_dtype_device(
                out, q.shape[:-1] + v.shape[-1:], q.dtype, q.device, "out"
            )
        if is_float8(q):
            logging.warning(
                "Our current prefill kernel implementation needs f16 input, the f8 inputs "
                " are casted to f16, which could result in performance degradation."
            )
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        if self._custom_mask_buf is not None:
            mask_mode = MaskMode.CUSTOM.value
        else:
            if self._causal:
                mask_mode = MaskMode.CAUSAL.value
            else:
                mask_mode = MaskMode.NON_CAUSAL.value

        run_args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr_buf,
            self._kv_indptr_buf,
            out,
            lse,
            mask_mode,
            TensorLayout[self._kv_layout].value,
            window_left,
            enable_pdl,
        ]
        if self._jit_module is not None:
            run_args.extend(list(args))
        else:
            run_args += [
                self._custom_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                self._prefix_len_ptr,
                self._token_pos_in_items_ptr,
                self._max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
                self._token_pos_in_items_len,
            ]
            if self._backend == "aiter":
                run_args += [self._max_q_len, self._max_kv_len]

        assert self._cached_module is not None, "cached module is not initialized"
        self._cached_module.ragged_run(*run_args)
        return (out, lse) if return_lse else out

    run_return_lse = functools.partialmethod(run, return_lse=True)
