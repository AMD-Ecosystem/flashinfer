# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List

import jinja2
import torch

from ..core import JitSpec, gen_jit_spec, logger
from ..env import FLASHINFER_CSRC_DIR, FLASHINFER_GEN_SRC_DIR
from ..utils import (
    dtype_map_hip,
    filename_safe_dtype_map,
    mask_mode_literal,
    pos_encoding_mode_literal,
    write_if_different,
)


def _generate_additional_params_hip(
    additional_tensor_names,
    additional_tensor_dtypes,
    additional_scalar_names,
    additional_scalar_dtypes,
    is_sm90_template: bool = False,
):
    """HIP-specific variant: generates PyTorch at::Tensor / std::optional<at::Tensor>
    parameter types instead of the TVM ffi::Tensor types used by the CUDA backend."""
    additional_params_decl = "".join(
        [
            f"{dtype}* {var};\n"
            for dtype, var in zip(
                additional_tensor_dtypes,
                additional_tensor_names,
                strict=False,
            )
        ]
        + [
            f"{dtype} {var};\n"
            for dtype, var in zip(
                additional_scalar_dtypes, additional_scalar_names, strict=False
            )
        ]
    )
    additional_func_params = "".join(
        [
            (
                f", std::optional<at::Tensor> {var}"
                if var.startswith("maybe")
                else f", at::Tensor {var}"
            )
            for var in additional_tensor_names
        ]
        + [
            f", {dtype} {var}"
            for dtype, var in zip(
                additional_scalar_dtypes, additional_scalar_names, strict=False
            )
        ]
    )
    if is_sm90_template:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.additional_params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.additional_params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(
                    additional_tensor_dtypes, additional_tensor_names, strict=False
                )
            ]
            + [
                f"params.additional_params.{var} = {var};"
                for var in additional_scalar_names
            ]
        )
    else:
        additional_params_setter = " \\\n".join(
            [
                (
                    f"params.{var} = {var} ? static_cast<{dtype}*>({var}->data_ptr()): nullptr;"
                    if var.startswith("maybe")
                    else f"params.{var} = static_cast<{dtype}*>({var}.data_ptr());"
                )
                for dtype, var in zip(
                    additional_tensor_dtypes, additional_tensor_names, strict=False
                )
            ]
            + [f"params.{var} = {var};" for var in additional_scalar_names]
        )
    return (additional_params_decl, additional_func_params, additional_params_setter)


generate_additional_params = _generate_additional_params_hip


def get_single_decode_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        f"single_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def get_batch_decode_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> str:
    return (
        f"batch_decode_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}"
    )


def get_single_prefill_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    if backend == "fa3":
        logger.warning(
            "FA3 backend not supported on ROCm. The backend argument will be ignored."
        )
    if backend == "aiter":
        # AITER handles mask / sliding-window / logits-cap / fp16-qk-reduction
        # at runtime — one .so per (dtype, head_dim) combo is enough.
        return (
            f"single_prefill_with_kv_cache_aiter_"
            f"dtype_q_{filename_safe_dtype_map[dtype_q]}_"
            f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
            f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
            f"head_dim_qk_{head_dim_qk}_"
            f"head_dim_vo_{head_dim_vo}"
        )
    return (
        f"single_prefill_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def get_batch_prefill_uri(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> str:
    if backend == "fa3":
        logger.warning(
            "FA3 backend not supported on ROCm. The backend argument will be ignored."
        )
    if backend == "aiter":
        # Single .so hosts both paged_run and ragged_run; URI is shared.
        return (
            f"batch_prefill_with_kv_cache_aiter_"
            f"dtype_q_{filename_safe_dtype_map[dtype_q]}_"
            f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
            f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
            f"head_dim_qk_{head_dim_qk}_"
            f"head_dim_vo_{head_dim_vo}"
        )
    return (
        f"batch_prefill_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}_"
        f"posenc_{pos_encoding_mode}_"
        f"use_swa_{use_sliding_window}_"
        f"use_logits_cap_{use_logits_soft_cap}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def gen_single_decode_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
):
    uri = get_single_decode_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    )
    return gen_customize_single_decode_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["double", "double", "double", "double"],  # additional_scalar_dtypes
        f"DefaultAttention<false, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>",  # variant_name
        "#include<flashinfer/generic/attention/variants.cuh>",  # variant_decl
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )


def gen_single_prefill_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> JitSpec:
    uri = get_single_prefill_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    )
    if backend in ("fa2", "aiter"):
        # AITER shares FA2's additional-param signature; the C++ shim ignores
        # parameters AITER doesn't consume (rope, custom_mask).
        additional_tensor_names = ["maybe_custom_mask", "maybe_alibi_slopes"]
        additional_tensor_dtypes = ["uint8_t", "float"]
        additional_scalar_names = [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ]
        additional_scalar_dtypes = ["double", "double", "double", "double"]
        variant_name = f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        variant_decl = "#include<flashinfer/generic/attention/variants.cuh>"
    else:
        additional_tensor_names = []
        additional_tensor_dtypes = []
        additional_scalar_names = ["logits_soft_cap", "sm_scale"]
        additional_scalar_dtypes = ["double", "double"]
        variant_name = f"DefaultAttention<{str(use_logits_soft_cap).lower()}>"
        variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"

    return gen_customize_single_prefill_module(
        backend,
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name,
        variant_decl,
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
    )


def get_batch_decode_aiter_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
) -> str:
    # AITER PA v1 bakes (dtype, head_dim, gqa_ratio, block_size, partition_size,
    # logits_cap_enabled, sliding_window_enabled, alibi_enabled, npar_loops, mtp)
    # into its own .so via md5-hashed func_name at plan() time. Our wrapper .so
    # is dtype-/head_dim-agnostic — one shared variant covers any combination.
    # Including dtype/head_dim in the URI keeps it consistent with other URIs
    # and harmless (still one .so per concrete combo, but build is cheap).
    return (
        f"batch_decode_with_paged_kv_cache_aiter_"
        f"dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_qk_{head_dim_qk}_"
        f"head_dim_vo_{head_dim_vo}"
    )


def gen_batch_decode_aiter_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
) -> JitSpec:
    import aiter as _aiter_mod

    aiter_jit_dir = os.path.join(os.path.dirname(_aiter_mod.__file__), "jit")

    uri = get_batch_decode_aiter_uri(
        dtype_q, dtype_kv, dtype_o, head_dim_qk, head_dim_vo
    )
    gen_directory = FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []
    for filename in [
        "batch_decode_aiter.cu",
        "batch_decode_aiter_jit_pybind.cu",
        "aiter_loader.cc",
    ]:
        src_path = FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    return gen_jit_spec(
        uri,
        source_paths,
        extra_cflags=[f'-DFLASHINFER_AITER_JIT_DIR=\\"{aiter_jit_dir}\\"'],
        extra_ldflags=["-ldl"],
    )


def gen_batch_decode_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
):
    uri = get_batch_decode_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    )
    return gen_customize_batch_decode_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["double", "double", "double", "double"],  # additional_scalar_dtypes
        f"DefaultAttention<false, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>",  # variant_name
        "#include<flashinfer/generic/attention/variants.cuh>",  # variant_decl
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )


def gen_batch_prefill_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool,
) -> JitSpec:
    uri = get_batch_prefill_uri(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
    )

    if backend == "aiter":
        return gen_customize_batch_prefill_module(
            backend,
            uri,
            dtype_q,
            dtype_kv,
            dtype_o,
            dtype_idx,
            head_dim_qk,
            head_dim_vo,
            [],
            [],
            [],
            [],
            "",
            "",
            pos_encoding_mode=pos_encoding_mode,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
            use_fp16_qk_reduction=use_fp16_qk_reduction,
        )

    if backend == "fa2":
        additional_tensor_names = [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
        ]
        additional_tensor_dtypes = [
            "uint8_t",
            "int32_t",
            "float",
        ]  # NOTE(Zihao): int32_t should follow dtype_idx
        additional_scalar_names = [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ]
        additional_scalar_dtypes = ["double", "double", "double", "double"]
        variant_name = f"DefaultAttention<use_custom_mask, {str(use_sliding_window).lower()}, {str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>"
        variant_decl = "#include<flashinfer/generic/attention/variants.cuh>"
    else:
        additional_tensor_names = []
        additional_tensor_dtypes = []
        additional_scalar_names = ["logits_soft_cap", "sm_scale"]
        additional_scalar_dtypes = ["double", "double"]
        variant_name = f"DefaultAttention<{str(use_logits_soft_cap).lower()}>"
        variant_decl = "#include<flashinfer/attention/hopper/variants.cuh>"

    return gen_customize_batch_prefill_module(
        backend,
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name,
        variant_decl,
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
    )


def gen_customize_single_decode_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
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
) -> JitSpec:
    gen_directory = FLASHINFER_GEN_SRC_DIR / uri
    (
        additional_params_decl,
        additional_func_params,
        additional_params_setter,
    ) = generate_additional_params(
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
    )

    with open(FLASHINFER_CSRC_DIR / "single_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(FLASHINFER_CSRC_DIR / "single_decode_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    kwargs = {
        "additional_func_params": additional_func_params,
        "additional_params_decl": additional_params_decl,
        "additional_params_setter": additional_params_setter,
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map_hip[dtype_q],
        "dtype_kv": dtype_map_hip[dtype_kv],
        "dtype_o": dtype_map_hip[dtype_o],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
    }

    generated_inc_str = config_templ.render(
        **kwargs,
    )

    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []

    dest_path = gen_directory / "single_decode_kernel.cu"
    source_paths.append(dest_path)
    source = kernel_inst_templ.render(
        **kwargs,
    )
    write_if_different(dest_path, source)
    for filename in [
        "single_decode.cu",
        "single_decode_jit_pybind.cu",
    ]:
        src_path = FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "single_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)

    return gen_jit_spec(uri, source_paths)


def gen_customize_single_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
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
):
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map_hip[dtype_q],
        "dtype_kv": dtype_map_hip[dtype_kv],
        "dtype_o": dtype_map_hip[dtype_o],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    elif backend == "fa2":
        gen_directory = FLASHINFER_GEN_SRC_DIR / uri
        additional_params_decl, additional_func_params, additional_params_setter = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(FLASHINFER_CSRC_DIR / "single_prefill_customize_config.jinja") as f:
            config_templ = jinja2.Template(f.read())

        with open(FLASHINFER_CSRC_DIR / "single_prefill_kernel_inst.jinja") as f:
            kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_func_params": additional_func_params,
            "additional_params_decl": additional_params_decl,
            "additional_params_setter": additional_params_setter,
        }

        generated_inc_str = config_templ.render(
            **kwargs,
        )
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        for mask_mode in [0, 1, 2]:
            filename = f"single_prefill_kernel_mask_{mask_mode}.cu"
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "single_prefill.cu",
            "single_prefill_jit_pybind.cu",
        ]:
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "single_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)

        return gen_jit_spec(uri, source_paths)
    elif backend == "aiter":
        # Native C++ AITER backend: calls aiter::mha_fwd directly via dlopen/dlsym.
        # No c10::Dispatcher or IValue boxing — pure C++ call through a cached fn ptr.
        import aiter as _aiter_mod

        aiter_jit_dir = os.path.join(os.path.dirname(_aiter_mod.__file__), "jit")

        gen_directory = FLASHINFER_GEN_SRC_DIR / uri
        additional_params_decl, additional_func_params, additional_params_setter = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(FLASHINFER_CSRC_DIR / "single_prefill_customize_config.jinja") as f:
            config_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_func_params": additional_func_params,
            "additional_params_decl": additional_params_decl,
            "additional_params_setter": additional_params_setter,
        }

        generated_inc_str = config_templ.render(**kwargs)
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        for filename in [
            "single_prefill_aiter.cu",
            "single_prefill_aiter_jit_pybind.cu",
            "aiter_loader.cc",
        ]:
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "single_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)

        # -DFLASHINFER_AITER_JIT_DIR bakes the default jit dir at compile time;
        # runtime AITER_JIT_DIR env var overrides it (see aiter_loader.cc).
        return gen_jit_spec(
            uri,
            source_paths,
            extra_cflags=[f'-DFLASHINFER_AITER_JIT_DIR=\\"{aiter_jit_dir}\\"'],
            extra_ldflags=["-ldl"],
        )
    elif backend == "fa3":
        raise ValueError("FA3 backend not currently supported for ROCm")
    else:
        raise ValueError(f"Invalid backend: {backend}")


def gen_customize_batch_decode_module(
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
):
    gen_directory = FLASHINFER_GEN_SRC_DIR / uri
    (additional_params_decl, additional_func_params, additional_params_setter) = (
        generate_additional_params(
            additional_tensor_names,
            additional_tensor_dtypes,
            additional_scalar_names,
            additional_scalar_dtypes,
        )
    )

    kwargs = {
        "additional_params_decl": additional_params_decl,
        "additional_func_params": additional_func_params,
        "additional_params_setter": additional_params_setter,
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map_hip[dtype_q],
        "dtype_kv": dtype_map_hip[dtype_kv],
        "dtype_o": dtype_map_hip[dtype_o],
        "idtype": dtype_map_hip[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
    }

    with open(FLASHINFER_CSRC_DIR / "batch_decode_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())

    with open(FLASHINFER_CSRC_DIR / "batch_decode_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    generated_inc_str = config_templ.render(
        **kwargs,
    )

    source_paths = []

    dest_path = gen_directory / "batch_decode_kernel.cu"
    source_paths.append(dest_path)
    source = kernel_inst_templ.render(
        **kwargs,
    )
    write_if_different(dest_path, source)

    for filename in [
        "batch_decode.cu",
        "batch_decode_jit_pybind.cu",
    ]:
        src_path = FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    generated_config_path = gen_directory / "batch_decode_config.inc"
    write_if_different(generated_config_path, generated_inc_str)

    return gen_jit_spec(uri, source_paths)


def gen_customize_batch_prefill_module(
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
):
    kwargs = {
        "variant_decl": variant_decl,
        "variant_name": variant_name,
        "dtype_q": dtype_map_hip[dtype_q],
        "dtype_kv": dtype_map_hip[dtype_kv],
        "dtype_o": dtype_map_hip[dtype_o],
        "idtype": dtype_map_hip[idtype],
        "head_dim_qk": head_dim_qk,
        "head_dim_vo": head_dim_vo,
        "pos_encoding_mode": pos_encoding_mode_literal[pos_encoding_mode],
        "use_sliding_window": str(use_sliding_window).lower(),
        "use_logits_soft_cap": str(use_logits_soft_cap).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }
    if backend == "auto":
        raise ValueError("backend should not be auto when jit_args is provided")
    elif backend == "fa2":
        gen_directory = FLASHINFER_GEN_SRC_DIR / uri
        (additional_params_decl, additional_func_params, additional_params_setter) = (
            generate_additional_params(
                additional_tensor_names,
                additional_tensor_dtypes,
                additional_scalar_names,
                additional_scalar_dtypes,
            )
        )

        with open(FLASHINFER_CSRC_DIR / "batch_prefill_customize_config.jinja") as f:
            config_templ = jinja2.Template(f.read())

        with open(FLASHINFER_CSRC_DIR / "batch_prefill_paged_kernel_inst.jinja") as f:
            paged_kernel_inst_templ = jinja2.Template(f.read())

        with open(FLASHINFER_CSRC_DIR / "batch_prefill_ragged_kernel_inst.jinja") as f:
            ragged_kernel_inst_templ = jinja2.Template(f.read())

        kwargs |= {
            "additional_params_decl": additional_params_decl,
            "additional_func_params": additional_func_params,
            "additional_params_setter": additional_params_setter,
        }

        generated_inc_str = config_templ.render(
            **kwargs,
        )
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        for mask_mode in [0, 1, 2]:
            dest_path = (
                gen_directory / f"batch_prefill_paged_kernel_mask_{mask_mode}.cu"
            )
            source_paths.append(dest_path)
            source = paged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

            dest_path = (
                gen_directory / f"batch_prefill_ragged_kernel_mask_{mask_mode}.cu"
            )
            source_paths.append(dest_path)
            source = ragged_kernel_inst_templ.render(
                mask_mode=mask_mode_literal[mask_mode],
                **kwargs,
            )
            write_if_different(dest_path, source)

        for filename in [
            "batch_prefill.cu",
            "batch_prefill_jit_pybind.cu",
        ]:
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_config.inc"
        write_if_different(generated_config_path, generated_inc_str)
        return gen_jit_spec(
            uri,
            source_paths,
        )
    elif backend == "aiter":
        import aiter as _aiter_mod

        aiter_jit_dir = os.path.join(os.path.dirname(_aiter_mod.__file__), "jit")

        gen_directory = FLASHINFER_GEN_SRC_DIR / uri

        with open(
            FLASHINFER_CSRC_DIR / "batch_prefill_aiter_customize_config.jinja"
        ) as f:
            config_templ = jinja2.Template(f.read())

        generated_inc_str = config_templ.render(
            dtype_q=dtype_map_hip[dtype_q],
            dtype_kv=dtype_map_hip[dtype_kv],
            dtype_o=dtype_map_hip[dtype_o],
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
        )
        os.makedirs(gen_directory, exist_ok=True)

        source_paths = []
        # Both paged_run and ragged_run share the .so — single TORCH_LIBRARY
        # extension, single aiter_loader.cc dlopen cache, one config .inc.
        for filename in [
            "batch_prefill_paged_aiter.cu",
            "batch_prefill_paged_aiter_jit_pybind.cu",
            "batch_ragged_prefill_aiter.cu",
            "batch_ragged_prefill_aiter_jit_pybind.cu",
            "aiter_loader.cc",
        ]:
            src_path = FLASHINFER_CSRC_DIR / filename
            dest_path = gen_directory / filename
            source_paths.append(dest_path)
            with open(src_path, "r") as f:
                source = f.read()
            write_if_different(dest_path, source)

        generated_config_path = gen_directory / "batch_prefill_aiter_config.inc"
        write_if_different(generated_config_path, generated_inc_str)

        return gen_jit_spec(
            uri,
            source_paths,
            extra_cflags=[f'-DFLASHINFER_AITER_JIT_DIR=\\"{aiter_jit_dir}\\"'],
            extra_ldflags=["-ldl"],
        )
    elif backend == "fa3":
        raise ValueError("FA3 backend not currently supported for ROCm")
    else:
        raise ValueError(f"Invalid backend: {backend}")


def get_pod_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode_p: int,
    use_sliding_window_p: bool,
    use_logits_soft_cap_p: bool,
    use_fp16_qk_reduction: bool,
    dtype_idx: torch.dtype,
    pos_encoding_mode_d: int,
    use_sliding_window_d: bool,
    use_logits_soft_cap_d: bool,
) -> str:
    return (
        f"pod_with_kv_cache_dtype_q_{filename_safe_dtype_map[dtype_q]}_"
        f"dtype_kv_{filename_safe_dtype_map[dtype_kv]}_"
        f"dtype_o_{filename_safe_dtype_map[dtype_o]}_"
        f"head_dim_{head_dim}_"
        f"posenc_p_{pos_encoding_mode_p}_"
        f"use_swa_p_{use_sliding_window_p}_"
        f"use_logits_cap_p_{use_logits_soft_cap_p}_"
        f"posenc_d_{pos_encoding_mode_d}_"
        f"use_swa_d_{use_sliding_window_d}_"
        f"use_logits_cap_d_{use_logits_soft_cap_d}_"
        f"dtype_idx_{filename_safe_dtype_map[dtype_idx]}_"
        f"f16qk_{use_fp16_qk_reduction}"
    )


def get_batch_pod_uri(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode_p: int,
    use_sliding_window_p: bool,
    use_logits_soft_cap_p: bool,
    use_fp16_qk_reduction: bool,
    dtype_idx: torch.dtype,
    pos_encoding_mode_d: int,
    use_sliding_window_d: bool,
    use_logits_soft_cap_d: bool,
) -> str:
    return "batch_" + get_pod_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
        pos_encoding_mode_p,
        use_sliding_window_p,
        use_logits_soft_cap_p,
        use_fp16_qk_reduction,
        dtype_idx,
        pos_encoding_mode_d,
        use_sliding_window_d,
        use_logits_soft_cap_d,
    )


def gen_pod_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode_p: int,
    use_sliding_window_p: bool,
    use_logits_soft_cap_p: bool,
    use_fp16_qk_reduction: bool,
    dtype_idx: torch.dtype,
    pos_encoding_mode_d: int,
    use_sliding_window_d: bool,
    use_logits_soft_cap_d: bool,
) -> JitSpec:
    uri = get_pod_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
        pos_encoding_mode_p,
        use_sliding_window_p,
        use_logits_soft_cap_p,
        use_fp16_qk_reduction,
        dtype_idx,
        pos_encoding_mode_d,
        use_sliding_window_d,
        use_logits_soft_cap_d,
    )
    variant_name_p = f"DefaultAttention<use_custom_mask_p, {str(use_sliding_window_p).lower()}, {str(use_logits_soft_cap_p).lower()}, {str(pos_encoding_mode_p == 2).lower()}>"
    variant_name_d = f"DefaultAttention<use_custom_mask_d, {str(use_sliding_window_d).lower()}, {str(use_logits_soft_cap_d).lower()}, {str(pos_encoding_mode_d == 2).lower()}>"
    return gen_customize_pod_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        variant_name_p,
        variant_name_d,
        pos_encoding_mode_p=pos_encoding_mode_p,
        use_sliding_window_p=use_sliding_window_p,
        use_logits_soft_cap_p=use_logits_soft_cap_p,
        pos_encoding_mode_d=pos_encoding_mode_d,
        use_sliding_window_d=use_sliding_window_d,
        use_logits_soft_cap_d=use_logits_soft_cap_d,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
    )


def gen_batch_pod_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim: int,
    pos_encoding_mode_p: int,
    use_sliding_window_p: bool,
    use_logits_soft_cap_p: bool,
    use_fp16_qk_reduction: bool,
    dtype_idx: torch.dtype,
    pos_encoding_mode_d: int,
    use_sliding_window_d: bool,
    use_logits_soft_cap_d: bool,
) -> JitSpec:
    uri = get_batch_pod_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim,
        pos_encoding_mode_p,
        use_sliding_window_p,
        use_logits_soft_cap_p,
        use_fp16_qk_reduction,
        dtype_idx,
        pos_encoding_mode_d,
        use_sliding_window_d,
        use_logits_soft_cap_d,
    )
    variant_name_p = f"DefaultAttention<use_custom_mask_p, {str(use_sliding_window_p).lower()}, {str(use_logits_soft_cap_p).lower()}, {str(pos_encoding_mode_p == 2).lower()}>"
    variant_name_d = f"DefaultAttention<use_custom_mask_d, {str(use_sliding_window_d).lower()}, {str(use_logits_soft_cap_d).lower()}, {str(pos_encoding_mode_d == 2).lower()}>"
    return gen_customize_batch_pod_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        variant_name_p,
        variant_name_d,
        pos_encoding_mode_p=pos_encoding_mode_p,
        use_sliding_window_p=use_sliding_window_p,
        use_logits_soft_cap_p=use_logits_soft_cap_p,
        pos_encoding_mode_d=pos_encoding_mode_d,
        use_sliding_window_d=use_sliding_window_d,
        use_logits_soft_cap_d=use_logits_soft_cap_d,
        use_fp16_qk_reduction=use_fp16_qk_reduction,
    )


def _gen_customize_pod_like_module(
    prefix: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    variant_name_p: str,
    variant_name_d: str,
    pos_encoding_mode_p: int,
    use_sliding_window_p: bool,
    use_logits_soft_cap_p: bool,
    pos_encoding_mode_d: int,
    use_sliding_window_d: bool,
    use_logits_soft_cap_d: bool,
    use_fp16_qk_reduction: bool,
) -> JitSpec:
    """Shared body for single (prefix="pod") and batch (prefix="batch_pod") POD JIT modules."""
    gen_directory = FLASHINFER_GEN_SRC_DIR / uri

    kwargs = {
        "variant_name_p": variant_name_p,
        "variant_name_d": variant_name_d,
        "dtype_q": dtype_map_hip[dtype_q],
        "dtype_kv": dtype_map_hip[dtype_kv],
        "dtype_o": dtype_map_hip[dtype_o],
        "idtype": dtype_map_hip[dtype_idx],
        "head_dim_qk": head_dim,
        "head_dim_vo": head_dim,
        "pos_encoding_mode_p": pos_encoding_mode_literal[pos_encoding_mode_p],
        "pos_encoding_mode_d": pos_encoding_mode_literal[pos_encoding_mode_d],
        "use_sliding_window_p": str(use_sliding_window_p).lower(),
        "use_logits_soft_cap_p": str(use_logits_soft_cap_p).lower(),
        "use_sliding_window_d": str(use_sliding_window_d).lower(),
        "use_logits_soft_cap_d": str(use_logits_soft_cap_d).lower(),
        "use_fp16_qk_reduction": str(use_fp16_qk_reduction).lower(),
    }

    with open(FLASHINFER_CSRC_DIR / f"{prefix}_customize_config.jinja") as f:
        config_templ = jinja2.Template(f.read())
    with open(FLASHINFER_CSRC_DIR / f"{prefix}_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    os.makedirs(gen_directory, exist_ok=True)
    write_if_different(
        gen_directory / f"{prefix}_config.inc", config_templ.render(**kwargs)
    )

    source_paths = []
    for mask_mode_p in [0, 1, 2]:
        for mask_mode_d in [0, 1, 2]:
            dest_path = (
                gen_directory / f"{prefix}_kernel_mask_{mask_mode_p}p_{mask_mode_d}d.cu"
            )
            source_paths.append(dest_path)
            write_if_different(
                dest_path,
                kernel_inst_templ.render(
                    mask_mode_p=mask_mode_literal[mask_mode_p],
                    mask_mode_d=mask_mode_literal[mask_mode_d],
                    **kwargs,
                ),
            )

    for filename in [f"{prefix}.cu", f"{prefix}_jit_pybind.cu"]:
        src_path = FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            write_if_different(dest_path, f.read())

    return gen_jit_spec(uri, source_paths)


def gen_customize_pod_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    variant_name_p: str,
    variant_name_d: str,
    pos_encoding_mode_p: int = 0,
    use_sliding_window_p: bool = False,
    use_logits_soft_cap_p: bool = False,
    pos_encoding_mode_d: int = 0,
    use_sliding_window_d: bool = False,
    use_logits_soft_cap_d: bool = False,
    use_fp16_qk_reduction: bool = False,
) -> JitSpec:
    return _gen_customize_pod_like_module(
        "pod",
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        variant_name_p,
        variant_name_d,
        pos_encoding_mode_p,
        use_sliding_window_p,
        use_logits_soft_cap_p,
        pos_encoding_mode_d,
        use_sliding_window_d,
        use_logits_soft_cap_d,
        use_fp16_qk_reduction,
    )


def gen_customize_batch_pod_module(
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim: int,
    variant_name_p: str,
    variant_name_d: str,
    pos_encoding_mode_p: int = 0,
    use_sliding_window_p: bool = False,
    use_logits_soft_cap_p: bool = False,
    pos_encoding_mode_d: int = 0,
    use_sliding_window_d: bool = False,
    use_logits_soft_cap_d: bool = False,
    use_fp16_qk_reduction: bool = False,
) -> JitSpec:
    return _gen_customize_pod_like_module(
        "batch_pod",
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim,
        variant_name_p,
        variant_name_d,
        pos_encoding_mode_p,
        use_sliding_window_p,
        use_logits_soft_cap_p,
        pos_encoding_mode_d,
        use_sliding_window_d,
        use_logits_soft_cap_d,
        use_fp16_qk_reduction,
    )
