import torch

from flashinfer.device_utils import IS_HIP
from flashinfer.testing.utils import set_seed
from flashinfer.utils import get_compute_capability

# Re-exported: attention.py imports these from here.
from .rocm_utils import (  # noqa: F401
    HIP_DECODE_GQA_GROUP_SIZES,
    aiter_serves,
    fa2_backed_backends,
    get_device_arch,
    rocm_supported_backends,
)

# Output columns for the test results.
output_column_dict = {
    "perf": [
        "routine",
        "median_time",
        "std_time",
        "tflops",
        "tb_per_sec",
        "backend",
        # What backend="auto" actually resolved to, and why it declined AITER.
        # Empty on CUDA and for explicitly-named backends.
        "backend_resolved",
        "backend_fallback_reason",
    ],
    "attention": [
        "page_size",
        "batch_size",
        "s_qo",
        "s_kv",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim_qk",
        "head_dim_vo",
        "head_dim_ckv",
        "head_dim_kpe",
        "causal",
        "q_dtype",
        "kv_dtype",
        "avg_actual_seq_len",
        "random_actual_seq_len",
    ],
    "gemm": [
        "m",
        "n",
        "k",
        "group_size",
        "tile_size",
        "scale_major_mode",
        "out_dtype",
        "mma_sm",
        "use_128x4_sf_layout",
        "use_nvfp4",
    ],
    "moe": [
        "num_tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "local_expert_offset",
        "local_num_experts",
        "routing_method",
        "use_shuffled_weight",
        "weight_layout",
        "use_routing_bias",
        "use_routing_scales_on_input",
        "input_dtype",
        "weight_dtype",
        "gated_act",
        # CUTLASS fused MoE specific
        "cutlass_variant",
        "quantized_input",
        "tp_size",
        "tp_rank",
        "ep_size",
        "ep_rank",
    ],
    "general": [
        "refcheck",
        "no_cuda_graph",
        "use_cupti",
        "allow_output_mismatch",
        "random_seed",
        "case_tag",
        "generate_repro_command",
        "repro_command",
    ],
}

full_output_columns = (
    output_column_dict["perf"]
    + output_column_dict["attention"]
    + output_column_dict["gemm"]
    + output_column_dict["moe"]
    + output_column_dict["general"]
)

benchmark_apis = {
    "attention": [
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchPrefillWithRaggedKVCacheWrapper",
        "BatchMLAPagedAttentionWrapper",
    ],
    "gemm": [
        "gemm_fp8_nt_groupwise",
        "group_gemm_fp8_nt_groupwise",
        "bmm_fp8",
        "mm_fp4",
    ],
    "moe": [
        "trtllm_fp4_block_scale_moe",
        "trtllm_fp8_block_scale_moe",
        "trtllm_fp8_per_tensor_scale_moe",
        "cutlass_fused_moe",
    ],
}


def print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec):
    output_backend_width = 15
    print(
        f"[PERF] {backend.ljust(output_backend_width)[:output_backend_width]}:: median time {median_time:.3f} ms; std {std_time:.3f} ms; achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
    )


def get_device(args):
    # Synchronize to ensure that the device is ready after previous tests
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {gpu_name = }")
    return device


def is_close_stats(input, other, rtol=1e-5, atol=1e-8):
    close_tensor = torch.isclose(input, other, rtol=rtol, atol=atol)
    num_elements = close_tensor.numel()
    num_different_elements = num_elements - close_tensor.sum().item()
    return (
        num_different_elements,  # number of different elements
        num_elements,  # total number of elements in tensor
        num_different_elements / num_elements * 100.0,
    )


def dtype_str_to_torch_dtype(dtype_str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "fp8_e4m3":
        return torch.float8_e4m3fn
    elif dtype_str == "fp8_e5m2":
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


routine_cc_to_supported_backends = {
    # ATTENTION
    "BatchDecodeWithPagedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_decode_with_kv_cache
        "7.5": ["fa2"],
        "8.0": ["fa2", "fa2_tc", "cudnn"],
        "8.6": ["fa2", "fa2_tc", "cudnn"],
        "8.9": ["fa2", "fa2_tc", "cudnn"],
        "9.0": ["fa2", "fa2_tc", "cudnn", "trtllm-native"],
        "10.0": ["fa2", "fa2_tc", "cudnn", "trtllm-gen", "trtllm-native"],
        "10.3": ["fa2", "fa2_tc", "cudnn", "trtllm-gen", "trtllm-native"],
        "12.0": ["fa2", "fa2_tc", "cudnn", "trtllm-native"],
    },
    "BatchPrefillWithPagedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_context_with_kv_cache
        "7.5": [],
        "8.0": ["fa2", "cudnn"],
        "8.6": ["fa2", "cudnn"],
        "8.9": ["fa2", "cudnn"],
        "9.0": ["fa2", "fa3", "cudnn"],
        "10.0": ["fa2", "cudnn", "trtllm-gen", "trtllm-native"],
        "10.3": ["fa2", "cudnn", "trtllm-gen", "trtllm-native"],
        "12.0": ["fa2", "cudnn"],
    },
    "BatchPrefillWithRaggedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_ragged_attention_deepseek
        "7.5": [],
        "8.0": ["fa2", "cudnn"],
        "8.6": ["fa2", "cudnn"],
        "8.9": ["fa2", "cudnn"],
        "9.0": ["fa2", "fa3", "cudnn"],
        "10.0": ["fa2", "cudnn", "cutlass", "trtllm-native"],
        "10.3": ["fa2", "cudnn", "cutlass", "trtllm-native"],
        "12.0": ["fa2", "cudnn"],
    },
    "BatchMLAPagedAttentionWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_decode_with_kv_cache_mla
        "7.5": [],
        "8.0": ["fa2"],
        "8.6": ["fa2"],
        "8.9": ["fa2"],
        "9.0": ["fa2", "fa3"],
        "10.0": ["fa2", "cutlass", "trtllm-native"],
        "10.3": ["fa2", "cutlass", "trtllm-native"],
        "12.0": ["fa2"],
    },
    # GEMM
    "gemm_fp8_nt_groupwise": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass"],
        "10.3": ["cutlass"],
        "12.0": [],
    },
    "group_gemm_fp8_nt_groupwise": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass"],
        "10.3": ["cutlass"],
        "12.0": [],
    },
    "bmm_fp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": ["cudnn", "cublas"],
        "9.0": ["cudnn", "cublas"],
        "10.0": ["cudnn", "cublas", "cutlass"],
        "10.3": ["cudnn", "cublas", "cutlass"],
        "12.0": ["cudnn", "cublas"],
    },
    "mm_fp4": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cudnn", "trtllm", "cutlass"],
        "10.3": ["cudnn", "trtllm", "cutlass"],
        "12.0": ["cudnn", "cutlass"],
        "12.1": ["cudnn", "cutlass"],
    },
    # MOE
    "trtllm_fp4_block_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm"],
        "10.3": ["trtllm"],
        "12.0": [],
    },
    "trtllm_fp8_block_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm"],
        "10.3": ["trtllm"],
        "12.0": [],
    },
    "trtllm_fp8_per_tensor_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm"],
        "10.3": ["trtllm"],
        "12.0": [],
    },
    "cutlass_fused_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass"],
        "10.3": ["cutlass"],
        "12.0": [],
    },
}


def l2_flush_size_mb():
    """Flush-buffer size large enough to evict the device's last-level cache.

    CDNA's 256 MB Infinity Cache exactly equals the upstream buffer, which would
    leave its own tail resident.
    """
    return 512 if IS_HIP else 256


def bench_timing_kwargs(args, device):
    """Timing arguments shared by every bench_gpu_time call site.

    `bench_gpu_time` honours a `*_time_ms` budget only when the matching
    `*_iters` is None, so the two are mutually exclusive per phase.
    """
    kwargs = {
        "dry_run_iters": args.dry_run_iters,
        "repeat_iters": args.num_iters,
        "l2_flush": True,
        "l2_flush_size_mb": l2_flush_size_mb(),
        "l2_flush_device": device,
    }
    if getattr(args, "dry_run_time_ms", None) is not None:
        kwargs["dry_run_iters"] = None
        kwargs["dry_run_time_ms"] = args.dry_run_time_ms
    if getattr(args, "repeat_time_ms", None) is not None:
        kwargs["repeat_iters"] = None
        kwargs["repeat_time_ms"] = args.repeat_time_ms
    return kwargs


def as_nhd_paged_kv_cache(kv_cache):
    """View an HND-shaped ``[pages, 2, heads, page_size, dim]`` paged cache as NHD.

    ``result[p, i, s, h]`` is ``kv_cache[p, i, h, s]`` -- the same logical entry.
    Zero-copy, and contiguous when the caller built the cache NHD-ordered.
    """
    return kv_cache.transpose(2, 3)


def record_backend_resolution(cur_res, wrapper):
    """Record what ``auto`` resolved to, and why it declined AITER.

    Only meaningful for ``auto``: an explicit backend resolves to itself.
    """
    if wrapper is None or cur_res.get("backend") != "auto":
        return
    # `or ""` matters: the attribute exists and is None when auto did not decline,
    # and str(None) would write the literal "None" into the CSV.
    cur_res["backend_resolved"] = getattr(wrapper, "backend", "") or ""
    cur_res["backend_fallback_reason"] = (
        getattr(wrapper, "backend_fallback_reason", "") or ""
    )


def filter_backends_by_compute_capability(backends, routine, device):
    # FlashInfer currently does not have an isSupported() function that checks support.
    # WAR: Use helper function to check support.
    if IS_HIP:
        # gfx942/gfx950 report compute capability 9.4/9.5, which match no entry
        # in the NVIDIA table -- every backend would be stripped, including fa2.
        target = get_device_arch(device)
        label = f"architecture {target}"
        supported_backends = rocm_supported_backends(routine, device)
    else:
        major, minor = get_compute_capability(device)
        target = f"{major}.{minor}"
        label = f"compute capability {target}"
        # If the compute capability is not supported, return an empty list.
        supported_backends = routine_cc_to_supported_backends[routine].get(target, [])

    backends_to_remove = []
    for backend in backends:
        if backend not in supported_backends:
            backends_to_remove.append(backend)
    for backend in backends_to_remove:
        backends.remove(backend)
        print(
            f"[WARNING] {backend} for routine {routine} is not supported on {label}. Skipping."
        )
    return backends
