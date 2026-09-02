# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The public surface `import flashinfer` exposes on ROCm.

Lives here rather than in an ``elif IS_HIP:`` arm of ``flashinfer/__init__.py``
so the upstream file carries a one-line delegation instead of ~115 lines that
conflict on every sync.
"""

from .._version import __version__ as __version__
from . import gate_cuda_only_modules
from .hip_utils import check_torch_rocm_compatibility

# Checks compatibility with installed torch
check_torch_rocm_compatibility()

# Before any of the imports below, so a gated name cannot be resolved on the way
# in. flashinfer.comm also calls this, but `import flashinfer` never reaches it
# on HIP, which left the gate uninstalled for every caller that did not import
# comm first.
gate_cuda_only_modules()

# ========================================
# HIP/ROCm Imports (AMD-ported modules)
# ========================================
from .. import jit as jit
from ..activation import gelu_and_mul as gelu_and_mul
from ..activation import gelu_tanh_and_mul as gelu_tanh_and_mul
from ..activation import silu_and_mul as silu_and_mul
from .fused_moe import aiter_fused_moe as aiter_fused_moe
from .fused_moe import moe_fp8_dtype as moe_fp8_dtype
from .fused_moe import quantize_moe_weight as quantize_moe_weight
from .fused_moe import shuffle_moe_weight as shuffle_moe_weight
from ..get_include_paths import get_csrc_dir as get_csrc_dir
from ..get_include_paths import get_include as get_include
from ..norm import fused_add_rmsnorm as fused_add_rmsnorm
from ..norm import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm
from ..norm import gemma_rmsnorm as gemma_rmsnorm
from ..norm import layernorm as layernorm
from ..norm import rmsnorm as rmsnorm
from ..page import append_paged_kv_cache as append_paged_kv_cache
from ..page import append_paged_mla_kv_cache as append_paged_mla_kv_cache
from ..page import get_batch_indices_positions as get_batch_indices_positions
from ..page import get_seq_lens as get_seq_lens
from .prefill import (
    single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
)
from ..quantization.packbits import packbits as packbits
from ..quantization.packbits import segment_packbits as segment_packbits
from ..rope import apply_llama31_rope as apply_llama31_rope
from ..rope import apply_llama31_rope_inplace as apply_llama31_rope_inplace
from ..rope import apply_llama31_rope_pos_ids as apply_llama31_rope_pos_ids
from ..rope import (
    apply_llama31_rope_pos_ids_inplace as apply_llama31_rope_pos_ids_inplace,
)
from ..rope import apply_rope as apply_rope
from ..rope import apply_rope_inplace as apply_rope_inplace
from ..rope import apply_rope_pos_ids as apply_rope_pos_ids
from ..rope import apply_rope_pos_ids_inplace as apply_rope_pos_ids_inplace
from ..rope import (
    apply_rope_with_cos_sin_cache as apply_rope_with_cos_sin_cache,
)
from ..rope import (
    apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace,
)
from ..sampling import chain_speculative_sampling as chain_speculative_sampling
from ..sampling import min_p_sampling_from_probs as min_p_sampling_from_probs
from ..sampling import sampling_from_logits as sampling_from_logits
from ..sampling import sampling_from_probs as sampling_from_probs
from ..sampling import softmax as softmax
from ..sampling import top_k_mask_logits as top_k_mask_logits
from ..sampling import top_k_renorm_probs as top_k_renorm_probs
from ..sampling import top_k_sampling_from_probs as top_k_sampling_from_probs
from ..sampling import (
    top_k_top_p_sampling_from_logits as top_k_top_p_sampling_from_logits,
)
from ..sampling import (
    top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs,
)
from ..sampling import top_p_renorm_probs as top_p_renorm_probs
from ..sampling import top_p_sampling_from_probs as top_p_sampling_from_probs

# ========================================
# Module Aliases (for CUDA API compatibility)
# ========================================
from . import install_shadow_modules as _install_shadow_modules

# Binds flashinfer.decode / .prefill / .mla as attributes as well as in
# sys.modules, so both `import flashinfer.mla` and `flashinfer.mla.X` reach
# the ROCm module.
_install_shadow_modules()

# Cascade imports must come after the sys.modules injection above so that
# cascade.py's relative imports of flashinfer.decode / flashinfer.prefill
# resolve to the ROCm implementations.
from ..cascade import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper as BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
)
from ..cascade import (
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper as BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
)
from ..cascade import (
    MultiLevelCascadeAttentionWrapper as MultiLevelCascadeAttentionWrapper,
)
from ..cascade import merge_state as merge_state
from ..cascade import merge_state_in_place as merge_state_in_place
from ..cascade import merge_states as merge_states
from ..pod import PODWithPagedKVCacheWrapper as PODWithPagedKVCacheWrapper
from ..pod import BatchPODWithPagedKVCacheWrapper as BatchPODWithPagedKVCacheWrapper

from ..utils import next_positive_power_of_2 as next_positive_power_of_2
from .torch_compile import use_torch_custom_ops_enabled as use_torch_custom_ops_enabled
