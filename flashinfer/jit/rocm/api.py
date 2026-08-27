# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What ``flashinfer.jit`` re-exports on ROCm.

Lives here rather than in an ``elif IS_HIP:`` arm of ``flashinfer/jit/__init__.py``
so the upstream file carries a delegation instead of ~45 conflicting lines.
"""

from .. import env as env
from ..activation import gen_act_and_mul_module as gen_act_and_mul_module
from ..activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from ..attention import gen_batch_decode_module as gen_batch_decode_module
from ..attention import (
    gen_batch_decode_aiter_module as gen_batch_decode_aiter_module,
)
from ..attention import (
    get_batch_decode_aiter_uri as get_batch_decode_aiter_uri,
)
from ..attention import gen_batch_prefill_module as gen_batch_prefill_module
from ..attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from ..attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from ..attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from ..attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from ..attention import gen_single_decode_module as gen_single_decode_module
from ..attention import (
    gen_single_prefill_module as gen_single_prefill_module,
)
from ..attention import gen_pod_module as gen_pod_module
from ..attention import gen_batch_pod_module as gen_batch_pod_module
from ..attention import get_batch_decode_uri as get_batch_decode_uri
from ..attention import get_batch_prefill_uri as get_batch_prefill_uri
from ..attention import get_single_decode_uri as get_single_decode_uri
from ..attention import get_single_prefill_uri as get_single_prefill_uri
from ..attention import get_pod_uri as get_pod_uri
from ..attention import get_batch_pod_uri as get_batch_pod_uri
from ..core import JitSpec as JitSpec
from ..core import JitSpecStatus as JitSpecStatus
from ..core import JitSpecRegistry as JitSpecRegistry
from ..core import jit_spec_registry as jit_spec_registry
from ..core import MissingJITCacheError as MissingJITCacheError
from ..core import build_jit_specs as build_jit_specs
from ..core import clear_cache_dir as clear_cache_dir
from ..core import gen_jit_spec as gen_jit_spec
from ..norm import gen_norm_module as gen_norm_module
from ..page import gen_page_module as gen_page_module
from ..quantization import gen_quantization_module as gen_quantization_module
from ..rope import gen_rope_module as gen_rope_module
from ..sampling import gen_sampling_module as gen_sampling_module
