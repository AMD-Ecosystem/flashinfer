"""
Copyright (c) 2025 by FlashInfer team.

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

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_rope_module() -> JitSpec:
    return gen_jit_spec(
        "rope",
        [
            jit_env.FLASHINFER_CSRC_DIR / "rope.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_rope_binding.cu",
        ],
    )


def gen_rope_aiter_module() -> JitSpec:
    from .rocm.aiter_source import aiter_jitspec_flags, refresh_aiter_jitspec

    # AITER split the monolithic rope module by variant. Through 0.1.10 the whole
    # forward path lived in "module_rope_pos_fwd"; from 0.1.16 that name is not
    # registered at all, and the entry point this shim calls
    # (rope_cached_positions_2c_fwd_impl) is built by the 2c cached-positions
    # module. Asking for the old name does not error usefully -- AITER hands back
    # an empty source list and the JIT dies on `assert len(sources) > 0`.
    extra_include_paths, extra_ldflags = aiter_jitspec_flags(
        "module_rope_2c_cached_positions_fwd"
    )
    return refresh_aiter_jitspec(
        gen_jit_spec(
            "rope_aiter",
            [
                jit_env.FLASHINFER_CSRC_DIR / "rope_aiter.cu",
                jit_env.FLASHINFER_CSRC_DIR / "rope_aiter_jit_pybind.cu",
            ],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
        )
    )
