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


def gen_page_module() -> JitSpec:
    return gen_jit_spec(
        "page",
        [
            jit_env.FLASHINFER_CSRC_DIR / "page.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_page_binding.cu",
        ],
    )


def gen_page_aiter_module() -> JitSpec:
    from .rocm.aiter_source import aiter_jitspec_flags, refresh_aiter_jitspec

    extra_include_paths, extra_ldflags = aiter_jitspec_flags("module_cache")
    # Required: build.ninja is written only when missing, so without this a
    # cached module keeps linking whichever AITER library it first saw.
    return refresh_aiter_jitspec(
        gen_jit_spec(
            "page_aiter",
            [
                jit_env.FLASHINFER_CSRC_DIR / "page_aiter.cu",
                jit_env.FLASHINFER_CSRC_DIR / "page_aiter_jit_pybind.cu",
            ],
            extra_include_paths=extra_include_paths,
            extra_ldflags=extra_ldflags,
        )
    )
