"""
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2025 by AMD ROCm team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Global compilation context management for FlashInfer on ROCm.
"""

import logging


from . import hip_utils

logger = logging.getLogger(__name__)


class CompilationContext:
    """Manages ROCm compilation targets with comprehensive validation."""

    COMMON_HIPCC_FLAGS = [
        "-DFLASHINFER_ENABLE_HIP",
        "-DFLASHINFER_ENABLE_FP8",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
        # Required from torch 2.12 on. The c10::hip / at::hip compatibility
        # namespaces in c10/hip/HIPStream.h ("hipify v2 backward compat in
        # external projects") are wrapped in `#ifdef USE_ROCM`. Without this,
        # c10::hip::getCurrentHIPStream() -- used by every *_aiter.cu shim --
        # fails to resolve, while the namespace itself still exists via other
        # headers, so the error reads "no member named ... in namespace
        # 'c10::hip'" rather than a missing include. Earlier torch releases
        # declared the block unconditionally, which is why this was not needed
        # before. AITER's own builds already pass -DUSE_ROCM=1.
        "-DUSE_ROCM=1",
    ]

    def __init__(self):
        """
        Initialize and validate ROCm architectures once.

        Performs comprehensive validation:
        1. System ROCm version compatibility
        2. FlashInfer AMD port availability
        3. PyTorch ROCm compilation support
        """
        import torch.utils.cpp_extension as torch_cpp_ext

        # One resolver for every path that asks "what are we building for", so
        # this cannot disagree with the validation in hip_utils -- it used to,
        # returning gfx950 here while validation checked gfx942.
        arch_list = hip_utils.resolve_target_archs()
        logger.info(f"Target ROCm architectures: {arch_list}")

        # Comprehensive validation (all 3 checks)
        self.arch_flags, self.TARGET_ROCM_ARCHS = (
            hip_utils.validate_flashinfer_rocm_arch(
                arch_list=arch_list, torch_cpp_ext_module=torch_cpp_ext, verbose=False
            )
        )

    def get_hipcc_flags_list(self) -> list[str]:
        """
        Generate hipcc compiler flags for target architectures.

        Returns:
            List of flags like ["--offload-arch=gfx942", "--offload-arch=gfx90a", ...]
        """
        return self.arch_flags + self.COMMON_HIPCC_FLAGS

    def get_target_archs(self) -> set[str]:
        """
        Get the set of target architectures.

        Returns:
            Set of architecture strings like {"gfx942", "gfx90a"}
        """
        return self.TARGET_ROCM_ARCHS.copy()

    def has_arch(self, arch: str) -> bool:
        """
        Check if a specific architecture is targeted.

        Args:
            arch: Architecture string like "gfx942"

        Returns:
            True if the architecture is in the target set
        """
        return arch in self.TARGET_ROCM_ARCHS
