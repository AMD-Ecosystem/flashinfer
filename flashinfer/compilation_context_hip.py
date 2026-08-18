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
import os

import torch

from . import hip_utils
from .arch_caps import normalize_arch

logger = logging.getLogger(__name__)


class CompilationContext:
    """Manages ROCm compilation targets with comprehensive validation."""

    COMMON_HIPCC_FLAGS = [
        "-DFLASHINFER_ENABLE_HIP",
        "-DFLASHINFER_ENABLE_FP8",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
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

        # Get architecture list from env or auto-detect
        arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST")
        if arch_list is None:
            arch_list = self._auto_detect_archs()
            if arch_list:
                logger.info(f"Auto-detected ROCm architectures: {arch_list}")

        # Comprehensive validation (all 3 checks)
        self.arch_flags, self.TARGET_ROCM_ARCHS = (
            hip_utils.validate_flashinfer_rocm_arch(
                arch_list=arch_list, torch_cpp_ext_module=torch_cpp_ext, verbose=False
            )
        )

    def _auto_detect_archs(self) -> str:
        """Auto-detect ROCm architectures from supported system devices.

        Only devices whose gcnArchName is in FLASHINFER_SUPPORTED_ROCM_ARCHS are
        considered, so unsupported integrated GPUs are silently ignored here rather
        than being passed on to validate_flashinfer_rocm_arch for filtering.
        """
        try:
            indices = hip_utils.get_supported_device_indices()
            if indices:
                archs = {
                    normalize_arch(torch.cuda.get_device_properties(i).gcnArchName)
                    for i in indices
                }
                return ",".join(sorted(archs))
            logger.warning("No supported ROCm devices detected, defaulting to gfx942")
            return "gfx942"
        except Exception as e:
            logger.warning(f"Failed to auto-detect ROCm device architectures: {e}")
            return "gfx942"

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
