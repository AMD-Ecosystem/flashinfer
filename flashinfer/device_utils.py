# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device detection and capability utilities for CUDA/ROCm backends.

This module provides a central location for device detection and backend
selection, avoiding scattered checks throughout the codebase.
"""

from torch import version


def is_hip_available() -> bool:
    """
    Check if ROCm/HIP backend is available.

    Returns:
        bool: True if PyTorch was built with ROCm/HIP support
    """
    return hasattr(version, "hip") and version.hip is not None


def is_cuda_available() -> bool:
    """
    Check if CUDA backend is available (and not HIP).

    Returns:
        bool: True if PyTorch was built with CUDA (not ROCm) support
    """
    return hasattr(version, "cuda") and version.cuda is not None


# Global constants - evaluated once at module import
# Use these throughout the codebase for device-specific logic
IS_HIP = is_hip_available()
IS_CUDA = is_cuda_available()
