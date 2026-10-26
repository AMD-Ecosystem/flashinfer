# SPDX-FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier : Apache-2.0

"""AMD FlashInfer JIT Cache Package

This package provides pre-compiled HIP kernels for FlashInfer on AMD ROCm platforms.
"""

from pathlib import Path

# Get the path to the AOT modules directory within this package
jit_cache_dir = Path(__file__).parent / "jit_cache"


def get_jit_cache_dir() -> str:
    """Get the path to the jit_cache directory containing pre-compiled kernels.

    Returns:
        str: Absolute path to the jit_cache directory
    """
    return str(jit_cache_dir)


def get_aiter_variant_dir() -> str:
    """Path to the prebuilt AITER attention variants, if this wheel carries any.

    Holds one subdirectory per ``<arch>__aiter-<version>__rocm-<version>`` tag,
    so a wheel built for several architectures can ship several and the consumer
    selects the one matching its own install. The directory need not exist: a
    wheel built without running the prebuild simply has none, and FlashInfer
    falls back to building variants on demand as it always did.

    Returns:
        str: Absolute path to the aiter_variants directory
    """
    return str(jit_cache_dir / "aiter_variants")


try:
    from ._version import __version__ as __version__
except (ModuleNotFoundError, ImportError):
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "get_aiter_variant_dir",
    "get_jit_cache_dir",
]
