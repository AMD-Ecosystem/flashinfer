"""
Copyright (c) 2024 by FlashInfer team.
Copyright (c) 2025-2026 Advanced Micro Devices, Inc.

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

# NOTE(lequn): Do not "from .jit.env import xxx".
# Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
# This helps AOT script to override envs.

import os
import pathlib
import re

from ..arch_caps import normalize_arch
from ..device_utils import IS_CUDA, IS_HIP


def has_flashinfer_jit_cache() -> bool:
    """
    Check if flashinfer_jit_cache module is available.

    Returns:
        True if flashinfer_jit_cache exists, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("flashinfer_jit_cache") is not None


def has_flashinfer_cubin() -> bool:
    """
    Check if flashinfer_cubin module is available.

    Returns:
        True if flashinfer_cubin exists, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("flashinfer_cubin") is not None


def has_amd_flashinfer_jit_cache() -> bool:
    """
    Check if amd_flashinfer_jit_cache module is available.

    Returns:
        True if amd_flashinfer_jit_cache exists, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("amd_flashinfer_jit_cache") is not None


FLASHINFER_BASE_DIR: pathlib.Path = pathlib.Path(
    os.getenv("FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

FLASHINFER_CACHE_DIR: pathlib.Path = FLASHINFER_BASE_DIR / ".cache" / "flashinfer"
_package_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]

if IS_CUDA:
    from ..compilation_context import CompilationContext
    from ..version import __version__ as flashinfer_version

    def _get_cubin_dir():
        """
        Get the cubin directory path with the following priority:
        1. flashinfer-cubin package if installed
        2. Environment variable FLASHINFER_CUBIN_DIR
        3. Default cache directory
        """
        if has_flashinfer_cubin():
            import flashinfer_cubin

            flashinfer_cubin_version = flashinfer_cubin.__version__
            if (
                not os.getenv("FLASHINFER_DISABLE_VERSION_CHECK")
                and flashinfer_version != flashinfer_cubin_version
            ):
                raise RuntimeError(
                    f"flashinfer-cubin version ({flashinfer_cubin_version}) does not match "
                    f"flashinfer version ({flashinfer_version}). "
                    "Please install the same version of both packages. "
                    "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
                )

            return pathlib.Path(flashinfer_cubin.get_cubin_dir())

        env_dir = os.getenv("FLASHINFER_CUBIN_DIR")
        if env_dir:
            return pathlib.Path(env_dir)

        return FLASHINFER_CACHE_DIR / "cubins"

    def _get_aot_dir():
        """
        Get the AOT directory path with the following priority:
        1. flashinfer-jit-cache package if installed
        2. Default fallback to _package_root / "data" / "aot"
        """
        if has_flashinfer_jit_cache():
            import flashinfer_jit_cache

            flashinfer_jit_cache_version = flashinfer_jit_cache.__version__
            # NOTE(Zihao): we don't use exact version match here because the version of flashinfer-jit-cache
            # contains the CUDA version suffix: e.g. 0.3.1+cu129.
            if not os.getenv(
                "FLASHINFER_DISABLE_VERSION_CHECK"
            ) and not flashinfer_jit_cache_version.startswith(flashinfer_version):
                raise RuntimeError(
                    f"flashinfer-jit-cache version ({flashinfer_jit_cache_version}) does not match "
                    f"flashinfer version ({flashinfer_version}). "
                    "Please install the same version of both packages. "
                    "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
                )

            return pathlib.Path(flashinfer_jit_cache.get_jit_cache_dir())

        return _package_root / "data" / "aot"

    def _get_workspace_dir_name() -> pathlib.Path:
        compilation_context = CompilationContext()
        # NOTE(Zihao): sorted() is crucial here to ensure deterministic directory names.
        arch = "_".join(
            f"{major}{minor}"
            for major, minor in sorted(compilation_context.TARGET_CUDA_ARCHS)
        )
        return FLASHINFER_CACHE_DIR / flashinfer_version / arch

    FLASHINFER_CUBIN_DIR: pathlib.Path = _get_cubin_dir()
    FLASHINFER_AOT_DIR: pathlib.Path = _get_aot_dir()
    FLASHINFER_WORKSPACE_DIR: pathlib.Path = _get_workspace_dir_name()
    FLASHINFER_JIT_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "cached_ops"
    FLASHINFER_GEN_SRC_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "generated"
    FLASHINFER_DATA: pathlib.Path = _package_root / "data"
    FLASHINFER_INCLUDE_DIR: pathlib.Path = _package_root / "data" / "include"
    FLASHINFER_CSRC_DIR: pathlib.Path = _package_root / "data" / "csrc"
    # FLASHINFER_SRC_DIR = _package_root / "data" / "src"
    CUTLASS_INCLUDE_DIRS: list = [
        _package_root / "data" / "cutlass" / "include",
        _package_root / "data" / "cutlass" / "tools" / "util" / "include",
    ]
    SPDLOG_INCLUDE_DIR: pathlib.Path = _package_root / "data" / "spdlog" / "include"

    def get_nvshmem_include_dirs():
        paths = os.environ.get("NVSHMEM_INCLUDE_PATH")
        if paths is not None:
            return [pathlib.Path(p) for p in paths.split(os.pathsep) if p]

        import nvidia.nvshmem

        path = pathlib.Path(nvidia.nvshmem.__path__[0]) / "include"
        return [path]

    def get_nvshmem_lib_dirs():
        paths = os.environ.get("NVSHMEM_LIBRARY_PATH")
        if paths is not None:
            return [pathlib.Path(p) for p in paths.split(os.pathsep) if p]

        import nvidia.nvshmem

        path = pathlib.Path(nvidia.nvshmem.__path__[0]) / "lib"
        return [path]

elif IS_HIP:
    from .._version import __version__ as flashinfer_version
    from ..get_include_paths import get_csrc_dir, get_include

    def _get_aot_dir_hip():
        """
        Get the HIP AOT directory path with the following priority:
        1. amd-flashinfer-jit-cache package if installed (with version check)
        2. Default fallback to _package_root / "data" / "aot"
        """
        if has_amd_flashinfer_jit_cache():
            import amd_flashinfer_jit_cache

            amd_jit_cache_version = amd_flashinfer_jit_cache.__version__
            # Version check mirrors the CUDA _get_aot_dir() pattern.
            # The AMD JIT cache version may include ROCm suffix (e.g. 0.5.3+rocm6.4),
            # so we check startswith rather than exact match.
            if not os.getenv(
                "FLASHINFER_DISABLE_VERSION_CHECK"
            ) and not amd_jit_cache_version.startswith(flashinfer_version):
                raise RuntimeError(
                    f"amd-flashinfer-jit-cache version ({amd_jit_cache_version}) does not match "
                    f"flashinfer version ({flashinfer_version}). "
                    "Please install the same version of both packages. "
                    "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
                )

            return pathlib.Path(amd_flashinfer_jit_cache.get_jit_cache_dir())

        return _package_root / "data" / "aot"

    def _get_workspace_dir_name() -> pathlib.Path:
        try:
            import torch

            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gcn_arch = props.gcnArchName
            # Extract gfx arch (e.g., "gfx942:sramecc+:xnack-" -> "gfx942").
            # The `gfx\d` guard only checks that this *looks* like an
            # architecture name before trusting it; the value itself comes from
            # normalize_arch, which keeps letter suffixes ("gfx90a") that the
            # previous `(gfx\d+)` capture silently truncated to "gfx90".
            arch = normalize_arch(gcn_arch)
            if not re.match(r"gfx\d", arch):
                from torch.utils.cpp_extension import _get_rocm_arch_flags

                flags = _get_rocm_arch_flags()
                archs = [
                    flag.replace("--offload-arch=", "")
                    for flag in flags
                    if flag.startswith("--offload-arch=")
                ]
                arch = archs[0] if archs else "noarch"

            # Validate that the current device is supported by FlashInfer.
            # Setting the current device is the caller's responsibility; we
            # detect a misconfiguration early with a clear error message.
            from ..hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS

            if arch != "noarch" and arch not in FLASHINFER_SUPPORTED_ROCM_ARCHS:
                raise RuntimeError(
                    f"torch.cuda.current_device() is device {torch.cuda.current_device()} "
                    f"with unsupported ROCm architecture '{arch}'. "
                    f"Please set the current device to a supported GPU before importing "
                    f"flashinfer (e.g. torch.cuda.set_device(<device_index>)). "
                    f"Supported architectures: {', '.join(FLASHINFER_SUPPORTED_ROCM_ARCHS)}"
                )
        except RuntimeError:
            raise
        except Exception:
            arch = "noarch"
        # e.g.: $HOME/.cache/flashinfer/0.5.3/gfx942/
        return FLASHINFER_CACHE_DIR / flashinfer_version / arch

    FLASHINFER_AOT_DIR: pathlib.Path = _get_aot_dir_hip()  # type: ignore[no-redef]
    FLASHINFER_WORKSPACE_DIR: pathlib.Path = _get_workspace_dir_name()  # type: ignore[no-redef]
    FLASHINFER_JIT_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "cached_ops"  # type: ignore[no-redef]
    FLASHINFER_GEN_SRC_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "generated"  # type: ignore[no-redef]
    FLASHINFER_INCLUDE_DIR: pathlib.Path = pathlib.Path(get_include())  # type: ignore[no-redef]
    FLASHINFER_CSRC_DIR: pathlib.Path = pathlib.Path(get_csrc_dir())  # type: ignore[no-redef]
