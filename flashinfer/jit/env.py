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


if IS_CUDA:
    # These must stay gated, unlike the helpers above: jit/comm.py imports on
    # ROCm, so gen_nvshmem_module() would reach them and fail on an absent
    # nvidia.nvshmem -- or, with NVSHMEM_* set, on a source csrc_rocm lacks.
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


if IS_CUDA:
    FLASHINFER_CUBIN_DIR: pathlib.Path = _get_cubin_dir()
    FLASHINFER_AOT_DIR: pathlib.Path = _get_aot_dir()
    FLASHINFER_WORKSPACE_DIR: pathlib.Path = _get_workspace_dir_name()
    FLASHINFER_JIT_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "cached_ops"
    FLASHINFER_GEN_SRC_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "generated"
    FLASHINFER_DATA: pathlib.Path = _package_root / "data"
    FLASHINFER_INCLUDE_DIR: pathlib.Path = _package_root / "data" / "include"
    FLASHINFER_CSRC_DIR: pathlib.Path = _package_root / "data" / "csrc"
    # FLASHINFER_SRC_DIR = _package_root / "data" / "src"
    CUTLASS_INCLUDE_DIRS: list[pathlib.Path] = [
        _package_root / "data" / "cutlass" / "include",
        _package_root / "data" / "cutlass" / "tools" / "util" / "include",
    ]
    SPDLOG_INCLUDE_DIR: pathlib.Path = _package_root / "data" / "spdlog" / "include"
elif IS_HIP:
    from ..get_include_paths import get_csrc_dir, get_include
    from .rocm.env import get_aot_dir as _get_aot_dir_hip
    from .rocm.env import get_workspace_dir as _get_workspace_dir_hip

    FLASHINFER_AOT_DIR: pathlib.Path = _get_aot_dir_hip(  # type: ignore[no-redef]
        _package_root, has_amd_flashinfer_jit_cache
    )
    FLASHINFER_WORKSPACE_DIR: pathlib.Path = _get_workspace_dir_hip(  # type: ignore[no-redef]
        FLASHINFER_CACHE_DIR
    )
    FLASHINFER_JIT_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "cached_ops"  # type: ignore[no-redef]
    FLASHINFER_GEN_SRC_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "generated"  # type: ignore[no-redef]
    FLASHINFER_INCLUDE_DIR: pathlib.Path = pathlib.Path(get_include())  # type: ignore[no-redef]
    FLASHINFER_CSRC_DIR: pathlib.Path = pathlib.Path(get_csrc_dir())  # type: ignore[no-redef]
