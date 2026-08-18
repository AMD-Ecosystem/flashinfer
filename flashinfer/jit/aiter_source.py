# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shared plumbing for FlashInfer's C++-level AITER backends (ROCm).

FlashInfer wraps AITER kernels by compiling a small ``csrc_rocm/*_aiter.cu`` shim
that calls AITER's C++ entry point directly and links the symbol-visible AITER
``.so``. The shim forward-declares the entry point rather than ``#include``-ing
AITER's public header, because that header pulls in pybind11, which clashes with
FlashInfer's ``-DPy_LIMITED_API`` build; ``torch::Tensor`` is ``at::Tensor``, so
the linker resolves the symbol from the AITER ``.so``.

AITER's installed wheel builds its modules with ``-fvisibility=hidden``, so the
kernel symbols (e.g. ``rope_cached_positions_2c_fwd_impl``) are not linkable. This
helper rebuilds the needed AITER module once with ``AITER_SYMBOL_VISIBLE=1`` via
AITER's own ``aiter.jit.core.build_module`` (which also runs CK blob codegen for CK
ops), caches the result under FlashInfer's cache dir as ``lib<module>.so``, and
hands back the include/link flags for ``gen_jit_spec``.
"""

import functools
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ..arch_caps import normalize_arch
from . import env as jit_env
from .core import logger


_DEFAULT_BUILD_ARCH = "gfx942"


def _env_arch_list() -> List[str]:
    """FLASHINFER_ROCM_ARCH_LIST as a normalized list, accepting ',' or ';'."""
    import re

    raw = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "")
    return [normalize_arch(a) for a in re.split(r"[;,]", raw) if a.strip()]


def _detected_device_arch() -> Optional[str]:
    """The running device's architecture, or None if no GPU is visible.

    torch is imported lazily: this module is imported during JIT setup, and a
    GPU-less wheel build must not require a working HIP runtime.
    """
    try:
        import torch

        if torch.cuda.device_count() == 0:
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return normalize_arch(props.gcnArchName)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def resolve_aiter_build_arch() -> str:
    """Return the **single** GPU architecture to build the AITER shim for.

    Deliberately one architecture, never a list. AITER's ``get_gfx()`` resolves
    to the *last* ``GPU_ARCHS`` entry rather than the running device, so a
    multi-arch value makes AITER's Python-level dispatch believe it is on that
    architecture no matter what the hardware is -- and it also flips global
    compile flags. One entry makes that failure mode unreachable.

    Resolution order: ``FLASHINFER_ROCM_ARCH_LIST`` -> detected device ->
    ``gfx942``. When the environment names several architectures, the running
    device's is preferred among them; when the environment names architectures
    that exclude the running device, the environment still wins (cross-compiling
    is legitimate) but the mismatch is reported, because the resulting shim will
    fault on this machine.
    """
    env_archs = _env_arch_list()
    device_arch = _detected_device_arch()

    if env_archs:
        if device_arch and device_arch in env_archs:
            return device_arch
        if device_arch:
            logger.warning(
                "FLASHINFER_ROCM_ARCH_LIST=%s does not include this device's "
                "architecture (%s); building the AITER shim for %s. It will not "
                "run on this GPU.",
                ",".join(env_archs),
                device_arch,
                env_archs[0],
            )
        return env_archs[0]

    if device_arch:
        return device_arch

    logger.warning(
        "No ROCm device detected and FLASHINFER_ROCM_ARCH_LIST is unset; "
        "building the AITER shim for %s. Set FLASHINFER_ROCM_ARCH_LIST to "
        "target a different architecture.",
        _DEFAULT_BUILD_ARCH,
    )
    return _DEFAULT_BUILD_ARCH


def _aiter_cache_tag() -> str:
    """A filesystem-safe tag keying the lib cache by target arch and AITER version.

    Without the arch component, a lib built for one arch would be silently reused
    on a machine with a different arch. Without the version component, an AITER
    upgrade (which can change the C++ ABI the FlashInfer shim links against) would
    silently reuse the stale .so. The FlashInfer JIT dir already keys by its own
    version+arch, but this cache sits outside it.

    Keyed on the *resolved* architecture -- the one actually compiled for -- so
    the tag cannot disagree with the contents of the directory it names."""
    arch = resolve_aiter_build_arch()
    try:
        import importlib.metadata as _md

        version = _md.version("amd-aiter")
    except Exception:
        version = "unknown"
    return f"{arch}__aiter-{version}"


@functools.lru_cache(maxsize=1)
def _aiter_libs_dir() -> Path:
    # Keyed by arch + AITER version so a cached lib is never reused across an
    # incompatible arch or a changed AITER ABI.
    d = jit_env.FLASHINFER_CACHE_DIR / "aiter_libs" / _aiter_cache_tag()
    d.mkdir(parents=True, exist_ok=True)
    return d


@functools.lru_cache(maxsize=1)
def _aiter_csrc_include_dir() -> Path:
    """The aiter_meta C++ public header dir (rmsnorm.h / activation.h / rope.h)."""
    # aiter ships its C++ sources/headers in the sibling aiter_meta package,
    # which is a namespace package (no __file__) — resolve via __path__.
    import aiter_meta

    for p in aiter_meta.__path__:
        inc = Path(p) / "csrc" / "include"
        if inc.exists():
            return inc
    raise RuntimeError(
        "Could not locate aiter_meta/csrc/include; is the aiter source package installed?"
    )


def ensure_aiter_lib(module_name: str) -> Path:
    """
    Build (once, cached) a symbol-visible AITER module and return the path to the
    linkable ``lib<module_name>.so`` under FlashInfer's cache.

    Idempotent: if the cached lib already exists it is returned without rebuilding.
    """
    libs_dir = _aiter_libs_dir()
    lib_path = libs_dir / f"lib{module_name}.so"
    if lib_path.exists():
        return lib_path

    # Build into a FlashInfer-owned dir so we never mutate the AITER install.
    aiter_build_dir = libs_dir / "build"
    aiter_build_dir.mkdir(parents=True, exist_ok=True)

    # AITER reads these from the environment at build time.
    from ..hip_utils import get_rocm_home

    prev = {
        "AITER_SYMBOL_VISIBLE": os.environ.get("AITER_SYMBOL_VISIBLE"),
        "AITER_JIT_DIR": os.environ.get("AITER_JIT_DIR"),
        "GPU_ARCHS": os.environ.get("GPU_ARCHS"),
        "ROCM_HOME": os.environ.get("ROCM_HOME"),
    }
    os.environ["AITER_SYMBOL_VISIBLE"] = "1"
    os.environ["AITER_JIT_DIR"] = str(aiter_build_dir)
    # AITER splits GPU_ARCHS on ';' and validates each entry, so a comma-joined
    # list reaches it as one unparseable token. A single architecture sidesteps
    # the separator entirely -- and is required regardless; see
    # resolve_aiter_build_arch.
    os.environ["GPU_ARCHS"] = resolve_aiter_build_arch()
    os.environ["ROCM_HOME"] = get_rocm_home()

    built: Optional[Path] = None
    try:
        from aiter.jit import core as aiter_core
        from aiter.jit.core import build_module, get_args_of_build

        # AITER's cpp_extension bakes -fvisibility=hidden into COMMON_HIPCC_FLAGS
        # at import time when AITER_SYMBOL_VISIBLE is unset. If aiter was already
        # imported (e.g. the MHA/MLA path ran first), setting the env var now is
        # too late, so force default visibility via the per-build flags — they are
        # appended after COMMON_HIPCC_FLAGS, and the later flag wins. Without this,
        # the rebuilt .so hides the kernel symbols and the FlashInfer shim fails to
        # link with "undefined symbol".
        a = get_args_of_build(module_name)
        flags_extra_hip = list(a["flags_extra_hip"]) + ["-fvisibility=default"]
        build_module(
            md_name=module_name,
            srcs=a["srcs"],
            flags_extra_cc=a["flags_extra_cc"],
            flags_extra_hip=flags_extra_hip,
            blob_gen_cmd=a["blob_gen_cmd"],
            extra_include=a["extra_include"],
            extra_ldflags=a["extra_ldflags"],
            verbose=os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1",
            is_python_module=a["is_python_module"],
            is_standalone=a["is_standalone"],
            torch_exclude=a["torch_exclude"],
            hipify=a.get("hipify", False),
        )

        # AITER decides the output dir from a module-level `bd_dir` global that is
        # frozen at import time, so the .so does not reliably land in
        # aiter_build_dir when aiter was imported before we set AITER_JIT_DIR.
        # Resolve the produced file from AITER's own get_user_jit_dir() (which
        # re-reads the env), falling back to a recursive search of our build dir.
        built = _find_built_so(
            module_name, aiter_build_dir, Path(aiter_core.get_user_jit_dir())
        )
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    if built is None:
        raise RuntimeError(
            f"AITER build for {module_name!r} did not produce a {module_name}.so. "
            "Check that aiter is installed and ROCm is available."
        )
    # Copy (not symlink) the artifact into our arch-keyed cache so it survives
    # cleanup of AITER's build tree, and publish it atomically via os.replace so a
    # concurrent loader never observes a partial/missing file.
    tmp_lib = lib_path.with_name(f".{lib_path.name}.{os.getpid()}.tmp")
    shutil.copy2(built, tmp_lib)
    os.replace(tmp_lib, lib_path)
    return lib_path


def _find_built_so(module_name: str, *search_dirs: Path) -> Optional[Path]:
    """Locate the freshly built ``<module_name>.so`` across AITER's output dirs."""
    name = f"{module_name}.so"
    for d in search_dirs:
        candidate = d / name
        if candidate.is_file():
            return candidate
    for d in search_dirs:
        if d.exists():
            for found in d.rglob(name):
                if found.is_file():
                    return found
    return None


def aiter_jitspec_flags(
    module_name: str,
) -> Tuple[List[Union[str, Path]], List[str]]:
    """
    Build the AITER lib if needed and return ``(extra_include_paths, extra_ldflags)``
    to pass to ``gen_jit_spec`` so the FlashInfer shim can find AITER's header and
    link the kernel.
    """
    ensure_aiter_lib(module_name)
    libs_dir = _aiter_libs_dir()
    extra_include_paths: List[Union[str, Path]] = [str(_aiter_csrc_include_dir())]
    extra_ldflags = [
        f"-L{libs_dir}",
        f"-l{module_name}",
        f"-Wl,-rpath,{libs_dir}",
    ]
    return extra_include_paths, extra_ldflags
