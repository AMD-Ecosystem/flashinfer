# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Where the ROCm JIT puts its AOT and workspace directories."""

import json
import os
import pathlib
import re
import warnings
from typing import Callable

from ..._version import __version__ as flashinfer_version
from ...arch_caps import normalize_arch

#: Records which architectures an AOT kernel tree was compiled for. Lives here
#: rather than in aot_hip so the reader owns the name and the writer imports it.
AOT_MANIFEST_NAME = "aot_manifest.json"


def get_aot_dir(
    package_root: pathlib.Path, has_jit_cache: Callable[[], bool]
) -> pathlib.Path:
    """Prefer the amd-flashinfer-jit-cache wheel, else the in-tree data dir.

    `has_jit_cache` is passed in rather than imported: env.py calls this while
    still executing its own module body, so importing back into it would cycle.
    The wheel's version may carry a ROCm suffix (0.5.3+rocm6.4), hence startswith.
    """
    if not has_jit_cache():
        return package_root / "data" / "aot"

    import amd_flashinfer_jit_cache

    cache_version = amd_flashinfer_jit_cache.__version__
    if not os.getenv(
        "FLASHINFER_DISABLE_VERSION_CHECK"
    ) and not cache_version.startswith(flashinfer_version):
        raise RuntimeError(
            f"amd-flashinfer-jit-cache version ({cache_version}) does not match "
            f"flashinfer version ({flashinfer_version}). "
            "Please install the same version of both packages. "
            "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
        )

    cache_dir = pathlib.Path(amd_flashinfer_jit_cache.get_jit_cache_dir())
    if _aot_arch_matches(cache_dir):
        return cache_dir
    # Fall back to the in-tree dir, which is normally empty, so JitSpec.is_aot
    # goes False and every module JITs for the architecture actually present.
    return package_root / "data" / "aot"


def _aot_arch_matches(cache_dir: pathlib.Path) -> bool:
    """Whether prebuilt kernels in ``cache_dir`` target an arch we can run.

    ``JitSpec.is_aot`` is a bare ``aot_path.exists()`` and AOT beats JIT
    unconditionally, while the wheel tag carries no gfx architecture -- so
    nothing else stops a gfx942 wheel serving gfx950. That costs a wrong-ISA
    load on ops in every forward pass, not just attention.

    A wheel with no manifest predates this check; assume it is right rather
    than breaking every install that already works.
    """
    if os.getenv("FLASHINFER_DISABLE_VERSION_CHECK"):
        return True

    manifest = cache_dir / AOT_MANIFEST_NAME
    if not manifest.exists():
        return True

    try:
        built_for = json.loads(manifest.read_text())["rocm_arch_list"]
    except (OSError, ValueError, KeyError):
        warnings.warn(
            f"Could not read {manifest}; using its prebuilt kernels unchecked.",
            stacklevel=2,
        )
        return True

    from ...hip_utils import resolve_target_archs

    built = {normalize_arch(a) for a in built_for.split(",") if a}
    target = {normalize_arch(a) for a in resolve_target_archs().split(",") if a}
    if target <= built:
        return True

    warnings.warn(
        f"amd-flashinfer-jit-cache was built for {sorted(built)} but this "
        f"system needs {sorted(target)}; ignoring the prebuilt kernels and "
        f"compiling from source. Install a matching wheel to avoid this. "
        f"Set FLASHINFER_DISABLE_VERSION_CHECK=1 to use them anyway.",
        stacklevel=2,
    )
    return False


def get_workspace_dir(cache_dir: pathlib.Path) -> pathlib.Path:
    """``<cache_dir>/<version>/<arch>``.

    With no GPU visible the arch comes from FLASHINFER_ROCM_ARCH_LIST instead, so
    importing does not need a device (#316). An unsupported *live* device still
    raises rather than degrading, so a wrong-arch cache hit cannot happen quietly.
    """
    try:
        import torch

        if torch.cuda.device_count() == 0:
            # Importing flashinfer must not require a live GPU. Keep the arch in
            # the path anyway: build.ninja is only written when absent, so one
            # shared directory would serve two ISAs the same objects.
            from ...hip_utils import resolve_target_archs

            archs = sorted(set(resolve_target_archs().split(",")))
            return cache_dir / flashinfer_version / "_".join(archs)

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # normalize_arch keeps letter suffixes ("gfx90a"); the gfx\d guard only
        # checks the value looks like an arch name before it is trusted.
        arch = normalize_arch(props.gcnArchName)
        if not re.match(r"gfx\d", arch):
            from torch.utils.cpp_extension import _get_rocm_arch_flags

            archs = [
                f.replace("--offload-arch=", "")
                for f in _get_rocm_arch_flags()
                if f.startswith("--offload-arch=")
            ]
            arch = archs[0] if archs else "noarch"

        # Setting the current device is the caller's job; catch a misconfigured
        # one here rather than at kernel launch.
        from ...hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS

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
    return cache_dir / flashinfer_version / arch
