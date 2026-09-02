# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""How ROCm substitutes itself for the CUDA implementation.

:func:`install_shadow_modules` makes an upstream module name resolve to its ROCm
twin; :func:`gate_cuda_only_modules` makes an unported module raise a catchable
ImportError. Ops whose only difference is the kernel need neither — the
``FLASHINFER_CSRC_DIR`` redirect already covers them.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Dict, FrozenSet, Iterable, cast

# Upstream module name -> the ROCm module that replaces it.
SHADOW_MODULES: Dict[str, str] = {
    "flashinfer.decode": "flashinfer.rocm.decode",
    "flashinfer.mla": "flashinfer.rocm.mla",
    "flashinfer.prefill": "flashinfer.rocm.prefill",
}

# Submodules with no ROCm implementation. cuda_ipc and trtllm_ar bind to
# libcudart at import time; vllm_ar, nvshmem and nvshmem_allreduce import but
# fail when the JIT fires; mnnvl needs pynvml. mixed_comm is where v0.6.18
# folded the NVSHMEM build: it renders .cu templates into FLASHINFER_GEN_SRC_DIR
# before reaching `import nvidia.nvshmem`, so without the gate a ROCm caller
# writes CUDA sources on its way to the failure.
CUDA_ONLY_MODULES = frozenset(
    {
        "flashinfer.comm.cuda_ipc",
        "flashinfer.comm.mixed_comm",
        "flashinfer.comm.mnnvl",
        "flashinfer.comm.nvshmem",
        "flashinfer.comm.nvshmem_allreduce",
        "flashinfer.comm.trtllm_alltoall",
        "flashinfer.comm.trtllm_ar",
        "flashinfer.comm.trtllm_mnnvl_ar",
        "flashinfer.comm.vllm_ar",
        # v0.6.18 made quantization a package; these two reach CUDA-only jit
        # exports at import. Gating them turns an obscure sm121a_nvcc_flags
        # ImportError into the uniform CUDA-only one, stubs included.
        "flashinfer.quantization.fp4_quantization",
        "flashinfer.quantization.fp8_quantization",
        "flashinfer.fp4_quantization",
        "flashinfer.fp8_quantization",
    }
)


def install_shadow_modules() -> Dict[str, ModuleType]:
    """Point each upstream module name at its ROCm twin, and return them by name.

    Both halves are needed: ``sys.modules`` serves ``import flashinfer.mla``,
    while the attribute on the parent package serves ``flashinfer.mla.X``.
    """
    # Import every twin before aliasing any, so a twin that imports another
    # shadowed name cannot resolve it against a half-installed registry.
    imported = {
        up: importlib.import_module(rocm) for up, rocm in SHADOW_MODULES.items()
    }
    package = sys.modules[__name__.rsplit(".", 1)[0]]
    for upstream, module in imported.items():
        sys.modules[upstream] = module
        setattr(package, upstream.rsplit(".", 1)[1], module)
    return imported


class _CudaOnlyLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(
            f"{module.__spec__.name} is CUDA-only and not available on ROCm"
        )


class _CudaOnlyFinder(importlib.abc.MetaPathFinder):
    _is_flashinfer_cuda_only_finder = True

    def __init__(self, names: FrozenSet[str]) -> None:
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names:
            return importlib.machinery.ModuleSpec(fullname, _CudaOnlyLoader())
        return None


def gate_cuda_only_modules(names: Iterable[str] = CUDA_ONLY_MODULES) -> None:
    """Make the named modules raise a uniform ImportError. Idempotent.

    Deliberately supplies no stub module: ``hasattr`` is how downstream engines
    feature-detect, and a stub would answer yes. Note this does not fix
    ``importlib.util.find_spec``, which still reports the module as present.
    """
    wanted = frozenset(names)
    for finder in sys.meta_path:
        # Marker, not isinstance: importlib.reload redefines the class, so the
        # installed finder must still be recognised and widened in place.
        if not getattr(finder, "_is_flashinfer_cuda_only_finder", False):
            continue
        existing = getattr(finder, "names", None)
        if existing is None:
            continue  # marker without our shape; fall through and install ours
        cast(_CudaOnlyFinder, finder).names = existing | wanted
        return
    sys.meta_path.insert(0, _CudaOnlyFinder(wanted))
