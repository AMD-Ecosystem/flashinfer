# SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache 2.0

# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import os
import shlex
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    _get_num_workers,
    _get_pybind11_abi_build_flags,
)

from flashinfer.hip_utils import get_rocm_home

from . import env as jit_env

ROCM_HOME = get_rocm_home()


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def _env_flags(name: str) -> List[str]:
    """Split an env var of extra compiler flags, tolerating unbalanced quotes."""
    raw = os.environ.get(name)
    if not raw:
        return []
    try:
        return shlex.split(raw)
    except ValueError as exc:
        print(
            f"Warning: Could not parse {name} with shlex: {exc}. "
            "Falling back to simple split.",
            file=sys.stderr,
        )
        return raw.split()


def _own_headers_non_system() -> bool:
    """True when this fork's own headers should be -I rather than -isystem."""
    return os.environ.get("FLASHINFER_OWN_HEADERS_NON_SYSTEM", "0") == "1"


def _for_host(flags: List[str]) -> List[str]:
    """Rewrite compile flags for the plain host compiler.

    Drops `--offload-arch`, which only the HIP driver understands, and unwraps
    `-Xarch_host X` to `X` -- the host rule is not an offload compile, so the
    driver rejects the prefixed form outright. A dropped flag takes its argument
    with it; leaving a bare `gfx942` behind reads as a missing linker input.
    """
    out: List[str] = []
    i = 0
    while i < len(flags):
        flag = flags[i]
        if flag.startswith("--offload-arch="):
            i += 1
        elif flag == "--offload-arch":
            i += 2  # separate-argument form; past the end is a no-op
        elif flag == "-Xarch_host" and i + 1 < len(flags):
            out.append(flags[i + 1])
            i += 2
        elif flag == "-Xarch_device" and i + 1 < len(flags):
            i += 2
        else:
            out.append(flag)
            i += 1
    return out


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
) -> str:
    system_includes = [
        sysconfig.get_path("include"),
        "$torch_home/include",
        "$torch_home/include/torch/csrc/api/include",
        "$cuda_home/include",
    ]
    # Ours, and normally also -isystem to keep third-party warnings quiet -- which
    # silences them in our own headers too, and suppresses instrumentation there.
    # FLASHINFER_OWN_HEADERS_NON_SYSTEM=1 makes them -I so both apply.
    own_includes = [
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]

    common_cflags = [
        "-DTORCH_EXTENSION_NAME=$name",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DPy_LIMITED_API=0x03090000",
    ]
    common_cflags += _get_pybind11_abi_build_flags()
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            common_cflags.append(f"-I{extra_dir.resolve()}")
    for sys_dir in system_includes:
        common_cflags.append(f"-isystem {sys_dir}")
    own_include_flag = "-I" if _own_headers_non_system() else "-isystem "
    for own_dir in own_includes:
        common_cflags.append(f"{own_include_flag}{own_dir}")

    cflags = [
        "$common_cflags",
        "-fPIC",
    ]
    if extra_cflags is not None:
        cflags += extra_cflags

    cuda_cflags: List[str] = []
    cuda_cflags += [
        "$common_cflags",
        "-fPIC",
    ]
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags
    cuda_cflags += _env_flags("FLASHINFER_EXTRA_CUDAFLAGS")

    ldflags = [
        "-shared",
        "-L$torch_home/lib",
        "-lc10",
        "-ltorch_cpu",
        "-ltorch",
    ]
    ldflags += [
        "-L$rocm_home/lib",
        "-lc10_hip",
        "-ltorch_hip",
        "-lamdhip64",
    ]

    ldflags += _env_flags("FLASHINFER_EXTRA_LDFLAGS")

    if extra_ldflags is not None:
        ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    rocm_home = ROCM_HOME
    amdclang = os.environ.get("PYTORCH_AMDCLANG", "$rocm_home/bin/amdclang++")

    # Mirrors FLASHINFER_EXTRA_CFLAGS/CUDAFLAGS on the CUDA path (cpp_ext.py);
    # the HIP path previously had a hook for link flags only.
    #
    # CFLAGS is folded in here, on the host rule alone, rather than into
    # `cflags`: the hip_compile rule below ends with `$cflags`, so anything put
    # there would also reach device codegen and, being last, would outrank both
    # -O3 and FLASHINFER_EXTRA_CUDAFLAGS. On CUDA the var is host-only; keep it
    # so. It goes *through* _for_host rather than after it -- copying a flag
    # list off a HIP driver line is the likeliest way to acquire an
    # `-Xarch_host`, and this rule is the one that cannot take it.
    host_cflags = _for_host(cflags + _env_flags("FLASHINFER_EXTRA_CFLAGS"))

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
    ]
    lines += [
        f"rocm_home = {rocm_home}",
        f"torch_home = {_TORCH_PATH}",
        f"cxx = {cxx}",
        f"amdclang = {amdclang}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "cflags = " + join_multiline(cflags),
        "host_cflags = " + join_multiline(host_cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "ldflags = " + join_multiline(ldflags),
        "",
        "rule compile",
        "  command = $cxx -MD -MF $out.d $host_cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule hip_compile",
        "  command = $amdclang -xhip -MD -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags $cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
    ]

    # Add nvcc linking rule for device code
    if needs_device_linking:
        raise ValueError("Device linking unimplemented for ROCm backend")
    else:
        lines.extend(
            [
                "rule link",
                "  command = $cxx $in $ldflags -o $out",
                "",
            ]
        )

    objects = []
    for source in sources:
        is_hip = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_hip else ".o"
        cmd = "hip_compile" if is_hip else "compile"
        obj_name = source.with_suffix(object_suffix).name
        obj = f"$name/{obj_name}"
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {source.resolve()}")

    lines.append("")
    link_rule = "link"
    lines.append(f"build $name/$name.so: {link_rule} " + " ".join(objects))
    lines.append("default $name/$name.so")
    lines.append("")

    return "\n".join(lines)


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
