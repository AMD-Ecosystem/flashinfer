# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Flag plumbing in the HIP JIT ninja generator.

The failure modes are quiet ones: `-isystem` on our own headers silently drops
every warning and any instrumentation in them, and an un-unwrapped
``-Xarch_host`` makes the host rule fail in a way that reads as a compiler
problem rather than a flag problem.

Pure functions over flag lists -- no GPU, no torch import beyond the module's
own, so this runs in the CPU conformance lane.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    """Load cpp_ext_hip with torch and the ROCm probe stubbed out.

    The module imports torch and resolves ROCM_HOME at import time, neither of
    which the flag helpers touch. Stubbing keeps this runnable on a CPU box --
    otherwise the whole file skips and the CPU lane silently covers nothing.
    """
    stubs = {
        "torch": types.SimpleNamespace(
            _C=types.SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True)
        ),
        "torch.utils": types.ModuleType("torch.utils"),
        "torch.utils.cpp_extension": types.SimpleNamespace(
            _TORCH_PATH="/stub/torch",
            _get_num_workers=lambda verbose: 1,
            _get_pybind11_abi_build_flags=lambda: [],
        ),
        "flashinfer.hip_utils": types.SimpleNamespace(
            get_rocm_home=lambda: "/stub/rocm"
        ),
        # The module does `from . import env`, so it needs a real package chain.
        "flashinfer": types.ModuleType("flashinfer"),
        "flashinfer.jit": types.ModuleType("flashinfer.jit"),
        "flashinfer.jit.env": types.SimpleNamespace(
            FLASHINFER_INCLUDE_DIR=Path("/stub/include"),
            FLASHINFER_CSRC_DIR=Path("/stub/csrc/rocm"),
        ),
    }
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "flashinfer.jit.cpp_ext_hip",
            _REPO_ROOT / "flashinfer" / "jit" / "cpp_ext_hip.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, was in saved.items():
            if was is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = was
    return module


ext = _load_module()


class TestHostFlagRewrite:
    def test_offload_arch_is_dropped(self):
        assert ext._for_host(["-O3", "--offload-arch=gfx942", "-fPIC"]) == [
            "-O3",
            "-fPIC",
        ]

    def test_xarch_host_is_unwrapped_not_dropped(self):
        """The host rule is not an offload compile; the driver rejects the prefix."""
        flags = ["-Xarch_host", "-fprofile-instr-generate", "-O3"]

        assert ext._for_host(flags) == ["-fprofile-instr-generate", "-O3"]

    def test_xarch_device_is_removed_with_its_argument(self):
        flags = ["-O3", "-Xarch_device", "-ffast-math", "-fPIC"]

        assert ext._for_host(flags) == ["-O3", "-fPIC"]

    def test_trailing_xarch_without_argument_is_not_swallowed(self):
        """A malformed tail must not silently drop the flag or IndexError."""
        assert ext._for_host(["-O3", "-Xarch_host"]) == ["-O3", "-Xarch_host"]

    def test_offload_arch_separate_argument_takes_its_value_with_it(self):
        """`--offload-arch gfx942` -- dropping only the flag strands `gfx942`,
        which the host driver reads as a missing linker input, not a bad flag."""
        assert ext._for_host(["-O3", "--offload-arch", "gfx942", "-fPIC"]) == [
            "-O3",
            "-fPIC",
        ]

    def test_offload_arch_lookalike_is_not_dropped(self):
        """Matching on the `=` form stops a longer flag being eaten by prefix."""
        assert ext._for_host(["--offload-arch-list=x"]) == ["--offload-arch-list=x"]

    def test_trailing_offload_arch_without_a_value_is_not_an_error(self):
        assert ext._for_host(["-O3", "--offload-arch"]) == ["-O3"]

    def test_trailing_xarch_device_is_not_swallowed(self):
        assert ext._for_host(["-O3", "-Xarch_device"]) == ["-O3", "-Xarch_device"]

    def test_joined_xarch_form_is_left_alone(self):
        """Not valid clang syntax; pass it through rather than guess at it."""
        assert ext._for_host(["-Xarch_host=-DA"]) == ["-Xarch_host=-DA"]

    def test_repeated_pairs_all_unwrap(self):
        flags = [
            "-Xarch_host",
            "-fprofile-instr-generate",
            "-Xarch_host",
            "-fcoverage-mapping",
        ]

        assert ext._for_host(flags) == [
            "-fprofile-instr-generate",
            "-fcoverage-mapping",
        ]


class TestEnvFlagHooks:
    def test_absent_env_yields_nothing(self, monkeypatch):
        monkeypatch.delenv("FLASHINFER_EXTRA_CFLAGS", raising=False)

        assert ext._env_flags("FLASHINFER_EXTRA_CFLAGS") == []

    def test_quoted_values_split_like_a_shell(self, monkeypatch):
        monkeypatch.setenv("FLASHINFER_EXTRA_CFLAGS", '-DA="x y" -O0')

        assert ext._env_flags("FLASHINFER_EXTRA_CFLAGS") == ["-DA=x y", "-O0"]

    def test_unbalanced_quote_falls_back_instead_of_raising(self, monkeypatch):
        """A bad value must not abort every JIT build in the process."""
        monkeypatch.setenv("FLASHINFER_EXTRA_CFLAGS", '-DA="unterminated')

        assert ext._env_flags("FLASHINFER_EXTRA_CFLAGS") == ['-DA="unterminated']


class TestOwnHeaderIncludeMode:
    def test_defaults_to_system_includes(self, monkeypatch):
        monkeypatch.delenv("FLASHINFER_OWN_HEADERS_NON_SYSTEM", raising=False)

        assert ext._own_headers_non_system() is False

    def test_opt_in_switches_to_plain_include(self, monkeypatch):
        """`-isystem` suppresses warnings and instrumentation in our own headers."""
        monkeypatch.setenv("FLASHINFER_OWN_HEADERS_NON_SYSTEM", "1")

        assert ext._own_headers_non_system() is True

    @pytest.mark.parametrize("value", ["0", "", "no", "true"])
    def test_only_exactly_one_enables_it(self, monkeypatch, value):
        monkeypatch.setenv("FLASHINFER_OWN_HEADERS_NON_SYSTEM", value)

        assert ext._own_headers_non_system() is False


def _ninja(tmp_path, extra_cflags=None, **env):
    """Render a real ninja file for one .cu source under the given environment."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
    try:
        return ext.generate_ninja_build_for_op(
            name="probe",
            sources=[tmp_path / "probe.cu"],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=None,
            extra_ldflags=None,
            extra_include_dirs=None,
        )
    finally:
        for key, was in saved.items():
            if was is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = was


def _var(text, name):
    """Pull one ninja variable's value, rejoining its line continuations."""
    body = text.split(f"\n{name} = ", 1)[1]
    out = []
    for line in body.splitlines():
        out.append(line.rstrip(" $"))
        if not line.endswith("$"):
            break
    return " ".join(part.strip() for part in out)


class TestGeneratedNinja:
    """The generator must actually use the helpers -- unit-testing them is not enough."""

    def test_host_rule_gets_xarch_unwrapped(self, tmp_path):
        """The spec's own flags carry the prefix; the host rule must not see it."""
        text = _ninja(tmp_path, extra_cflags=["-Xarch_host", "-DHOSTONLY"])

        assert "-Xarch_host" in _var(text, "cflags")
        host = _var(text, "host_cflags")
        assert "-DHOSTONLY" in host
        assert "-Xarch_host" not in host, "host rule cannot take the offload prefix"

    def test_offload_arch_still_stripped_from_host(self, tmp_path):
        text = _ninja(tmp_path, extra_cflags=["--offload-arch=gfx942"])

        assert "--offload-arch" not in _var(text, "host_cflags")

    def test_env_cflags_are_host_only(self, tmp_path):
        """hip_compile ends with `$cflags`, so anything there also hits device
        codegen and, being last, outranks -O3. CUDA scopes this var to the host;
        putting it on host_cflags directly is what keeps the two the same."""
        text = _ninja(tmp_path, FLASHINFER_EXTRA_CFLAGS="-DHOSTONLY")

        assert "-DHOSTONLY" in _var(text, "host_cflags")
        assert "-DHOSTONLY" not in _var(text, "cflags")
        assert "-DHOSTONLY" not in _var(text, "cuda_cflags")

    def test_env_cflags_go_through_the_host_rewrite(self, tmp_path):
        """Copying a flag list off a HIP driver line is how an -Xarch_host is
        acquired, and the host rule is exactly the one that cannot take it."""
        text = _ninja(
            tmp_path,
            FLASHINFER_EXTRA_CFLAGS="-Xarch_host -DHOSTONLY --offload-arch=gfx950",
        )

        host = _var(text, "host_cflags")
        assert "-DHOSTONLY" in host
        assert "-Xarch_host" not in host
        assert "--offload-arch=gfx950" not in host

    def test_env_ldflags_still_reach_the_link_rule(self, tmp_path):
        """The one already-shipping hook this change refactored."""
        text = _ninja(tmp_path, FLASHINFER_EXTRA_LDFLAGS="-L/opt/x -lfoo")

        ld = _var(text, "ldflags")
        assert "-L/opt/x" in ld and "-lfoo" in ld

    def test_own_headers_are_isystem_by_default(self, tmp_path):
        text = _ninja(tmp_path, FLASHINFER_OWN_HEADERS_NON_SYSTEM=None)

        assert "-isystem /stub/include" in text
        assert "-I/stub/include" not in text

    def test_non_system_mode_switches_own_headers_to_plain_include(self, tmp_path):
        """Measured: -Wall -Wextra reports 4 warnings under -I and 0 under -isystem."""
        text = _ninja(tmp_path, FLASHINFER_OWN_HEADERS_NON_SYSTEM="1")

        assert "-I/stub/include" in text
        assert "-I/stub/csrc/rocm" in text
        assert "-isystem /stub/include" not in text
        # Third-party headers must stay -isystem regardless (ninja keeps the var).
        assert "-isystem $torch_home/include" in text

    def test_cuda_flags_reach_the_hip_rule(self, tmp_path):
        text = _ninja(tmp_path, FLASHINFER_EXTRA_CUDAFLAGS="-fcoverage-mapping")

        assert "-fcoverage-mapping" in _var(text, "cuda_cflags")


def test_the_cuda_path_still_reads_the_same_names():
    """Only the CUDA half is a text check -- it is not imported here.

    The HIP half is asserted by rendering a ninja, since a source grep passes
    on a comment that merely mentions the variable.
    """
    cuda = (_REPO_ROOT / "flashinfer" / "jit" / "cpp_ext.py").read_text(
        encoding="utf-8"
    )

    for name in ("FLASHINFER_EXTRA_CFLAGS", "FLASHINFER_EXTRA_CUDAFLAGS"):
        assert name in cuda, f"{name} vanished from the CUDA path"


def test_environment_is_not_mutated_by_import(tmp_path):
    """The module must not set these itself; the caller owns them.

    Compares the whole environment across a render -- the previous version
    asserted `os.environ[name] is not None`, which no str value can fail.
    """
    before = dict(os.environ)

    _ninja(tmp_path)

    assert dict(os.environ) == before
