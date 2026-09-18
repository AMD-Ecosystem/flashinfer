# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Degradation paths in the ROCm JIT layer and the small utility modules.

These are the arms that only run when something is already wrong -- a broken
toolchain, a GPU-less host, a missing jit-cache wheel -- so a normal run never
reaches them and a regression there surfaces as a traceback with no message
rather than the diagnostic each one was written to give.
"""

import subprocess
from pathlib import Path

import pytest
import torch

from flashinfer.jit.rocm import aiter_variants as av
from flashinfer.jit.rocm import core as jit_core
from flashinfer.jit.rocm import cpp_ext, env as jit_env
from flashinfer.rocm import api_compat, compilation_context


class TestNinjaInvocation:
    def test_a_failed_build_reports_ninja_s_own_output(self, tmp_path, monkeypatch):
        """Without the output the caller gets "Ninja build failed." and nothing
        to act on."""

        def boom(*_a, **_k):
            raise subprocess.CalledProcessError(
                1, "ninja", output="ninja: error: no rule to make target"
            )

        monkeypatch.setattr(cpp_ext.subprocess, "run", boom)
        with pytest.raises(RuntimeError, match="no rule to make target"):
            cpp_ext.run_ninja(tmp_path, tmp_path / "build.ninja", verbose=False)

    def test_device_linking_is_refused_rather_than_silently_skipped(self):
        with pytest.raises(ValueError, match="Device linking unimplemented"):
            cpp_ext.generate_ninja_build_for_op(
                name="demo",
                sources=[Path("a.cu")],
                extra_cflags=None,
                extra_cuda_cflags=None,
                extra_ldflags=None,
                extra_include_dirs=None,
                needs_device_linking=True,
            )


class TestArchValidation:
    def test_a_rejected_arch_says_what_failed(self, monkeypatch):
        """The wrapper exists to name the stage; a bare RuntimeError from the
        validator reads as a compiler problem."""
        from flashinfer.rocm import hip_utils

        def boom(**_k):
            raise RuntimeError("gfx000 is not supported")

        monkeypatch.setattr(hip_utils, "validate_flashinfer_rocm_arch", boom)
        with pytest.raises(RuntimeError, match="ROCm architecture validation failed"):
            jit_core.check_rocm_arch()


class TestDeviceArchProbe:
    def test_a_host_with_no_visible_device_probes_as_none(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert jit_env._live_device_arch() is None

    def test_a_probe_that_raises_is_not_fatal(self, monkeypatch):
        """This runs during `import flashinfer`; raising here would make the
        package unimportable on an odd host."""

        def boom():
            raise RuntimeError("no HIP runtime")

        monkeypatch.setattr(torch.cuda, "device_count", boom)
        assert jit_env._live_device_arch() is None


class TestJitCacheWheelLookup:
    def test_a_wheel_without_the_accessor_is_ignored(self, monkeypatch):
        import sys
        import types

        stub = types.ModuleType("amd_flashinfer_jit_cache")
        monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", stub)
        assert av._wheel_store() is None

    def test_an_accessor_that_raises_is_ignored(self, monkeypatch):
        import sys
        import types

        stub = types.ModuleType("amd_flashinfer_jit_cache")

        def boom():
            raise RuntimeError("wheel is corrupt")

        stub.get_aiter_variant_dir = boom
        monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", stub)
        assert av._wheel_store() is None

    def test_an_accessor_naming_a_missing_directory_is_ignored(
        self, tmp_path, monkeypatch
    ):
        import sys
        import types

        stub = types.ModuleType("amd_flashinfer_jit_cache")
        stub.get_aiter_variant_dir = lambda: str(tmp_path / "absent")
        monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", stub)
        assert av._wheel_store() is None

        # Distinguish this arm from the `except Exception` above it: the same
        # accessor pointed at a directory that *does* exist must be accepted.
        present = tmp_path / av.variant_store_dir().name
        present.mkdir(parents=True)
        stub.get_aiter_variant_dir = lambda: str(tmp_path)
        assert av._wheel_store() == present


class TestCompilationContext:
    def test_the_target_set_is_handed_out_as_a_copy(self):
        ctx = compilation_context.CompilationContext()
        archs = ctx.get_target_archs()

        assert archs
        archs.add("gfx000")
        assert "gfx000" not in ctx.get_target_archs(), "callers must not mutate it"

    def test_has_arch_agrees_with_the_target_set(self):
        ctx = compilation_context.CompilationContext()
        known = next(iter(ctx.get_target_archs()))

        assert ctx.has_arch(known)
        assert not ctx.has_arch("gfx000")


class TestCudaOnlyRejection:
    def test_the_default_and_the_neutral_value_are_both_accepted(self):
        api_compat.reject_cuda_only("fixed_split_size", None, None)
        api_compat.reject_cuda_only("enable_pdl", False, None, neutral=False)

    def test_a_non_default_value_names_what_to_pass_instead(self):
        with pytest.raises(NotImplementedError, match="pass fixed_split_size=None"):
            api_compat.reject_cuda_only("fixed_split_size", 128, None)

    def test_a_tensor_against_a_scalar_default_does_not_compare_elementwise(self):
        """`torch.ones(2) == 1.0` is a tensor, and `if` on it raises "Boolean
        value of Tensor is ambiguous" -- which is why the compare is guarded by
        isinstance. Against a `None` default `==` already yields a plain bool,
        so that case would pass with the guard removed."""
        with pytest.raises(NotImplementedError, match="CUDA-only"):
            api_compat.reject_cuda_only("k_scale", torch.ones(2), 1.0)


class TestCudaOnlyModuleGate:
    def test_runpy_gets_an_import_error_rather_than_an_attribute_error(self):
        """`python -m flashinfer.aot` asks the loader for code, not exec_module."""
        import importlib

        spec = importlib.util.find_spec("flashinfer.aot")
        assert spec is not None and spec.loader is not None
        with pytest.raises(ImportError, match="flashinfer.aot"):
            spec.loader.get_code("flashinfer.aot")
        with pytest.raises(ImportError, match="flashinfer.aot"):
            spec.loader.get_source("flashinfer.aot")


class TestUpstreamBaseIO:
    def test_an_unreadable_base_file_names_itself(self, tmp_path, monkeypatch):
        import importlib.util
        import sys

        target = Path(__file__).resolve().parents[2] / "scripts" / "upstream_base.py"
        spec = importlib.util.spec_from_file_location("_fi_upstream_base", target)
        module = importlib.util.module_from_spec(spec)
        # setitem, not assignment: a hand-loaded module left in sys.modules
        # outlives the test for the rest of the worker.
        monkeypatch.setitem(sys.modules, "_fi_upstream_base", module)
        spec.loader.exec_module(module)

        # A directory where the file should be: an OSError that is not
        # FileNotFoundError, which is the only arm "absent" does not cover.
        (tmp_path / module.FILENAME).mkdir()
        with pytest.raises(module.UpstreamBaseError, match="cannot read"):
            module.read_worktree(str(tmp_path))
