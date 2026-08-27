# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for AOT HIP kernel compilation.

Tests the flashinfer.aot_hip module to ensure:
1. JIT specs are generated correctly
2. Kernels compile successfully
3. .so files are created and can be loaded
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from flashinfer import aot_hip
from flashinfer.aot_hip import (
    compile_and_package_modules,
    gen_all_modules,
    get_default_config,
)

# Skip all tests if HIP is not available
pytestmark = pytest.mark.skipif(
    not hasattr(torch.version, "hip") or torch.version.hip is None,
    reason="HIP not available",
)


def test_get_default_config():
    """Test that default configuration is properly formed."""
    config = get_default_config()

    assert "fa2_head_dim" in config
    assert "f16_dtype" in config
    assert "use_sliding_window" in config
    assert "use_logits_soft_cap" in config

    # Verify structure
    assert isinstance(config["fa2_head_dim"], list)
    assert len(config["fa2_head_dim"]) > 0
    assert all(
        isinstance(dim, tuple) and len(dim) == 2 for dim in config["fa2_head_dim"]
    )

    assert isinstance(config["f16_dtype"], list)
    assert all(
        dtype in [torch.float16, torch.bfloat16] for dtype in config["f16_dtype"]
    )

    assert isinstance(config["use_sliding_window"], list)
    assert isinstance(config["use_logits_soft_cap"], list)


def test_gen_all_modules_minimal():
    """Test generating JIT specs with minimal configuration."""
    # Use minimal config to reduce test time
    f16_dtype = [torch.float16]
    fa2_head_dim = [(128, 128)]  # Single head dimension
    use_sliding_window = [False]
    use_logits_soft_cap = [False]

    jit_specs = gen_all_modules(
        f16_dtype,
        fa2_head_dim,
        use_sliding_window,
        use_logits_soft_cap,
    )

    # Should generate multiple specs (single_decode, single_prefill, batch_decode, batch_prefill)
    assert len(jit_specs) > 0

    # Verify JitSpec structure
    for spec in jit_specs:
        assert hasattr(spec, "name")
        assert hasattr(spec, "sources")  # Fixed: it's 'sources' not 'source_files'
        assert isinstance(spec.name, str)
        assert len(spec.name) > 0


def test_gen_all_modules_deduplication():
    """Test that generated modules are deduplicated by name."""
    # Use config that might generate duplicates
    f16_dtype = [torch.float16]
    fa2_head_dim = [(128, 128)]
    use_sliding_window = [False, False]  # Duplicate values
    use_logits_soft_cap = [False, False]

    jit_specs = gen_all_modules(
        f16_dtype,
        fa2_head_dim,
        use_sliding_window,
        use_logits_soft_cap,
    )

    # Check no duplicate names
    names = [spec.name for spec in jit_specs]
    assert len(names) == len(set(names)), "Found duplicate module names"


def test_compile_and_package_minimal():
    """Test the full compile and package workflow with minimal config.

    This tests the complete AOT pipeline without slow compilation.
    Uses skip_prebuilt=True to avoid actual compilation.
    """
    # Create temporary directories
    build_dir = Path(tempfile.mkdtemp())
    out_dir = Path(tempfile.mkdtemp())
    project_root = Path(__file__).parent.parent

    try:
        # Minimal config to avoid heavy compilation
        config = {
            "fa2_head_dim": [(128, 128)],
            "f16_dtype": [torch.float16],
            "use_sliding_window": [False],
            "use_logits_soft_cap": [False],
        }

        # Test with skip_prebuilt=True to avoid actual compilation
        # This verifies the pipeline works without spending time compiling
        compile_and_package_modules(
            out_dir=None,  # Don't copy, just test generation
            build_dir=build_dir,
            project_root=project_root,
            config=config,
            verbose=False,
            skip_prebuilt=True,  # Skip actual compilation
        )

    finally:
        # Cleanup
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_failed_validation_leaves_the_environment_alone(monkeypatch):
    """A raise mid-build must not leave FLASHINFER_ROCM_ARCH_LIST behind.

    compile_and_package_modules publishes the resolved list into the process
    environment on purpose -- the AITER shim reads it from there
    (jit/aiter_source.py) and an AOT build has no other channel to reach it.
    That side effect outlives the call, so it must only happen once the list is
    known good; otherwise a build that dies on an unsupported ROCm version
    silently repoints whatever runs next in the same process.

    The variable starts *unset* and resolution comes from detection, so the
    published value differs from the starting state. Seeding it with the value
    the resolver would return makes the write a no-op and the test vacuous --
    it then passes with the bug present.
    """
    import flashinfer.aot_hip as aot_hip

    monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
    monkeypatch.setattr(
        "flashinfer.hip_utils.rocminfo_gpu_agents",
        lambda: (("gfx950", ""),),
    )

    class _Boom:
        def __init__(self):
            raise RuntimeError("ROCm version 0.0 is not recognized")

    monkeypatch.setattr("flashinfer.compilation_context_hip.CompilationContext", _Boom)

    with pytest.raises(RuntimeError, match="not recognized"):
        aot_hip.compile_and_package_modules(
            out_dir=None,
            build_dir=Path(tempfile.mkdtemp()),
            project_root=Path(__file__).parent.parent,
            config={
                "fa2_head_dim": [(128, 128)],
                "f16_dtype": [torch.float16],
                "use_sliding_window": [False],
                "use_logits_soft_cap": [False],
            },
            verbose=False,
            skip_prebuilt=True,
        )

    assert "FLASHINFER_ROCM_ARCH_LIST" not in os.environ


def test_publishes_the_validated_list_not_the_requested_one(monkeypatch):
    """What reaches the AITER shim must be what the kernels were built for.

    Validation filters as well as raises: ``validate_flashinfer_rocm_arch``
    drops architectures this ROCm or this PyTorch cannot build, warning instead
    of failing, so the context's target set can be a strict subset of the
    resolver's answer. Publishing the wider list hands the shim an architecture
    the kernels do not target -- the exact disagreement this change exists to
    remove, arriving through the environment instead of through a hard-coded
    default.

    gfx950 is requested and accepted by the resolver; the context reports only
    gfx942, standing in for a ROCm or PyTorch that cannot build gfx950. The
    published value must follow the context.
    """
    import flashinfer.aot_hip as aot_hip

    monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx950,gfx942")

    class _FilteringContext:
        """Accepts the request, targets a subset -- what a filter looks like."""

        arch_flags = ["--offload-arch=gfx942"]
        TARGET_ROCM_ARCHS = {"gfx942"}

        def __init__(self):
            pass

    monkeypatch.setattr(
        "flashinfer.compilation_context_hip.CompilationContext", _FilteringContext
    )

    aot_hip.compile_and_package_modules(
        out_dir=None,
        build_dir=Path(tempfile.mkdtemp()),
        project_root=Path(__file__).parent.parent,
        config={
            "fa2_head_dim": [(128, 128)],
            "f16_dtype": [torch.float16],
            "use_sliding_window": [False],
            "use_logits_soft_cap": [False],
        },
        verbose=False,
        skip_prebuilt=True,
    )

    assert os.environ["FLASHINFER_ROCM_ARCH_LIST"] == "gfx942"


def test_publishing_preserves_the_requested_order(monkeypatch):
    """The caller's first choice must stay first.

    ``resolve_aiter_build_arch()`` takes ``env_archs[0]`` when no device is
    visible, so the order of the published list decides which architecture the
    single-arch AITER shim is built for. ``TARGET_ROCM_ARCHS`` is a ``set`` --
    ``arch_set = set(requested_archs)`` in hip_utils -- so reading order from it
    is impossible, and sorting it is not order-neutral: it would turn a request
    for "gfx950,gfx942" into a shim built for gfx942. ``arch_flags`` iterates
    ``requested_archs`` in order and is the only ordered survivor.
    """
    import flashinfer.aot_hip as aot_hip

    monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx950,gfx942")

    class _OrderedContext:
        # Deliberately not alphabetical: sorted() would reverse this.
        arch_flags = ["--offload-arch=gfx950", "--offload-arch=gfx942"]
        TARGET_ROCM_ARCHS = {"gfx942", "gfx950"}

        def __init__(self):
            pass

    monkeypatch.setattr(
        "flashinfer.compilation_context_hip.CompilationContext", _OrderedContext
    )

    aot_hip.compile_and_package_modules(
        out_dir=None,
        build_dir=Path(tempfile.mkdtemp()),
        project_root=Path(__file__).parent.parent,
        config={
            "fa2_head_dim": [(128, 128)],
            "f16_dtype": [torch.float16],
            "use_sliding_window": [False],
            "use_logits_soft_cap": [False],
        },
        verbose=False,
        skip_prebuilt=True,
    )

    assert os.environ["FLASHINFER_ROCM_ARCH_LIST"] == "gfx950,gfx942"


def test_module_naming_convention():
    """Test that generated module names follow expected conventions."""
    f16_dtype = [torch.float16]
    fa2_head_dim = [(128, 128)]
    use_sliding_window = [False]
    use_logits_soft_cap = [False]

    jit_specs = gen_all_modules(
        f16_dtype,
        fa2_head_dim,
        use_sliding_window,
        use_logits_soft_cap,
    )

    # Check naming patterns
    expected_patterns = [
        "single_decode",
        "single_prefill",
        "batch_decode",
        "batch_prefill",
    ]

    found_patterns = {pattern: False for pattern in expected_patterns}
    for spec in jit_specs:
        for pattern in expected_patterns:
            if pattern in spec.name:
                found_patterns[pattern] = True

    # At least some expected patterns should be found
    assert any(found_patterns.values()), (
        f"No expected module patterns found in: {[s.name for s in jit_specs]}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gen_modules_with_different_dtypes(dtype):
    """Test generating modules with different float16 dtypes."""
    f16_dtype = [dtype]
    fa2_head_dim = [(128, 128)]
    use_sliding_window = [False]
    use_logits_soft_cap = [False]

    jit_specs = gen_all_modules(
        f16_dtype,
        fa2_head_dim,
        use_sliding_window,
        use_logits_soft_cap,
    )

    assert len(jit_specs) > 0
    # Verify dtype is reflected in module names
    dtype_str = "f16" if dtype == torch.float16 else "bf16"
    assert any(dtype_str in spec.name for spec in jit_specs)


@pytest.mark.parametrize("head_dim", [(64, 64), (128, 128), (256, 256)])
def test_gen_modules_with_different_head_dims(head_dim):
    """Test generating modules with different head dimensions."""
    f16_dtype = [torch.float16]
    fa2_head_dim = [head_dim]
    use_sliding_window = [False]
    use_logits_soft_cap = [False]

    jit_specs = gen_all_modules(
        f16_dtype,
        fa2_head_dim,
        use_sliding_window,
        use_logits_soft_cap,
    )

    assert len(jit_specs) > 0
    # Verify head dim is reflected in module names
    assert any(f"head_dim_qk_{head_dim[0]}" in spec.name for spec in jit_specs)
    assert any(f"head_dim_vo_{head_dim[1]}" in spec.name for spec in jit_specs)


def _register_arch_list_undo(monkeypatch):
    """Make monkeypatch unwind FLASHINFER_ROCM_ARCH_LIST, which the build publishes.

    delenv records no undo entry when the name is already absent, so seed it
    first -- otherwise the build's write survives into the rest of the worker.
    """
    monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "")
    monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST")


def _jit_env_paths():
    """Everything _redirected_jit_env mutates, including the env var."""
    from flashinfer.jit import env as jit_env

    return (
        jit_env.FLASHINFER_WORKSPACE_DIR,
        jit_env.FLASHINFER_JIT_DIR,
        jit_env.FLASHINFER_GEN_SRC_DIR,
        os.environ.get("FLASHINFER_WORKSPACE_BASE"),
    )


def test_build_restores_the_jit_workspace_paths(monkeypatch, tmp_path):
    """An AOT build must not repoint the JIT workspace for the rest of the process."""
    import flashinfer.aot_hip as aot_hip

    before = _jit_env_paths()
    _register_arch_list_undo(monkeypatch)
    monkeypatch.setattr(aot_hip, "gen_all_modules", lambda *a, **k: [])
    monkeypatch.setattr("flashinfer.jit.build_jit_specs", lambda *a, **k: None)

    aot_hip.compile_and_package_modules(
        out_dir=None,
        build_dir=tmp_path / "build",
        project_root=Path(__file__).parent.parent,
        config={
            "fa2_head_dim": [(128, 128)],
            "f16_dtype": [torch.float16],
            "use_sliding_window": [False],
            "use_logits_soft_cap": [False],
        },
        verbose=False,
        skip_prebuilt=True,
    )

    assert _jit_env_paths() == before


def test_unusable_build_dir_restores_the_jit_workspace_paths(tmp_path):
    """The paths are repointed before the mkdirs, so a failing mkdir must unwind too."""
    from flashinfer.aot_hip import _redirected_jit_env

    before = _jit_env_paths()
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("")

    with pytest.raises(OSError), _redirected_jit_env(not_a_dir):
        pytest.fail("the context manager should not have yielded")

    assert _jit_env_paths() == before


def test_failed_build_restores_the_jit_workspace_paths(monkeypatch, tmp_path):
    """The restore has to survive the error path too, not just a clean return."""
    import flashinfer.aot_hip as aot_hip

    before = _jit_env_paths()

    class _Boom:
        def __init__(self):
            raise RuntimeError("ROCm version 0.0 is not recognized")

    _register_arch_list_undo(monkeypatch)
    monkeypatch.setattr("flashinfer.compilation_context_hip.CompilationContext", _Boom)

    with pytest.raises(RuntimeError, match="not recognized"):
        aot_hip.compile_and_package_modules(
            out_dir=None,
            build_dir=tmp_path / "build",
            project_root=Path(__file__).parent.parent,
            config={
                "fa2_head_dim": [(128, 128)],
                "f16_dtype": [torch.float16],
                "use_sliding_window": [False],
                "use_logits_soft_cap": [False],
            },
            verbose=False,
            skip_prebuilt=True,
        )

    assert _jit_env_paths() == before


class TestArgumentParsers:
    @pytest.mark.parametrize("text", ["true", "TRUE", "True", "1"])
    def test_truthy_spellings(self, text):
        assert aot_hip.parse_bool(text) is True

    @pytest.mark.parametrize("text", ["false", "FALSE", "False", "0"])
    def test_falsy_spellings(self, text):
        assert aot_hip.parse_bool(text) is False

    @pytest.mark.parametrize("text", ["yes", "", "2", "none"])
    def test_anything_else_is_rejected_rather_than_assumed_false(self, text):
        with pytest.raises(ValueError, match="Invalid boolean value"):
            aot_hip.parse_bool(text)

    def test_head_dim_splits_into_qo_and_kv(self):
        assert aot_hip.parse_head_dim("192,128") == (192, 128)

    @pytest.mark.parametrize("text", ["128", "128,", "a,b", "1,2,3"])
    def test_malformed_head_dim_is_rejected(self, text):
        with pytest.raises(ValueError):
            aot_hip.parse_head_dim(text)


class TestSkippedCombinations:
    """gen_fa2 declines two combinations; both would otherwise fail at build."""

    def test_mixed_dtypes_of_equal_width_are_skipped(self):
        specs = list(
            aot_hip.gen_fa2(
                dtype_qo=torch.float16,
                dtype_kv=torch.bfloat16,
                head_dim_qk=128,
                head_dim_vo=128,
                use_sliding_window=False,
                use_logits_soft_cap=False,
            )
        )
        assert specs == []

    def test_fp8_is_skipped_because_fa2_has_no_fp8_tensor_cores(self):
        specs = list(
            aot_hip.gen_fa2(
                dtype_qo=torch.float8_e4m3fn,
                dtype_kv=torch.float8_e4m3fn,
                head_dim_qk=128,
                head_dim_vo=128,
                use_sliding_window=False,
                use_logits_soft_cap=False,
            )
        )
        assert specs == []


class _Spec:
    def __init__(self, name):
        self.name = name


def _built(build_dir, *names):
    """Lay out build_dir as a finished build of `names`."""
    for name in names:
        so = build_dir / "cached_ops" / name / f"{name}.so"
        so.parent.mkdir(parents=True)
        so.write_bytes(b"\x7fELF fake")
    return [_Spec(n) for n in names]


class TestCopyBuiltKernels:
    def test_each_kernel_lands_under_its_own_name(self, tmp_path):
        build_dir, out_dir = tmp_path / "build", tmp_path / "out"
        specs = _built(build_dir, "mod_a", "mod_b")

        aot_hip.copy_built_kernels(specs, out_dir, build_dir)

        for name in ("mod_a", "mod_b"):
            assert (out_dir / name / f"{name}.so").read_bytes() == b"\x7fELF fake"

    def test_a_missing_kernel_is_named_rather_than_silently_skipped(self, tmp_path):
        build_dir, out_dir = tmp_path / "build", tmp_path / "out"
        specs = _built(build_dir, "mod_a") + [_Spec("never_built")]

        with pytest.raises(FileNotFoundError, match="never_built"):
            aot_hip.copy_built_kernels(specs, out_dir, build_dir)

    def test_a_previous_output_directory_is_replaced_not_merged(self, tmp_path):
        """A stale .so left from an earlier build would ship in the wheel."""
        build_dir, out_dir = tmp_path / "build", tmp_path / "out"
        (out_dir / "stale_mod").mkdir(parents=True)
        (out_dir / "stale_mod" / "stale_mod.so").write_bytes(b"old")
        specs = _built(build_dir, "mod_a")

        aot_hip.copy_built_kernels(specs, out_dir, build_dir)

        assert not (out_dir / "stale_mod").exists()
        assert (out_dir / "mod_a" / "mod_a.so").exists()


class TestRegisterDefaultModules:
    def test_reports_the_same_count_as_the_default_config(self):
        config = get_default_config()
        expected = len(
            gen_all_modules(
                config["f16_dtype"],
                config["fa2_head_dim"],
                config["use_sliding_window"],
                config["use_logits_soft_cap"],
            )
        )

        assert aot_hip.register_default_modules() == expected
        assert expected > 0


@pytest.fixture
def cli(monkeypatch):
    """Run main() with a given argv, capturing what it hands the builder."""
    seen = {}
    monkeypatch.setattr(
        aot_hip, "compile_and_package_modules", lambda **kw: seen.update(kw)
    )

    def _run(*args):
        monkeypatch.setattr(sys, "argv", ["aot_hip.py", *args])
        aot_hip.main()
        return seen

    return _run


class TestMain:
    def test_bare_invocation_builds_the_default_config_in_place(self, cli):
        seen = cli()

        assert seen["out_dir"] is None
        assert seen["build_dir"] == Path.cwd()
        assert seen["config"] == get_default_config()
        assert (seen["verbose"], seen["skip_prebuilt"]) == (True, False)

    def test_directories_are_taken_from_the_command_line(self, cli, tmp_path):
        seen = cli("--out-dir", str(tmp_path / "o"), "--build-dir", str(tmp_path / "b"))

        assert seen["out_dir"] == tmp_path / "o"
        assert seen["build_dir"] == tmp_path / "b"

    def test_head_dims_are_parsed_into_pairs(self, cli):
        seen = cli("--fa2-head-dim", "64,64", "192,128")
        assert seen["config"]["fa2_head_dim"] == [(64, 64), (192, 128)]

    def test_dtype_names_resolve_to_torch_dtypes(self, cli):
        seen = cli("--f16-dtype", "bfloat16")
        assert seen["config"]["f16_dtype"] == [torch.bfloat16]

    def test_boolean_axes_are_parsed(self, cli):
        seen = cli("--use-sliding-window", "true", "--use-logits-soft-cap", "0")

        assert seen["config"]["use_sliding_window"] == [True]
        assert seen["config"]["use_logits_soft_cap"] == [False]

    def test_unset_axes_keep_their_defaults(self, cli):
        """An empty list must not narrow the build to nothing."""
        default = get_default_config()
        seen = cli("--use-sliding-window")

        assert seen["config"]["use_sliding_window"] == default["use_sliding_window"]

    def test_an_unknown_dtype_is_rejected_by_argparse(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["aot_hip.py", "--f16-dtype", "float64"])
        with pytest.raises(SystemExit):
            aot_hip.main()


def test_verbose_summary_names_the_arch_list_it_published(
    tmp_path, monkeypatch, capsys
):
    """The summary is the only place an AOT run reports which architectures it
    resolved to, and the AITER shim reads that same list from the environment."""
    import flashinfer.jit as jit

    monkeypatch.setattr(jit, "build_jit_specs", lambda specs, **kw: None)
    monkeypatch.setattr(aot_hip, "copy_built_kernels", lambda *a: None)
    out_dir = tmp_path / "out"

    compile_and_package_modules(
        out_dir=out_dir,
        build_dir=tmp_path / "build",
        project_root=tmp_path,
        config={"fa2_head_dim": [(64, 64)], "f16_dtype": [torch.float16]},
        verbose=True,
        skip_prebuilt=True,
    )
    out = capsys.readouterr().out

    assert "AOT build summary:" in out
    assert f"out_dir: {out_dir}" in out
    assert "FLASHINFER_ROCM_ARCH_LIST:" in out
    assert os.environ["FLASHINFER_ROCM_ARCH_LIST"] in out
    assert "AOT kernels saved to:" in out
