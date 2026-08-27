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
import tempfile
from pathlib import Path

import pytest
import torch

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


def _jit_env_paths():
    from flashinfer.jit import env as jit_env

    return (
        jit_env.FLASHINFER_WORKSPACE_DIR,
        jit_env.FLASHINFER_JIT_DIR,
        jit_env.FLASHINFER_GEN_SRC_DIR,
    )


def test_build_restores_the_jit_workspace_paths(monkeypatch, tmp_path):
    """An AOT build must not repoint the JIT workspace for the rest of the process."""
    import flashinfer.aot_hip as aot_hip

    before = _jit_env_paths()
    # The build publishes this itself; register it so monkeypatch unwinds it.
    monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
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

    monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
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
