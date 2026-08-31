"""
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2025 by AMD ROCm team.

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

import os
import sys
import sysconfig
from pathlib import Path

from setuptools import build_meta as _orig
from wheel.bdist_wheel import bdist_wheel

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip version check when building amd-flashinfer-jit-cache package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


def _compile_jit_cache(output_dir: Path, verbose: bool = True):
    """Compile AOT modules using flashinfer.rocm.aot functions directly."""
    from flashinfer import aot_hip

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Set up build directory
    build_dir = project_root / "build" / "aot_hip"

    # Use the centralized compilation function from aot_hip.py
    aot_hip.compile_and_package_modules(
        out_dir=output_dir,
        build_dir=build_dir,
        project_root=project_root,
        config=None,  # Use default config
        verbose=verbose,
        skip_prebuilt=False,
    )


def _build_aot_modules():
    """Build AOT HIP modules."""
    # First, ensure AOT modules are compiled
    aot_package_dir = Path(__file__).parent / "amd_flashinfer_jit_cache" / "jit_cache"
    aot_package_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Compile AOT modules
        _compile_jit_cache(aot_package_dir)

        # Verify that some modules were actually compiled
        so_files = list(aot_package_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError("No .so files were generated during AOT compilation")

        print(f"Successfully compiled {len(so_files)} AOT HIP modules")

    except Exception as e:
        print(f"Failed to compile AOT modules: {e}")
        raise


def _prepare_build():
    """Shared preparation logic for both wheel and editable builds."""
    _build_aot_modules()


class PlatformSpecificBdistWheel(bdist_wheel):
    """Custom wheel builder that uses py_limited_api for cp310+."""

    def finalize_options(self):
        super().finalize_options()
        # Force platform-specific wheel (not pure Python)
        self.root_is_pure = False
        # Use py_limited_api for cp310 (Python 3.10+)
        self.py_limited_api = "cp310"

    def get_tag(self):
        # Use py_limited_api tags
        python_tag = "cp310"
        abi_tag = "abi3"  # Stable ABI tag

        # Get platform tag using sysconfig (PEP 425 compliant)
        plat_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")

        return python_tag, abi_tag, plat_tag


class _MonkeyPatchBdistWheel:
    """Context manager to temporarily monkey-patch setuptools bdist_wheel."""

    def __enter__(self):
        import setuptools.command.bdist_wheel as setuptools_bdist_wheel

        self.original_bdist_wheel = setuptools_bdist_wheel.bdist_wheel
        setuptools_bdist_wheel.bdist_wheel = PlatformSpecificBdistWheel
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import setuptools.command.bdist_wheel as setuptools_bdist_wheel

        setuptools_bdist_wheel.bdist_wheel = self.original_bdist_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with custom AOT module compilation."""
    print("Building amd-flashinfer-jit-cache wheel...")

    _prepare_build()

    with _MonkeyPatchBdistWheel():
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable install with custom AOT module compilation."""
    print("Building amd-flashinfer-jit-cache in editable mode...")

    _prepare_build()

    # Now build the editable install using setuptools
    _orig_build_editable = getattr(_orig, "build_editable", None)
    if _orig_build_editable is None:
        raise RuntimeError("build_editable not supported by setuptools backend")

    result = _orig_build_editable(wheel_directory, config_settings, metadata_directory)

    return result


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata with platform-specific wheel tags."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable install."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )


# Export the required interface
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = getattr(
    _orig, "get_requires_for_build_editable", None
)
