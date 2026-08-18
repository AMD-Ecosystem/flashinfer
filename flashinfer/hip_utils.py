# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools

# arch_caps imports nothing (in particular, not torch), so importing it here
# keeps this module safe to import before the HIP runtime starts -- which
# tests/conftest.py relies on to set HIP_VISIBLE_DEVICES first.
from .arch_caps import normalize_arch

# AMDGPU archs supported by amd-flashinfer
FLASHINFER_SUPPORTED_ROCM_ARCHS = ["gfx942", "gfx950"]


def get_rocm_home():
    """
    Get the ROCM_HOME directory from environment variables or default path.

    Returns:
        str: Path to ROCm installation (e.g., "/opt/rocm")
    """
    import os

    return os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"


def is_therock_build() -> bool:
    """
    Check if ROCm was built using TheRock build system.

    Returns:
        bool: True if TheRock build is detected, False otherwise
    """
    import os

    # First, try checking for rocm_sdk package
    try:
        import rocm_sdk

        if hasattr(rocm_sdk, "__version__") and rocm_sdk.__version__:
            return True
    except ImportError:
        pass

    # Fall back to checking for TheRock manifest file
    rocm_home = get_rocm_home()
    manifest_path = os.path.join(rocm_home, "share", "therock", "therock_manifest.json")
    return os.path.isfile(manifest_path)


def get_system_rocm_version_from_info_file():
    """
    Try to get ROCm version from .info/version file located in ROCM_HOME.

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import os

    rocm_home = get_rocm_home()
    version_file = os.path.join(rocm_home, ".info", "version")
    try:
        with open(version_file, "r") as f:
            version = f.read().strip()
            return ".".join(version.split(".")[:3])
    except (FileNotFoundError, IOError):
        return None


def get_system_rocm_version_from_hipconfig():
    """
    Try to get ROCm version from hipconfig --version command.

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["hipconfig", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            match = re.search(r"(\d+\.\d+(?:\.\d+)?)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_system_rocm_version_from_amd_smi():
    """
    Try to get ROCm version from amd-smi command.

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["amd-smi"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            match = re.search(r"ROCm version:\s*(\d+\.\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_system_rocm_version_from_dpkg():
    """
    Try to get ROCm version from dpkg (Ubuntu/Debian package manager).

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["dpkg", "-l", "rocm-core"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            match = re.search(r"rocm-core\s+(\d+\.\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_system_rocm_version():
    """
    Attempt to detect the system ROCm version.

    For standard ROCm installations, detection falls back through several
    methods in order of reliability: ``ROCM_HOME/.info/version``, ``amd-smi``,
    ``dpkg``, and finally ``hipconfig``.

    For TheRock builds, ``hipconfig`` is used directly because it reports the
    HIP runtime version (consistent with ``torch.version.hip``), unlike
    ``.info/version`` which reports the TheRock SDK version (for example,
    ``"7.12.0"`` when HIP is ``7.3``).

    Returns:
        str: ROCm version like "7.1.0" or None if not detectable
    """
    if is_therock_build():
        return get_system_rocm_version_from_hipconfig()

    # Try standard detection methods in order of reliability
    detection_methods = [
        get_system_rocm_version_from_info_file,
        get_system_rocm_version_from_amd_smi,
        get_system_rocm_version_from_dpkg,
        get_system_rocm_version_from_hipconfig,
    ]

    for method in detection_methods:
        version = method()
        if version:
            return version
        print(f"ROCm version not found using {method.__name__}. Trying next method...")

    return None


def validate_rocm_arch(arch_list: str = None, verbose: bool = False) -> str:
    """
    Validate ROCm architecture against system ROCm version.

    Args:
        arch_list: Comma-separated list of architectures (e.g., "gfx942,gfx90a").
                   If None, reads from FLASHINFER_ROCM_ARCH_LIST env var or defaults to "gfx942"
        verbose: Whether to print validation messages

    Returns:
        Validated architecture list string

    Raises:
        RuntimeError: If ROCm not found or architectures not supported
    """
    import os

    # ROCm compatibility matrix: version -> supported gfx architectures
    # Refer: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
    #        https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md#rocm-on-linux
    # Update lists for adding or removing a version or arch
    # Add new tuple for adding a new version group
    _ROCM_ARCH_GROUPS = [
        (
            ["7.13", "7.12", "7.11", "7.3", "7.2", "7.1", "7.0"],
            [
                "gfx950",
                "gfx1201",
                "gfx1200",
                "gfx1101",
                "gfx1100",
                "gfx1030",
                "gfx942",
                "gfx90a",
                "gfx908",
            ],
        ),
        (
            ["6.4", "6.3"],
            ["gfx1100", "gfx1030", "gfx942", "gfx90a", "gfx908"],
        ),
    ]

    # Build the compatibility matrix
    ROCM_COMPAT_MATRIX = {
        version: archs for versions, archs in _ROCM_ARCH_GROUPS for version in versions
    }

    # Get architecture list from parameter, env var, or default
    if arch_list is None:
        arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "gfx942")

    # Validate system has ROCm installed
    system_rocm_version = get_system_rocm_version()
    if system_rocm_version is None:
        raise RuntimeError(
            "Could not detect ROCm installation. Please ensure ROCm is installed and "
            "accessible (check ROCM_PATH, ROCM_HOME or /opt/rocm)."
        )

    # Parse version to major.minor for compatibility check
    version_parts = system_rocm_version.split(".")
    rocm_version_key = f"{version_parts[0]}.{version_parts[1]}"

    # Validate architectures against compatibility matrix
    requested_archs = [arch.strip() for arch in arch_list.split(",")]
    supported_archs = ROCM_COMPAT_MATRIX.get(rocm_version_key, [])

    if not supported_archs:
        raise RuntimeError(
            f"ROCm version {system_rocm_version} is not recognized in the ROCm "
            f"compatibility matrix. Requested architectures: {', '.join(requested_archs)}.\n"
            f"See compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html"
        )

    # Check each requested arch is supported; filter out unsupported ones rather than
    # failing hard so that a system with a mix of supported dGPUs and unsupported
    # integrated GPUs still works.
    unsupported = [arch for arch in requested_archs if arch not in supported_archs]
    if unsupported:
        supported_in_request = [
            arch for arch in requested_archs if arch in supported_archs
        ]
        if not supported_in_request:
            raise RuntimeError(
                f"ROCm version {system_rocm_version} does not support any of the provided "
                f"architectures: {', '.join(unsupported)}.\n"
                f"See compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html"
            )
        import warnings

        warnings.warn(
            f"ROCm version {system_rocm_version} does not support the following "
            f"architecture(s): {', '.join(unsupported)}. "
            f"They will be excluded from compilation. "
            f"Supported architecture(s) found: {', '.join(supported_in_request)}.",
            UserWarning,
            stacklevel=2,
        )
        arch_list = ",".join(supported_in_request)

    if verbose:
        print(f"Validated ROCm {system_rocm_version} with architecture(s): {arch_list}")

    return arch_list


def validate_flashinfer_rocm_arch(
    arch_list: str = None, torch_cpp_ext_module=None, verbose: bool = False
) -> tuple:
    """
    Comprehensive ROCm architecture validation for FlashInfer compilation.

    Validates in order:
    1. System ROCm version supports the architectures (ROCM_COMPAT_MATRIX)
    2. FlashInfer has AMD ports for the architectures (FLASHINFER_SUPPORTED_ROCM_ARCHS)
    3. PyTorch was compiled with the architectures (torch.utils.cpp_extension)

    Args:
        arch_list: Comma-separated list (e.g., "gfx942,gfx90a") or None for default
        torch_cpp_ext_module: torch.utils.cpp_extension module for PyTorch validation
        verbose: Print validation messages

    Returns:
        tuple: (arch_flags, arch_set)
            - arch_flags: List like ["--offload-arch=gfx942"]
            - arch_set: Set like {"gfx942"}

    Raises:
        RuntimeError: If any validation step fails with clear error message
    """
    import os

    # Get architecture list from parameter, env var, or default
    if arch_list is None:
        arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "gfx942")

    # Step 1: Validate against system ROCm version (reuse existing logic)
    validated_arch_list = validate_rocm_arch(arch_list=arch_list, verbose=verbose)
    requested_archs = [arch.strip() for arch in validated_arch_list.split(",")]

    # Step 2: Validate against AMD-ported FlashInfer architectures.
    # Filter out unsupported archs rather than failing hard so that a system with a mix
    # of supported dGPUs and unsupported integrated GPUs (e.g. gfx942 + gfx1035) still works.
    unsupported_by_flashinfer = [
        arch for arch in requested_archs if arch not in FLASHINFER_SUPPORTED_ROCM_ARCHS
    ]
    if unsupported_by_flashinfer:
        supported_in_request = [
            arch for arch in requested_archs if arch in FLASHINFER_SUPPORTED_ROCM_ARCHS
        ]
        if not supported_in_request:
            raise RuntimeError(
                f"FlashInfer does not support any of the requested ROCm architectures: "
                f"{', '.join(unsupported_by_flashinfer)}.\n"
                f"Currently supported by FlashInfer: {', '.join(FLASHINFER_SUPPORTED_ROCM_ARCHS)}"
            )
        import warnings

        warnings.warn(
            f"FlashInfer does not support the following ROCm architecture(s): "
            f"{', '.join(unsupported_by_flashinfer)}. "
            f"They will be excluded from JIT compilation. "
            f"Supported architecture(s) found: {', '.join(supported_in_request)}.",
            UserWarning,
            stacklevel=2,
        )
        requested_archs = supported_in_request

    # Step 3: Validate against PyTorch's available architectures (if module provided)
    arch_flags = [f"--offload-arch={arch}" for arch in requested_archs]
    if torch_cpp_ext_module is not None:
        pytorch_arch_flags = torch_cpp_ext_module._get_rocm_arch_flags()
        missing_in_pytorch = [
            flag for flag in arch_flags if flag not in pytorch_arch_flags
        ]
        if missing_in_pytorch:
            raise RuntimeError(
                f"PyTorch does not support the following architectures: {', '.join(missing_in_pytorch)}.\n"
                f"PyTorch was compiled with: {', '.join(pytorch_arch_flags)}"
            )

    if verbose:
        print(f"FlashInfer validated architectures: {', '.join(requested_archs)}")

    # Return both the flags list and the set
    arch_set = set(requested_archs)
    return arch_flags, arch_set


def get_available_gpu_count() -> int:
    """
    Query the number of AMD GPUs visible to the current process.

    Uses torch.cuda.device_count(), which maps to the HIP runtime on ROCm and
    respects HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES.  torch is a required
    dependency of amd-flashinfer, so no fallback is needed.

    Returns:
        int: Number of visible GPUs (0 if none are visible to this process).
    """
    import torch

    return torch.cuda.device_count()


@functools.cache
def get_supported_device_indices() -> tuple:
    """
    Return the indices of AMD GPUs whose architecture is supported by FlashInfer.

    Uses rocminfo (subprocess) rather than torch.cuda so this function is safe
    to call before the HIP runtime is initialized (e.g. before HIP_VISIBLE_DEVICES
    is set in xdist workers).  rocminfo enumerates GPU agents in the same order as
    the HIP runtime, so the Nth GPU agent = HIP device index N.

    The result is cached so rocminfo is invoked at most once per process.

    Returns:
        tuple[int, ...]: Device indices of supported GPUs. Empty tuple if none
                         found or rocminfo is unavailable.
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10, check=False
        )
        if result.returncode != 0:
            return ()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ()

    supported = []
    gpu_index = 0
    current_name = None
    current_is_gpu = False

    def _commit():
        nonlocal gpu_index
        if current_is_gpu:
            # rocminfo's agent-level "Name:" is normally bare ("gfx942"), but
            # normalize defensively: an unstripped qualifier here would drop
            # every GPU from the supported list, and callers would silently
            # fall back to the default architecture.
            if normalize_arch(current_name or "") in FLASHINFER_SUPPORTED_ROCM_ARCHS:
                supported.append(gpu_index)
            gpu_index += 1

    for line in result.stdout.splitlines():
        s = line.strip()
        if re.match(r"^Agent \d+", s):
            _commit()
            current_name = None
            current_is_gpu = False
        elif s.startswith("Name:") and current_name is None:
            current_name = s.split(":", 1)[1].strip()
        elif s.startswith("Device Type:") and "GPU" in s:
            current_is_gpu = True
    _commit()  # process last agent

    return tuple(supported)


# A "primary" CPX sibling reports the full physical card capacity; the other
# three siblings report ~25% of it. 0.95 separates the two cleanly without
# tying the test to an exact GB value.
_PRIMARY_VRAM_RATIO = 0.95


@functools.cache
def get_physical_card_device_indices() -> tuple:
    """
    Return one supported device index per physical AMD card.

    On CDNA3 CPX systems each physical card exposes 4 logical XCD-sized
    devices that share the card's HBM. Running one xdist worker per logical
    device causes intermittent HSA hardware exceptions when multiple workers
    on the same physical card concurrently allocate large tensors. We pick
    one "primary" index per card (the one that reports the full card
    capacity via rocm-smi) so callers can spread workloads one-per-card.

    On non-CPX systems all supported devices report identical VRAM and the
    helper returns them unchanged. Falls back to the supported-device list
    if rocm-smi is unavailable or its output cannot be parsed.

    Cached per-process; xdist spawns workers as separate processes, so the
    cache is per-worker (the underlying rocm-smi call is paid once per
    Python process, not once per test).
    """
    import json
    import subprocess

    supported = get_supported_device_indices()
    if not supported:
        return ()

    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return supported
        data = json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return supported

    vram_by_idx: dict[int, int] = {}
    for key, val in data.items():
        if not key.startswith("card") or not isinstance(val, dict):
            continue
        try:
            idx = int(key[4:])
            vram_by_idx[idx] = int(val["VRAM Total Memory (B)"])
        except (KeyError, TypeError, ValueError):
            continue

    supported_vram = {idx: vram_by_idx[idx] for idx in supported if idx in vram_by_idx}
    if not supported_vram:
        return supported

    threshold = int(max(supported_vram.values()) * _PRIMARY_VRAM_RATIO)
    primary = tuple(idx for idx in supported if supported_vram.get(idx, 0) >= threshold)
    return primary or supported


def check_torch_rocm_compatibility() -> None:
    """
    Verify that PyTorch is installed with compatible ROCm support.

    This function checks:
    1. PyTorch is installed
    2. PyTorch has ROCm/HIP support (not CPU-only)
    3. PyTorch ROCm version matches system ROCm version (if detectable)

    Provides helpful error messages to guide users to correct installation.

    Raises:
        ImportError: If PyTorch is not installed
        RuntimeError: If PyTorch doesn't have ROCm support
    """
    import warnings

    from torch import version

    # Check for torch package with rocm support
    if not hasattr(version, "hip") or version.hip is None:
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "ERROR: PyTorch does NOT have ROCm support.\n\n"
            "You installed the CPU-only version from PyPI.\n"
            "amd-flashinfer requires PyTorch compiled with ROCm support.\n\n"
            "Fix this by:\n"
            "  1. Uninstall current PyTorch:\n"
            "     pip uninstall torch\n\n"
            "  2. Install PyTorch for ROCm:\n"
            "     pip install torch==2.7.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/\n\n"
            "See https://github.com/rocm/flashinfer for detailed installation instructions.\n"
            + "="
            * 70
        )

    # ROCm version compatibility warning
    torch_rocm = version.hip
    torch_rocm_major_minor = ".".join(torch_rocm.split(".")[:2])

    # Try to detect system ROCm version
    system_rocm = get_system_rocm_version()

    if system_rocm:
        system_rocm_major_minor = ".".join(system_rocm.split(".")[:2])
        if torch_rocm_major_minor != system_rocm_major_minor:
            warnings.warn(
                f"\n{'=' * 70}\n"
                f"WARNING: ROCm version mismatch detected!\n\n"
                f"  System ROCm version: {system_rocm}\n"
                f"  PyTorch ROCm version: {torch_rocm_major_minor}\n\n"
                f"This may cause runtime errors or crashes.\n\n"
                f"To fix, reinstall PyTorch for your ROCm version:\n"
                f"  pip install torch==2.7.1 -f "
                f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{system_rocm}/\n\n"
                f"Or if using uv:\n"
                f"  export FLASHINFER_ROCM_VERSION={system_rocm}\n"
                f"  uv pip install torch==2.7.1 -f "
                f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{system_rocm}/\n"
                f"{'=' * 70}",
                RuntimeWarning,
                stacklevel=2,
            )
