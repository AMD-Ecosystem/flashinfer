# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os

# arch_caps imports nothing (in particular, not torch), so importing it here
# keeps this module importable without torch -- which the hardware-less
# arch-caps conformance job relies on.
from .arch_caps import normalize_arch

logger = logging.getLogger(__name__)

# AMDGPU archs supported by amd-flashinfer
FLASHINFER_SUPPORTED_ROCM_ARCHS = ["gfx942", "gfx950"]


# The architecture assumed when nothing else can be determined. Must stay equal
# to ``jit/aiter_source.py``'s ``_DEFAULT_BUILD_ARCH``: the two are consulted
# independently on a GPU-less host, and a shim built for one architecture beside
# kernels built for another faults at run time.
_GPULESS_FALLBACK_ARCH = "gfx942"


@functools.cache
def _detected_supported_archs() -> tuple:
    """Supported architectures rocminfo reports, cached for the process.

    ``rocminfo_gpu_agents`` is deliberately uncached -- "the caller decides" --
    and this caller is a hot one: ``CompilationContext`` is constructed from five
    places and each construction resolves the target list. The code this replaces
    reached rocminfo through the cached ``get_supported_device_indices``, so
    calling the uncached probe directly would re-run the subprocess (with its
    timeout) on every construction. Hardware cannot change under a running
    process, so caching costs nothing in fidelity.

    Tests that patch ``rocminfo_gpu_agents`` must call ``cache_clear()``.
    """
    return tuple(
        sorted(
            {
                arch
                for arch, _ in rocminfo_gpu_agents()
                if arch in FLASHINFER_SUPPORTED_ROCM_ARCHS
            }
        )
    )


def _canonical_arch_list(raw: str) -> str:
    """``"gfx950:sramecc+; gfx942,,gfx942"`` -> ``"gfx950,gfx942"``.

    Caller- and environment-supplied lists arrive in whatever shape the operator
    typed. The validators below split on ``","`` only and compare tokens against
    :data:`FLASHINFER_SUPPORTED_ROCM_ARCHS` verbatim, so an unnormalized value is
    not merely untidy -- ``"gfx942;gfx950"`` becomes the single token
    ``"gfx942;gfx950"``, matches nothing, and
    :func:`validate_flashinfer_rocm_arch` raises "FlashInfer does not support any
    of the requested ROCm architectures". ``";"`` is worth accepting because
    ``jit/aiter_source.py`` already documents it for this same variable, and the
    two must not disagree about their own env var.

    Only *syntax* is normalized. Unknown architectures are passed through so the
    validators can report them; silently dropping one here would turn a clear
    error into a build that quietly targets less than was asked for.

    Order is preserved (first occurrence wins) rather than sorted: it is the
    caller's stated preference, and it is what ends up on the hipcc command line.
    """
    seen = []
    for token in raw.replace(";", ",").split(","):
        arch = normalize_arch(token)
        if arch and arch not in seen:
            seen.append(arch)
    return ",".join(seen)


def resolve_target_archs(arch_list: str = None) -> str:
    """Return the architectures to build for, as a comma-separated string.

    The single answer to "what are we compiling for". Resolution order:

    1. ``arch_list``, when a caller passes one explicitly.
    2. ``FLASHINFER_ROCM_ARCH_LIST``.
    3. The architectures of the supported GPUs actually present.
    4. ``gfx942``, warned -- the pre-existing default, kept deliberately (see
       the comment at the fallback).

    The bug this fixes is that steps 1-3 were open-coded three times and
    disagreed. On a CDNA4 host, ``validate_flashinfer_rocm_arch(arch_list=None)``
    returned ``{"gfx942"}`` on a gfx950 device while ``CompilationContext``
    compiled for gfx950 -- so the check that exists to catch "your PyTorch was
    not built for this architecture" was validating an architecture nobody was
    building for. Vacuous on a PyTorch carrying both; a spurious hard failure on
    an arch-specific build that carries only gfx950. gfx942 was the one
    architecture where the hard-coded literal happened to be right, which is why
    CDNA3 never noticed.

    Detection uses rocminfo rather than ``torch.cuda`` so this stays callable
    before the HIP runtime starts, and keeps this module importable without
    torch -- see ``tests/rocm/test_arch_caps.py``.
    """
    import os

    # Canonicalize the two operator-supplied paths. A value that normalizes away
    # entirely (";;", whitespace) falls through to detection rather than
    # returning "", which would otherwise reach the validators as a single empty
    # token and fail as an unsupported architecture.
    if arch_list:
        canonical = _canonical_arch_list(arch_list)
        if canonical:
            return canonical

    from_env = os.environ.get("FLASHINFER_ROCM_ARCH_LIST")
    if from_env:
        canonical = _canonical_arch_list(from_env)
        if canonical:
            return canonical

    detected = _detected_supported_archs()
    if detected:
        return ",".join(detected)

    # Deliberately the *existing* default, not "every architecture we support".
    #
    # A fat list looks like the safer answer here and is not. `aot_hip` publishes
    # this value into FLASHINFER_ROCM_ARCH_LIST, and `resolve_aiter_build_arch()`
    # returns `env_archs[0]` when no device is visible -- so "gfx942,gfx950"
    # ships gfx950 HIP kernels beside a gfx942-only AITER shim, which faults on a
    # gfx950 card. Today both sides independently fall back to gfx942 and
    # therefore agree; widening one of them alone is a regression.
    #
    # This keeps that agreement while fixing the bug this function exists for --
    # the *disagreement* between the validator and the compiler on a host where a
    # GPU is visible. Whether a GPU-less host should instead raise, build fat, or
    # teach the shim to follow the list is a real question with its own blast
    # radius (it breaks build hosts that work today), and belongs in its own
    # change.
    logger.warning(
        "No supported AMD GPU detected and FLASHINFER_ROCM_ARCH_LIST is unset; "
        "falling back to %s. Set FLASHINFER_ROCM_ARCH_LIST to the architecture "
        "you are building for -- otherwise the result will not run on %s.",
        _GPULESS_FALLBACK_ARCH,
        ", ".join(
            a for a in FLASHINFER_SUPPORTED_ROCM_ARCHS if a != _GPULESS_FALLBACK_ARCH
        ),
    )
    return _GPULESS_FALLBACK_ARCH


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
                   If None, resolved by :func:`resolve_target_archs`.
        verbose: Whether to print validation messages

    Returns:
        Validated architecture list string

    Raises:
        RuntimeError: If ROCm not found or architectures not supported
    """

    # ROCm compatibility matrix: version -> supported gfx architectures
    # Refer: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
    #        https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md#rocm-on-linux
    # Update lists for adding or removing a version or arch
    # Add new tuple for adding a new version group
    _ROCM_ARCH_GROUPS = [
        (
            # Both names for ROCm 10: a stock install reports "10.0" via
            # .info/version, while a pip-SDK build short-circuits to hipconfig,
            # which gives the HIP version instead.
            [
                "10.0",
                "7.15",
                "7.14",
                "7.13",
                "7.12",
                "7.11",
                "7.3",
                "7.2",
                "7.1",
                "7.0",
            ],
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
        arch_list = resolve_target_archs()

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
    3. PYTORCH_ROCM_ARCH, if set, permits the architectures (skipped when unset,
       where torch reports the visible cards rather than its build)

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

    # Get architecture list from parameter, env var, or default
    if arch_list is None:
        arch_list = resolve_target_archs()

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

    # Step 3: Validate against PyTorch's architectures -- but only when PyTorch was
    # actually told which ones to build. With PYTORCH_ROCM_ARCH unset,
    # _get_rocm_arch_flags() enumerates the *visible cards* rather than anything
    # about the build: it yields ["--offload-arch=", ...] when no device is
    # visible, and lists only the local card otherwise. Enforcing against that
    # turns an import on a GPU-free host into a hard error, and rejects
    # cross-compiling for gfx950 from a gfx942 box -- which is precisely what an
    # ahead-of-time build is for.
    arch_flags = [f"--offload-arch={arch}" for arch in requested_archs]
    if torch_cpp_ext_module is not None and os.environ.get("PYTORCH_ROCM_ARCH"):
        pytorch_arch_flags = torch_cpp_ext_module._get_rocm_arch_flags()
        missing_in_pytorch = [
            flag for flag in arch_flags if flag not in pytorch_arch_flags
        ]
        if missing_in_pytorch:
            raise RuntimeError(
                f"PYTORCH_ROCM_ARCH excludes the following architectures: {', '.join(missing_in_pytorch)}.\n"
                f"It restricts extension builds to: {', '.join(pytorch_arch_flags)}\n"
                "Unset it to build for whatever FLASHINFER_ROCM_ARCH_LIST requests."
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


def rocminfo_gpu_agents() -> tuple:
    """
    Return ``(arch, marketing_name)`` for each GPU agent rocminfo reports.

    Uses rocminfo (subprocess) rather than torch.cuda so this is safe to call
    before the HIP runtime is initialized (e.g. before HIP_VISIBLE_DEVICES is set
    in xdist workers).  rocminfo enumerates GPU agents in the same order as the
    HIP runtime, so entry N corresponds to HIP device index N.

    ``arch`` is normalized ("gfx942"); ``marketing_name`` is rocminfo's
    "Marketing Name" (e.g. "AMD Instinct MI350X") or "" when absent. CPU agents
    also carry a Marketing Name, so agents are filtered on ``Device Type: GPU``.

    Not cached: the caller decides.  ``get_supported_device_indices`` caches its
    own result, and one-shot consumers pay a single subprocess.

    Returns:
        tuple[tuple[str, str], ...]: One (arch, marketing_name) pair per GPU
                                     agent. Empty if rocminfo is unavailable.
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

    agents = []
    name = marketing = None
    is_gpu = False

    def _commit():
        if is_gpu:
            # rocminfo's agent-level "Name:" is normally bare ("gfx942"), but
            # normalize defensively: an unstripped qualifier would drop every
            # GPU from the supported list, and callers would silently fall back
            # to the default architecture.
            agents.append((normalize_arch(name or ""), marketing or ""))

    for line in result.stdout.splitlines():
        s = line.strip()
        if re.match(r"^Agent \d+", s):
            _commit()
            name = marketing = None
            is_gpu = False
        elif s.startswith("Name:") and name is None:
            name = s.split(":", 1)[1].strip()
        elif s.startswith("Marketing Name:") and marketing is None:
            marketing = s.split(":", 1)[1].strip()
        elif s.startswith("Device Type:") and "GPU" in s:
            is_gpu = True
    _commit()  # process last agent

    return tuple(agents)


@functools.cache
def get_supported_device_indices() -> tuple:
    """
    Return the indices of AMD GPUs whose architecture is supported by FlashInfer.

    The result is cached so rocminfo is invoked at most once per process.

    Returns:
        tuple[int, ...]: Device indices of supported GPUs. Empty tuple if none
                         found or rocminfo is unavailable.
    """
    return tuple(
        index
        for index, (arch, _) in enumerate(rocminfo_gpu_agents())
        if arch in FLASHINFER_SUPPORTED_ROCM_ARCHS
    )


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
            "  1. pip uninstall torch\n"
            "  2. pip install torch==<version> -f "
            "https://repo.radeon.com/rocm/manylinux/rocm-rel-<your-rocm>/\n\n"
            "Pin the version. -f only *adds* to PyPI, so an unpinned install\n"
            "can pick a newer CPU-only wheel from there and land you back here.\n"
            "Which version pairs with your ROCm release, and what to do when\n"
            "repo.radeon.com publishes no rocm-rel- directory for it (as for\n"
            "the 10.0 the development image uses), are in the Quick start:\n"
            "https://github.com/AMD-Ecosystem/flashinfer#quick-start\n" + "=" * 70
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
                f"To fix, reinstall PyTorch built for ROCm "
                f"{system_rocm_major_minor}. Which wheel, and where from,\n"
                f"depends on that release -- repo.radeon.com publishes no\n"
                f"rocm-rel- directory for some of them, so there is no one\n"
                f"command that is right for all.\n\n"
                f"See https://github.com/AMD-Ecosystem/flashinfer#quick-start\n"
                f"{'=' * 70}",
                RuntimeWarning,
                stacklevel=2,
            )
