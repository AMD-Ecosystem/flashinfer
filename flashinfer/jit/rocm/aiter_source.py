# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Shared plumbing for FlashInfer's C++-level AITER backends (ROCm).

FlashInfer wraps AITER kernels by compiling a small ``csrc/rocm/*_aiter.cu`` shim
that calls AITER's C++ entry point directly and links the symbol-visible AITER
``.so``. Prefer ``#include``-ing AITER's real header, so a signature change is a
compile error rather than a load-time ``undefined symbol``. Fall back to a
forward declaration only for headers that still pull in pybind11, which clashes
with FlashInfer's ``-DPy_LIMITED_API`` build (``rope.h``, ``rmsnorm.h`` as of
0.1.16); there ``torch::Tensor`` is ``at::Tensor``, so the linker still resolves.

AITER's installed wheel builds its modules with ``-fvisibility=hidden``, so the
kernel symbols (e.g. ``rope_cached_positions_2c_fwd_impl``) are not linkable. This
helper rebuilds the needed AITER module once with ``AITER_SYMBOL_VISIBLE=1`` via
AITER's own ``aiter.jit.core.build_module`` (which also runs CK blob codegen for CK
ops), caches the result under FlashInfer's cache dir as ``lib<module>.so``, and
hands back the include/link flags for ``gen_jit_spec``.

A shim may link more than one AITER module (fused MoE needs both ``moe_sorting``
and a CK GEMM module), and a module may be built in a *specialized* form -- see
:class:`AiterModule`.
"""

import functools
import inspect
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from filelock import FileLock

from ...rocm.arch_caps import normalize_arch
from .. import env as jit_env
from ..core import JitSpec, logger


_DEFAULT_BUILD_ARCH = "gfx942"

# An architecture name and nothing else: "gfx942", "gfx950", "gfx90a". Anchored
# so a token that merely starts with "gfx" cannot smuggle in a path separator.
_ARCH_RE = re.compile(r"^gfx[0-9a-f]+$")

# The linked library name becomes an -l flag and a filename, so it may not smuggle
# in a path separator or a flag-looking prefix. \Z, not $: $ also matches before a
# trailing newline, which would reach the ninja link line intact.
_LIB_NAME_RE = re.compile(r"\A[A-Za-z0-9_][A-Za-z0-9_.+-]*\Z")

# Guards the env-mutating build in _build_aiter_lib; see ensure_aiter_lib.
_BUILD_LOCK = threading.Lock()


@dataclass(frozen=True)
class AiterModule:
    """One AITER module to build symbol-visible and link into a FlashInfer shim.

    ``name`` keys AITER's ``optCompilerConfig.json``, which supplies the sources
    and compile flags. ``md_name`` and ``blob_gen_cmd`` override the two fields
    that select *which instances* that source tree generates.

    Specialization is not an optimization. ``module_moe_ck2stages`` builds every
    dtype/activation/quant combination by default -- 280 MB and far past a
    tolerable first-call build -- while one configuration needs a handful of
    instances. AITER specializes it the same way (``get_moe_stage_module`` in
    ``aiter/ops/moe_op.py``).

    ``md_name`` also names the cached ``lib<md_name>.so``, so two specializations
    of one module cannot overwrite each other.
    """

    name: str
    md_name: Optional[str] = None
    blob_gen_cmd: Optional[Union[str, Tuple[str, ...]]] = None

    @property
    def lib_name(self) -> str:
        """The library basename: ``lib<lib_name>.so``, linked as ``-l<lib_name>``."""
        # `is None`, not falsiness: an empty md_name must not silently resolve to
        # the unspecialized name, which is how two specializations would collide.
        return self.name if self.md_name is None else self.md_name

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("md_name", self.md_name)):
            if value is not None and not _LIB_NAME_RE.match(value):
                raise ValueError(
                    f"AITER {label} {value!r} is not usable as a library name; "
                    "expected something like 'module_moe_sorting'"
                )


def _as_modules(
    modules: Sequence[Union[str, AiterModule]],
) -> Tuple[AiterModule, ...]:
    return tuple(AiterModule(m) if isinstance(m, str) else m for m in modules)


def _env_arch_list() -> List[str]:
    """FLASHINFER_ROCM_ARCH_LIST as a normalized list, accepting ',' or ';'.

    Tokens are checked *after* normalization, and anything that is not an
    architecture name is dropped with a warning. Two reachable reasons, both of
    which end up naming a directory:

    - A token that is all qualifier and no architecture (``":sramecc+"``, or a
      bare ``":"``) is non-empty as written but normalizes to ``""``, which
      would become the build architecture and name a cache directory
      ``__aiter-<version>``.
    - An arbitrary string reaches that same directory name, so ``"../../tmp"``
      escapes the cache root once the tag is joined onto it.

    ``validate_rocm_arch`` already rejects such a token for the main JIT, but it
    only *warns and excludes* unless every entry is bad -- so in a mixed list the
    bad entry survives to here.

    Empty tokens are dropped silently: a trailing separator is benign, not a typo
    worth reporting.
    """
    raw = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "")
    archs = []
    for token in re.split(r"[;,]", raw):
        arch = normalize_arch(token)
        if not arch:
            continue
        if not _ARCH_RE.match(arch):
            logger.warning(
                "Ignoring %r in FLASHINFER_ROCM_ARCH_LIST: not a GPU architecture "
                "name (expected e.g. 'gfx942').",
                token,
            )
            continue
        archs.append(arch)
    return archs


def _detected_device_arch() -> Optional[str]:
    """The running device's architecture, or None if no GPU is visible.

    torch is imported lazily: this module is imported during JIT setup, and a
    GPU-less wheel build must not require a working HIP runtime.
    """
    try:
        import torch

        if torch.cuda.device_count() == 0:
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return normalize_arch(props.gcnArchName)
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def resolve_aiter_build_arch() -> str:
    """Return the **single** GPU architecture to build the AITER shim for.

    Deliberately one architecture, never a list. AITER's ``get_gfx()`` resolves
    to the *last* ``GPU_ARCHS`` entry rather than the running device, so a
    multi-arch value makes AITER's Python-level dispatch believe it is on that
    architecture no matter what the hardware is -- and it also flips global
    compile flags. One entry makes that failure mode unreachable.

    Resolution order: ``FLASHINFER_ROCM_ARCH_LIST`` -> detected device ->
    ``gfx942``. When the environment names several architectures, the running
    device's is preferred among them; when the environment names architectures
    that exclude the running device, the environment still wins (cross-compiling
    is legitimate) but the mismatch is reported, because the resulting shim will
    fault on this machine.
    """
    env_archs = _env_arch_list()
    device_arch = _detected_device_arch()

    if env_archs:
        if device_arch and device_arch in env_archs:
            return device_arch
        if device_arch:
            logger.warning(
                "FLASHINFER_ROCM_ARCH_LIST=%s does not include this device's "
                "architecture (%s); building the AITER shim for %s. It will not "
                "run on this GPU.",
                ",".join(env_archs),
                device_arch,
                env_archs[0],
            )
        return env_archs[0]

    if device_arch:
        return device_arch

    logger.warning(
        "No ROCm device detected and FLASHINFER_ROCM_ARCH_LIST is unset; "
        "building the AITER shim for %s. Set FLASHINFER_ROCM_ARCH_LIST to "
        "target a different architecture.",
        _DEFAULT_BUILD_ARCH,
    )
    return _DEFAULT_BUILD_ARCH


def _aiter_cache_tag() -> str:
    """A filesystem-safe tag keying the lib cache by target arch and AITER version.

    Without the arch component, a lib built for one arch would be silently reused
    on a machine with a different arch. Without the version component, an AITER
    upgrade (which can change the C++ ABI the FlashInfer shim links against) would
    silently reuse the stale .so. The FlashInfer JIT dir already keys by its own
    version+arch, but this cache sits outside it.

    Keyed on the *resolved* architecture -- the one actually compiled for -- so
    the tag cannot disagree with the contents of the directory it names."""
    arch = resolve_aiter_build_arch()
    # The tag is joined onto the cache root to create a directory, so
    # "filesystem-safe" above has to be enforced, not just asserted in prose.
    # _env_arch_list already rejects anything that is not an architecture name;
    # this keeps the guarantee true for the other two sources of `arch` (the
    # device probe and the default) and for any future caller. Loud rather than
    # silently sanitized: an arch that needs rewriting means the resolver is
    # wrong, and a quietly renamed cache directory would hide that.
    if not arch or arch != Path(arch).name or arch.startswith("."):
        raise ValueError(
            f"refusing to build a cache directory name from architecture "
            f"{arch!r}: not a single safe path component"
        )
    try:
        import importlib.metadata as _md

        version = _md.version("amd-aiter")
    except Exception:
        version = "unknown"
    return f"{arch}__aiter-{version}"


@functools.lru_cache(maxsize=1)
def _aiter_libs_dir() -> Path:
    # Keyed by arch + AITER version so a cached lib is never reused across an
    # incompatible arch or a changed AITER ABI.
    d = jit_env.FLASHINFER_CACHE_DIR / "aiter_libs" / _aiter_cache_tag()
    d.mkdir(parents=True, exist_ok=True)
    return d


def refresh_aiter_jitspec(spec: JitSpec) -> JitSpec:
    """Regenerate ``build.ninja`` so a changed AITER library path takes effect.

    The AITER shim libs live outside the JIT tree, under
    ``aiter_libs/<arch>__aiter-<version>/``, and reach the module only as an
    ``-L``/``-rpath`` on the link line. ``JitSpec.build()`` writes ``build.ninja``
    only when it is missing, so once a module has been built the recorded link
    line is never revisited -- the module keeps loading whichever AITER lib it
    was first built against, even after the resolved architecture changes.

    That is not a stale-build annoyance: the ``.so`` retains a RUNPATH into the
    old directory, so a module cached under ``.../gfx950/`` can go on loading a
    gfx942 library and **segfault**, and clearing ``FLASHINFER_ROCM_ARCH_LIST``
    or setting it correctly does not fix it. Only deleting the cache does.

    ``write_ninja`` funnels through ``write_if_different``, so this is free when
    nothing changed and rewrites exactly when the link line moves; ninja then
    relinks on its own.

    Only the JIT path is touched. An AOT-prebuilt module is loaded straight from
    ``aot_path`` and a ``FLASHINFER_DISABLE_JIT`` run raises before ninja is
    consulted, so in both cases the manifest has no reader and rewriting it would
    be pure filesystem noise.

    The write takes ``spec.lock_path`` -- the same lock ``JitSpec.build()`` holds
    while ninja runs -- because ``write_if_different`` truncates in place. Without
    it a concurrent builder (``pytest -n auto`` shares one JIT cache across
    processes) could have the manifest emptied under it mid-read.

    Args:
        spec: The freshly created :class:`~flashinfer.jit.core.JitSpec`.

    Returns:
        The same spec, for use as ``return refresh_aiter_jitspec(gen_jit_spec(...))``.
    """
    # v0.6.18 made JitSpec an ABC; is_aot and write_ninja live on JitSpecNvcc,
    # which is what gen_jit_spec() actually returns under its JitSpec annotation.
    if spec.is_aot or os.environ.get("FLASHINFER_DISABLE_JIT"):  # type: ignore[attr-defined]
        return spec
    with FileLock(spec.lock_path, thread_local=False):
        spec.write_ninja()  # type: ignore[attr-defined]
    return spec


@functools.lru_cache(maxsize=1)
def _aiter_csrc_include_dir() -> Path:
    """The aiter_meta C++ public header dir (rmsnorm.h / activation.h / rope.h)."""
    # aiter ships its C++ sources/headers in the sibling aiter_meta package,
    # which is a namespace package (no __file__) — resolve via __path__.
    import aiter_meta

    for p in aiter_meta.__path__:
        inc = Path(p) / "csrc" / "include"
        if inc.exists():
            return inc
    raise RuntimeError(
        "Could not locate aiter_meta/csrc/include; is the aiter source package installed?"
    )


def ensure_aiter_lib(module: Union[str, AiterModule]) -> Path:
    """
    Build (once, cached) a symbol-visible AITER module and return the path to the
    linkable ``lib<name>.so`` under FlashInfer's cache.

    Idempotent: if the cached lib already exists it is returned without rebuilding.
    """
    if isinstance(module, str):
        module = AiterModule(module)
    module_name = module.name
    md_name = module.lib_name

    libs_dir = _aiter_libs_dir()
    lib_path = libs_dir / f"lib{md_name}.so"
    if lib_path.exists():
        return lib_path

    # One lock for every module, not one per module. _build_aiter_lib mutates
    # process-global env (AITER_JIT_DIR, GPU_ARCHS, ROCM_HOME) and restores a
    # snapshot taken on entry, so two builds of *different* modules in one process
    # interleave: the first to finish rolls the environment back under the one
    # still running. The thread lock covers that; the file lock covers `pytest
    # -n auto`, which shares a cache dir across processes. os.replace below makes
    # publishing atomic; these make producing safe.
    with _BUILD_LOCK, FileLock(str(libs_dir / ".aiter-build.lock"), thread_local=False):
        if lib_path.exists():
            return lib_path
        return _build_aiter_lib(module_name, md_name, module.blob_gen_cmd, lib_path)


def _build_aiter_lib(
    module_name: str,
    md_name: str,
    blob_gen_cmd_override: Optional[Union[str, Tuple[str, ...]]],
    lib_path: Path,
) -> Path:
    libs_dir = lib_path.parent
    # Build into a FlashInfer-owned dir so we never mutate the AITER install.
    # Deliberately one dir for every module, not one per md_name: AITER_JIT_DIR
    # is process-global, and a shim now builds two modules per spec. A per-module
    # value makes concurrent builders overwrite each other's env; a constant one
    # makes that write benign. AITER already isolates each build under
    # {bd_dir}/{md_name} anyway.
    aiter_build_dir = libs_dir / "build"
    aiter_build_dir.mkdir(parents=True, exist_ok=True)

    prev = {
        "AITER_SYMBOL_VISIBLE": os.environ.get("AITER_SYMBOL_VISIBLE"),
        "AITER_JIT_DIR": os.environ.get("AITER_JIT_DIR"),
        "GPU_ARCHS": os.environ.get("GPU_ARCHS"),
        "ROCM_HOME": os.environ.get("ROCM_HOME"),
    }

    built: Optional[Path] = None
    try:
        # Inside the try so a failure here still restores the environment: a
        # leaked AITER_JIT_DIR sends the *next* module's .so hunt to the wrong
        # directory.
        from ...rocm.hip_utils import get_rocm_home

        os.environ["AITER_SYMBOL_VISIBLE"] = "1"
        os.environ["AITER_JIT_DIR"] = str(aiter_build_dir)
        # AITER splits GPU_ARCHS on ';' and validates each entry, so a
        # comma-joined list reaches it as one unparseable token. A single
        # architecture sidesteps the separator entirely -- and is required
        # regardless; see resolve_aiter_build_arch.
        os.environ["GPU_ARCHS"] = resolve_aiter_build_arch()
        os.environ["ROCM_HOME"] = get_rocm_home()

        from aiter.jit import core as aiter_core
        from aiter.jit.core import build_module, get_args_of_build

        # AITER's cpp_extension bakes -fvisibility=hidden into COMMON_HIPCC_FLAGS
        # at import time when AITER_SYMBOL_VISIBLE is unset. If aiter was already
        # imported (e.g. the MHA/MLA path ran first), setting the env var now is
        # too late, so force default visibility via the per-build flags — they are
        # appended after COMMON_HIPCC_FLAGS, and the later flag wins. Without this,
        # the rebuilt .so hides the kernel symbols and the FlashInfer shim fails to
        # link with "undefined symbol".
        a = get_args_of_build(module_name)
        flags_extra_hip = list(a["flags_extra_hip"]) + ["-fvisibility=default"]
        blob_gen_cmd = a["blob_gen_cmd"]
        if blob_gen_cmd_override is not None:
            blob_gen_cmd = (
                blob_gen_cmd_override
                if isinstance(blob_gen_cmd_override, str)
                else list(blob_gen_cmd_override)
            )
        logger.info(
            "Building symbol-visible AITER module %s for %s (first use; this can "
            "take several minutes for CK modules)",
            md_name,
            os.environ["GPU_ARCHS"],
        )
        kwargs = {
            "md_name": md_name,
            "srcs": a["srcs"],
            "flags_extra_cc": a["flags_extra_cc"],
            "flags_extra_hip": flags_extra_hip,
            "blob_gen_cmd": blob_gen_cmd,
            "extra_include": a["extra_include"],
            "extra_ldflags": a["extra_ldflags"],
            "verbose": os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1",
            "is_python_module": a["is_python_module"],
            "is_standalone": a["is_standalone"],
            "torch_exclude": a["torch_exclude"],
            "hipify": a.get("hipify", False),
            # Added after 0.1.10; required (no default) from 0.1.16 on, where it
            # selects third-party sources AITER clones per build (CK,
            # HipKittens). get_args_of_build supplies the right value.
            "third_party": a.get("third_party"),
        }
        # AITER's build_module signature moves between releases, so pass only what
        # the installed one accepts: 0.1.10 has no `third_party` and would raise
        # TypeError on it, while 0.1.16+ makes it a required positional. Filtering
        # here keeps a single code path working across both.
        accepted = inspect.signature(build_module).parameters
        build_module(**{k: v for k, v in kwargs.items() if k in accepted})

        # AITER decides the output dir from a module-level `bd_dir` global that is
        # frozen at import time, so the .so does not reliably land in
        # aiter_build_dir when aiter was imported before we set AITER_JIT_DIR.
        # Resolve the produced file from AITER's own get_user_jit_dir() (which
        # re-reads the env), falling back to a recursive search of our build dir.
        built = _find_built_so(
            md_name, aiter_build_dir, Path(aiter_core.get_user_jit_dir())
        )
    finally:
        for k, v in prev.items():
            if v is None:
                # GPU_ARCHS is the exception: AITER's own Python ops build
                # outside this scope and assert on an unset value, and this is
                # the arch _ensure_aiter_gpu_archs would resolve anyway.
                if k != "GPU_ARCHS":
                    os.environ.pop(k, None)
            else:
                os.environ[k] = v

    if built is None:
        raise RuntimeError(
            f"AITER build for {module_name!r} did not produce a {md_name}.so. "
            "Check that aiter is installed and ROCm is available."
        )
    # Copy (not symlink) the artifact into our arch-keyed cache so it survives
    # cleanup of AITER's build tree, and publish it atomically via os.replace so a
    # concurrent loader never observes a partial/missing file.
    tmp_lib = lib_path.with_name(f".{lib_path.name}.{os.getpid()}.tmp")
    shutil.copy2(built, tmp_lib)
    os.replace(tmp_lib, lib_path)
    return lib_path


def _find_built_so(md_name: str, *search_dirs: Path) -> Optional[Path]:
    """Locate the freshly built ``<md_name>.so`` across AITER's output dirs."""
    name = f"{md_name}.so"
    for d in search_dirs:
        candidate = d / name
        if candidate.is_file():
            return candidate
    for d in search_dirs:
        if d.exists():
            for found in d.rglob(name):
                if found.is_file():
                    return found
    return None


def aiter_jitspec_flags(
    *modules: Union[str, AiterModule],
) -> Tuple[List[Union[str, Path]], List[str]]:
    """
    Build the AITER libs if needed and return ``(extra_include_paths, extra_ldflags)``
    to pass to ``gen_jit_spec`` so the FlashInfer shim can find AITER's headers and
    link the kernels.

    Accepts more than one module: a shim that spans two AITER modules (fused MoE
    calls ``moe_sorting`` and a CK GEMM module) links them all against the one
    ``-L``/``-rpath`` pair, since every lib lands in the same arch-keyed cache dir.
    """
    if not modules:
        raise ValueError("aiter_jitspec_flags needs at least one AITER module")
    resolved = _as_modules(modules)
    for module in resolved:
        ensure_aiter_lib(module)
    libs_dir = _aiter_libs_dir()
    extra_include_paths: List[Union[str, Path]] = [str(_aiter_csrc_include_dir())]
    extra_ldflags = [f"-L{libs_dir}"]
    extra_ldflags += [f"-l{module.lib_name}" for module in resolved]
    extra_ldflags.append(f"-Wl,-rpath,{libs_dir}")
    return extra_include_paths, extra_ldflags
