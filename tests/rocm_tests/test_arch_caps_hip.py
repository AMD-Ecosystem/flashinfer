# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for flashinfer.arch_caps.

Deliberately GPU-free: every assertion here holds on any machine, so this file
protects the architectures we cannot physically test against as well as the one
we can. It is also *dependency*-free -- see the loader below -- which is what
lets it run as a hardware-less CI job on every pull request
(``.github/workflows/arch-caps-conformance.yml``).
"""

import importlib.util
import pathlib
import subprocess
import sys
import types

import pytest

# --------------------------------------------------------------------------
# Load the modules under test without importing the `flashinfer` package.
#
# flashinfer/__init__.py ends in `raise RuntimeError("FlashInfer requires either
# CUDA or ROCm/HIP backend. Detected CPU-only PyTorch installation.")`, so
# `from flashinfer import arch_caps` needs a GPU-capable torch build even though
# nothing in this file touches a GPU or a tensor. Loading the two modules
# directly keeps the suite runnable with nothing installed but pytest.
#
# That is not merely a convenience. Both modules are torch-free at module scope
# *by contract* -- hip_utils sits on the pre-HIP_VISIBLE_DEVICES path in
# tests/conftest.py, and arch_caps is imported by hip_utils -- and this loader
# makes the contract structural: if either grows a module-scope `import torch`,
# the conformance job stops being installable rather than quietly regressing.
#
# They reference each other relatively (hip_utils:10 at module scope,
# arch_caps:354 inside _live_versions), so they are registered under a synthetic
# package. Two unrelated top-level modules would leave those imports unresolvable
# and silently degrade _live_versions to its except-branch.
# --------------------------------------------------------------------------

_PKG_NAME = "_arch_caps_conformance"
_PKG_DIR = pathlib.Path(__file__).resolve().parents[2] / "flashinfer"


def _load_without_package_init(*module_names):
    """Import ``flashinfer.<name>`` modules without running the package __init__."""
    package = types.ModuleType(_PKG_NAME)
    package.__path__ = [str(_PKG_DIR)]
    sys.modules[_PKG_NAME] = package

    loaded = []
    for name in module_names:
        qualified = f"{_PKG_NAME}.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified, _PKG_DIR / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        # Register before exec so a relative import back into this synthetic
        # package resolves rather than re-executing the module.
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


arch_caps, hip_utils = _load_without_package_init("arch_caps", "hip_utils")
normalize_arch = arch_caps.normalize_arch
FLASHINFER_SUPPORTED_ROCM_ARCHS = hip_utils.FLASHINFER_SUPPORTED_ROCM_ARCHS


class TestNormalizeArch:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            # The forms torch's gcnArchName actually produces.
            ("gfx942:sramecc+:xnack-", "gfx942"),
            ("gfx950:sramecc+:xnack-", "gfx950"),
            # Already-normalized input must round-trip, so the helper is safe to
            # apply twice.
            ("gfx942", "gfx942"),
            ("gfx950", "gfx950"),
            # A single qualifier, and one with no value.
            ("gfx942:xnack-", "gfx942"),
            ("gfx90a:sramecc+", "gfx90a"),
        ],
    )
    def test_strips_qualifiers(self, raw, expected):
        assert normalize_arch(raw) == expected

    @pytest.mark.parametrize("arch", ["gfx90a", "gfx1201", "gfx1100"])
    def test_preserves_letter_suffixes_and_four_digit_archs(self, arch):
        """Regression: the previous ``re.match(r"(gfx\\d+)")`` form truncated
        ``gfx90a`` to ``gfx90``, naming an architecture that does not exist."""
        assert normalize_arch(arch) == arch

    def test_strips_surrounding_whitespace(self):
        assert normalize_arch("  gfx942  ") == "gfx942"

    @pytest.mark.parametrize("raw", ["", "   ", ":"])
    def test_degenerate_input_does_not_raise(self, raw):
        """Callers gate on the result rather than on an exception, so empty
        input must come back empty instead of blowing up."""
        assert normalize_arch(raw) == ""


def test_suite_loads_without_torch():
    """Neither arch_caps nor hip_utils may import torch at module scope.

    hip_utils sits on a path that must not touch torch: tests/conftest.py calls
    it to choose a GPU *before* pinning HIP_VISIBLE_DEVICES, which is why it
    probes with rocminfo rather than torch.cuda. arch_caps is imported by
    hip_utils at module scope, so it inherits the same constraint.

    Asserting it against *this file's* loader rather than against arch_caps
    alone covers both modules at once and pins the exact precondition the
    hardware-less CI job depends on: if this fails, that job cannot run.

    Executed in a subprocess so the verdict does not depend on what the calling
    pytest session happened to import first -- under the full ROCm suite, torch
    is long since loaded by the time this runs.
    """
    code = f"""
import importlib.util, sys
assert "torch" not in sys.modules, "torch preloaded; test would be meaningless"
spec = importlib.util.spec_from_file_location("_suite", r"{pathlib.Path(__file__).resolve()}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.normalize_arch("gfx950:sramecc+:xnack-") == "gfx950"
assert mod.FLASHINFER_SUPPORTED_ROCM_ARCHS, "hip_utils constant did not load"
assert mod.arch_caps.CAPABILITIES, "capability table did not load"
assert "torch" not in sys.modules, "arch_caps or hip_utils imported torch"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# Capability table
#
# Everything below is GPU-free by construction: the arch is monkeypatched.
# That is the point -- it lets the architecture we cannot physically reach be
# tested on whichever one we happen to have. Historically that meant simulating
# gfx950 on a CDNA3 box; today it is the reverse.
# --------------------------------------------------------------------------

# Derived, not hard-coded: adding an arch to the allowlist must automatically
# start exercising it in the routing tests below, not merely in the
# declaration check. Reading the real constant is what makes that automatic,
# and it is why the loader above bothers with hip_utils at all.
SUPPORTED_ARCHS = tuple(FLASHINFER_SUPPORTED_ROCM_ARCHS)


@pytest.fixture
def as_arch(monkeypatch):
    """Pretend the running device is a given architecture."""

    def _set(arch):
        monkeypatch.setattr(arch_caps, "_device_arch", lambda _device=None: arch)

    return _set


@pytest.fixture
def as_toolchain(monkeypatch):
    """Pretend a given (rocm, aiter) pair is installed."""

    def _set(rocm, aiter="0.1.10"):
        monkeypatch.setattr(arch_caps, "_live_versions", lambda: (rocm, aiter))

    return _set


class TestTableWellFormed:
    def test_keys_are_unique(self):
        keys = [(c.op, c.backend) for c in arch_caps.CAPABILITIES]
        assert len(keys) == len(set(keys)), "duplicate (op, backend) row"

    def test_duplicate_rows_are_refused_at_index_time(self):
        """A dict comprehension would let the later of two contradictory rows
        win silently. Raising means a duplicate cannot reach a routing decision
        even if it somehow reaches an interpreter without this suite."""
        rows = (
            arch_caps.Capability("rmsnorm", "aiter", {"gfx942": arch_caps._OK_942}),
            arch_caps.Capability("rmsnorm", "aiter", {"gfx942": arch_caps._OK_950}),
        )
        with pytest.raises(ValueError, match="duplicate capability row"):
            arch_caps._index(rows)

    def test_all_lists_every_public_name(self):
        """A public name absent from ``__all__`` is invisible to ``import *``
        and to doc tooling. ``Capability`` was, even though ``CAPABILITIES`` is
        exported and every element of it is one."""
        defined_here = {
            name
            for name, obj in vars(arch_caps).items()
            if not name.startswith("_")
            and not isinstance(obj, types.ModuleType)
            # Classes and functions carry __module__; module-level constants
            # do not, so admit those by their naming convention.
            and (
                getattr(obj, "__module__", None) == arch_caps.__name__ or name.isupper()
            )
        }
        assert defined_here == set(arch_caps.__all__)

    def test_backends_are_known(self):
        assert {c.backend for c in arch_caps.CAPABILITIES} == {"aiter", "hip"}

    def test_every_supported_arch_declared_in_every_row(self):
        """The guard rail: adding an arch to FLASHINFER_SUPPORTED_ROCM_ARCHS
        must fail here until each op declares it, rather than silently
        inheriting support."""
        for cap in arch_caps.CAPABILITIES:
            missing = set(FLASHINFER_SUPPORTED_ROCM_ARCHS) - set(cap.archs)
            assert not missing, f"{cap.op}/{cap.backend} does not declare {missing}"

    def test_no_arch_keys_outside_the_supported_list(self):
        for cap in arch_caps.CAPABILITIES:
            extra = set(cap.archs) - set(FLASHINFER_SUPPORTED_ROCM_ARCHS)
            assert not extra, f"{cap.op}/{cap.backend} declares unknown {extra}"

    def test_table_is_immutable(self):
        """``frozen=True`` only stops fields being rebound -- a plain dict in
        ``archs`` would still let any importer edit the global table in place and
        silently change what every later caller is allowed to route."""
        cap = arch_caps.CAPABILITIES[0]
        arch = next(iter(cap.archs))
        with pytest.raises(TypeError):
            cap.archs[arch] = None
        with pytest.raises(TypeError):
            del cap.archs[arch]

    def test_immutability_does_not_depend_on_the_construction_helper(self):
        """The coercion lives on ``Capability`` rather than on ``_archs`` so a
        row built any other way is protected too."""
        cap = arch_caps.Capability("op", "hip", {"gfx942": arch_caps._OK_942})
        with pytest.raises(TypeError):
            cap.archs["gfx950"] = arch_caps._OK_950

    def test_every_op_the_library_asks_for_is_declared(self):
        """The table is only useful if its row names match what callers pass.

        ``require_capability`` raises for an unknown op, so a mismatch turns
        into a hard failure at the call site -- which is how the first version
        of this table was caught declaring "activation" while
        ``activation.py`` passes "silu_and_mul". Scan the source so that drift
        fails here, cheaply and without a GPU, instead of in whichever op
        happens to be exercised first.
        """
        import re

        pkg = pathlib.Path(arch_caps.__file__).parent
        # Both the aiter_utils wrappers and the capability API they delegate to:
        # mla_rocm and page.py call require_capability / capability_available
        # directly, and those op strings drift just as easily. All five take the
        # op second, so one alternation covers them.
        pattern = re.compile(
            r"(?:require_aiter|is_aiter_available|require_capability"
            r'|capability_available|capability_reason)\([^,()]*(?:\([^()]*\))?[^,()]*,\s*"([a-z_]+)"'
        )
        used = set()
        for src in pkg.rglob("*.py"):  # subpackages too, not just the top level
            used |= set(pattern.findall(src.read_text()))

        # Coverage is not total, and pretending otherwise would be worse than
        # the gap: prefill_rocm and decode_rocm pass `op` as a variable, so no
        # literal-matching scan can see batch_prefill / batch_decode /
        # single_prefill. Those are reached by the runtime tests instead.
        assert used, "scan found no op names; the pattern has rotted"
        # These two are reached *only* through require_capability /
        # capability_available (mla_rocm.py, page.py), never through the
        # aiter_utils wrappers. Their presence is what proves the alternation
        # still covers the capability API directly; drop it and the scan
        # silently narrows back to the wrappers while still passing.
        assert {"mla", "append_paged_kv_cache"} <= used, (
            "scan no longer sees ops called through the capability API: "
            f"{sorted({'mla', 'append_paged_kv_cache'} - used)}"
        )
        declared = {c.op for c in arch_caps.CAPABILITIES if c.backend == "aiter"}
        assert used <= declared, (
            f"ops passed by the library but absent from the table: "
            f"{sorted(used - declared)}"
        )

    def test_known_bad_rows_explain_themselves(self):
        """A gate with no detail is unactionable for whoever hits it."""
        for cap in arch_caps.CAPABILITIES:
            for arch, entry in cap.archs.items():
                for bad in entry.known_bad:
                    assert bad.detail, f"{cap.op}/{cap.backend}/{arch}: empty detail"


class TestVersionWindow:
    @pytest.mark.parametrize(
        "rocm,expected",
        [
            ("7.1", False),
            ("7.1.1", False),
            ("7.2", True),
            ("7.2.0", True),
            ("7.2.4", True),  # measured: bit-identical failure to 7.2.0
            ("7.3", False),
            ("7.14", False),  # (7,14) > (7,3): a later release, not 7.1.4
        ],
    )
    def test_rocm_window_is_half_open(self, rocm, expected):
        bad = arch_caps.KnownBad(rocm_min="7.2", rocm_max="7.3")
        assert bad.matches(rocm, None) is expected

    @pytest.mark.parametrize(
        "low,high,reported,expected",
        [
            # "7.2" and "7.2.0" name the same release, so the window must not
            # care which form the machine reports. Raw tuple comparison makes
            # (7, 2) < (7, 2, 0), which would drop the gate for the first row.
            ("7.2.0", "7.3.0", "7.2", True),
            ("7.2", "7.3", "7.2.0", True),
            ("7.2.0", "7.3.0", "7.2.0", True),
            # Padding must not blur the exclusive upper bound.
            ("7.2.0", "7.3", "7.3.0", False),
            ("7.2.0", "7.3.0", "7.3", False),
            # ...nor the inclusive lower one.
            ("7.2.0", "7.3.0", "7.1", False),
        ],
    )
    def test_absent_components_are_zero_not_lower(self, low, high, reported, expected):
        """`get_system_rocm_version_from_hipconfig` matches
        ``\\d+\\.\\d+(?:\\.\\d+)?`` -- the patch component is optional, and on
        TheRock builds that is the only detection method consulted, so a bare
        "7.2" is a state we can actually be handed."""
        bad = arch_caps.KnownBad(rocm_min=low, rocm_max=high)
        assert bad.matches(reported, None) is expected

    def test_unknown_version_does_not_match(self):
        """Refusing to route because a version could not be read would break
        machines that are probably fine."""
        bad = arch_caps.KnownBad(rocm_min="7.2", rocm_max="7.3")
        assert bad.matches(None, None) is False


class TestGating:
    @pytest.mark.parametrize("arch", SUPPORTED_ARCHS)
    @pytest.mark.parametrize("backend", ["aiter", "hip"])
    def test_declared_rows_are_routable_on_a_clean_toolchain(
        self, as_arch, as_toolchain, arch, backend
    ):
        as_arch(arch)
        as_toolchain("7.1")  # outside every known_bad window
        for cap in arch_caps.CAPABILITIES:
            if cap.backend != backend:
                continue
            assert arch_caps.capability_available(None, cap.op, cap.backend), (
                f"{cap.op}/{cap.backend} unexpectedly gated on {arch}"
            )

    def test_undeclared_arch_is_refused(self, as_arch):
        """An arch nobody declared grants nothing, even for a real op."""
        as_arch("gfx90a")
        assert not arch_caps.capability_available(None, "rmsnorm", "aiter")
        with pytest.raises(arch_caps.ArchCapabilityError, match="gfx90a"):
            arch_caps.require_capability(None, "rmsnorm", "aiter")

    def test_unknown_op_is_refused(self, as_arch):
        as_arch("gfx950")
        with pytest.raises(arch_caps.ArchCapabilityError, match="not a declared"):
            arch_caps.require_capability(None, "no_such_op", "aiter")

    def test_undeclared_arch_message_names_the_ones_that_work(self, as_arch):
        """A CPU tensor is the common way to land here, and "not declared for
        unknown" would be accurate but useless."""
        as_arch("unknown")
        reason = arch_caps.capability_reason(None, "silu_and_mul", "aiter")
        assert "gfx942" in reason and "gfx950" in reason

    @pytest.mark.parametrize(
        "op,suggested",
        [
            # prefill and decode reject 'native' outright -- they take 'fa2'.
            ("batch_prefill", "fa2"),
            ("single_prefill", "fa2"),
            ("batch_decode", "fa2"),
            ("rope", "native"),
            ("rmsnorm", "native"),
            ("silu_and_mul", "native"),
            ("append_paged_kv_cache", "native"),
        ],
    )
    def test_suggested_fallback_is_one_the_op_accepts(self, as_arch, op, suggested):
        """The advice has to be followable. A blanket "use backend='native'"
        would hand prefill and decode users a string their own validation
        rejects with "Unknown backend", trading one error for a worse one."""
        as_arch("unknown")
        reason = arch_caps.capability_reason(None, op, "aiter")
        assert f"backend={suggested!r}" in reason

    def test_no_fallback_is_suggested_when_none_exists(self, as_arch):
        """mla accepts only 'auto'/'aiter' (mla_rocm.py:112), so naming any
        alternative would be a dead end. Say nothing rather than something
        wrong."""
        as_arch("unknown")
        reason = arch_caps.capability_reason(None, "mla", "aiter")
        assert "gfx942" in reason
        assert "backend=" not in reason

    def test_declared_fallbacks_are_real_backend_strings(self):
        """Guards against a typo in the table turning into advice that cannot
        work. 'auto' is excluded deliberately: suggesting it as the escape from
        a gate that 'auto' itself already applied would be circular."""
        for cap in arch_caps.CAPABILITIES:
            if cap.fallback:
                assert cap.fallback in {"native", "fa2"}, (
                    f"{cap.op}/{cap.backend} suggests unknown backend {cap.fallback!r}"
                )


class TestRocm72CausalPrefill:
    """The one measured defect: gfx950 + ROCm 7.2.x miscompiles AITER causal
    batch prefill. gfx942 is fine on the same toolchain."""

    def test_gated_on_gfx950_under_rocm_72(self, as_arch, as_toolchain):
        as_arch("gfx950")
        as_toolchain("7.2.0")
        assert not arch_caps.capability_available(None, "batch_prefill", "aiter")
        with pytest.raises(arch_caps.ArchCapabilityError, match="known-broken"):
            arch_caps.require_capability(None, "batch_prefill", "aiter")

    def test_still_gated_on_the_latest_affected_patch(self, as_arch, as_toolchain):
        as_arch("gfx950")
        as_toolchain("7.2.4")
        assert not arch_caps.capability_available(None, "batch_prefill", "aiter")

    def test_open_on_gfx950_under_rocm_71(self, as_arch, as_toolchain):
        """Measured clean: max_abs_err 0.000250, 12/12 parametrizations pass."""
        as_arch("gfx950")
        as_toolchain("7.1")
        assert arch_caps.capability_available(None, "batch_prefill", "aiter")

    def test_gfx942_unaffected_on_the_same_toolchain(self, as_arch, as_toolchain):
        """This is the whole point of keying on arch as well as version."""
        as_arch("gfx942")
        as_toolchain("7.2.0")
        assert arch_caps.capability_available(None, "batch_prefill", "aiter")

    def test_hip_fallback_stays_open_where_aiter_is_gated(self, as_arch, as_toolchain):
        """The gate is only useful if `auto` has somewhere correct to fall back
        to -- fa2 was measured correct on the same hardware (2.6e-4)."""
        as_arch("gfx950")
        as_toolchain("7.2.0")
        assert arch_caps.capability_available(None, "batch_prefill", "hip")

    def test_escape_hatch_opts_in_to_danger(self, as_arch, as_toolchain, monkeypatch):
        """Opt in to the broken path, never opt in to safety."""
        as_arch("gfx950")
        as_toolchain("7.2.0")
        monkeypatch.setenv("FLASHINFER_ARCH_ALLOW_KNOWN_BAD", "1")
        assert arch_caps.capability_available(None, "batch_prefill", "aiter")


class TestVersionProbeIsCheap:
    """Version detection shells out (``amd-smi``, ``dpkg``, ``hipconfig``, each
    with a timeout). A per-routing-decision query must not pay that repeatedly."""

    def test_rows_without_a_window_never_probe(self, as_arch, monkeypatch):
        """23 of 24 rows have no ``known_bad``, so the probe is skipped outright
        rather than merely being fast the second time."""
        calls = []

        def counted():
            calls.append(1)
            return ("7.2.0", "0.1.10")

        monkeypatch.setattr(arch_caps, "_live_versions", counted)
        as_arch("gfx950")
        assert arch_caps.capability_available(None, "rmsnorm", "aiter")
        assert calls == []

        # ...but a row that does carry a window still consults it.
        assert not arch_caps.capability_available(None, "batch_prefill", "aiter")
        assert calls == [1]

    def test_detection_runs_once_per_process(self):
        """Guards the ``lru_cache``: without it every gated query would re-run
        the subprocess probes."""
        arch_caps._live_versions.cache_clear()
        try:
            first = arch_caps._live_versions()
            second = arch_caps._live_versions()
            assert first == second
            assert arch_caps._live_versions.cache_info().misses == 1
        finally:
            arch_caps._live_versions.cache_clear()


class TestArchCapabilityError:
    def test_satisfies_every_existing_catcher(self):
        """Routing three divergent exception types through one class only works
        if the old ones still catch it.

        ValueError: test_activation_aiter_hip.py:67 asserts on it.
        RuntimeError: test_batch_prefill_bf16_custom_mask_hip.py:157 catches it.
        """
        err = arch_caps.ArchCapabilityError("boom")
        assert isinstance(err, ValueError)
        assert isinstance(err, RuntimeError)

    def test_is_not_an_import_error(self):
        """A missing aiter package is a different condition and keeps its own
        exception type."""
        assert not isinstance(arch_caps.ArchCapabilityError("x"), ImportError)
