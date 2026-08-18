# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for AITER shim build-architecture resolution.

GPU-free by construction: the device probe is monkeypatched, so both CDNA3 and
CDNA4 behaviour is exercised on whichever card happens to be present -- including
the architecture the host does not have.
"""

import pytest

from flashinfer.jit import aiter_source


@pytest.fixture(autouse=True)
def _clear_resolver_cache():
    """resolve_aiter_build_arch is lru_cached; each case needs a clean slate."""
    aiter_source.resolve_aiter_build_arch.cache_clear()
    yield
    aiter_source.resolve_aiter_build_arch.cache_clear()


@pytest.fixture
def warnings_logged(monkeypatch):
    """Capture logger.warning calls from this module.

    caplog cannot see them: FlashInferJITLogger does not propagate to the root
    logger, so intercepting the call is both simpler and independent of how
    logging happens to be configured.
    """
    recorded = []

    def _warn(msg, *args, **kwargs):
        recorded.append(msg % args if args else msg)

    monkeypatch.setattr(aiter_source.logger, "warning", _warn)
    return recorded


@pytest.fixture
def device_arch(monkeypatch):
    """Pretend the machine has a given architecture (or none)."""

    def _set(arch):
        monkeypatch.setattr(aiter_source, "_detected_device_arch", lambda: arch)

    return _set


class TestResolveBuildArch:
    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    def test_follows_the_device_when_env_unset(self, monkeypatch, device_arch, arch):
        """The bug this fixes: an unset env built for a hardcoded gfx942 on any
        machine, so a gfx950 host silently produced CDNA3 code objects."""
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        device_arch(arch)
        assert aiter_source.resolve_aiter_build_arch() == arch

    def test_defaults_when_no_device_and_no_env(self, monkeypatch, device_arch):
        """GPU-less wheel builds are real, so a last-resort default is kept."""
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        device_arch(None)
        assert aiter_source.resolve_aiter_build_arch() == "gfx942"

    @pytest.mark.parametrize("sep", [",", ";"])
    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    def test_multi_arch_env_resolves_to_the_running_device(
        self, monkeypatch, device_arch, sep, arch
    ):
        """Both separators are accepted, and the result is a single architecture.

        AITER splits GPU_ARCHS on ';' -- a comma-joined list reaches it as one
        unparseable token and it raises. Accept either form from the user.
        """
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", sep.join(["gfx942", "gfx950"]))
        device_arch(arch)
        assert aiter_source.resolve_aiter_build_arch() == arch

    def test_env_wins_but_warns_when_it_excludes_the_device(
        self, monkeypatch, device_arch, warnings_logged
    ):
        """Cross-compiling is legitimate, so the environment is honoured -- but
        the resulting shim will fault here, so say so."""
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942")
        device_arch("gfx950")
        assert aiter_source.resolve_aiter_build_arch() == "gfx942"
        assert any("gfx950" in w for w in warnings_logged), warnings_logged

    def test_no_warning_when_env_matches_the_device(
        self, monkeypatch, device_arch, warnings_logged
    ):
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942,gfx950")
        device_arch("gfx950")
        assert aiter_source.resolve_aiter_build_arch() == "gfx950"
        assert warnings_logged == []

    def test_result_is_never_a_list(self, monkeypatch, device_arch):
        """AITER's get_gfx() takes the *last* GPU_ARCHS entry rather than the
        running device, so more than one entry is never safe to pass."""
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx942,gfx950")
        device_arch(None)
        resolved = aiter_source.resolve_aiter_build_arch()
        assert "," not in resolved and ";" not in resolved
        assert resolved in ("gfx942", "gfx950")

    def test_qualifiers_in_env_are_normalized(self, monkeypatch, device_arch):
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx950:sramecc+:xnack-")
        device_arch(None)
        assert aiter_source.resolve_aiter_build_arch() == "gfx950"

    @pytest.mark.parametrize("env", [":", ":sramecc+", ":,gfx942", "gfx942,:", ",,"])
    def test_qualifier_only_tokens_never_become_the_arch(
        self, monkeypatch, device_arch, env
    ):
        """A token that is all qualifier is non-empty as written but normalizes
        to '', so filtering the raw token is not enough.

        Left unfiltered it is returned verbatim -- ':,gfx942' on a gfx950 host
        resolved to '' and tagged the cache directory '__aiter-<version>'. The
        upstream arch validation warns about such a token and drops it rather
        than raising, so it does reach this resolver.
        """
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", env)
        device_arch("gfx950")
        assert "" not in aiter_source._env_arch_list()
        assert aiter_source.resolve_aiter_build_arch() in ("gfx942", "gfx950")


class TestCacheTag:
    @pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
    def test_tag_follows_the_resolved_arch(self, monkeypatch, device_arch, arch):
        """The tag must name what was actually compiled.

        Previously it re-read the environment independently of the build, so an
        unset env tagged the directory 'gfx942' on a gfx950 machine -- and that
        wrong-architecture .so was then reused on every later run.
        """
        monkeypatch.delenv("FLASHINFER_ROCM_ARCH_LIST", raising=False)
        device_arch(arch)
        assert aiter_source._aiter_cache_tag().startswith(f"{arch}__aiter-")

    def test_tag_is_filesystem_safe(self, monkeypatch, device_arch):
        monkeypatch.setenv("FLASHINFER_ROCM_ARCH_LIST", "gfx950:sramecc+:xnack-")
        device_arch(None)
        tag = aiter_source._aiter_cache_tag()
        assert not (set(tag) & set("/:;, "))
