# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Probe and lookup helpers in jit/aiter_source.py.

Everything here runs during JIT setup, before any HIP runtime is required, so
each probe has a not-available arm that a working box never reaches: no visible
device, no amd-aiter metadata, a header tree that is not where it should be.
Getting those wrong turns a clear message into a traceback out of the JIT.

No build is performed and no library is loaded.
"""

import pytest

from flashinfer.jit import aiter_source


class TestDetectedDeviceArch:
    """torch is imported lazily here: a GPU-less wheel build must not need a
    working HIP runtime, so every failure is None rather than an exception."""

    def test_no_visible_device_reports_none(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert aiter_source._detected_device_arch() is None

    def test_a_failing_probe_reports_none(self, monkeypatch):
        import torch

        monkeypatch.setattr(
            torch.cuda,
            "device_count",
            lambda: (_ for _ in ()).throw(RuntimeError("no runtime")),
        )
        assert aiter_source._detected_device_arch() is None

    def test_a_live_device_reports_a_gfx_name(self):
        import torch

        if torch.cuda.device_count() == 0:
            pytest.skip("no device")
        assert aiter_source._detected_device_arch().startswith("gfx")


class TestCacheTag:
    def test_the_tag_carries_the_arch_and_the_aiter_version(self, monkeypatch):
        monkeypatch.setattr(aiter_source, "resolve_aiter_build_arch", lambda: "gfx942")

        tag = aiter_source._aiter_cache_tag()

        assert tag.startswith("gfx942__aiter-")

    def test_an_unreadable_version_still_produces_a_tag(self, monkeypatch):
        """A cache key with no version is still better than failing the build."""
        import importlib.metadata

        monkeypatch.setattr(aiter_source, "resolve_aiter_build_arch", lambda: "gfx942")
        monkeypatch.setattr(
            importlib.metadata,
            "version",
            lambda name: (_ for _ in ()).throw(RuntimeError("no metadata")),
        )

        assert aiter_source._aiter_cache_tag() == "gfx942__aiter-unknown"

    @pytest.mark.parametrize("arch", ["", ".", "..", "a/b", ".hidden"])
    def test_an_unsafe_arch_never_becomes_a_directory_name(self, monkeypatch, arch):
        """The tag is used as a path component; a traversal or a dotfile here
        would put the cache somewhere nobody looks."""
        monkeypatch.setattr(aiter_source, "resolve_aiter_build_arch", lambda: arch)

        with pytest.raises(ValueError, match="single safe path component"):
            aiter_source._aiter_cache_tag()


class TestCsrcIncludeDir:
    @pytest.fixture(autouse=True)
    def _uncached(self):
        aiter_source._aiter_csrc_include_dir.cache_clear()
        yield
        aiter_source._aiter_csrc_include_dir.cache_clear()

    def test_a_missing_header_tree_names_the_package(self, monkeypatch):
        import aiter_meta

        monkeypatch.setattr(aiter_meta, "__path__", ["/nonexistent"])

        with pytest.raises(RuntimeError, match="aiter_meta/csrc/include"):
            aiter_source._aiter_csrc_include_dir()


class TestFindBuiltSo:
    """AITER writes its output to whichever of several directories its own JIT
    chose; a miss here reads as 'the build produced nothing'."""

    def test_a_direct_hit_wins_over_a_nested_one(self, tmp_path):
        shallow, deep = tmp_path / "a", tmp_path / "b"
        (shallow).mkdir()
        (deep / "nested").mkdir(parents=True)
        (shallow / "mod.so").write_bytes(b"x")
        (deep / "nested" / "mod.so").write_bytes(b"y")

        found = aiter_source._find_built_so("mod", deep, shallow)

        assert found == shallow / "mod.so"

    def test_a_nested_artifact_is_found_by_recursive_search(self, tmp_path):
        root = tmp_path / "out"
        (root / "deep" / "deeper").mkdir(parents=True)
        target = root / "deep" / "deeper" / "mod.so"
        target.write_bytes(b"x")

        assert aiter_source._find_built_so("mod", root) == target

    def test_a_missing_directory_is_skipped_not_fatal(self, tmp_path):
        assert aiter_source._find_built_so("mod", tmp_path / "never") is None

    def test_nothing_built_reports_none(self, tmp_path):
        (tmp_path / "other.so").write_bytes(b"x")
        assert aiter_source._find_built_so("mod", tmp_path) is None

    def test_a_directory_named_like_the_library_is_not_a_hit(self, tmp_path):
        (tmp_path / "mod.so").mkdir()
        assert aiter_source._find_built_so("mod", tmp_path) is None
