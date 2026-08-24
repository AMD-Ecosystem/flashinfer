# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``refresh_aiter_jitspec``, the AITER link-line refresh.

GPU-free by construction: a stub spec stands in for :class:`JitSpec`, so nothing
here compiles or touches a device.

What is being protected. The AITER shim libs live outside the JIT tree and reach
a module only as an ``-L``/``-rpath`` on the link line. ``JitSpec.build()`` writes
``build.ninja`` only when it is missing, so without this refresh a module cached
under ``.../gfx950/`` keeps loading whichever AITER lib it was first built
against -- a wrong-architecture load that segfaults rather than failing cleanly.
"""

import threading
import time

import pytest
from filelock import FileLock

from flashinfer.jit.aiter_source import refresh_aiter_jitspec


class StubSpec:
    """Minimal stand-in for JitSpec: records write_ninja(), real lock on disk."""

    def __init__(self, lock_path, is_aot=False):
        self.name = "stub_aiter"
        self.lock_path = lock_path
        self.is_aot = is_aot
        self.writes = 0

    def write_ninja(self):
        self.writes += 1


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "stub_aiter.lock"


def test_refresh_rewrites_on_the_jit_path(lock_path):
    spec = StubSpec(lock_path)
    assert refresh_aiter_jitspec(spec) is spec
    assert spec.writes == 1


def test_refresh_skips_aot_prebuilt(lock_path):
    """build_and_load() loads straight from aot_path, so ninja never reads it."""
    spec = StubSpec(lock_path, is_aot=True)
    refresh_aiter_jitspec(spec)
    assert spec.writes == 0


def test_refresh_skips_when_jit_disabled(lock_path, monkeypatch):
    """build() raises before consulting the manifest, so writing it is noise."""
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    spec = StubSpec(lock_path)
    refresh_aiter_jitspec(spec)
    assert spec.writes == 0


def test_refresh_waits_for_the_build_lock(lock_path):
    """The write must not land while a concurrent builder holds the lock.

    write_if_different truncates in place, so an unlocked rewrite could empty
    build.ninja while another process's ninja is mid-read. Reachable in practice:
    pytest -n auto shares one JIT cache across processes.
    """
    hold_for = 1.0
    acquired = threading.Event()

    def holder():
        with FileLock(lock_path, thread_local=False):
            acquired.set()
            time.sleep(hold_for)

    t = threading.Thread(target=holder)
    t.start()
    try:
        assert acquired.wait(timeout=10), "holder thread never took the lock"
        spec = StubSpec(lock_path)
        start = time.monotonic()
        refresh_aiter_jitspec(spec)
        waited = time.monotonic() - start
    finally:
        t.join()

    assert spec.writes == 1
    # Generous lower bound: proves it blocked, without being timing-flaky.
    assert waited > hold_for / 2, (
        f"refresh did not wait on the lock (waited {waited:.3f}s)"
    )


def test_every_aiter_linked_generator_refreshes():
    """Each gen_*_aiter_module must route its spec through refresh_aiter_jitspec.

    The tests above prove the helper works; nothing proved the generators call
    it. page_aiter shipped without it and the whole suite stayed green, because
    the symptom only appears after an arch or AITER-version change against an
    already-cached module.

    Source inspection rather than execution: calling a generator runs
    aiter_jitspec_flags(), which builds the AITER library for real.

    Matched on the AST, not on text: a substring scan is satisfied by the
    comment *explaining* the call, so it passes with the call removed.
    """
    import ast
    import inspect
    import textwrap

    from flashinfer.jit import activation, norm, page, rope

    generators = [
        activation.gen_silu_and_mul_aiter_module,
        norm.gen_norm_aiter_module,
        page.gen_page_aiter_module,
        rope.gen_rope_aiter_module,
    ]

    def _calls_refresh(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "refresh_aiter_jitspec"
            for node in ast.walk(tree)
        )

    missing = [
        f"{fn.__module__}.{fn.__name__}" for fn in generators if not _calls_refresh(fn)
    ]
    assert not missing, (
        "AITER-linked generators that do not refresh their link line, so a "
        f"cached module keeps linking a stale AITER library: {missing}"
    )


def test_generator_list_covers_every_aiter_module():
    """The list above must not silently fall behind a newly added shim."""
    import pkgutil

    import flashinfer.jit as jit_pkg

    found = set()
    for mod in pkgutil.iter_modules(jit_pkg.__path__):
        src_path = f"{jit_pkg.__path__[0]}/{mod.name}.py"
        try:
            with open(src_path) as fh:
                src = fh.read()
        except OSError:
            continue
        for line in src.splitlines():
            if line.startswith("def gen_") and line.rstrip().endswith(
                "_aiter_module() -> JitSpec:"
            ):
                found.add(f"flashinfer.jit.{mod.name}.{line[4:].split('(')[0]}")

    covered = {
        "flashinfer.jit.activation.gen_silu_and_mul_aiter_module",
        "flashinfer.jit.norm.gen_norm_aiter_module",
        "flashinfer.jit.page.gen_page_aiter_module",
        "flashinfer.jit.rope.gen_rope_aiter_module",
    }
    assert found == covered, (
        f"AITER generators not covered by the refresh test: {sorted(found - covered)}; "
        f"listed but gone: {sorted(covered - found)}"
    )
