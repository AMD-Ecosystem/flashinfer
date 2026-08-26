# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record which ``csrc_rocm/`` translation units a test run actually loaded.

JIT-built HIP has no line coverage; this is the honest substitute, reported
separately from the Python percentage. Enable with ``-p jit_reach_plugin``
(``tests/`` is on ``pythonpath``); inert unless ``FLASHINFER_JIT_REACH_DIR``
is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_ENV_DIR = "FLASHINFER_JIT_REACH_DIR"

_reached: set[str] = set()


def _record(sources) -> None:
    for src in sources or ():
        name = Path(str(src)).name
        if name.endswith((".cu", ".cc")):
            # Match on basename: jit/attention/modules_hip.py copies each source
            # into FLASHINFER_GEN_SRC_DIR and registers the copy, so the
            # recorded path points into the JIT cache, not into csrc_rocm/.
            _reached.add(name)


def pytest_configure(config) -> None:
    # JitSpec.load, not jit_spec_registry: `is_compiled` is true off a warm
    # on-disk cache, so the registry reports modules no test touched.
    if not os.environ.get(_ENV_DIR):
        return
    try:
        from flashinfer.jit.core import JitSpec
    except Exception:  # noqa: BLE001 -- no flashinfer, nothing to instrument
        return

    original = JitSpec.load

    def load(self, *args, **kwargs):
        _record(getattr(self, "sources", None))
        return original(self, *args, **kwargs)

    if getattr(JitSpec.load, "_fi_reach_wrapped", False):
        return
    load._fi_reach_wrapped = True  # type: ignore[attr-defined]
    JitSpec.load = load  # type: ignore[method-assign]


def pytest_sessionfinish(session, exitstatus) -> None:
    # Not pytest_terminal_summary: xdist unregisters the terminal reporter in
    # workers, so under -n auto that hook never fires and the file stays empty.
    out_dir = os.environ.get(_ENV_DIR)
    if not out_dir or not _reached:
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"jit-reach.{worker}.json").write_text(
        json.dumps(sorted(_reached)), encoding="utf-8"
    )
