# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

from flashinfer.rocm.arch_caps import normalize_arch
from flashinfer.rocm.hip_utils import (
    get_physical_card_device_indices,
    get_system_rocm_version,
    rocminfo_gpu_agents,
)

# The "-n auto" worker count lives in the repo-root conftest.py: xdist resolves
# it from the initial conftests, which do not include this one unless the
# invocation names tests/rocm.

_UNKNOWN = "unknown"

# "AMD Instinct MI350X" -> "MI350X".
_SKU_RE = re.compile(r"\bMI\d+[A-Za-z]*\b")


def _probe(fn, default=_UNKNOWN):
    """Run a probe, degrading to ``default`` rather than failing the session.

    A descriptive header must never be the reason a test run dies, so each field
    is probed independently and a broken probe costs exactly one field.
    """
    try:
        return fn() or default
    except Exception:
        return default


def _devices() -> str:
    """Device indices this session runs on.

    HIP_VISIBLE_DEVICES wins when set: tests/conftest.py assigns each xdist
    worker a card through it, so it -- not the full card list -- is the card
    this session is meant to be using.
    """
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
    if visible:
        return visible
    indices = get_physical_card_device_indices()
    # "none", not the "unknown" _probe would otherwise supply for an empty
    # string: no supported GPU is a determinate answer, and a run on a host
    # without one should not be mistaken for a run whose probe broke.
    return ",".join(str(i) for i in indices) if indices else "none"


def _arch_and_sku() -> tuple:
    """Return ``(arch, sku)``: architecture from torch, SKU from rocminfo.

    torch reports the architecture exactly via ``gcnArchName``, but its device
    ``name`` is not a marketing name -- on an MI350X it is the generic
    "AMD Radeon Graphics" -- so the SKU has to come from rocminfo's
    "Marketing Name". The two are matched on architecture so that a mixed-arch
    host can never attribute one card's SKU to another card's architecture.

    Within one architecture the agents must also agree on the board. rocminfo
    ignores HIP_VISIBLE_DEVICES -- it reports every physical agent whatever this
    session was pinned to -- so on a host mixing boards that share an
    architecture (MI300X and MI325X are both gfx942) there is nothing to tell us
    which one we got. Naming either would be a guess wearing the clothes of an
    answer, so disagreement reports ``unknown``.
    """
    arch = _UNKNOWN
    try:
        import torch

        if torch.cuda.device_count():
            # normalize_arch returns "" for degenerate input rather than
            # raising. Fold that back into _UNKNOWN, or the rocminfo fallback
            # below is skipped and the header prints a blank arch.
            arch = normalize_arch(torch.cuda.get_device_properties(0).gcnArchName) or (
                _UNKNOWN
            )
    except Exception:
        pass

    agents = rocminfo_gpu_agents()
    if arch == _UNKNOWN:
        arch = next((a for a, _ in agents if a), _UNKNOWN)

    # Only report a name we can actually resolve to a SKU, and only when it is
    # the sole candidate. Some environments -- this repo's own ROCm container
    # among them -- return the generic "AMD Radeon Graphics" for an Instinct
    # part, which leaves the set empty; a same-arch mix leaves it ambiguous.
    # Either way, echoing something would read like an answer rather than a
    # failure to identify the board.
    matches = (_SKU_RE.search(m) for a, m in agents if a == arch)
    skus = {m.group(0) for m in matches if m}
    return arch, (skus.pop() if len(skus) == 1 else _UNKNOWN)


def _header_line() -> str:
    """Build the one-line hardware/toolchain description.

    Pasted test output is otherwise not self-describing: a result or a timing
    only means something alongside the architecture, the SKU, and the
    ROCm/torch/AITER versions that produced it.

    The SKU is named separately from the architecture on purpose. MI300X and
    MI325X are both gfx942, and MI350X and MI355X are both gfx950 -- correctness
    may be inherited across a shared architecture, performance may not.
    """
    arch, sku = _probe(_arch_and_sku, (_UNKNOWN, _UNKNOWN))

    def _rocm():
        # get_system_rocm_version narrates each detection method it falls
        # through, which would otherwise interleave with the header.
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            return get_system_rocm_version()

    def _torch():
        import torch

        return str(torch.__version__)

    def _aiter():
        import importlib.metadata as _md

        return _md.version("amd-aiter")

    return (
        f"rocm: arch={arch} sku={sku} devices={_probe(_devices)} "
        f"rocm={_probe(_rocm)} torch={_probe(_torch)} aiter={_probe(_aiter)}"
    )


def pytest_report_header(config):
    """Emit the description in pytest's session header."""
    return _header_line()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Re-emit the description at the end when the session header is suppressed.

    This repo runs with ``-q`` by default (``addopts`` in pyproject.toml), and
    quiet mode drops the session header entirely -- so pytest_report_header
    alone would never be seen in a normal run. The terminal summary survives
    ``-q``, which is what makes the run self-describing in practice.
    """
    if config.option.verbose < 0:  # quiet: the header above was suppressed
        terminalreporter.write_line(_header_line())
