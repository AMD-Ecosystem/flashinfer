#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the per-architecture support matrix in README.md from the capability table.

``flashinfer/arch_caps.py`` is the single source of truth for which
``(op, backend)`` pairs are usable on which GPU architecture, and -- through
``evidence`` -- for which of those claims anyone has actually measured. The
README used to restate that by hand as the string "gfx942/gfx950", which cannot
express either an architecture-specific defect or the difference between a
measured row and a declared one. It went stale the moment the two architectures
stopped behaving identically.

Usage::

    python3 scripts/gen_arch_support_matrix.py            # rewrite README.md
    python3 scripts/gen_arch_support_matrix.py --check     # fail if out of date

``--check`` runs as a pre-commit hook, so a table edit that does not regenerate
the README is caught before review rather than after someone trusts the wrong
row.

Deliberately imports nothing from the ``flashinfer`` package: ``__init__.py``
raises on a CPU-only torch build, and this script must run in the same
hardware-less environment as pre-commit. The loader below mirrors the one in
``tests/rocm_tests/test_arch_caps_hip.py``; the duplication is intentional, as
each has to stand alone in an environment where the other may not be importable.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "flashinfer"
README = REPO_ROOT / "README.md"

BEGIN = "<!-- BEGIN GENERATED: arch support matrix -- scripts/gen_arch_support_matrix.py -->"
END = "<!-- END GENERATED: arch support matrix -->"

# Column headers. Plain text on purpose: markdownlint's MD033 rejects inline
# HTML, so no <br>/<sub> here. The SKUs each architecture covers are already
# spelled out in the "Supported GPUs" line directly above the table.
# An arch missing from this map still renders, under its bare gfx name.
ARCH_LABELS = {
    "gfx942": "gfx942 (CDNA3)",
    "gfx950": "gfx950 (CDNA4)",
}

# markdownlint's configured list style is `*`, and it rewrites `-` in place --
# which would leave the generated block failing its own --check on the next run.
BULLET = "*"

VALIDATED = "✅"
DECLARED = "◻️"
KNOWN_BAD = "⚠️"
UNSUPPORTED = "❌"


def _load_arch_caps():
    """Load flashinfer.arch_caps without executing the package __init__."""
    pkg_name = "_arch_matrix_pkg"
    package = types.ModuleType(pkg_name)
    package.__path__ = [str(PKG_DIR)]
    sys.modules[pkg_name] = package

    qualified = f"{pkg_name}.arch_caps"
    source = PKG_DIR / "arch_caps.py"
    # Two separate failures, both of which otherwise surface inside a pre-commit
    # hook as a traceback that does not say which file was being loaded:
    #   - the file is missing or unreadable. spec_from_file_location happily
    #     returns a spec for a path that does not exist, so this has to be
    #     checked here rather than inferred from the spec.
    #   - the spec or its loader is None, which is what happens if the module is
    #     ever renamed to something importlib has no source loader for.
    if not source.is_file():
        raise SystemExit(f"cannot read the capability table at {source}")
    spec = importlib.util.spec_from_file_location(qualified, source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"no Python source loader for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def _archs_in_order(arch_caps) -> list[str]:
    """Every architecture any row declares, in first-seen order.

    Derived rather than hard-coded so adding an architecture to the table adds a
    column here without anyone remembering to.
    """
    seen: list[str] = []
    for cap in arch_caps.CAPABILITIES:
        for arch in cap.archs:
            if arch not in seen:
                seen.append(arch)
    return seen


def _window(bad) -> str:
    """Render a KnownBad's bounds as a half-open interval, e.g. ``ROCm [7.2, 7.3)``."""
    parts = []
    for label, low, high in (
        ("ROCm", bad.rocm_min, bad.rocm_max),
        ("amd-aiter", bad.aiter_min, bad.aiter_max),
    ):
        if low is None and high is None:
            continue
        left = f"[{low}" if low is not None else "(-∞"
        right = f"{high})" if high is not None else "+∞)"
        parts.append(f"{label} {left}, {right}")
    return " and ".join(parts) or "all versions"


def _cell(entry, footnote_ids: dict[int, int]) -> str:
    # `type(...)` reaches the Support enum itself. Reading the member off another
    # member (entry.support.SUPPORTED) resolves to the same object today, but
    # only because enums still allow that lookup -- it has been on and off the
    # deprecation list, and it reads as though SUPPORTED were an attribute of
    # the value rather than a sibling member.
    if entry is None or entry.support is not type(entry.support).SUPPORTED:
        return UNSUPPORTED
    if entry.known_bad:
        refs = "".join(f"[^kb{footnote_ids[id(bad)]}]" for bad in entry.known_bad)
        return f"{KNOWN_BAD}{refs}"
    return VALIDATED if entry.evidence else DECLARED


def render(arch_caps) -> str:
    archs = _archs_in_order(arch_caps)

    # Number the known-bad windows in table order so footnote order matches
    # reading order.
    footnote_ids: dict[int, int] = {}
    footnotes: list[tuple[int, object, str, str]] = []
    for cap in arch_caps.CAPABILITIES:
        for arch in archs:
            entry = cap.archs.get(arch)
            for bad in getattr(entry, "known_bad", ()) or ():
                if id(bad) not in footnote_ids:
                    footnote_ids[id(bad)] = len(footnote_ids) + 1
                    footnotes.append(
                        (footnote_ids[id(bad)], bad, f"{cap.op}/{cap.backend}", arch)
                    )

    lines = [
        BEGIN,
        "",
        f"| Op | Backend | {' | '.join(ARCH_LABELS.get(a, a) for a in archs)} |",
        f"| :--- | :--- | {' | '.join(':---:' for _ in archs)} |",
    ]
    for cap in arch_caps.CAPABILITIES:
        cells = " | ".join(_cell(cap.archs.get(a), footnote_ids) for a in archs)
        lines.append(f"| `{cap.op}` | `{cap.backend}` | {cells} |")

    lines += [
        "",
        f"{BULLET} {VALIDATED} **validated** — a suite was run on that board, and the "
        "evidence is recorded below.",
        f"{BULLET} {DECLARED} **declared** — supported, but no per-op measurement has "
        "been recorded on that architecture.",
        f"{BULLET} {KNOWN_BAD} **broken on some toolchains** — usable, but not on every "
        "ROCm/AITER version; see the footnote.",
        f"{BULLET} {UNSUPPORTED} **not available** on that architecture.",
        "",
    ]

    for number, bad, where, arch in footnotes:
        detail = bad.detail.rstrip(".")
        link = f" <{bad.url}>" if bad.url else ""
        lines.append(
            f"[^kb{number}]: `{where}` on {arch}, {_window(bad)}: {detail}. "
            f"Override with `FLASHINFER_ARCH_ALLOW_KNOWN_BAD=1` if you have "
            f"validated it yourself.{link}"
        )

    # Evidence lines last: they are the provenance for every ✅ above, and are
    # what makes "validated" a checkable claim rather than a badge.
    evidence = sorted(
        {
            f"`{arch}` — {entry.evidence}"
            for cap in arch_caps.CAPABILITIES
            for arch, entry in cap.archs.items()
            if entry.evidence
        }
    )
    if evidence:
        lines += ["", "Measured on:", ""]
        lines += [f"{BULLET} {item}" for item in evidence]

    lines += ["", END]
    return "\n".join(lines)


def splice(readme_text: str, block: str) -> str:
    start = readme_text.find(BEGIN)
    stop = readme_text.find(END)
    if start == -1 or stop == -1:
        raise SystemExit(
            f"{README}: missing generated-block markers.\n"
            f"  expected {BEGIN}\n  ...and    {END}"
        )
    return readme_text[:start] + block + readme_text[stop + len(END) :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if README.md does not match the capability table",
    )
    # pre-commit passes the matched filenames; the script always regenerates the
    # whole block, so accept and ignore them rather than failing on argv.
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()

    # The rendered block contains emoji, so the encoding cannot be left to the
    # locale: a pre-commit run under LC_ALL=C would raise UnicodeDecodeError
    # here and UnicodeEncodeError on the write below.
    current = README.read_text(encoding="utf-8")
    updated = splice(current, render(_load_arch_caps()))

    if args.check:
        if current != updated:
            # Show the diff. "Out of date" alone leaves whoever hit this
            # guessing, and the answer is not always an arch_caps.py edit --
            # another formatter rewriting the generated block looks identical
            # from the outside.
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile="README.md (committed)",
                tofile="README.md (generated)",
                n=1,
            )
            print(
                "README.md arch support matrix is out of date with "
                "flashinfer/arch_caps.py.\n"
                "Regenerate it with:\n"
                "    python3 scripts/gen_arch_support_matrix.py\n",
                file=sys.stderr,
            )
            sys.stderr.writelines(diff)
            return 1
        return 0

    if current != updated:
        README.write_text(updated, encoding="utf-8")
        print(f"updated {README.relative_to(REPO_ROOT)}")
    else:
        print(f"{README.relative_to(REPO_ROOT)} already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
