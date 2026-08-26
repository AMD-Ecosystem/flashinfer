#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Report what an upstream sync would cost, without performing one.

Prints how far the fork and upstream have moved from the merge base, the
conflicts a merge would raise, and upstream churn on the headers that
``generic/`` forked -- those conflict with nothing and go stale silently.

Usage::

    python3 scripts/upstream_canary.py --upstream-ref upstream/main
    python3 scripts/upstream_canary.py --fail-over 14    # ratchet for CI

Exit 0 clean, 1 if ``--fail-over`` is exceeded, 2 if the tool could not run.
Leaves the working tree and index untouched, but does write unreferenced loose
objects, so a long-lived CI checkout wants a periodic ``git gc``.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import posixpath
import subprocess
import sys
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

EXIT_OK, EXIT_RATCHET, EXIT_ERROR = 0, 1, 2

# The port's own docs, packaging metadata, and regenerated sample output. Not a
# general "docs and config" exemption: an edit to .github/workflows/ or docker/
# is a real in-place edit to an upstream file, so it stays in the count.
_EXPECTED_BASENAME_GLOBS = ("*.md", "*.toml")
_EXPECTED_EXACT = (".gitignore", ".pre-commit-config.yaml", "version.txt")
_EXPECTED_PREFIXES = ("benchmarks/samples/",)

_FORKED_DIR = "include/flashinfer/rocm"

# rocm/ mirrors upstream's layout but flattens one level: rocm/attention/ holds
# headers upstream keeps both in attention/ and directly under flashinfer/.
_UPSTREAM_HEADER_DIRS = ("include/flashinfer/attention", "include/flashinfer")
_HEADER_SUFFIXES = (".cuh", ".hpp", ".h")


class ToolError(Exception):
    """The canary could not produce a report -- distinct from a conflicting merge."""


class Churn(NamedTuple):
    added: int = 0
    deleted: int = 0

    def __str__(self) -> str:
        return f"+{self.added}/-{self.deleted}"

    @property
    def total(self) -> int:
        return self.added + self.deleted


class Conflict(NamedTuple):
    path: str
    ours: Churn
    theirs: Churn
    kind: str

    @property
    def weight(self) -> int:
        """Both sides' churn multiplied, so two-sided conflicts rank first."""
        return self.ours.total * self.theirs.total

    @property
    def expected(self) -> bool:
        base = posixpath.basename(self.path)
        return (
            any(fnmatch.fnmatch(base, g) for g in _EXPECTED_BASENAME_GLOBS)
            or self.path in _EXPECTED_EXACT
            or self.path.startswith(_EXPECTED_PREFIXES)
        )


def _run(
    repo: Optional[str], *args: str, check: bool = True
) -> subprocess.CompletedProcess:
    """Run git at the repo root, in the C locale, tolerating undecodable paths."""
    cmd = ["git"] + (["-C", repo] if repo else []) + list(args)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        env={**os.environ, "LC_ALL": "C"},
    )
    if check and proc.returncode != 0:
        raise ToolError(f"{' '.join(cmd)} failed:\n{proc.stderr.strip()}")
    return proc


def _git(repo: Optional[str], *args: str) -> str:
    return _run(repo, *args).stdout.strip()


def _ls_tree(repo: str, ref: str, path: str) -> Set[str]:
    """Paths under `path` in `ref`. ``-z`` because the default C-quotes non-ASCII."""
    args = ["ls-tree", "-r", "-z", "--name-only", ref]
    if path:
        args += ["--", path]
    return {f for f in _run(repo, *args).stdout.split("\0") if f}


def _churn_map(
    repo: str, base: str, tip: str
) -> Tuple[Dict[str, Churn], Dict[str, str]]:
    """Per-path churn for a revision range, plus a new->old map of renames.

    ``-M`` because rename detection is otherwise subject to diff.renames, and no
    pathspec because filtering by path suppresses it outright.
    """
    fields = _run(repo, "diff", "--numstat", "-z", "-M", f"{base}..{tip}").stdout.split(
        "\0"
    )
    churn: Dict[str, Churn] = {}
    renames: Dict[str, str] = {}
    i = 0
    while i < len(fields):
        record = fields[i]
        i += 1
        if not record:
            continue
        parts = record.split("\t")
        if len(parts) < 3:
            raise ToolError(
                f"malformed numstat -z record at field {i - 1}: {record!r}; "
                "expected added, deleted and path"
            )
        added = int(parts[0]) if parts[0].isdigit() else 0  # "-" marks a binary file
        deleted = int(parts[1]) if parts[1].isdigit() else 0
        if parts[2]:
            path = parts[2]
        else:
            # Rename/copy: path field empty, next two fields are old then new.
            if i + 1 >= len(fields):
                raise ToolError(
                    f"truncated numstat -z rename record at field {i}: expected an "
                    "old and a new path but the stream ended"
                )
            renames[fields[i + 1]] = fields[i]
            path = fields[i + 1]
            i += 2
        churn[path] = Churn(added, deleted)
    return churn, renames


def _merge_tree(repo: str, ours: str, theirs: str) -> List[Tuple[str, str]]:
    """(path, conflict-type) for a merge that is never made.

    The ``-z`` info section is ``<n-paths> NUL <path>... NUL <type> NUL <message>``,
    so the type is a field rather than something scraped from git's English prose.
    """
    proc = _run(repo, "merge-tree", "--write-tree", "-z", ours, theirs, check=False)
    if proc.returncode == 0:
        return []
    if proc.returncode != 1:
        raise ToolError(f"git merge-tree failed:\n{proc.stderr.strip()}")

    fields = proc.stdout.split("\0")
    idx = 1  # skip the tree OID
    while idx < len(fields) and fields[idx]:
        idx += 1
    idx += 1

    kinds: Dict[str, str] = {}
    order: List[str] = []
    while idx < len(fields):
        if not fields[idx]:
            idx += 1
            continue
        try:
            n_paths = int(fields[idx])
        except ValueError as exc:
            # Never truncate silently: an under-count reads as good news.
            raise ToolError(
                f"unparsed merge-tree -z record at field {idx}: {fields[idx]!r}; "
                f"the informational format may have changed ({_git(repo, 'version')})"
            ) from exc
        paths = fields[idx + 1 : idx + 1 + n_paths]
        type_idx = idx + 1 + n_paths
        # A slice silently truncates, so check the count rather than infer it.
        # Every record is <n-paths> <path>... <type> <message>; a short one means
        # the stream was cut, and continuing would under-report.
        if len(paths) != n_paths or type_idx + 1 >= len(fields):
            raise ToolError(
                f"truncated merge-tree -z record at field {idx}: expected {n_paths} "
                f"path(s), a conflict type and a message, but the stream ended"
            )
        kind = fields[type_idx]
        idx = type_idx + 2  # skip the human-readable message
        if not kind.startswith("CONFLICT"):
            continue  # "Auto-merging" and friends
        label = kind[len("CONFLICT (") : -1] if kind.endswith(")") else kind
        for path in paths:
            if path not in kinds:
                order.append(path)
            kinds[path] = label
    return [(p, kinds[p]) for p in order]


def _report_position(
    base_desc: str, ours: Tuple[int, int], theirs: Tuple[int, int]
) -> None:
    print("== position ==")
    print(f"merge base : {base_desc}")
    print(f"ours       : {ours[0]:>6} commits, {ours[1]:>5} files changed")
    print(f"upstream   : {theirs[0]:>6} commits, {theirs[1]:>5} files changed")
    print()


def _report_conflicts(conflicts: List[Conflict]) -> int:
    """Print the ranked conflict table; return the count of non-expected paths."""
    print("== conflicts ==")
    if not conflicts:
        print("clean merge -- nothing to do")
        print()
        return 0

    rows = sorted(conflicts, key=lambda c: (c.expected, -c.weight))
    width = max(len(c.path) for c in rows)
    for c in rows:
        tag = "" if c.kind == "contents" else f" [{c.kind}]"
        if c.expected:
            tag += " [expected]"
        print(
            f"  {c.path:<{width}}  ours {str(c.ours):>12}"
            f"  upstream {str(c.theirs):>12}{tag}"
        )

    code = sum(1 for c in rows if not c.expected)
    print()
    print(f"  {len(rows)} conflicted, {code} of them code")
    print()
    return code


def _report_drift(repo: str, ours: str, theirs: str, churn: Dict[str, Churn]) -> None:
    """Upstream churn on the headers that generic/ forked.

    Read from `ours`, not the working tree, so --ours means what it says and a
    checkout that predates or relocates generic/ still reports correctly.
    """
    print("== forked-header drift ==")
    forked = sorted(
        p for p in _ls_tree(repo, ours, _FORKED_DIR) if p.endswith(_HEADER_SUFFIXES)
    )
    if not forked:
        print(f"no forked headers under {_FORKED_DIR} in {ours[:12]}")
        print()
        return

    upstream_files = _ls_tree(repo, theirs, "include/flashinfer")
    rows: List[Tuple[str, Churn]] = []
    orphans: List[str] = []
    for path in forked:
        name = posixpath.basename(path)
        candidates = [
            f"{d}/{name}"
            for d in _UPSTREAM_HEADER_DIRS
            if f"{d}/{name}" in upstream_files
        ]
        if not candidates:
            orphans.append(name)
            continue
        if len(candidates) > 1:
            print(
                f"  NOTE: {name} matches {len(candidates)} upstream paths; using {candidates[0]}"
            )
        c = churn.get(candidates[0])
        if c and c.total:
            rows.append((candidates[0], c))

    tracked = len(forked) - len(orphans)
    if rows:
        rows.sort(key=lambda r: -r[1].total)
        width = max(len(r[0]) for r in rows)
        for path, c in rows:
            print(f"  {path:<{width}}  upstream {c}")
        print()
        print(
            f"  {len(rows)} of {tracked} forked headers have upstream changes to review"
        )
    else:
        print(f"  no upstream changes to any of the {tracked} forked headers")
    if orphans:
        print(
            f"  ({len(orphans)} ROCm-only, no upstream counterpart: {', '.join(orphans)})"
        )
    print()


def _resolve(repo: str, ref: str, label: str) -> str:
    probe = _run(
        repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False
    )
    if probe.returncode:
        hint = (
            "\nIf it names a remote you have not configured, add it first:\n"
            "  git remote add upstream https://github.com/flashinfer-ai/flashinfer.git\n"
            "  git fetch upstream"
            if label == "--upstream-ref"
            else ""
        )
        raise ToolError(f"unknown ref '{ref}' passed to {label}.{hint}")
    return _git(repo, "rev-parse", ref)


def run(args: argparse.Namespace) -> int:
    repo = _git(None, "rev-parse", "--show-toplevel")
    ours = _resolve(repo, args.ours, "--ours")
    theirs = _resolve(repo, args.upstream_ref, "--upstream-ref")
    base = _git(repo, "merge-base", ours, theirs)

    if len(_git(repo, "merge-base", "--all", ours, theirs).split()) > 1:
        print("NOTE: multiple merge bases -- churn is measured against one of them,")
        print("      while the merge itself resolves against a virtual base.\n")

    our_churn, _ = _churn_map(repo, base, ours)
    their_churn, their_renames = _churn_map(repo, base, theirs)

    _report_position(
        _git(repo, "log", "-1", "--format=%h %ad %s", "--date=short", base),
        (int(_git(repo, "rev-list", "--count", f"{base}..{ours}")), len(our_churn)),
        (int(_git(repo, "rev-list", "--count", f"{base}..{theirs}")), len(their_churn)),
    )

    merged = _merge_tree(repo, ours, theirs)
    at_base = _ls_tree(repo, base, "") if merged else set()
    conflicts = []
    for path, kind in merged:
        # Follow an upstream rename back to the name our side still edits, or the
        # most expensive case reports +0/-0 and sorts last.
        old = their_renames.get(path)
        ours_churn = our_churn.get(path) or (
            our_churn.get(old, Churn()) if old else Churn()
        )
        if kind == "contents" and path not in at_base:
            kind = "rename" if old else "add/add"
        conflicts.append(
            Conflict(path, ours_churn, their_churn.get(path, Churn()), kind)
        )

    code = _report_conflicts(conflicts)
    _report_drift(repo, ours, theirs, their_churn)

    if args.fail_over is not None and code > args.fail_over:
        print(f"FAIL: {code} code conflicts exceeds --fail-over {args.fail_over}")
        return EXIT_RATCHET
    return EXIT_OK


def main() -> int:
    summary = (__doc__ or "Report what an upstream sync would cost.").splitlines()[0]
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "--upstream-ref",
        default="upstream/main",
        help="upstream ref to measure against (default: upstream/main)",
    )
    parser.add_argument(
        "--ours", default="HEAD", help="our ref to measure (default: HEAD)"
    )
    parser.add_argument(
        "--fail-over",
        type=int,
        default=None,
        metavar="N",
        help="exit 1 if more than N conflicts are outside the port's own docs "
        "and packaging metadata",
    )
    try:
        return run(parser.parse_args())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
