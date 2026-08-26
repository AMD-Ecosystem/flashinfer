#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Report line coverage for the code this fork added or changed.

Ownership is derived from the diff against the upstream merge base on every
run, so there is no list to go stale: a file added since the last run is picked
up automatically. Files we added are scored whole; upstream files we merely
edited are scored only on the lines our diff touched.

Usage::

    python3 scripts/amd_coverage.py --run -- -m "not slow"   # run tests, then score
    python3 scripts/amd_coverage.py                          # score an existing .coverage
    python3 scripts/amd_coverage.py --json-out cov.json --fail-under 60

Exit 0 clean, 1 if ``--fail-under`` is missed, 2 if the tool could not run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

EXIT_OK, EXIT_RATCHET, EXIT_ERROR = 0, 1, 2

_MANIFEST = "scripts/coverage_ownership.toml"

# Python we ship or run in CI. tests/ and benchmarks/ are out -- pytest never
# runs the benchmarks, so a permanent 0% would describe the harness, not the
# code. The report prints both exclusions rather than leaving them implicit.
_SURFACE = ("flashinfer", "scripts", "build_backend_rocm.py")

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

_CSRC_DIR = "flashinfer/csrc_rocm"
_JIT_REACH_ENV = "FLASHINFER_JIT_REACH_DIR"

_TIER_LABEL = {
    "A": "AMD-added, whole file",
    "B": "upstream-edited, changed lines",
    "C": "redirect-owned, whole file",
}


class ToolError(Exception):
    """The report could not be produced -- distinct from a low coverage number."""


class Owned(NamedTuple):
    """What we own in a file, before any coverage data is consulted."""

    path: str
    tier: str
    changed: Optional[Set[int]]  # None means the whole file (tiers A and C)
    reason: str = ""


class Score(NamedTuple):
    """One file after scoring. `owned` is statements only, exclusions removed."""

    path: str
    tier: str
    reason: str
    owned: Set[int]
    covered: Set[int]
    import_time: Set[int]
    excluded: int

    @property
    def exec_total(self) -> int:
        return len(self.owned - self.import_time)

    @property
    def exec_covered(self) -> int:
        return len((self.covered & self.owned) - self.import_time)


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


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # requires-python allows 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise ToolError(
                "reading TOML on Python 3.10 needs tomli: pip install tomli"
            ) from exc
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _resolve_base(repo: str, upstream_ref: str) -> str:
    probe = _run(
        repo,
        "rev-parse",
        "--verify",
        "--quiet",
        f"{upstream_ref}^{{commit}}",
        check=False,
    )
    if probe.returncode:
        raise ToolError(
            f"unknown ref '{upstream_ref}'. If it names a remote you have not "
            "configured, add it first:\n"
            "  git remote add upstream https://github.com/flashinfer-ai/flashinfer.git\n"
            "  git fetch upstream main"
        )
    return _git(repo, "merge-base", "HEAD", upstream_ref)


def _diff_status(repo: str, base: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Per-path status over the measured surface, plus a new->old rename map.

    Diffs the working tree, not HEAD, so uncommitted edits are attributed. ``-M``
    reports a rename as ``R``; without a row for it a renamed file would fall
    through to unowned and silently stop being measured.
    """
    fields = _run(
        repo, "diff", "--name-status", "-z", "-M", base, "--", *_SURFACE
    ).stdout.split("\0")
    status: Dict[str, str] = {}
    renames: Dict[str, str] = {}
    i = 0
    while i < len(fields):
        code = fields[i]
        i += 1
        if not code:
            continue
        kind = code[0]
        if kind in ("R", "C"):
            if i + 1 >= len(fields):
                raise ToolError(f"truncated name-status rename record at field {i}")
            old, new = fields[i], fields[i + 1]
            i += 2
            if new.endswith(".py"):
                # Only a .py that was already .py carries upstream lines we do not
                # own. Anything else becomes Python here for the first time, so
                # every line is ours and it is scored whole.
                status[new] = "R" if old.endswith(".py") else "A"
                if old.endswith(".py"):
                    renames[new] = old
            continue
        path = fields[i]
        i += 1
        if path.endswith(".py"):
            status[path] = kind
    return status, renames


def _changed_lines(repo: str, base: str, path: str, old: Optional[str]) -> Set[int]:
    """Post-image line numbers our diff added or changed in `path`.

    ``@@ -a,b +c,d @@`` -- c..c+d-1 are lines in the file as it is now, which is
    what coverage reports against. A hunk with d == 0 is a pure deletion.
    """
    args = ["diff", "-U0", "-M", base, "--", path]
    if old:
        args = ["diff", "-U0", "-M", base, "--", old, path]
    lines: Set[int] = set()
    for row in _run(repo, *args).stdout.splitlines():
        match = _HUNK.match(row)
        if not match:
            continue
        start = int(match.group(1))
        count = 1 if match.group(2) is None else int(match.group(2))
        lines.update(range(start, start + count))
    return lines


def _surface_python(repo: str) -> Tuple[Set[str], Set[str]]:
    """Python on the measured surface: (all files, the untracked subset).

    Untracked-but-not-ignored files count: leaving them out would contradict
    diffing the working tree, and drop exactly the code someone is writing.
    """
    tracked = {
        f
        for f in _run(repo, "ls-files", "-z", "--", *_SURFACE).stdout.split("\0")
        if f.endswith(".py")
    }
    untracked = {
        f
        for f in _run(
            repo, "ls-files", "-z", "--others", "--exclude-standard", "--", *_SURFACE
        ).stdout.split("\0")
        if f.endswith(".py")
    }
    return tracked | untracked, untracked


def classify(
    repo: str, base: str, manifest: dict
) -> Tuple[Dict[str, Owned], Dict[str, str], List[str], Dict[str, str]]:
    """Assign every tracked file on the surface to a tier, or to unowned.

    Returns the owned map, path->reason for deliberate exclusions, the sorted
    unowned paths, and path->reason for the ones ruled unowned by the manifest.
    The last is separate because "we decided this is upstream's" and "no diff
    ever touched it" are different claims, and only the first was reviewed.
    """
    tracked, untracked = _surface_python(repo)
    root = Path(repo)

    def _entries(key: str, must_be_on_surface: bool) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for item in manifest.get(key, []):
            path, reason = item["path"], item.get("reason", "")
            if not (root / path).exists():
                raise ToolError(
                    f"{_MANIFEST}: [{key}] names '{path}', which does not exist. "
                    "Remove the entry or fix the path."
                )
            # A tier ruling only takes effect inside classify()'s loop over the
            # surface, so one naming a file outside it is a silent no-op.
            # `excluded` is different: it documents a file coverage measures but
            # git does not track -- a generated, gitignored module is the case
            # it exists for -- so requiring it on the surface would reject the
            # very entries it is meant to carry.
            if must_be_on_surface and path not in tracked:
                raise ToolError(
                    f"{_MANIFEST}: [{key}] names '{path}', which exists but is "
                    f"not on the measured surface, so the ruling would do "
                    f"nothing. Surface: {', '.join(_SURFACE)}."
                )
            if not reason.strip():
                raise ToolError(
                    f"{_MANIFEST}: [{key}] entry '{path}' has no reason. Every "
                    "ruling is hand-made, so it has to say why."
                )
            out[path] = reason
        return out

    redirect = _entries("redirect_owned", must_be_on_surface=True)
    ruled_unowned = _entries("unowned", must_be_on_surface=True)
    excluded = _entries("excluded", must_be_on_surface=False)

    overlap = set(redirect) & set(ruled_unowned)
    if overlap:
        raise ToolError(
            f"{_MANIFEST}: {sorted(overlap)} listed as both redirect_owned and unowned"
        )

    status, renames = _diff_status(repo, base)
    owned: Dict[str, Owned] = {}
    unowned: List[str] = []

    for path in sorted(tracked):
        if path in excluded:
            continue
        kind = status.get(path)
        if path in redirect:
            tier, reason = "C", redirect[path]
        elif path in ruled_unowned:
            unowned.append(path)
            continue
        elif path in untracked:
            # Not in any diff, because git does not track it yet. Ours all the same.
            tier, reason = "A", "untracked"
        elif kind == "A":
            tier, reason = "A", ""
        elif kind in ("M", "R"):
            tier, reason = "B", "renamed" if kind == "R" else ""
        else:
            unowned.append(path)
            continue

        changed = (
            _changed_lines(repo, base, path, renames.get(path)) if tier == "B" else None
        )
        owned[path] = Owned(path, tier, changed, reason)

    return owned, excluded, sorted(unowned), ruled_unowned


def _canonical(measured: str, repo: Path, owned: Sequence[str]) -> Optional[str]:
    """Map an absolute path from a coverage data file back to a repo-relative one."""
    p = Path(measured)
    try:
        return p.resolve().relative_to(repo).as_posix()
    except (ValueError, OSError):
        pass
    posix = p.as_posix()
    for candidate in owned:
        if posix.endswith("/" + candidate):
            return candidate
    return None


def _executed(
    data_file: Path, repo: Path, owned: Sequence[str]
) -> Tuple[Dict[str, Set[int]], Set[str]]:
    """Executed lines per owned path, plus the source roots outside this tree.

    A suffix match is what makes a site-packages or bind-mounted run scorable,
    but it also silently accepts a *different* checkout -- the main one, whose
    editable install shadows the worktree when PYTHONPATH is unset. Statements
    would come from here and executed lines from there, so the roots are
    returned for the report to name rather than being quietly folded in.
    """
    import coverage

    data = coverage.CoverageData(basename=str(data_file))
    try:
        data.read()
    except Exception as exc:  # noqa: BLE001 -- coverage raises several unrelated types
        raise ToolError(f"cannot read coverage data {data_file}: {exc}") from exc

    out: Dict[str, Set[int]] = {}
    foreign: Set[str] = set()
    for measured in data.measured_files():
        rel = _canonical(measured, repo, owned)
        if rel is None:
            continue
        posix = Path(measured).as_posix()
        if posix.endswith("/" + rel) and not posix.startswith(f"{repo.as_posix()}/"):
            foreign.add(posix[: -len(rel) - 1])
        out.setdefault(rel, set()).update(data.lines(measured) or ())
    return out, foreign


def score(
    repo: Path,
    owned: Dict[str, Owned],
    data_file: Path,
    baseline: Optional[Path],
) -> Tuple[List[Score], Set[str]]:
    import coverage

    cov = coverage.Coverage(
        data_file=str(data_file), config_file=str(repo / "pyproject.toml")
    )
    cov.load()
    measured = list(cov.get_data().measured_files())

    # Executed lines come from the data keyed by our own path matching, not from
    # analysis2's lookup: a run bind-mounted at /wt records absolute paths that
    # never resolve against this checkout, and analysis2 would then report every
    # line missing. Statement lists still come from analysis2, which parses the
    # file and needs no data.
    executed, foreign = _executed(data_file, repo, list(owned))
    # The baseline's foreign roots matter as much as the run's: import-time
    # lines are *subtracted*, so a baseline captured against a different
    # checkout quietly shrinks the numerator with no warning at all.
    import_lines: Dict[str, Set[int]] = {}
    if baseline:
        import_lines, baseline_foreign = _executed(baseline, repo, list(owned))
        foreign |= baseline_foreign

    scores: List[Score] = []
    for path, entry in sorted(owned.items()):
        try:
            _, statements, excluded, _, _ = cov.analysis2(str(repo / path))
        except Exception as exc:  # noqa: BLE001 -- NoSource and friends
            raise ToolError(f"cannot analyse {path}: {exc}") from exc
        # `statements` already has the `if IS_CUDA:` arms removed, via
        # exclude_also in [tool.coverage.report]. Intersecting with the diff
        # hunks also drops the blank lines and comments a hunk spans.
        stmts = set(statements)
        selected = stmts if entry.changed is None else stmts & entry.changed
        # Scope the exclusion count to the lines we own, not the whole file: for
        # tier B, a guard outside our diff was never in our denominator, so
        # counting it claims the exclusion removed more than it did. Intersect
        # with `changed` rather than `selected` -- excluded lines are not
        # statements, so they were already taken out of `stmts`.
        dropped = excluded if entry.changed is None else set(excluded) & entry.changed
        scores.append(
            Score(
                path=path,
                tier=entry.tier,
                reason=entry.reason,
                owned=selected,
                covered=executed.get(path, set()) & selected,
                import_time=import_lines.get(path, set()) & selected,
                excluded=len(dropped),
            )
        )

    if not any(s.covered for s in scores):
        sample = next(iter(measured), "<nothing>")
        raise ToolError(
            f"no owned line was recorded as executed across {len(scores)} files "
            f"({len(measured)} files in {data_file}; e.g. {sample}).\n"
            "coverage's `source` is a directory, so a run that imported the "
            "package from site-packages measures nothing and still exits 0. "
            "Point PYTHONPATH at this tree and re-run."
        )
    return scores, foreign


def _is_dirty(repo: str) -> bool:
    """Whether the tree differs from HEAD, untracked files included.

    Untracked files are scored as tier A, so ignoring them would let a tree
    whose only change is a new owned module report itself as clean.
    """
    return bool(_git(repo, "status", "--porcelain"))


def _stale_sources(repo: Path, data_file: Path, owned: Sequence[str]) -> List[str]:
    """Owned files edited after the coverage data was written.

    Scoring an old .coverage otherwise reports a confident number for code that
    never ran, and exits 0 -- the same false green the other guards exist for.
    """
    try:
        recorded = data_file.stat().st_mtime
    except OSError:
        return []
    newer = []
    for rel in owned:
        try:
            if (repo / rel).stat().st_mtime > recorded:
                newer.append(rel)
        except OSError:
            continue
    return sorted(newer)


def _junit_counts(path: Path) -> Optional[Dict[str, int]]:
    """Test outcome counts, so a skipped-heavy run cannot read as untested code."""
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    suites = [root] if root.tag == "testsuite" else list(root)
    total = sum(int(s.get("tests", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    return {
        "total": total,
        "skipped": skipped,
        "failed": failed,
        # Clamped: --reruns makes a retried test several <testcase> entries, so
        # the arithmetic is an estimate and a negative would be nonsense.
        "passed": max(0, total - skipped - failed),
    }


_REACH_STAMP = "jit-reach.stamp"


def _stamp_of(data_file: Path) -> str:
    """Identity of a data file: path, size and mtime.

    Size as well as mtime because NFS timestamp granularity is coarse enough
    that a rewrite moments later can keep the same mtime.
    """
    st = data_file.stat()
    return f"{data_file.resolve()} {st.st_size} {st.st_mtime_ns}"


def _write_stamp(out_dir: Path, data_file: Path) -> None:
    """Tie the reach shards in `out_dir` to the data file this run just wrote."""
    # Suppressed: the reach figure is optional and never worth failing a run.
    with contextlib.suppress(OSError):
        (out_dir / _REACH_STAMP).write_text(
            _stamp_of(data_file) + "\n", encoding="utf-8"
        )


def _stamp_matches(out_dir: Path, data_file: Path) -> bool:
    try:
        recorded = (out_dir / _REACH_STAMP).read_text(encoding="utf-8").strip()
        return recorded == _stamp_of(data_file)
    except OSError:
        return False


def _jit_reach(
    repo: Path, out_dir: Path, data_file: Path
) -> Optional[Tuple[int, List[str]]]:
    """(reached, unreached) csrc_rocm translation units, merged across xdist workers.

    Not a coverage figure: the HIP sources are JIT-compiled and have no line
    data. It says only which of them the Python tests caused to be built at all.
    """
    # Report only shards from the run that produced this data. An mtime cutoff
    # cannot express that -- pytest-cov combines .coverage after the workers
    # write their shards, so the shards are always fractionally older -- hence
    # the explicit stamp _run_pytest leaves behind.
    if not _stamp_matches(out_dir, data_file):
        return None
    shards = sorted(out_dir.glob("jit-reach.*.json"))
    if not shards:
        return None
    reached: Set[str] = set()
    for shard in shards:
        try:
            reached.update(json.loads(shard.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    units = {p.name for p in (repo / _CSRC_DIR).glob("*") if p.suffix in (".cu", ".cc")}
    if not units:
        return None
    return len(reached & units), sorted(units - reached)


def _pct(covered: int, total: int) -> str:
    return f"{100.0 * covered / total:6.2f}%" if total else "   n/a"


def _report(
    repo: Path,
    base_desc: str,
    dirty: bool,
    arch: str,
    scores: List[Score],
    excluded: Dict[str, str],
    unowned: List[str],
    ruled: Dict[str, str],
    tests: Optional[Dict[str, int]],
    reach: Optional[Tuple[int, List[str]]],
    stale: List[str],
    foreign: Set[str],
    show_files: bool,
) -> float:
    print("== amd coverage ==")
    print(f"base  : {base_desc}")
    print(f"tree  : {repo}{' (uncommitted changes)' if dirty else ''}")
    print(f"arch  : {arch}")
    if stale:
        head = ", ".join(stale[:3]) + (
            f", +{len(stale) - 3} more" if len(stale) > 3 else ""
        )
        print(f"STALE : {len(stale)} owned files changed after this run: {head}")
        print("        re-run with --run; the number below describes the older code")
    for root in sorted(foreign):
        print(f"NOTE  : executed lines came from {root}, not this tree")
        print("        set PYTHONPATH if that is a different checkout")
    if tests:
        print(
            f"tests : {tests['passed']} passed, {tests['skipped']} skipped, "
            f"{tests['failed']} failed"
        )
        if tests["skipped"]:
            print("        skips lower the number without meaning the code is untested")
    print()

    if show_files:
        print("== per file ==")
        width = max(len(s.path) for s in scores)
        for s in sorted(scores, key=lambda s: (s.tier, s.path)):
            note = f"  ({s.excluded} excluded)" if s.excluded else ""
            print(
                f"  {s.tier}  {s.path:<{width}}  "
                f"{s.exec_covered:>5}/{s.exec_total:<5} {_pct(s.exec_covered, s.exec_total)}{note}"
            )
        print()

    print("== by tier ==")
    for tier in ("A", "B", "C"):
        rows = [s for s in scores if s.tier == tier]
        if not rows:
            continue
        cov = sum(s.exec_covered for s in rows)
        tot = sum(s.exec_total for s in rows)
        print(
            f"  tier {tier}  {_TIER_LABEL[tier]:<32}{len(rows):>4} files  "
            f"{cov:>6}/{tot:<6} {_pct(cov, tot)}"
        )

    exec_cov = sum(s.exec_covered for s in scores)
    exec_tot = sum(s.exec_total for s in scores)
    import_tot = sum(len(s.import_time) for s in scores)
    excl_tot = sum(s.excluded for s in scores)
    excl_files = sum(1 for s in scores if s.excluded)

    print()
    print(
        f"  {'execution coverage':<47}{exec_cov:>6}/{exec_tot:<6} {_pct(exec_cov, exec_tot)}"
    )
    if import_tot:
        total_cov, total_tot = exec_cov + import_tot, exec_tot + import_tot
        print(f"  {'import-time lines (always covered)':<47}{import_tot:>6}")
        print(
            f"  {'total, conventional':<47}{total_cov:>6}/{total_tot:<6} {_pct(total_cov, total_tot)}"
        )
    else:
        print(
            "  (no import-time baseline: the headline includes lines that run at import)"
        )
    print()

    if reach is not None:
        reached, unreached = reach
        total = reached + len(unreached)
        print("== csrc_rocm reach ==")
        print(
            f"  {reached} of {total} translation units were built and loaded by a test"
        )
        print("  (not a coverage figure -- JIT-built HIP has no line data)")
        if unreached:
            head = ", ".join(unreached[:6])
            more = f", +{len(unreached) - 6} more" if len(unreached) > 6 else ""
            print(f"  never loaded: {head}{more}")
        print()

    print("== not counted ==")
    if excl_tot:
        print(
            f"  {excl_tot} lines in {excl_files} files excluded by "
            "[tool.coverage.report], mostly `if IS_CUDA:` arms"
        )
    for path, reason in sorted({**excluded, **ruled}.items()):
        print(f"  {path} -- {reason}")
    print(
        f"  {len(unowned)} upstream files on the measured surface, attributed to no tier"
    )
    print("  tests/ and benchmarks/ are outside the measured surface")
    print("  C++/HIP under csrc_rocm/ and include/ has no line coverage (JIT-built)")
    print()

    return 100.0 * exec_cov / exec_tot if exec_tot else 0.0


def _detect_arch() -> str:
    env = os.environ.get("FLASHINFER_ROCM_ARCH_LIST")
    if env:
        return env
    try:
        out = subprocess.run(
            ["rocm_agent_enumerator"], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    archs = sorted(
        {a.strip() for a in out.split() if a.startswith("gfx") and a != "gfx000"}
    )
    return ",".join(archs) or "unknown"


def _capture_baseline(repo: Path, out: Path) -> Path:
    """Record the lines that run merely from `import flashinfer`.

    tests/conftest.py imports the package at collection, so without this every
    module-level line counts as covered before a test body runs.
    """
    # pid-namespaced: two concurrent runs (the documented cross-arch recipe)
    # would otherwise truncate and unlink each other's probe.
    probe = out.parent / f"_import_probe_{os.getpid()}.py"
    probe.write_text("import flashinfer  # noqa: F401\n", encoding="utf-8")
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                f"--data-file={out}",
                f"--rcfile={repo / 'pyproject.toml'}",
                str(probe),
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            # `coverage run <script>` puts the script's directory on sys.path,
            # not the cwd, so with --out-dir the repo is not importable.
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [str(repo), os.environ.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            },
        )
    finally:
        probe.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise ToolError(
            "could not capture the import-time baseline, so the headline would "
            "silently include lines that run at import. Fix the import, or pass "
            f"--no-baseline to accept the conventional number:\n"
            f"{proc.stderr.strip()[-800:]}"
        )
    return out


def _run_pytest(
    repo: Path, out_dir: Path, data_file: Path, pytest_args: Sequence[str]
) -> Path:
    junit = out_dir / "junit.xml"
    for stale in out_dir.glob("jit-reach.*.json"):
        stale.unlink()  # a previous run's shards would inflate the reach count
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov",
        "--cov-report=",
        f"--junitxml={junit}",
        "-p",
        "jit_reach_plugin",
        *pytest_args,
    ]
    print(f"+ {' '.join(cmd)}", file=sys.stderr)

    # Compare against a snapshot rather than a wall-clock stamp: NFS mtime
    # granularity can make a file written moments after `time.time()` look older
    # than it, which would reject a perfectly good run. Size as well as
    # nanoseconds for the same reason -- on a coarse clock a rewrite inside one
    # tick leaves the timestamp identical, and that would read as "wrote
    # nothing". Same identity tuple as the reach stamp.
    def _identity(p: Path) -> Optional[Tuple[int, int]]:
        try:
            st = p.stat()
        except OSError:
            return None
        return st.st_size, st.st_mtime_ns

    before = {p: _identity(p) for p in (junit, data_file)}
    proc = subprocess.run(
        cmd,
        cwd=repo,
        # COVERAGE_FILE, or --data-file would write one file and score another.
        env={
            **os.environ,
            _JIT_REACH_ENV: str(out_dir),
            "COVERAGE_FILE": str(data_file),
        },
    )
    # pytest exits non-zero on test failures; a partially-failing run is still
    # worth scoring. But exit 1 also covers a plugin ImportError, which runs no
    # tests and writes nothing -- and leftover artifacts would then be scored as
    # if they were this run's. Trust the artifacts, not the exit code.
    if proc.returncode >= 4:
        raise ToolError(f"pytest could not run (exit {proc.returncode})")
    for artefact, was in before.items():
        now = _identity(artefact)
        if now is None or now == was:
            raise ToolError(
                f"pytest exited {proc.returncode} without writing {artefact.name}; "
                "it collected no tests (a plugin import error looks like this). "
                "Refusing to score the previous run's data."
            )
    _write_stamp(out_dir, data_file)
    return junit


def run(args: argparse.Namespace) -> int:
    repo = Path(_git(None, "rev-parse", "--show-toplevel"))
    base = _resolve_base(str(repo), args.upstream_ref)
    manifest_path = repo / _MANIFEST
    if not manifest_path.exists():
        raise ToolError(f"missing {_MANIFEST}")
    owned, excluded, unowned, ruled = classify(
        str(repo), base, _load_toml(manifest_path)
    )
    if not owned:
        raise ToolError("no owned files found -- is this the amd-integration fork?")

    # Anchored to the repo, not the caller's cwd: pytest runs with cwd=repo, so
    # a relative path would be written in one place and read from another.
    def _anchored(value: Optional[str], default: Path) -> Path:
        if not value:
            return default
        given = Path(value)
        return given if given.is_absolute() else repo / given

    out_dir = _anchored(args.out_dir, repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_file = _anchored(args.data_file, repo / ".coverage")
    # Not ".coverage.import-baseline": that matches coverage's parallel-data
    # glob `.coverage.*`, so pytest-cov's combine() absorbs and unlinks it, and
    # the import-time split silently vanishes into the headline.
    baseline = out_dir / "import-baseline.coverage"

    junit = out_dir / "junit.xml"
    baseline_result: Optional[Path] = None
    if args.no_baseline:
        pass  # honoured in every mode, including --run and a leftover file on disk
    elif args.run or not baseline.exists():
        baseline_result = _capture_baseline(repo, baseline)
    else:
        baseline_result = baseline
    if args.run:
        junit = _run_pytest(repo, out_dir, data_file, args.pytest_args)

    if not data_file.exists():
        raise ToolError(
            f"no coverage data at {data_file}. Run the suite first:\n"
            '  python3 scripts/amd_coverage.py --run -- -n auto -m "not slow"'
        )

    scores, foreign = score(repo, owned, data_file, baseline_result)
    dirty = _is_dirty(str(repo))
    arch = _detect_arch()
    base_desc = _git(str(repo), "log", "-1", "--format=%h %ad %s", "--date=short", base)
    tests = _junit_counts(junit)
    reach = _jit_reach(repo, out_dir, data_file)
    # --run has just written the data, so any mtime skew there is noise.
    stale = [] if args.run else _stale_sources(repo, data_file, list(owned))

    pct = _report(
        repo,
        base_desc,
        dirty,
        arch,
        scores,
        excluded,
        unowned,
        ruled,
        tests,
        reach,
        stale,
        foreign,
        args.show_files,
    )

    if args.json_out:
        payload = {
            "base": base,
            "base_desc": base_desc,
            "dirty": dirty,
            "arch": arch,
            "stale_sources": stale,
            "foreign_source_roots": sorted(foreign),
            "tests": tests,
            "execution_percent": round(pct, 2),
            "csrc_reach": (
                {"loaded": reach[0], "never_loaded": reach[1]} if reach else None
            ),
            "files": [
                {
                    "path": s.path,
                    "tier": s.tier,
                    "reason": s.reason,
                    "owned": s.exec_total,
                    "covered": s.exec_covered,
                    "import_time": len(s.import_time),
                    "excluded": s.excluded,
                    "missing": sorted(s.owned - s.covered - s.import_time),
                }
                for s in sorted(scores, key=lambda s: s.path)
            ],
            "excluded": excluded,
            "unowned": unowned,
            "ruled_unowned": ruled,
        }
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_out}")

    if args.fail_under is not None and pct < args.fail_under:
        print(
            f"FAIL: execution coverage {pct:.2f}% is under --fail-under {args.fail_under}"
        )
        return EXIT_RATCHET
    return EXIT_OK


def main() -> int:
    summary = (__doc__ or "").splitlines()[0]
    parser = argparse.ArgumentParser(
        description=summary,
        epilog="Arguments after `--` are passed to pytest when --run is given.",
    )
    parser.add_argument(
        "--run", action="store_true", help="run pytest under coverage before scoring"
    )
    parser.add_argument(
        "--upstream-ref",
        default="upstream/main",
        help="ref to take the merge base against",
    )
    parser.add_argument(
        "--data-file", default=None, help="coverage data file (default: ./.coverage)"
    )
    parser.add_argument(
        "--out-dir", default=None, help="where to write junit.xml and the baseline"
    )
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help="write the machine-readable report",
    )
    parser.add_argument(
        "--show-files", action="store_true", help="print the per-file table"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the import-time split (the headline then includes import-only lines)",
    )
    parser.add_argument("--fail-under", type=float, default=None, metavar="PCT")
    parser.add_argument("pytest_args", nargs="*", help=argparse.SUPPRESS)

    try:
        return run(parser.parse_args())
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
