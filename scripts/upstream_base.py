#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The upstream commit this fork is synced to, and the base derived from it.

Recorded in a file rather than read off ancestry: a squash-merged sync leaves no
merge parent, so ``merge-base(HEAD, <upstream tag>)`` silently returns the
*previous* fork point. Both callers must agree on the rule, so it lives here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Tuple

FILENAME = "upstream-base"

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")

# (repo, *args, check=) -> CompletedProcess. Each caller passes its own runner:
# amd_coverage adds `safe.directory`, the canary does not.
Runner = Callable[..., subprocess.CompletedProcess]


class UpstreamBaseError(Exception):
    """The recorded base could not be read or trusted."""


class MissingBaseObject(UpstreamBaseError):
    """The recorded commit is not in the object store.

    Distinct because the remedy inverts the usual one: an explicit base needs the
    object *present*, not *reachable*, so fetching the tag does help here.
    """


class UpstreamBase(NamedTuple):
    ref: str
    sha: str


def parse(text: str, source: str) -> UpstreamBase:
    """First non-comment line as ``<ref> <40-hex-sha>``."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise UpstreamBaseError(
                f"{source}: expected '<ref> <sha>', got {line!r}. A malformed "
                "record is never guessed at -- the wrong base reports a "
                "plausible number rather than an error."
            )
        ref, sha = parts
        if not _SHA_RE.fullmatch(sha):
            raise UpstreamBaseError(
                f"{source}: {sha!r} is not a 40-character lowercase hex sha"
            )
        if not _REF_RE.fullmatch(ref):
            # Both fields reach git as arguments, and a leading '-' is parsed
            # there as an option -- which then reads as "ref not present".
            raise UpstreamBaseError(
                f"{source}: {ref!r} is not a plain ref name "
                "([A-Za-z0-9] then letters, digits, '.', '_', '-', '/')"
            )
        return UpstreamBase(ref, sha)
    raise UpstreamBaseError(f"{source}: no '<ref> <sha>' line")


def read_worktree(repo: str) -> Optional[UpstreamBase]:
    """Read the working-tree copy. ``None`` if absent; raises if unreadable."""
    path = Path(repo) / FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise UpstreamBaseError(f"cannot read {path}: {exc}") from exc
    return parse(text, str(path))


def read_ref(run: Runner, repo: str, ref: str) -> Optional[UpstreamBase]:
    """Read the copy committed at ``ref``, so ``--ours`` means what it says."""
    proc = run(repo, "show", f"{ref}:{FILENAME}", check=False)
    if proc.returncode:
        return None
    return parse(proc.stdout, f"{ref}:{FILENAME}")


def _require_present(run: Runner, repo: str, recorded: UpstreamBase) -> None:
    """The recorded commit must be in the object store; it need not be reachable."""
    probe = run(
        repo,
        "rev-parse",
        "--verify",
        "--quiet",
        f"{recorded.sha}^{{commit}}",
        check=False,
    )
    if probe.returncode:
        raise MissingBaseObject(
            f"recorded base {recorded.sha[:12]} ({recorded.ref}) is not in this "
            f"clone. It is reachable from no branch by design, so fetch the tag "
            f"that carries it:\n  git fetch origin tag upstream-base/{recorded.ref}"
        )
    named = run(
        repo,
        "rev-parse",
        "--verify",
        "--quiet",
        f"{recorded.ref}^{{commit}}",
        check=False,
    )
    if not named.returncode and named.stdout.strip() != recorded.sha:
        raise UpstreamBaseError(
            f"{FILENAME} records {recorded.ref} as {recorded.sha[:12]}, but that "
            f"ref resolves to {named.stdout.strip()[:12]} here. Upstream moved the "
            "tag, or the file was hand-edited."
        )


def select(
    run: Runner,
    repo: str,
    ours: str,
    theirs: str,
    recorded: Optional[UpstreamBase],
) -> Tuple[str, str]:
    """``(base_sha, source)`` — the anchor is the recorded commit, not our tip.

    ``merge-base(recorded, theirs)`` degrades to the recorded commit itself for a
    release we have absorbed, and still finds the real divergence point for a
    moving branch like ``upstream/main``.
    """
    if recorded is None:
        anchor, source = ours, "ancestry"
    else:
        _require_present(run, repo, recorded)
        anchor, source = recorded.sha, "recorded"

    proc = run(repo, "merge-base", anchor, theirs, check=False)
    if proc.returncode:
        detail = "" if proc.returncode == 1 else "\n" + proc.stderr.strip()
        raise UpstreamBaseError(
            f"'{theirs}' shares no history with {anchor[:12]}.{detail}"
        )
    return proc.stdout.strip(), source


def multiple_bases(run: Runner, repo: str, base_anchor: str, theirs: str) -> bool:
    proc = run(repo, "merge-base", "--all", base_anchor, theirs, check=False)
    return not proc.returncode and len(proc.stdout.split()) > 1


def missing_note(where: str) -> str:
    """Announce the fallback. A quiet one is how #344 shipped a wrong number."""
    return (
        f"NOTE: no {FILENAME} at {where}; falling back to ancestry against our "
        "own tip.\n      That is correct only while the sync is still a merge "
        "commit."
    )
