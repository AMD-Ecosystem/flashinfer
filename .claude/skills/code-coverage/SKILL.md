---
name: code-coverage
description: Measure and interpret line coverage for the AMD-authored parts of amd-flashinfer — running scripts/amd_coverage.py, reading its tiers, and the container/worktree setup it needs. Load when asked to measure coverage, report a coverage number, or explain what the port's coverage actually covers.
---

# Coverage for the port, not for upstream

`scripts/amd_coverage.py` scores only the code this fork added or changed.
Ownership is recomputed on every run from the diff against the upstream release
the port is based on, read off this fork's own tag (`v0.5.3+amd.2` scores
against upstream `v0.5.3`), so the answer is always current for the tree in
front of you and there is no list to refresh. A fixed release rather than a
moving branch is what makes two runs weeks apart comparable.

## Run it

`origin` carries upstream's release tags, so no second remote is needed.

```bash
git fetch --tags origin                       # the base is computed, not stored
pip install -e ".[dev]"                       # brings pytest-cov
python3 scripts/amd_coverage.py --run --show-files \
    --json-out tmp/coverage/coverage-amd.json --out-dir tmp/coverage \
    -- -n auto --reruns 2 -m "not slow"
```

Scoring is separable from running — re-score an existing `.coverage` as often
as you like without paying for the suite again:

```bash
python3 scripts/amd_coverage.py --show-files
```

## Reading the output

| Tier | Means | Scored on |
|---|---|---|
| A | we added the file | every statement |
| B | upstream file we edited | **only the lines our diff touched** |
| C | zero-line Python diff, but ours via the `FLASHINFER_CSRC_DIR` redirect | every statement |

Tier C is declared in `scripts/coverage_ownership.toml` because no diff can
discover it. The tool fails if an entry there names a file that no longer
exists, so a rename cannot leave it quietly wrong.

**The headline is execution coverage** — owned lines reached beyond
`import flashinfer`. Import-time lines are counted separately rather than folded
in, because `tests/conftest.py` imports the package at collection and would
otherwise mark every module-level statement covered before a test body runs.
Roughly 45% of tier-B lines on this fork are of that kind.

**`if IS_CUDA:` bodies are excluded and the count is printed.** The port
re-indented upstream code under those guards, so git blames it on us even
though no ROCm box executes it. Detection is AST-based and matches only a bare
`IS_CUDA` test; a compound condition stays in the denominator rather than being
dropped on a guess.

**`csrc_rocm` reach is not coverage.** JIT-built HIP has no line data. The
report says how many of its translation units a run built and loaded, via the
`tests/jit_reach_plugin.py` hook on `JitSpec.load`. Do not quote it as a
percentage or add it to the Python number.

## Say which architecture the number came from

`arch_caps.py` gates behaviour per gfx942/gfx950, so a one-box run leaves the
other architecture's branches unexecuted. The arch is recorded in the report
header and in the JSON. For a union, `coverage combine` the two data files —
but note the mapping in `[tool.coverage.paths]` only applies when combine runs
**from a checkout that contains the sources**, because coverage aliases only
onto paths that exist on disk. Combining from anywhere else is a silent no-op
that leaves two half-covered copies of every file.

## Running against a worktree, in a container

Mount the **parent repository**, not just the worktree. A linked worktree's
`.git` is a file pointing into the main checkout, so without it `git
rev-parse` fails inside the container and the classifier cannot start. Mount
the repo root at its host path, and allow the ownership mismatch:

```bash
R=/path/to/flashinfer; WT=$R/tmp/worktrees/<branch>
docker run -d --name demandal-<branch> \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --shm-size 16g -e MAX_JOBS=16 \
  -v "$R":"$R" -w "$WT" -e PYTHONPATH="$WT" \
  -v "$HOME/.cache/fi-gfx942/<label>":/root/.cache/flashinfer \
  flashinfer-bench:rocm sleep infinity
docker exec <container> git config --global --add safe.directory '*'
```

There is **no GPU CI** in this repo, so no automation produces this number —
every workflow runs on `ubuntu-latest` or a CPU runner. The classifier's own
tests (`tests/rocm_tests/test_amd_coverage.py`) do run there, via
`arch-caps-conformance.yml`.

## Failure modes worth recognising

- **"no owned line was recorded as executed"** — the run imported flashinfer
  from somewhere other than this tree. `source` is a directory, so coverage
  measures nothing and would otherwise exit 0 with a clean 0%. Fix `PYTHONPATH`.
- **"pytest exited N without writing junit.xml"** — pytest collected no tests.
  Exit 1 means both "tests failed" and "a plugin failed to import", so the tool
  trusts the artifacts rather than the exit code and refuses to score the
  previous run's data.
- **"could not capture the import-time baseline"** — the headline would
  silently include lines that run at import, so this is an error rather than a
  warning. Fix the import, or pass `--no-baseline` to accept the conventional
  number.
- **`STALE :` in the header** — you are scoring a `.coverage` older than the
  sources it describes. Re-run with `--run`.
- **Never construct `coverage.Coverage(config_file=<repo pyproject>)` without
  an explicit `data_file`.** It initializes the configured data file, which
  erases a completed run. A fixture in the test file enforces this.
- **A file you just wrote shows up as tier A "untracked"** — expected; the tool
  includes untracked, non-ignored files so work in progress still counts.
- **"cannot tell which upstream release this fork is based on"** — no reachable
  `*+amd.*` tag. `git fetch --tags origin`, unless the message also says the
  clone is shallow, in which case only `git fetch --unshallow origin` helps:
  `describe` needs the tag *reachable*, and the graft is what severs it.
- **"shares no history with HEAD"** — the tag resolved but the fork point is not
  in this clone. Same shallow cause, same fix.
- **`--fail-under` is comparable across runs, but still not wired into CI.** The
  base is a release tag, so it does not move between rebases and two runs weeks
  apart do compare. What stops it is cost: the number needs the full GPU suite,
  which no PR lane runs. The JSON pins the base SHA either way.
