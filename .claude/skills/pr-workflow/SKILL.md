---
name: pr-workflow
description: How to create and edit PRs on the AMD-Ecosystem/flashinfer GitHub repo (which publishes the amd-flashinfer package) — gh CLI quirks and the project's PR-description conventions.
---

# PR Workflow (AMD-Ecosystem/flashinfer)

> The GitHub repo is `AMD-Ecosystem/flashinfer`; the Python package it publishes is
> `amd-flashinfer`. All `gh` commands below target the GitHub repo.

## CRITICAL: PR target safeguard (fail-closed)

`AMD-Ecosystem/flashinfer` is a **GitHub fork** of `flashinfer-ai/flashinfer` (the true
upstream). Because of this, `gh pr create` defaults the PR base to the
fork-parent `flashinfer-ai/flashinfer` unless explicitly overridden. **A PR must
NEVER be opened against `flashinfer-ai/flashinfer`.**

All PRs go to **`AMD-Ecosystem/flashinfer`**, base branch **`amd-integration`**.

**Before ANY `gh pr create`, run this pre-flight check and ABORT if it fails:**

```bash
gh repo set-default --view   # MUST print exactly: AMD-Ecosystem/flashinfer
```

If it prints anything else (or errors), STOP — do not create the PR. Report the
mismatch to the user instead. Never guess the target.

**Always pass the target and base explicitly** — never rely on gh defaults:

```bash
gh pr create --repo AMD-Ecosystem/flashinfer --base amd-integration \
  --title "<title>" --body-file "$(git rev-parse --show-toplevel)/tmp/pr/<branch>.md"
```

If the resolved owner of `--repo` is ever `flashinfer-ai`, abort. It is always
better to fail to raise a PR and explain why than to raise one against upstream.

### One-time setup after a fresh clone

These are local config (not checked in) and must be redone per clone:

```bash
git remote -v                        # origin should be AMD-Ecosystem/flashinfer; there must be NO flashinfer-ai remote
gh repo set-default AMD-Ecosystem/flashinfer  # pin gh base repo so it does not fall back to the fork parent
ex="$(git rev-parse --git-common-dir)/info/exclude"   # per-clone ignore patterns
mkdir -p "$(dirname "$ex")"                           # info/ is absent in some clones
grep -qxF '/tmp/' "$ex" || printf '/tmp/\n' >> "$ex"  # ignores the repo's own tmp/
```

If a remote pointing at `flashinfer-ai/flashinfer` exists, remove it:
`git remote remove <name>`.

## CRITICAL: never PR from `amd-integration` (fail-closed)

`amd-integration` is the **base** branch — it must never be the **head** of a
PR, and must never be pushed to directly via a PR flow. Before pushing a branch
for a PR or running `gh pr create`, check the current branch:

```bash
git branch --show-current   # if this is "amd-integration" (or empty/detached), do NOT push/PR from it
```

This command prints an **empty string** in a detached-HEAD state. Treat empty
output as an abort condition too — do NOT push/PR from a detached HEAD; STOP and
report so the user can check out a proper topic branch first.

If you are on `amd-integration` and have commits to ship, do NOT raise the PR
from it. Instead, relocate the commits to a fresh topic branch, restore
`amd-integration` to match the remote, then PR from the topic branch:

```bash
# 1. Capture the local-only commits onto a new branch at current HEAD
git branch <topic-branch>

# 2. Move amd-integration back to the pristine remote state (no commits lost —
#    they are preserved on <topic-branch>). Verify origin/amd-integration is
#    fetched and up to date first.
git fetch origin amd-integration
git reset --hard origin/amd-integration

# 3. Switch to the topic branch and proceed with the normal PR flow
git checkout <topic-branch>
```

Only `git reset --hard` here because the commits are already safe on
`<topic-branch>` (confirm with `git log <topic-branch>` before resetting). If
anything is ambiguous — uncommitted changes, unclear which commits are
local-only, the topic branch already exists — STOP and report rather than
reset. It is always better to fail to raise a PR than to push to or PR from
`amd-integration`.

## CRITICAL: do branch work in a git worktree, never in the main checkout

**Always** do branch work in a worktree under `tmp/worktrees/<branch-name>`.
Leave the main checkout parked on `amd-integration`. Never switch the main
checkout to a topic branch.

```bash
git worktree add -b <branch-name> tmp/worktrees/<branch-name> origin/amd-integration  # new branch
git worktree add tmp/worktrees/<branch-name> <branch-name>                            # existing branch
```

`git worktree add` creates the leading `tmp/worktrees/` directories itself, so
no `mkdir -p` is needed on a fresh clone.

Why this is the rule and not a preference:

- The main checkout owns the editable install and the in-tree compiled
  extensions. Switching it between branches invalidates them and produces
  confusing stale-binary failures.
- Multiple PRs are usually in flight at once. Worktrees keep them physically
  separate, so an unrelated in-progress edit cannot leak into a PR.
- `amd-integration` staying pristine is what makes the `git reset --hard`
  recovery above safe.

The JIT cache is the one thing worktrees do **not** isolate. It lives outside
the repo at `$HOME/.cache/flashinfer/<version>/<arch>` (rooted at
`FLASHINFER_WORKSPACE_BASE` if set) and is keyed by version and arch only — no
branch or checkout path — so every worktree shares one cache. Clear it with
`rm -rf ~/.cache/flashinfer/`, or point `FLASHINFER_WORKSPACE_BASE` somewhere
per-branch, when comparing kernel changes across branches.

Exclude the worktree root **per-clone** rather than in a tracked ignore file, so
it never appears as a diff against upstream. This is the same `/tmp/` entry that
hides PR drafts — run the snippet in *One-time setup after a fresh clone* once
and it covers the whole clone, worktrees included.

That snippet uses `--git-common-dir`, not a literal `.git/info/exclude`, because
inside a linked worktree `.git` is a *file* pointing at the real git dir, so the
literal path fails with `not a directory`.

### A fresh worktree is source-only

Two gitignored, generated files must be recreated or the JIT will not build:

```bash
cd tmp/worktrees/<branch-name>
rm -rf flashinfer/include                    # -f alone will not clear a real dir
ln -s ../include flashinfer/include          # MUST be relative
cp <main-checkout>/flashinfer/_version.py flashinfer/_version.py   # see below if absent
```

Clear the path first. `-f` replaces a dangling or stale *symlink*, but against
a real directory `ln` silently creates `flashinfer/include/include` **inside**
it and exits 0 — leaving a broken tree with no error to go on. Deleting
`flashinfer/include` is safe: it is gitignored and generated, and the real
headers live in `include/` at the repo root.

- `flashinfer/include` — `get_include_paths.get_include()` returns
  `<pkg>/include`, and the JIT passes that through `.resolve()` into
  `-isystem`. Both failure modes emit a well-formed flag pointing at a
  directory that isn't there, so the compile fails with
  `'flashinfer/rocm/attention/aiter/batch_prefill.cuh' file not found` rather than
  anything naming the include path:
  - missing entirely → `-isystem <pkg>/include`, a path that does not exist.
  - copied as an **absolute** symlink into a container mount point
    (e.g. `-> /fi/include`) → `.resolve()` follows it to `-isystem /fi/include`,
    which does not exist on the host or under a different mount.

  Create it **relative** so it resolves to `<worktree>/include` wherever the
  tree is mounted.
- `flashinfer/_version.py` — setuptools-scm generated. Without it,
  `import flashinfer` fails with `ModuleNotFoundError: No module named
  'flashinfer._version'`. If the `cp` fails because the source file is not
  there either, the main checkout has never been built: setuptools-scm only
  emits it during an install or build (`write_to` in `pyproject.toml`), so run
  `python -m pip install --no-build-isolation -ve.` there first. Don't reach
  for the `setuptools_scm` CLI — it is a build dependency and is generally not
  present in the environment you're running from.

Build artifacts and editable installs stay in the main checkout. If you need to
run tests against worktree code, bind-mount the **worktree** path into the
container and point `PYTHONPATH` at the mount — then confirm
`flashinfer.__file__` resolves there before trusting any result.

## Branch naming

Topic branches are created off `origin/amd-integration` and named with **plain
hyphenated words** describing the change:

- ✅ `pr-workflow-push-safety`, `timing-torch-events`, `aiter-solution-generator`
- ❌ **no `rocm/` prefix** — the entire fork is ROCm, so it's redundant noise.
- ❌ **no plan-phase labels** (`p1`, `p2`, …) from `ROCM_PORT_PLAN.md` — name the
  *change*, not the plan milestone.

## CRITICAL: ask before pushing to remote (fail-closed)

**Never `git push` (or `gh pr create`, which pushes) without first getting the
user's explicit "yes" for that specific push.** It publishes to a shared repo.
This applies to every push — the initial branch push, force-pushes after a
rebase, and follow-up pushes addressing review comments; a prior "yes" does not
authorize a later push. State exactly what will be pushed and where (branch →
`AMD-Ecosystem/flashinfer`) and wait. Local-only work (commits, the quality
gate, the local review) needs no confirmation — only the network push does. When
in doubt, hold the push and ask.

## GitHub CLI

`gh pr edit` fails with a "Projects (classic) is being deprecated" GraphQL error on this repo. Use the REST API instead:

```bash
# Update PR description
gh api repos/AMD-Ecosystem/flashinfer/pulls/<number> --method PATCH --field body="<body>"

# Or from a file
gh api repos/AMD-Ecosystem/flashinfer/pulls/<number> --method PATCH --field body="$(cat "$(git rev-parse --show-toplevel)/tmp/pr/<branch>.md")"
```

Ask the user to confirm before running `git push` or `gh pr create` — these
publish to a shared repo and shouldn't be triggered without explicit consent.
Before any `gh pr create`, also complete the fail-closed PR target safeguard
above.

## Before creating a PR: quality gate

Run this gate on the branch's full diff before `gh pr create`, in order:

1. **Simplify / make production-ready.** Review all changes on the branch and run
   `/simplify`: remove dead code, debug/scratch code, debug-only comments, and
   unused imports. Keep comments that carry real value (the *why*, hidden
   constraints, non-obvious invariants) — do not strip those.

2. **Code review.** Run `/code-review` on the diff, then apply the suggestions and
   recommendations you judge worthwhile.

3. **Run the relevant tests.** Run the pytests covering the changed code (see
   CLAUDE.md for commands, e.g. `pytest -n auto --reruns 2 -m "not slow"`) and
   make sure there are no failures after the changes. Docs/skill-only branches
   that touch no Python have no relevant tests — say so explicitly rather than
   claiming a run.

4. **Commit** the resulting changes.

Only after this gate passes do the pre-flight safeguards and `gh pr create`.

## After creating a PR: handle the Copilot review

Every PR on this repo gets an automated Copilot review. After `gh pr create`,
always run this loop before considering the PR done:

1. **Wait for all Copilot comments to land.** The review is not instant — Copilot
   posts a top-level review plus inline comments a short while after the PR (and
   after each later push). Poll until the review has arrived and the comment set
   is stable; don't evaluate a half-posted review. Copilot may also *auto-push*
   "Potential fix for pull request finding" commits to the branch — if so,
   `git fetch` and integrate them before adding your own (rebase; resolve
   conflicts keeping the more complete version).

2. **Evaluate each comment on its merits.** Decide per comment whether to fix it
   — Copilot is often right but not always. Use judgement; do not blanket-apply.

3. **Address the ones worth fixing**, commit, and push to the PR branch.

4. **Resolve every thread**, with the right closure for each:
   - *Fixed* → reply citing the commit SHA, then resolve the thread.
   - *Won't fix* → reply with the reason you decided not to address it, then
     resolve the thread.
   Either way the thread ends resolved with a written rationale.

**Keep replies short** — a verdict, a one-line reason, and the SHA. Reserve a
code block for comments you are *declining*, where the evidence is the argument.

### Suppressed comments have no thread

Copilot folds some findings into a collapsed **"Suppressed comments (N)"**
section of the review body. These are not review threads: there is no replies
endpoint and nothing to resolve, so the loop above cannot close them and they
are easy to miss entirely.

They are also **not cumulative** — each review body lists only its own. Sweep
every review, not just the latest:

```bash
gh api repos/AMD-Ecosystem/flashinfer/pulls/<PR>/reviews --paginate \
  --jq '.[] | select(.user.login=="copilot-pull-request-reviewer[bot]")
        | "\(.id) \(.submitted_at) \(.body)"'
```

Handle them like any other comment, then record the disposition in **one**
top-level comment per review — `gh api repos/AMD-Ecosystem/flashinfer/issues/<PR>/comments`.
Without it the fixes land silently and a reviewer cannot tell they were read.

Each push triggers a fresh review, so repeat until a review comes back with no
new comments *and* no suppressed block.

List threads with their resolved status, and resolve them, via GraphQL. GraphQL
is required because thread resolution and the thread-level `isResolved` flag are
not exposed via REST (replying to a comment is available over REST — see below):

```bash
# List threads (id + resolved + comment bodies).
# Bump `first:` if a PR may have more than 50 threads / 10 comments per thread —
# this query is not exhaustive beyond those limits.
gh api graphql -f query='
{ repository(owner:"AMD-Ecosystem", name:"flashinfer") {
    pullRequest(number: <PR>) {
      reviewThreads(first: 50) { nodes {
        id isResolved
        comments(first: 10) { nodes { databaseId author { login } path body } } } } } } }'

# Reply to a comment (use the databaseId from above)
gh api repos/AMD-Ecosystem/flashinfer/pulls/<PR>/comments/<commentDatabaseId>/replies \
  --method POST --field body="<reply>"

# Resolve a thread (use the thread node id, e.g. PRRT_...)
gh api graphql -f query='
mutation { resolveReviewThread(input:{threadId:"<threadId>"}) {
  thread { isResolved } } }'
```

Done = no unresolved Copilot threads remain, each carrying either a fix+SHA reply
or a won't-fix rationale.

## PR Description

**Draft it at `$(git rev-parse --show-toplevel)/tmp/pr/<branch>.md`** — the repo
root of whichever checkout you are in, main or worktree, so the draft sits with
the branch that it describes. Anchor to the top level rather than writing a
relative `tmp/pr/...`, which resolves wrong from any subdirectory. `tmp/pr/`
does not exist in a fresh clone or worktree, so create it as you go:

```bash
branch=$(git branch --show-current)   # empty on a detached HEAD
draft="$(git rev-parse --show-toplevel)/tmp/pr/${branch:-detached-$(git rev-parse --short HEAD)}.md"
mkdir -p "$(dirname "$draft")"
# write the body to "$draft", then pass it as --body-file "$draft"
```

A detached HEAD is an abort condition for the PR itself (see above); the
fallback name only keeps the snippet from producing `tmp/pr/.md` if you run it
before noticing.

Never the **system** `/tmp`: it is shared, and there is no remove access to it
on these nodes, so drafts left there cannot be cleaned up. Note the two are
easily confused — the `/tmp/` entry in `info/exclude` is a *repo-root-anchored*
pattern (the leading slash means repo root, not the filesystem root), so it
ignores this repo's own `tmp/` directory and has nothing to do with the system
one.

That entry comes from *One-time setup after a fresh clone*. Because
`info/exclude` lives in the git **common** dir, the single line covers the main
checkout and every linked worktree. If `git status` does show your draft, that
setup step has not been run in this clone.

- **Do not hard-wrap** the body to a fixed column. GitHub renders a single
  newline inside a paragraph as a `<br>`, so column-wrapped prose shows up as
  broken lines. Keep each paragraph/bullet on one line and use blank lines only
  to separate blocks. (Mirrors the `## PR / issue body formatting` rule in
  CLAUDE.md; note this is the opposite of how the repo's `.md`/`.py` source
  files — including this skill — are wrapped.)
- **Do not append a "Generated with Claude Code" footer** (or any other
  tool-authored attribution) to the body, even if a harness default suggests it.

**Body** — include sections that apply, skip the rest:

- `## Summary` — 1–3 sentences on what and why.
- `### What changed` with `####` per component when the PR spans multiple
  subsystems. Bullet by file: ``- **`path`** — one-line purpose``. Call out
  non-obvious design choices.
- `### Architecture / design notes` — only when there's a real choice to record.
  Tables for routing/dispatch logic; explain *why*.
- `## Benchmark results` — for perf-touching PRs. Shape line + table per entry
  point + mean overhead/speedup row.
- `## Test plan` — checklist of what was actually run (not aspirational), ending
  with `pre-commit run -a`.

Don't restate the diff and commits. Explain non-obvious decisions and surprising behaviors.
