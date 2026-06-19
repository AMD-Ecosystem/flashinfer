---
name: pr-workflow
description: How to create and edit PRs on the ROCm/flashinfer GitHub repo (which publishes the amd-flashinfer package) — gh CLI quirks and the project's PR-description conventions.
---

# PR Workflow (ROCm/flashinfer)

> The GitHub repo is `ROCm/flashinfer`; the Python package it publishes is
> `amd-flashinfer`. All `gh` commands below target the GitHub repo.

## CRITICAL: PR target safeguard (fail-closed)

`ROCm/flashinfer` is a **GitHub fork** of `flashinfer-ai/flashinfer` (the true
upstream). Because of this, `gh pr create` defaults the PR base to the
fork-parent `flashinfer-ai/flashinfer` unless explicitly overridden. **A PR must
NEVER be opened against `flashinfer-ai/flashinfer`.**

All PRs go to **`ROCm/flashinfer`**, base branch **`amd-integration`**.

**Before ANY `gh pr create`, run this pre-flight check and ABORT if it fails:**

```bash
gh repo set-default --view   # MUST print exactly: ROCm/flashinfer
```

If it prints anything else (or errors), STOP — do not create the PR. Report the
mismatch to the user instead. Never guess the target.

**Always pass the target and base explicitly** — never rely on gh defaults:

```bash
gh pr create --repo ROCm/flashinfer --base amd-integration \
  --title "<title>" --body "$(cat /tmp/pr_body.md)"
```

If the resolved owner of `--repo` is ever `flashinfer-ai`, abort. It is always
better to fail to raise a PR and explain why than to raise one against upstream.

### One-time setup after a fresh clone

These are local config (not checked in) and must be redone per clone:

```bash
git remote -v                        # origin should be ROCm/flashinfer; there must be NO flashinfer-ai remote
gh repo set-default ROCm/flashinfer  # pin gh base repo so it does not fall back to the fork parent
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

## GitHub CLI

`gh pr edit` fails with a "Projects (classic) is being deprecated" GraphQL error on this repo. Use the REST API instead:

```bash
# Update PR description
gh api repos/ROCm/flashinfer/pulls/<number> --method PATCH --field body="<body>"

# Or from a file
gh api repos/ROCm/flashinfer/pulls/<number> --method PATCH --field body="$(cat /tmp/pr_body.md)"
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

List threads with their resolved status, and resolve them, via GraphQL. GraphQL
is required because thread resolution and the thread-level `isResolved` flag are
not exposed via REST (replying to a comment is available over REST — see below):

```bash
# List threads (id + resolved + comment bodies).
# Bump `first:` if a PR may have more than 50 threads / 10 comments per thread —
# this query is not exhaustive beyond those limits.
gh api graphql -f query='
{ repository(owner:"ROCm", name:"flashinfer") {
    pullRequest(number: <PR>) {
      reviewThreads(first: 50) { nodes {
        id isResolved
        comments(first: 10) { nodes { databaseId author { login } path body } } } } } } }'

# Reply to a comment (use the databaseId from above)
gh api repos/ROCm/flashinfer/pulls/<PR>/comments/<commentDatabaseId>/replies \
  --method POST --field body="<reply>"

# Resolve a thread (use the thread node id, e.g. PRRT_...)
gh api graphql -f query='
mutation { resolveReviewThread(input:{threadId:"<threadId>"}) {
  thread { isResolved } } }'
```

Done = no unresolved Copilot threads remain, each carrying either a fix+SHA reply
or a won't-fix rationale.

## PR Description

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
