# CLAUDE.md

> AMD ROCm port of FlashInfer (`amd-flashinfer`), targeting gfx942/gfx950.
> The upstream CUDA repo at <https://github.com/flashinfer-ai/flashinfer> uses
> different env vars, paths, and toolchains — its patterns don't apply here.

## CRITICAL: PR target

All PRs go to **`AMD-Ecosystem/flashinfer`**, base branch **`amd-integration`** — NEVER
to `flashinfer-ai/flashinfer` (the fork parent / true upstream). `gh pr create`
defaults the base to the fork parent, so always pass
`--repo AMD-Ecosystem/flashinfer --base amd-integration` explicitly, and run `gh repo set-default --view` first —
it MUST print `AMD-Ecosystem/flashinfer` or you abort. If the target owner cannot be
positively confirmed as `AMD-Ecosystem`, do not create the PR; stop and report why.
Failing to raise a PR is always preferable to raising one against upstream.

Also NEVER push to or raise a PR from the `amd-integration` branch itself — it
is the base, never the head. If you are on `amd-integration` with commits to
ship, create a fresh topic branch at the current HEAD
(`git branch <topic-branch>`), then run `git fetch origin amd-integration`
followed by `git reset --hard origin/amd-integration` to restore
`amd-integration` (the commits stay safe on `<topic-branch>`), and PR from
`<topic-branch>`. Treat a detached HEAD (empty `git branch --show-current`) as
an abort condition too.
Full procedure: `pr-workflow` skill.

## CRITICAL: work in a git worktree

Do branch work in a worktree at `tmp/worktrees/<branch-name>` and leave the main
checkout on `amd-integration`. Never switch the main checkout to a topic branch —
it owns the editable install and build artifacts, and switching it produces
stale-binary failures.

```bash
git worktree add -b <branch-name> tmp/worktrees/<branch-name> origin/amd-integration
```

A fresh worktree is source-only: recreate `flashinfer/include` as a **relative**
symlink (`rm -rf flashinfer/include && ln -s ../include flashinfer/include` — the
`rm` matters, since `ln -f` will not clear a real directory) and copy
`flashinfer/_version.py` from the main checkout, or the JIT will not build.
`_version.py` only exists there once that checkout has been installed or built.
Details: `pr-workflow` skill.

## Essential Commands

| Task | Command |
|------|---------|
| Install for development | `python -m pip install --no-build-isolation -ve.` |
| Run tests (fast) | `pytest -n auto --reruns 2 -m "not slow"` |
| Run all tests | `pytest -n auto --reruns 2` |
| Clear JIT cache | `rm -rf ~/.cache/flashinfer/` |
| Set target arch | `export FLASHINFER_ROCM_ARCH_LIST="gfx942,gfx950"` |
| Limit parallel build | `export MAX_JOBS=4` |
| Verbose JIT output | `export FLASHINFER_JIT_VERBOSE=1` |
| Run linting | `pre-commit run -a` |

## Installing Torch

Torch must come from AMD's ROCm repo via `--index-url` (not `-f`, which can
silently install a CPU-only wheel from PyPI). See the
[GPU, ROCm, and PyTorch Support](README.md#gpu-rocm-and-pytorch-support) table
in `README.md` for the version and command.

## Non-Obvious Gotchas

**JIT build.ninja caching**: `JitSpec.build()` only writes `build.ninja` when
the file is missing. Changing env vars (`FLASHINFER_ROCM_ARCH_LIST`, extra
cflags) is a **silent no-op** unless you call `spec.write_ninja()` first.

**`FLASHINFER_JIT_DEBUG=1` is a no-op on ROCm/HIP**: the env var is read in
[`flashinfer/jit/core.py`](flashinfer/jit/core.py) only on the `IS_CUDA` branch
(where it adds `-O0 -g -G`). The `IS_HIP` branch ignores it entirely. To get a
debug build on ROCm, append `"-O0", "-g"` via `extra_cuda_cflags` in the op's
JIT generator (the HIP path injects `-O3` before `extra_cuda_cflags`, so trailing
`-O0` is what actually overrides it on the hipcc command line) and clear
`~/.cache/flashinfer/`.

**Framework separation**: Torch headers **must not** be included in `include/`
files. `include/` is framework-agnostic (raw pointers only);
`flashinfer/csrc_rocm/` is where PyTorch tensor handling lives. Violations
cause subtle build failures.

**Test parallelism**: `pytest -n auto` automatically halves the physical GPU
count to avoid HSA/HIPBLAS flakiness under concurrent load. The `slow` marker
gates 1M-trial sampling and 4 GB tensor tests — exclude with `-m "not slow"`
for fast iteration.

**AITER is a separate install**: The AITER backend (used by prefill attention
on gfx942) is not bundled. Install from source:

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter && python3 setup.py develop
```

Check availability in code: `from flashinfer.aiter_utils import is_aiter_supported`

## Arch ↔ codename

MI300X / MI325X = gfx942 = CDNA3; MI355X = gfx950 = CDNA4.

External tuning references (CK, AITER, HipKittens) live in the
`benchmark-kernel` skill; PR/`gh` workflow details live in the `pr-workflow`
skill; the interactive plan-review workflow lives in the `plan-review` skill.

## Model Usage Policy

- **Plan Mode** (`/plan` or Shift+Tab): Always use `opus` at max effort for architecture decisions, system design, and multi-file analysis.
- **Agent / Agentic tasks**: Use `opus` at high effort when running multi-step autonomous tasks (bash commands, multi-file edits).
- **Edit / Implementation**: Use `sonnet` for routine code implementation, test writing, and refactoring once a plan is approved.
- **Quick tasks**: Use `sonnet` for Q&A, simple lookups, and documentation.

Switch model before starting each phase:
  /model opus     → for planning and agent runs
  /model sonnet   → for implementation and quick tasks
