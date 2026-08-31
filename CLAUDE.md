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

A fresh worktree is source-only. Recreate both generated trees as **relative**
symlinks and copy `flashinfer/_version.py` from the main checkout, or the JIT
will not build:

```bash
rm -rf flashinfer/include flashinfer/csrc     # ln -f will not clear a real dir
ln -s ../include flashinfer/include
mkdir -p flashinfer/csrc && ln -s ../../csrc/rocm flashinfer/csrc/rocm
cp <main-checkout>/flashinfer/_version.py flashinfer/_version.py
```

`_version.py` only exists in the main checkout once it has been installed or
built. Details: `pr-workflow` skill.

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
| Extra JIT link flags | `export FLASHINFER_EXTRA_LDFLAGS="-L/path -lfoo"` |
| Extra host compile flags | `export FLASHINFER_EXTRA_CFLAGS="-DFOO"` |
| Extra HIP compile flags | `export FLASHINFER_EXTRA_CUDAFLAGS="-DBAR"` |
| Warn on our own headers | `export FLASHINFER_OWN_HEADERS_NON_SYSTEM=1` |
| Run linting | `pre-commit run -a` |

## Installing Torch

Torch must come from AMD's ROCm repo at `repo.radeon.com`, via `-f`
(`--find-links`) and **not** `--index-url` — that repo is a flat wheel listing
rather than a PEP 503 index, so `--index-url` fails with "No matching
distribution found".

```bash
pip install torch==2.9.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
```

`-f` also lets pip see PyPI, so verify you did not get the CPU-only wheel:
`python -c "import torch; assert torch.version.hip, 'not a ROCm build'"`.
In practice the ROCm wheel wins anyway, because its `+rocm<X.Y>` local version
ranks higher than a same-version PyPI wheel. Supported versions are in the
[Supported hardware and toolchain](README.md#supported-hardware-and-toolchain)
table in `README.md`.

Neither ROCm 7.14 nor 10.0 has a `rocm-rel-` directory on `repo.radeon.com`,
so there is no pip recipe for either. The devcontainer gets torch 2.12 from
its `rocm/pytorch:rocm10.0_*` base image instead. Torch 2.13 is not usable
here: it removes `c10::impl::cow::materialize_cow_storage`, which every
published `amd-aiter` build's prebuilt prefill kernels link against.

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
`csrc/rocm/` is where PyTorch tensor handling lives. Violations
cause subtle build failures.

**Test parallelism**: `pytest -n auto` automatically halves the physical GPU
count to avoid HSA/HIPBLAS flakiness under concurrent load. The `slow` marker
gates 1M-trial sampling and 4 GB tensor tests — exclude with `-m "not slow"`
for fast iteration.

**AITER is version-pinned, and the pin depends on the channel**: the
devcontainer bundles the wheel, so no separate install is needed there. CI and
the devcontainer differ because the two channels carry different builds.

On CPython 3.12 (the CI image, ROCm 7.1.1):

```bash
pip install amd-aiter==0.1.10 --extra-index-url https://pypi.amd.com/rocm-7.1.1/simple
```

`amd-aiter` is **not** on the top-level `pypi.amd.com/simple` index, and it
must be `--extra-index-url` rather than `--index-url` so AITER's own
dependencies still resolve from PyPI. **Only cp310 and cp312 wheels exist** on
that channel (verified 2026-08-20); on 3.11, 3.13 or 3.14 it fails with
`No matching distribution found`, and public PyPI tops out at a stale
`0.1.7.post2.dev18`.

The devcontainer runs CPython 3.12 and takes 0.1.20 from the nightlies index.
Every 0.1.20 build is cp312 only, and none targets ROCm 10.0 — they are one
source revision (build id `3135022`) retargeted to `+rocm10.1.0a`,
`+rocm7.14.0` and `+rocm7.2.3`:

```bash
pip install amd-aiter==0.1.20+rocm10.1.0a20260819.3135022 \
  --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl-multi-arch/
```

Spell the version out in full including the local `+rocm...` segment; pip will
not select a local version from a loose specifier. pip normalises the project
name to `amd-aiter/`, so the sibling `amd_aiter/` directory on that index —
which carries the other retargets — needs a direct wheel URL.
`prefill_rocm.py` records whatever is validated as `_AITER_LAST_VALIDATED`, and
[`docs/rocm/backends.md`](docs/rocm/backends.md) explains the index choice.
`aiter_utils.AITER_MIN_VERSION` is the hard floor below which the vendored
struct layouts stop matching.

A source build (`git clone --recursive https://github.com/ROCm/aiter.git &&
cd aiter && python3 setup.py develop`) tracks master, which is **many releases
ahead of the pin with a different C ABI** — symbols the shim expects are
renamed, hidden rather than `extern "C"`, or absent. Nothing stops you running
one, but treat it as untested here.

That gap is a trap when working on the C++ shim: read the **installed** tree,
never a source checkout, before designing against an AITER symbol.

```bash
nm -D --defined-only -C <site-packages>/aiter/jit/module_<x>.so | grep '<fn>('
sed -n '/<fn>/,/;/p' <site-packages>/aiter_meta/csrc/include/<hdr>.h
```

Check availability in code: `from flashinfer.rocm.aiter_utils import is_aiter_supported`

## Arch ↔ codename

MI300X / MI325X = gfx942 = CDNA3; MI350X / MI355X = gfx950 = CDNA4.

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
