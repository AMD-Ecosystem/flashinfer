# Contributing to FlashInfer+ROCm

This is the **AMD ROCm port** of FlashInfer (`amd-flashinfer`), targeting
AMD Instinct GPUs — gfx942 (MI300X / MI325X, CDNA3) and gfx950
(MI355X, CDNA4). The upstream CUDA repo at
<https://github.com/flashinfer-ai/flashinfer> uses different env vars,
paths, and toolchains; its contribution guide does not transfer here.

For the project overview, quick-start install, and the support matrix,
see [`README.md`](README.md); for backend routing and AITER setup, see
[`docs/rocm/backends.md`](docs/rocm/backends.md). This document covers
building from source and everything else specific to contributing code.

# Setting up a Development Environment

Build the development image with the repository's Dockerfile:

```bash
docker build -t flashinfer-dev:rocm10.0 -f docker/Dockerfile.rocm .
```

`ROCM_VERSION`, `UBUNTU_VERSION`, `PY_VERSION`, and `TORCH_VERSION` default to
10.0, 24.04, 3.12, and 2.12.0. They select the `rocm/pytorch` base image tag,
so they are not independent knobs — any override has to name a tag that exists
on Docker Hub. `AITER_VERSION` and `AITER_INDEX` pin the AITER wheel; every
0.1.20 build is cp312 only, so the interpreter and the AITER pin move together.
Do not raise `TORCH_VERSION` to 2.13 — it drops a `c10` symbol AITER's prebuilt
prefill kernels link against, and they fail to load.
Pass `--build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg
USER_GID=$(id -g)` to match container file ownership to your host user —
without them, build artifacts come out root-owned.

```bash
docker run -it \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --privileged --ipc=host --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  -v $PWD:/workspace \
  --name flashinfer-dev-container \
  flashinfer-dev:rocm10.0
```

`render` must be the **host's numeric GID**. Passing the name resolves against
the image's own `render` group, whose GID is assigned at build time and will
not match `/dev/dri/renderD*` — leaving the container user unable to open the
device, which surfaces as "No CUDA GPUs are available".

# Building and Installing

**Editable install** — what you want for day-to-day work:

```bash
python -m pip install --no-build-isolation -ve .
```

**Wheel:**

```bash
python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
pip install dist/amd_flashinfer-*.whl
```

`--no-deps` assumes dependencies are already present; omit it to resolve
them during the build. Nothing is compiled at build time — kernels are
JIT-compiled on first use and cached under `~/.cache/flashinfer/`.

**Ahead-of-time kernel build.** AOT is a separate step, not a flag on the
wheel build. It compiles the kernel set up front so the first call does
not pay for JIT:

```bash
python -m flashinfer.rocm.aot --help
```

Pair an AOT-built install with `FLASHINFER_DISABLE_JIT=1` to fail loudly
on a missing kernel instead of silently triggering a build.

# Running Tests

```bash
# Fast path -- skips the multi-GB speculative-sampling cases:
pytest -n auto --reruns 2 -m "not slow"

# Full coverage:
pytest -n auto --reruns 2

# Slow tests only -- -n 1 on purpose, that is what the marker means:
pytest -n 1 --reruns 2 -m "slow"

# One file, or a pattern:
pytest tests/rocm/test_batch_decode_kernels.py
pytest -k "test_batch_decode_kernels"
```

`testpaths` in [pyproject.toml](pyproject.toml) sets the default
selection. `pytest-rerunfailures` comes from the `dev` extra
(`pip install -e ".[dev]"`).

**Warm the JIT cache before you time anything.** On gfx950 at `a85540294`,
one MI350X and `-n 1`, the full suite took **118 min cold and 10 min warm**:
92% of a cold run is compilation, not test execution. A first run against an
empty `~/.cache/flashinfer` looks like a hung suite — there is no
`pytest-timeout` here, and a single AITER CK build can sit for tens of
minutes with no output. Expect less wall time with `-n auto` on a
multi-card host.

After this change `-m "not slow"` is within ~10 s of the full run — the
marker gates footprint, not time. That figure is the post-change split;
at `a85540294` the fast lane also excluded the 343 cases now unmarked.

**Worker count.** `pytest -n auto` spawns **half as many xdist workers as
physical AMD cards** (4 workers on a CPX-mode 8-card host) and assigns each
a card via `HIP_VISIBLE_DEVICES`, which is what scopes the subprocesses a
test spawns. One worker per card was tried first
and produced sporadic failures across rope, single_prefill, and logits_cap
under concurrent load. Pass an explicit `-n N` to override the halving.

**Reruns.** `--reruns 2` absorbs the residual ~0.01% of transient HIP
runtime crashes — HSA exceptions, HIPBLAS handle-pool exhaustion,
intermittent generator non-determinism — that worker pinning cannot
eliminate. Only failed tests are retried.

**`slow` marker.** Registered in [pyproject.toml](pyproject.toml). It tags
one thing: `test_chain_speculative_sampling`, whose worst case holds four
probability tensors live at once (~15 GB) and so cannot share a box with
other workers under `-n auto`.

**Nothing runs `-m slow`** — no CI job, no script — and `-m "not slow"` is
also what the coverage recipe below uses. Marking a test therefore retires
it rather than deferring it, so footprint is the only admissible reason.
Runtime never is: the whole suite is ~10 min warm, and case count is
actively misleading — `test_rope.py` is the largest matrix at 13,080 cases
and costs 1.2 ms each.

**HIPBLAS retry.** The reference attention helper in
`tests/attention_reference.py` wraps `torch.matmul` in a
`_hipblas_safe_matmul` retry that catches `HIPBLAS_STATUS_ALLOC_FAILED`
and backs off — needed under heavy concurrent xdist load.

# Measuring Coverage of the Port

`scripts/amd_coverage.py` reports line coverage for the code this fork added or changed, rather than for the whole inherited upstream tree. Ownership is recomputed on every run from the diff against the upstream release the port is based on — read off this fork's own tag, so `v0.5.3+amd.2` scores against upstream `v0.5.3`. There is no list to refresh, and a module added yesterday is picked up today. `origin` carries upstream's release tags, so no second remote is needed; override the base with `--upstream-ref` if you need a different one.

```bash
git fetch --tags origin                                   # the base is computed, not stored
pip install -e ".[dev]"                                   # brings pytest-cov
python3 scripts/amd_coverage.py --run --show-files -- -n auto --reruns 2 -m "not slow"
python3 scripts/amd_coverage.py                           # re-score an existing .coverage
```

**What gets counted.** Files we added are scored whole. Upstream files we merely edited are scored **only on the lines our diff touched**, so upstream's untested code neither flatters nor penalises the number. A third tier covers files with a zero-line Python diff whose implementation is ours anyway through the `FLASHINFER_CSRC_DIR` redirect — `sampling.py` and friends — which no diff can discover; they are declared in `scripts/coverage_ownership.toml`, one entry per file with a reason.

**What is deliberately left out, and why the report says so.** Lines inside `if IS_CUDA:` are excluded and counted in the output: the port re-indented upstream code under those guards, so git attributes it to us even though no ROCm box can execute it — in `flashinfer/jit/env.py` that is about half the owned lines. Lines that run at `import flashinfer` are reported as their own bucket rather than in the headline, because `tests/conftest.py` imports the package at collection and would otherwise credit every module-level statement before a test body runs. C++ under `csrc/rocm/` is JIT-compiled and has no line data at all; instead the report counts how many of its translation units a run actually built and loaded, labelled as reach, not coverage.

**Nothing is committed, on purpose.** A snapshot of the last run used to live in the repository; it recorded no HEAD sha, so a merge invalidated its per-file line numbers with nothing to detect that, and no tooling ever read it. Measure on demand instead — the run takes about 75 minutes under `--cov` instrumentation, against the ~10 min warm measured above — and use `--fail-under` if you want a threshold enforced.

**Say what the AITER lib cache was.** `jit/rocm/aiter_source.py`'s build path executes only when `ensure_aiter_lib` actually builds, so that file measures **32 lines higher** on a cold `~/.cache/flashinfer` than on a warm one — 120/124 against 88/124, observed across two runs on gfx942 on 2026-08-28. Neither is wrong, but two numbers only compare if the cache policy matched, and roughly 0.9 points of the headline turns on it — so state which you ran. (The two runs differed by other commits as well, so they do not isolate the effect at the headline level — the per-file figure is what was measured.)

**There is no GPU CI**, so nobody produces this number for you — every `.github/workflows` job runs on `ubuntu-latest` or a CPU self-hosted runner, and the `Jenkinsfile` is upstream's unused CUDA pipeline. Run it on a GPU box and say which architecture it came from: `arch_caps.py` gates behaviour per gfx942/gfx950, so a single-arch run leaves the other's branches unexecuted. The honest union is `coverage combine` across both, which needs `[tool.coverage.paths]` **and** must run from a checkout containing the sources — coverage only aliases onto a path that exists on disk, and is otherwise a silent no-op that halves the number.

**Running it in a container against a worktree** needs the *parent* repository mounted, not just the worktree: a linked worktree's `.git` is a file pointing into the main checkout, so git — and therefore the whole classifier — fails without it. Mount the repo root at its host path and set `git config --global --add safe.directory '*'` inside the container.

The classifier itself is covered by `tests/rocm/test_amd_coverage.py`, which needs no GPU and runs in the CPU conformance workflow.

# Code Structure

```text
flashinfer/
├── include/                  # framework-agnostic kernel headers (raw pointers only)
│   └── flashinfer/           # FlashInfer kernel implementations
│       └── rocm/             # fork-owned headers, incl. the HIP intrinsics
├── csrc/                     # upstream CUDA op registration (PyTorch bindings)
│   └── rocm/                 # HIP op registration — fork-owned, upstream has nothing here
├── flashinfer/
│   ├── jit/                  # Python JIT compilation infra (jit/rocm/ holds the HIP entry points)
│   └── *.py                  # Python user-facing API (e.g. attention.py, mla_rocm.py)
├── tests/rocm/         # HIP test suite (test_*.py)
├── benchmarks/rocm/  # ROCm-specific benchmarks
└── 3rdparty/                 # vendored dependencies (cutlass, composable_kernel, …)
```

**Framework separation.** `include/` files must remain framework-agnostic
— no PyTorch headers, raw pointers only. PyTorch tensor handling for HIP
ops lives in `csrc/rocm/`. Violating this causes subtle build
failures because the same headers are pulled into the JIT compilation
pipeline that has no PyTorch on its include path.

**`csrc/` vs `csrc/rocm/`.** `csrc/` is the upstream CUDA op registration
tree — keep it in sync with upstream where possible to reduce merge
conflicts. New HIP-specific op bindings go in `csrc/rocm/`, which upstream
owns no files in. Add an `_aiter` suffix when the file routes to AITER;
inside `rocm/` a `_hip` suffix is redundant.

`csrc/rocm/` is the *source*. `build_backend_rocm.py` materializes it as
`flashinfer/csrc/rocm` (a symlink for editable installs, a copy in the
wheel), which is what `FLASHINFER_CSRC_DIR` resolves to at runtime — so the
path is generated and gitignored, not somewhere to add files.

**HIP intrinsics.** `math.h`, `mma.h`, `memory_ops.h` and `vec_dtypes.h` under
`include/flashinfer/rocm/` wrap the HIP intrinsics. These once
sat behind a `gpu_iface` abstraction spanning CUDA too; that half is gone, so a
non-HIP compiler now gets an `#error` from `macros.hpp`. Put a new intrinsic in
the matching header if it is a general primitive. A kernel that needs
`hipcub` or one inline-asm builtin inline is fine — several under
`rocm/attention/` do — but anything a second kernel would want belongs in the
shared header.

Symbols live in `flashinfer::`, grouped by what they do — `flashinfer::math`
matches upstream's name, `flashinfer::memory` is ours (upstream calls the same
area `cp_async`). A fork header keeps its upstream namesake's symbol names and
carries an `#error` tripwire on upstream's include guard, so pulling both into
one translation unit is a compile error rather than a silent shadow.

The tripwire fires only when the upstream header is included **first**. The
reverse order is caught by the compiler as a redefinition instead. Do not try
to close that gap by defining upstream's guard here: that suppresses upstream's
body entirely and reintroduces the silent shadow the tripwire exists to stop.

# Additive-Only: the rule that keeps upstream syncs cheap

This is a **downstream fork** that syncs from
`flashinfer-ai/flashinfer` by merge. The cost of every sync is decided by one
number: how many upstream files the port has edited in place. Additions —
however large — are close to free at merge time, because upstream has nothing
to merge them against. The exception is a path upstream later adds too: that
conflicts as add/add, with no common ancestor to help resolve it, which is how
`CLAUDE.md` and `.claude/skills/benchmark-kernel/SKILL.md` got onto the
conflict list. For anything upstream might plausibly create, prefer a
`rocm/` subdirectory over a sibling file: `csrc/rocm/` and
`include/flashinfer/rocm/` collide with nothing even as upstream grows those
trees. A `_rocm`/`_aiter` suffix is the fallback only where a subdirectory does
not fit — inside a `rocm/` directory the suffix is redundant, so files there
carry the upstream name.

**So: add files, don't edit them.** Concretely, prefer in this order:

1. **Source-path redirect.** `FLASHINFER_CSRC_DIR` already points at
   `flashinfer/csrc/rocm/` on ROCm (see `flashinfer/jit/env.py` and
   `flashinfer/get_include_paths.py`), so a shared JIT generator naming
   `sampling.cu` picks up the HIP source with **zero Python diff**. This is why
   `flashinfer/sampling.py` and `flashinfer/quantization.py` contain no HIP
   references at all. Use it whenever only the kernel differs.
2. **A ROCm module beside the upstream one**, swapped into `sys.modules` in
   `flashinfer/__init__.py` — how `rocm/prefill.py` and `rocm/decode.py` are
   reached.
3. **A declared-unsupported gate**, so an op the port does not implement raises
   a named error rather than `AttributeError`. See the meta-path loader in
   `flashinfer/comm/__init__.py`.

**Last resort: an `if IS_HIP:` branch inside an upstream file.** Every one of
these is a recurring merge conflict, so it needs a reason that the three
mechanisms above cannot serve. `scripts/upstream_canary.py` is the authoritative
list — it reports exactly which files a merge would conflict on. The ownership
tiers that `scripts/amd_coverage.py` reports are *not* that list: they cover
`flashinfer/**/*.py` plus the build backend and `scripts/`, they count ROCm-only
additions such as `rocm/prefill.py` that are not edits to anything, and an
in-place edit under `csrc/` or `include/` never appears there at all.

**Forked headers are exempt from conflicts and therefore from warnings.**
Much of `include/flashinfer/rocm/` is a fork of an upstream header — the
`rocm/attention/` set, plus `sampling.cuh`, `quantization.cuh`, `layout.cuh`,
`fastdiv.cuh`, `utils.cuh` and `exception.h`. The HIP intrinsic headers and their
type headers have no upstream counterpart, so `upstream_canary.py` reports them
as orphans rather than pairing them. Their upstream
originals are byte-identical to the merge base and will merge cleanly forever,
so a fix landing upstream reaches the original and *not* the fork, with nothing
conflicting to tell you. The canary's drift report is the only signal, and a fix
to sampling or quantization belongs in the `rocm/` copy — the one ROCm
actually compiles.

**Check the cost before and after your change:**

```bash
git fetch upstream
python3 scripts/upstream_canary.py          # conflicts, ranked; plus forked-header drift
```

It builds nothing, needs no GPU, and leaves your working tree and index
untouched — the merge is resolved via `git merge-tree` into the object
database. (It does leave unreferenced loose objects behind, so a long-lived CI
checkout wants a periodic `git gc`.) A conflicting result is the normal steady
state; what matters is that your change does not lengthen the list.

## Syncing to a new upstream release

The fork's base is **recorded in `upstream-base`, not inferred from ancestry**.
Sync PRs are squash-merged like every other PR, so there is no merge parent for
`git merge-base` to find — anchoring on our own tip walks back to the *previous*
fork point and misreports the whole delta.

That makes `git merge v0.6.19` from `amd-integration` wrong: it resolves against
that stale base. Produce the merge from the recorded base instead:

```bash
BASE=$(python3 -c "import sys; sys.path.insert(0, 'scripts'); import upstream_base as u; print(u.read_worktree('.').sha)")
git switch -c sync/upstream-v0.6.19
git merge-recursive "$BASE" -- HEAD v0.6.19   # explicit base; populates the index
```

Read it with the tool's own parser, not an `awk` one-liner: a malformed record
has to fail here the same way it fails in `amd_coverage.py`, not resolve to
something plausible.

Then resolve, commit, and in the same PR update `upstream-base` to
`v0.6.19 <sha>`. After it merges, push the matching lightweight tag so the base
commit stays fetchable — it is reachable from no branch:

```bash
git push origin "$(git rev-parse v0.6.19^{commit})":refs/tags/upstream-base/v0.6.19
```

That tag is the only thing the *tools* need on `origin`: they resolve the base
by **sha**, and `upstream_base._require_present` wants the commit in the object
store, not reachable from anything.

### Then record the ancestry, in its own PR

A squash-merged sync leaves upstream's commits unreachable from
`amd-integration`, so GitHub's fork banner counts back to the original fork
point — 1192 commits behind instead of 114. No tooling depends on that
reachability, but the banner is the first thing anyone reads.

```bash
git merge -s ours v0.6.19 -m "chore: record v0.6.19 as an ancestor of the fork"
```

`-s ours` leaves our tree byte-identical; the commit exists only for the second
parent. Two things make or break it:

- **Merge that PR with a merge commit. Squashing it flattens the second parent
  and silently discards the ancestry** — which is the whole point of the PR.
- Push the plain `v0.6.19` tag to `origin` as well, so the parent stays
  identifiable by name. `origin` carries `v0.5.3` and `v0.6.18` for this reason.

`required_linear_history` on `amd-integration` rejects the merge commit, so it
takes the `pull_request` bypass. That is the standing trade: one merge commit
per upstream release buys an honest banner, and every other PR still squashes.

# Adding a Kernel

1. **Kernel implementation** — framework-agnostic header(s) in
   `include/flashinfer/rocm/`, using the intrinsic headers there for any
   HIP-specific intrinsic.
2. **PyTorch binding** — register the op in `csrc/rocm/`.
   The only layer that may include Torch headers.
3. **JIT generator** — add the op's JIT spec in `flashinfer/jit/*.py`.
4. **Python interface** — expose the user-facing API in `flashinfer/*.py`.
5. **Tests** — `test_*.py` under `tests/rocm/`. Reuse the
   fixtures in `tests/rocm/conftest.py`.
6. **(Optional) Benchmark** — script under `benchmarks/rocm/`.
7. **Pre-commit** — `pre-commit run -a` before submitting.

A step-by-step Claude Code skill (`add-rocm-kernel`) walks through this
with concrete examples.

# Build / JIT Gotchas

**JIT cache silently sticky.** `JitSpec.build()` only writes
`build.ninja` when the file is missing, so changing env vars
(`FLASHINFER_ROCM_ARCH_LIST`, extra cflags) is a **silent no-op**
unless you either `rm -rf ~/.cache/flashinfer/` or call
`spec.write_ninja()` explicitly. When debugging build flags, always
clear the cache first.

**Debug builds.** `FLASHINFER_JIT_DEBUG=1` is a no-op on ROCm/HIP — it
only injects debug flags on the CUDA branch. To get a debug build on
ROCm, append `"-O0", "-g"` via `extra_cuda_cflags` in the op's JIT
generator (the HIP path injects `-O3` before `extra_cuda_cflags`, so
the trailing `-O0` is what actually overrides it on the hipcc command
line) and clear `~/.cache/flashinfer/`.

# Pre-Commit

```bash
pre-commit install   # one-time, installs the git hook
pre-commit run -a    # run on all files
```

CI rejects PRs that don't pass `pre-commit run -a`.

# Submitting Changes

Open PRs against the `amd-integration` branch of
[`AMD-Ecosystem/flashinfer`](https://github.com/AMD-Ecosystem/flashinfer) —
never against `flashinfer-ai/flashinfer`, which `gh pr create` will otherwise
pick as the default base. For PR
description conventions (sections, benchmarks, test plan), see the
"PR Description" section of the `pr-workflow` skill
(`.claude/skills/pr-workflow/SKILL.md`).
