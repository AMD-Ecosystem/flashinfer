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
docker build -t flashinfer-dev:rocm7.14 -f .devcontainer/rocm/Dockerfile .
```

`ROCM_VERSION`, `UBUNTU_VERSION`, `PY_VERSION`, and `TORCH_VERSION` default to
7.14, 26.04, 3.14, and 2.12.0. They select the `rocm/pytorch` base image tag,
so they are not independent knobs — any override has to name a tag that exists
on Docker Hub. `AITER_VERSION` and `AITER_INDEX` pin the AITER wheel; the
default is the vllm-cdna nightly, which is the only cp314 build published.
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
  flashinfer-dev:rocm7.14
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
python -m flashinfer.aot_hip --help
```

Pair an AOT-built install with `FLASHINFER_DISABLE_JIT=1` to fail loudly
on a missing kernel instead of silently triggering a build.

# Running Tests

```bash
# Fast path -- skips 1M-trial sampling-frequency tests and 4 GB
# speculative-sampling cases (~7 min on a CPX 8-card host):
pytest -n auto --reruns 2 -m "not slow"

# Full coverage (~20 min):
pytest -n auto --reruns 2

# Slow tests only (~13 min):
pytest -n auto --reruns 2 -m "slow"

# One file, or a pattern:
pytest tests/rocm_tests/test_batch_decode_kernels_hip.py
pytest -k "test_batch_decode_kernels_hip"
```

`testpaths` in [pyproject.toml](pyproject.toml) sets the default
selection. `pytest-rerunfailures` comes from the `dev` extra
(`pip install -e ".[dev]"`).

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
the 1M-trial sampling-frequency tests, the 4 GB-tensor speculative-sampling
cases, and the whole `TestLogitsPipeCompilationHIP` class (each test runs
the sampling kernel twice, for `compile=True` and `False`).

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

**What is deliberately left out, and why the report says so.** Lines inside `if IS_CUDA:` are excluded and counted in the output: the port re-indented upstream code under those guards, so git attributes it to us even though no ROCm box can execute it — in `flashinfer/jit/env.py` that is about half the owned lines. Lines that run at `import flashinfer` are reported as their own bucket rather than in the headline, because `tests/conftest.py` imports the package at collection and would otherwise credit every module-level statement before a test body runs. C++ under `csrc_rocm/` is JIT-compiled and has no line data at all; instead the report counts how many of its translation units a run actually built and loaded, labelled as reach, not coverage.

**There is no GPU CI**, so nobody produces this number for you — every `.github/workflows` job runs on `ubuntu-latest` or a CPU self-hosted runner, and the `Jenkinsfile` is upstream's unused CUDA pipeline. Run it on a GPU box and say which architecture it came from: `arch_caps.py` gates behaviour per gfx942/gfx950, so a single-arch run leaves the other's branches unexecuted. The honest union is `coverage combine` across both, which needs `[tool.coverage.paths]` **and** must run from a checkout containing the sources — coverage only aliases onto a path that exists on disk, and is otherwise a silent no-op that halves the number.

**Running it in a container against a worktree** needs the *parent* repository mounted, not just the worktree: a linked worktree's `.git` is a file pointing into the main checkout, so git — and therefore the whole classifier — fails without it. Mount the repo root at its host path and set `git config --global --add safe.directory '*'` inside the container.

The classifier itself is covered by `tests/rocm_tests/test_amd_coverage.py`, which needs no GPU and runs in the CPU conformance workflow.

# Code Structure

```text
flashinfer/
├── include/                  # framework-agnostic kernel headers (raw pointers only)
│   ├── flashinfer/           # FlashInfer kernel implementations
│   └── gpu_iface/backend/    # GPU abstraction layer — cuda/ and hip/ shims
├── csrc/                     # upstream CUDA op registration (PyTorch bindings)
├── flashinfer/
│   ├── csrc_rocm/            # HIP op registration (PyTorch bindings) — the ROCm analog of csrc/
│   ├── jit/                  # Python JIT compilation infra (cpp_ext_hip.py is the HIP entry)
│   └── *.py                  # Python user-facing API (e.g. attention.py, mla_rocm.py)
├── tests/rocm_tests/         # HIP test suite (test_*_hip.py)
├── benchmarks/rocm_benchmarks/  # ROCm-specific benchmarks
└── 3rdparty/                 # vendored dependencies (cutlass, composable_kernel, …)
```

**Framework separation.** `include/` files must remain framework-agnostic
— no PyTorch headers, raw pointers only. PyTorch tensor handling for HIP
ops lives in `flashinfer/csrc_rocm/`. Violating this causes subtle build
failures because the same headers are pulled into the JIT compilation
pipeline that has no PyTorch on its include path.

**`csrc/` vs `flashinfer/csrc_rocm/`.** `csrc/` is the upstream CUDA op
registration tree — keep it in sync with upstream where possible to
reduce merge conflicts. New HIP-specific op bindings go in
`flashinfer/csrc_rocm/`, with a `_hip` or `_aiter` suffix when the file
routes to a HIP-specific code path or to AITER.

**`include/gpu_iface/`.** Hides CUDA/HIP divergence behind a common
header surface (`math_ops.hpp`, `mma_ops.hpp`, `memory_ops.hpp`, …).
When you need a new intrinsic, add the abstraction in `gpu_iface/` and
provide a HIP implementation under `gpu_iface/backend/hip/`. Don't
reach for `hipcub`, `__hip_*`, or inline asm from inside
`include/flashinfer/` — go through `gpu_iface`.

# Additive-Only: the rule that keeps upstream syncs cheap

This is a **downstream fork** that syncs from
`flashinfer-ai/flashinfer` by merge. The cost of every sync is decided by one
number: how many upstream files the port has edited in place. Additions —
however large — are close to free at merge time, because upstream has nothing
to merge them against. The exception is a path upstream later adds too: that
conflicts as add/add, with no common ancestor to help resolve it, which is how
`CLAUDE.md` and `.claude/skills/benchmark-kernel/SKILL.md` got onto the
conflict list. Prefer a `_rocm`/`_aiter`-suffixed name for anything upstream
might plausibly create.

**So: add files, don't edit them.** Concretely, prefer in this order:

1. **Source-path redirect.** `FLASHINFER_CSRC_DIR` already points at
   `flashinfer/csrc_rocm/` on ROCm (see `flashinfer/jit/env.py` and
   `flashinfer/get_include_paths.py`), so a shared JIT generator naming
   `sampling.cu` picks up the HIP source with **zero Python diff**. This is why
   `flashinfer/sampling.py` and `flashinfer/quantization.py` contain no HIP
   references at all. Use it whenever only the kernel differs.
2. **A ROCm module beside the upstream one**, swapped into `sys.modules` in
   `flashinfer/__init__.py` — how `prefill_rocm.py` and `decode_rocm.py` are
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
additions such as `prefill_rocm.py` that are not edits to anything, and an
in-place edit under `csrc/` or `include/` never appears there at all.

**Forked headers are exempt from conflicts and therefore from warnings.**
Everything under `include/flashinfer/rocm/` is a fork of an upstream header
re-expressed on `gpu_iface` — `rocm/attention/` for the attention headers,
plus `rocm/sampling.cuh` and `rocm/quantization.cuh`. Their upstream
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

# Adding a Kernel

1. **Kernel implementation** — framework-agnostic header(s) in
   `include/flashinfer/`, using `gpu_iface/` for any CUDA/HIP-divergent
   intrinsic.
2. **PyTorch binding** — register the op in `flashinfer/csrc_rocm/`.
   The only layer that may include Torch headers.
3. **JIT generator** — add the op's JIT spec in `flashinfer/jit/*.py`.
4. **Python interface** — expose the user-facing API in `flashinfer/*.py`.
5. **Tests** — `test_*_hip.py` under `tests/rocm_tests/`. Reuse the
   fixtures in `tests/rocm_tests/conftest.py`.
6. **(Optional) Benchmark** — script under `benchmarks/rocm_benchmarks/`.
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
