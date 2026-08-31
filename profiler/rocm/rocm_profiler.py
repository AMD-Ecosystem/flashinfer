"""
Copyright (c) 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

rocm_profiler.py — Generic ROCm kernel profiler with automated rocprofv3 subprocess.

Pipeline: timing (repeated GPU launches → median) → rocprofv3 counter collection →
roofline PNG.  Import in a driver script:

    from rocm_profiler import RocmProfiler, KernelConfig

    profiler = RocmProfiler(
        configs=[KernelConfig(name="s1024", run_fn=lambda: kernel(...),
                              theoretical_flops=..., theoretical_bytes=...)],
        counters="roofline",          # preset name or path to rocprofv3 YAML
        kernel_name_regex="SinglePrefill",
        output_dir="benchmarks/rocm",
        label="fa2",
    )
    if __name__ == "__main__":
        profiler.run()

CLI flags accepted by all driver scripts:
    --timing-only    Skip rocprofv3 and roofline
    --skip-roofline  Profile but skip plot
    --replot         Regenerate plot from existing CSVs (no GPU)
    --list-presets   Print counter preset names and exit
    --num-warmup N   Warmup launches per config in profile mode
    --dry-run-ms N   Timing dry-run window (ms)
    --repeat-ms N    Timing measurement window (ms)

rocprofv3 behaviour: replays the driver script once per PMC pass with
_ROCM_PROFILER_INTERNAL=1; each replay runs only the profile protocol and skips
timing and subprocess spawning.  rocprofv3 is invoked once per config (not once
for the whole sweep) to avoid ring-buffer overflow (~8 dispatches retained).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Counter presets (YAML, rocprofv3 native format)
# ---------------------------------------------------------------------------

_COUNTER_PRESETS: dict[str, str] = {
    # Two-pass roofline preset for gfx942 (MI300X).
    # Hardware constraints discovered empirically:
    #   - MemUnitBusy does not exist on gfx942
    #   - FetchSize and WriteSize CANNOT be in the same pass (same hw resource)
    #   - SQ_INSTS_VALU cannot be combined with FetchSize/WriteSize
    # Valid grouping verified with rocprofv3 error-code-38 testing:
    "roofline": """\
jobs:
  # Pass 1: L2 read traffic + MFMA throughput + HBM reads
  - pmc: [FetchSize, SQ_INSTS_VALU_MFMA_MOPS_F16, SQ_INSTS_MFMA, TCC_EA0_RDREQ_DRAM_sum]
  # Pass 2: L2 write traffic + LDS conflicts + HBM writes + MFMA duty-cycle cycles
  - pmc: [WriteSize, SQ_LDS_BANK_CONFLICT, TCC_EA0_WRREQ_DRAM_sum, SQ_VALU_MFMA_BUSY_CYCLES]
""",
    # Single-pass MFMA compute counters only
    "compute": """\
jobs:
  - pmc: [SQ_INSTS_VALU_MFMA_MOPS_F16, SQ_VALU_MFMA_BUSY_CYCLES, SQ_INSTS_VALU_MFMA_F16, SQ_INSTS_MFMA]
""",
    # Two-pass bandwidth and HBM counters
    # FetchSize and WriteSize must be in separate passes on gfx942 (error code 38)
    "memory": """\
jobs:
  - pmc: [FetchSize, SQ_INSTS_VALU, TCC_EA0_RDREQ_DRAM_sum, TCC_EA0_RDREQ_sum]
  - pmc: [WriteSize, SQ_LDS_BANK_CONFLICT, TCC_EA0_WRREQ_DRAM_sum, TCC_EA0_WRREQ_64B_sum]
""",
    # Two-pass minimal set — FetchSize and WriteSize split per gfx942 hw constraint
    "basic": """\
jobs:
  - pmc: [FetchSize, SQ_INSTS_VALU_MFMA_MOPS_F16]
  - pmc: [WriteSize]
""",
    # Two-pass occupancy preset — confirms the LDS-driven occupancy collapse
    # hypothesis and measures the true MFMA pipeline duty cycle.
    #
    # Derived metrics:
    #   SQ_VALU_MFMA_BUSY_CYCLES / SQ_BUSY_CYCLES  → MFMA duty cycle
    #   SQ_WAIT_INST_ANY / SQ_BUSY_CYCLES           → stalled fraction (barriers + waits)
    #   SQ_WAVES / grid_blocks                      → wavefronts per block (expect 4)
    "occupancy": """\
jobs:
  # Pass 1: wave count + GPU busy cycles + MFMA-active cycles + L2 reads
  - pmc: [SQ_WAVES, SQ_BUSY_CYCLES, SQ_VALU_MFMA_BUSY_CYCLES, FetchSize]
  # Pass 2: stall-all cycles + LDS instruction count + L2 writes
  - pmc: [SQ_WAIT_INST_ANY, SQ_INSTS_LDS, WriteSize]
""",
    # Two-pass stall-source breakdown — identifies whether pipeline stalls
    # originate from LDS latency, VMEM (HBM/L2) latency, or issue bubbles.
    #
    # Derived metrics:
    #   SQ_WAIT_INST_LDS  / SQ_BUSY_CYCLES  → LDS-not-ready stall fraction
    #   SQ_WAIT_INST_VMEM / SQ_BUSY_CYCLES  → VMEM-not-ready stall fraction
    "stall": """\
jobs:
  # Pass 1: MFMA instruction count + VMEM stall cycles + MFMA pipe busy + L2 reads
  - pmc: [SQ_INSTS_MFMA, SQ_WAIT_INST_VMEM, SQ_VALU_MFMA_BUSY_CYCLES, FetchSize]
  # Pass 2: LDS stall cycles + total busy cycles + L2 writes
  - pmc: [SQ_WAIT_INST_LDS, SQ_BUSY_CYCLES, WriteSize]
""",
    # -----------------------------------------------------------------------
    # Template for custom tuning presets — copy, rename, and modify.
    # Rules for gfx942 (MI300X):
    #   1. FetchSize and WriteSize must be in SEPARATE passes.
    #   2. Max ~4 counters per pass before risking error code 38.
    #   3. See https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/
    #      mi300-mi200-performance-counters.html for available counters.
    #
    # Example: occupancy + stall analysis alongside L2 bandwidth
    # "my_preset": """\
    # jobs:
    #   # Pass 1: L2 reads + MFMA ops + wave occupancy + GPU cycles
    #   - pmc: [FetchSize, SQ_INSTS_VALU_MFMA_MOPS_F16, SQ_WAVES, SQ_BUSY_CYCLES]
    #   # Pass 2: L2 writes + LDS conflicts + wait cycles
    #   - pmc: [WriteSize, SQ_LDS_BANK_CONFLICT, SQ_WAIT_INST_ANY]
    # """,
    # -----------------------------------------------------------------------
}

# ---------------------------------------------------------------------------
# Hardware ceiling lookup table
# ---------------------------------------------------------------------------


@dataclass
class HardwareCeilings:
    peak_tflops_fp16: float
    peak_bw_tbs: float
    gpu_name: str = ""

    @property
    def ridge_flops_per_byte(self) -> float:
        return self.peak_tflops_fp16 / self.peak_bw_tbs


# Peak FP16 matrix TFLOPS (dense) and HBM bandwidth. An arch key cannot separate
# SKUs sharing it, so each entry is its arch's slower member -- utilisation
# against these ceilings is an upper bound on the faster part.
KNOWN_HW: dict[str, HardwareCeilings] = {
    # gfx950 covers both MI350X (1 kW air) and MI355X (1.4 kW liquid); they
    # differ only in clock, so the lower MI350X matrix peak is used and a
    # roofline taken on an MI355X reads slightly optimistic.
    "gfx950": HardwareCeilings(2300.0, 8.0, "MI350X"),
    "gfx942": HardwareCeilings(1307.4, 5.3, "MI300X"),
    "gfx90a": HardwareCeilings(383.0, 3.277, "MI250X"),
    "gfx908": HardwareCeilings(185.0, 1.228, "MI100"),
}

# ---------------------------------------------------------------------------
# TensorSpec — convenience wrapper for auto-allocating tensors
# ---------------------------------------------------------------------------


@dataclass
class TensorSpec:
    """Descriptor for a single tensor argument to be allocated by the profiler."""

    shape: tuple
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    fill: Literal["randn", "zeros", "ones", "empty"] = "randn"

    def allocate(self) -> torch.Tensor:
        if self.fill == "randn":
            return torch.randn(self.shape, dtype=self.dtype, device=self.device)
        if self.fill == "zeros":
            return torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        if self.fill == "ones":
            return torch.ones(self.shape, dtype=self.dtype, device=self.device)
        return torch.empty(self.shape, dtype=self.dtype, device=self.device)


# ---------------------------------------------------------------------------
# KernelConfig — one benchmark configuration in the sweep
# ---------------------------------------------------------------------------


@dataclass
class KernelConfig:
    """
    One entry in the profiling sweep.

    Provide either:
      run_fn: Callable[[], Any]  — zero-arg closure that runs the kernel
    or:
      kernel_fn: Callable  + arg_specs: list[TensorSpec]
        — profiler allocates tensors from specs and builds run_fn automatically.

    theoretical_flops and theoretical_bytes are used to compute arithmetic intensity
    and to annotate the roofline plot.
    """

    name: str
    theoretical_flops: int
    theoretical_bytes: int
    run_fn: Callable[[], Any] | None = None
    label: str = ""
    num_tokens: int = 0  # when > 0, profiler also reports tokens/sec
    # Called once before each profiling phase to (re-)allocate tensors.
    # run_fn should close over a mutable container that setup_fn refreshes.
    setup_fn: Callable[[], None] | None = None
    # Convenience: auto-builds run_fn from kernel_fn + arg_specs
    kernel_fn: Callable | None = None
    arg_specs: list[TensorSpec] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name
        if self.run_fn is None:
            if self.kernel_fn is None or self.arg_specs is None:
                raise ValueError(
                    f"KernelConfig '{self.name}': provide 'run_fn' or both "
                    "'kernel_fn' and 'arg_specs'."
                )
            tensors = [spec.allocate() for spec in self.arg_specs]
            kfn = self.kernel_fn
            self.run_fn = lambda: kfn(*tensors)


# ---------------------------------------------------------------------------
# Sentinel env variable for subprocess detection
# ---------------------------------------------------------------------------

_ENV_SENTINEL = "_ROCM_PROFILER_INTERNAL"
# When set (value = "1"), _profile_mode() skips Phase 2 and exits after JIT
# warmup only.  Used by _run_rocprofv3() to pre-populate the JIT cache under
# rocprofv3's modified LD environment without incurring counter-collection
# overhead on the warmup dispatches.
_ENV_PREWARM = "_ROCM_PROFILER_PREWARM"
# When set to a decimal integer, _profile_mode() profiles ONLY the config at
# that index with a single dispatch.  rocprofv3 is invoked once per config to
# avoid filling the hardware counter ring buffer with multi-config sweeps.
_ENV_CONFIG_IDX = "_ROCM_PROFILER_CONFIG_IDX"

# ---------------------------------------------------------------------------
# RocmProfiler
# ---------------------------------------------------------------------------


class RocmProfiler:
    """
    Generic ROCm kernel profiler with automated rocprofv3 subprocess.

    Args:
        configs:            List of KernelConfig objects (the profiling sweep).
        num_warmup:         Warmup launches per config in profile mode.
        dry_run_ms:         Duration of the timing dry-run window (ms).
        repeat_ms:          Duration of the timing measurement window (ms).
        counters:           Counter preset name ("roofline", "compute", "memory",
                            "basic") or path to a YAML/text counter file.
        kernel_name_regex:  Regex passed to rocprofv3 --kernel-include-regex so
                            only the target kernel's dispatches are collected.
        output_dir:         Directory for output CSVs and PNGs.
        label:              Prefix for output filenames (e.g. "fa2").
        roofline:           If True, generate the roofline PNG after profiling.
        hw_ceilings:        Hardware ceilings for roofline. None = auto-detect
                            via rocminfo.
    """

    def __init__(
        self,
        configs: list[KernelConfig],
        *,
        num_warmup: int = 3,
        dry_run_ms: float = 100,
        repeat_ms: float = 1000,
        counters: str = "roofline",
        kernel_name_regex: str = "",
        output_dir: str = ".",
        label: str = "profile",
        roofline: bool = True,
        hw_ceilings: HardwareCeilings | None = None,
        rocprofv3_timeout: int = 600,
    ) -> None:
        self.configs = configs
        self.num_warmup = num_warmup
        self.dry_run_ms = dry_run_ms
        self.repeat_ms = repeat_ms
        self.counters = counters
        self.kernel_name_regex = kernel_name_regex
        self.output_dir = Path(output_dir)
        self.label = label
        self.roofline = roofline
        self._rocprofv3_timeout = rocprofv3_timeout
        if hw_ceilings is not None:
            self.hw_ceilings: HardwareCeilings | None = hw_ceilings
        elif os.environ.get(_ENV_SENTINEL) or os.environ.get(_ENV_PREWARM):
            # Skip rocminfo subprocess when running inside rocprofv3 (LD_PRELOAD
            # active) — spawning rocminfo there can corrupt rocprofv3's HSA state.
            self.hw_ceilings = None
        else:
            self.hw_ceilings = _detect_hw_ceilings()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main entry point.

        When called from the outer driver script, runs the full pipeline:
        timing → rocprofv3 subprocess → roofline.

        When called from inside a rocprofv3 subprocess replay (_ENV_SENTINEL
        set in the environment), runs only the profile protocol and returns.
        """
        if os.environ.get(_ENV_PREWARM):
            self._jit_warmup_only()
            return

        if os.environ.get(_ENV_SENTINEL):
            self._profile_mode()
            return

        args = self._parse_args()

        if args.list_presets:
            _print_presets()
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if args.replot:
            # --replot reads everything from existing CSVs; configs not required.
            timing_csv = self.output_dir / f"{self.label}_timing.csv"
            counter_csv = self.output_dir / f"{self.label}_counter_collection.csv"
            for p, desc in [(timing_csv, "timing CSV"), (counter_csv, "counter CSV")]:
                if not p.exists():
                    print(f"ERROR: {desc} not found: {p}", file=sys.stderr)
                    sys.exit(1)
            self._plot_roofline(timing_csv, counter_csv)
            return

        if not self.configs:
            raise ValueError(
                "configs must be non-empty for timing/profiling. "
                "Pass --replot to regenerate a plot from existing CSVs without a GPU."
            )

        timing_csv = self._timing_mode()

        if args.timing_only:
            return

        counter_csv = self._run_rocprofv3()

        if counter_csv is not None and self.roofline and not args.skip_roofline:
            self._plot_roofline(timing_csv, counter_csv)

    # ── Argparse ──────────────────────────────────────────────────────────────

    def _parse_args(self) -> argparse.Namespace:
        p = argparse.ArgumentParser(
            description=f"rocm_profiler — {self.label}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=True,
        )
        p.add_argument(
            "--timing-only",
            action="store_true",
            help="Run timing only; skip rocprofv3 and roofline plot.",
        )
        p.add_argument(
            "--skip-roofline",
            action="store_true",
            help="Skip roofline plot after profiling.",
        )
        p.add_argument(
            "--replot",
            action="store_true",
            help=(
                "Regenerate roofline plot from existing "
                f"{self.label}_timing.csv and {self.label}_counter_collection.csv "
                "(no GPU required)."
            ),
        )
        p.add_argument(
            "--list-presets",
            action="store_true",
            help="Print available counter presets and exit.",
        )
        p.add_argument(
            "--num-warmup",
            type=int,
            default=None,
            metavar="N",
            help=f"Warmup launches per config in profile mode (default: {self.num_warmup}).",
        )
        p.add_argument(
            "--dry-run-ms",
            type=float,
            default=None,
            metavar="MS",
            help=f"Timing dry-run window in ms (default: {self.dry_run_ms}).",
        )
        p.add_argument(
            "--repeat-ms",
            type=float,
            default=None,
            metavar="MS",
            help=f"Timing measurement window in ms (default: {self.repeat_ms}).",
        )
        # parse_known_args so the driver script can have its own unrelated flags
        args, _ = p.parse_known_args()

        if args.num_warmup is not None:
            self.num_warmup = args.num_warmup
        if args.dry_run_ms is not None:
            self.dry_run_ms = args.dry_run_ms
        if args.repeat_ms is not None:
            self.repeat_ms = args.repeat_ms

        return args

    # ── Timing mode ───────────────────────────────────────────────────────────

    def _timing_mode(self) -> Path:
        """
        Repeated GPU launches for each config → timing statistics.
        Writes {label}_timing.csv and {label}_meta.json; returns the CSV path.
        """
        from flashinfer.testing.utils import bench_gpu_time

        out_path = self.output_dir / f"{self.label}_timing.csv"
        print(
            f"[rocm_profiler] Timing {len(self.configs)} configs "
            f"(dry_run={self.dry_run_ms:.0f} ms, repeat={self.repeat_ms:.0f} ms) "
            f"→ {out_path}"
        )

        rows = []
        for cfg in self.configs:
            if cfg.setup_fn:
                cfg.setup_fn()

            # One pre-check launch: verify the kernel produces finite output
            # before entering the measurement loop.
            result = cfg.run_fn()
            torch.cuda.synchronize()
            _check_finite(result, cfg.name)

            times_ms = bench_gpu_time(
                cfg.run_fn,
                dry_run_time_ms=int(self.dry_run_ms),
                repeat_time_ms=int(self.repeat_ms),
            )
            arr = np.asarray(times_ms, dtype=np.float64)
            median_ms = float(np.median(arr))
            min_ms = float(np.min(arr))
            p5_ms = float(np.percentile(arr, 5))
            p95_ms = float(np.percentile(arr, 95))
            std_ms = float(np.std(arr))
            n_iters = int(len(arr))
            tflops = cfg.theoretical_flops / median_ms / 1e9
            ai_theory = cfg.theoretical_flops / cfg.theoretical_bytes
            tput_tok_per_s = (
                cfg.num_tokens / (median_ms * 1e-3) if cfg.num_tokens > 0 else 0
            )
            row: dict[str, Any] = {
                "name": cfg.name,
                "theoretical_flops": cfg.theoretical_flops,
                "theoretical_bytes": cfg.theoretical_bytes,
                "n_iters": n_iters,
                "min_ms": f"{min_ms:.4f}",
                "p5_ms": f"{p5_ms:.4f}",
                "median_ms": f"{median_ms:.4f}",
                "p95_ms": f"{p95_ms:.4f}",
                "std_ms": f"{std_ms:.4f}",
                "tflops": f"{tflops:.3f}",
                "ai_theory": f"{ai_theory:.2f}",
            }
            row["tput_tok_per_s"] = (
                f"{tput_tok_per_s:.0f}" if cfg.num_tokens > 0 else ""
            )
            rows.append(row)
            tput_str = (
                f"  {tput_tok_per_s / 1e3:7.1f} ktok/s" if cfg.num_tokens > 0 else ""
            )
            print(
                f"  {cfg.label:40s}  {median_ms:8.3f} ms  "
                f"[p5={p5_ms:.3f} p95={p95_ms:.3f} std={std_ms:.3f}]  "
                f"{tflops:7.2f} TFLOPS  AI={ai_theory:.1f}{tput_str}"
            )
            sys.stdout.flush()

        _timing_fields = [
            "name",
            "theoretical_flops",
            "theoretical_bytes",
            "n_iters",
            "min_ms",
            "p5_ms",
            "median_ms",
            "p95_ms",
            "std_ms",
            "tflops",
            "ai_theory",
            "tput_tok_per_s",
        ]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_timing_fields)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[rocm_profiler] Timing done → {out_path}\n")

        self._write_meta(out_path)
        return out_path

    def _write_meta(self, timing_csv: Path) -> None:
        """Write a sidecar JSON with provenance metadata for this bench run."""
        import platform

        meta: dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "label": self.label,
            "timing_csv": str(timing_csv),
            "num_configs": len(self.configs),
        }

        # Git provenance
        try:
            meta["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, timeout=5
            ).strip()
        except Exception:
            meta["git_commit"] = None
        try:
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, timeout=5
            ).strip()
            meta["git_dirty"] = bool(dirty)
        except Exception:
            meta["git_dirty"] = None

        # Python / Torch / FlashInfer versions
        meta["python_version"] = platform.python_version()
        try:
            meta["torch_version"] = torch.__version__
        except Exception:
            meta["torch_version"] = None
        try:
            import flashinfer

            meta["flashinfer_version"] = getattr(flashinfer, "__version__", None)
        except Exception:
            meta["flashinfer_version"] = None

        # ROCm version
        for vpath in ["/opt/rocm/VERSION", "/opt/rocm/.info/version"]:
            try:
                meta["rocm_version"] = Path(vpath).read_text().strip()
                break
            except Exception:
                pass
        else:
            meta["rocm_version"] = None

        # GPU info
        if self.hw_ceilings:
            meta["gpu_name"] = self.hw_ceilings.gpu_name
            meta["peak_tflops_fp16"] = self.hw_ceilings.peak_tflops_fp16
            meta["peak_bw_tbs"] = self.hw_ceilings.peak_bw_tbs

        # Relevant env vars
        meta["env"] = {
            k: os.environ.get(k)
            for k in [
                "FLASHINFER_ROCM_ARCH_LIST",
                "FLASHINFER_JIT_DEBUG",
                "FLASHINFER_JIT_VERBOSE",
                "MAX_JOBS",
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
            ]
        }

        out = self.output_dir / f"{self.label}_meta.json"
        out.write_text(json.dumps(meta, indent=2))
        print(f"[rocm_profiler] Metadata → {out}")

    # ── Profile mode (executes inside rocprofv3 subprocess) ───────────────────

    def _jit_warmup_only(self) -> None:
        """
        Run one silent kernel launch per config to populate the JIT cache,
        then exit.  Called when _ENV_PREWARM is set — this runs BEFORE
        rocprofv3 is spawned so no counter-collection overhead is incurred.
        """
        print("[rocm_profiler] JIT pre-warm pass (no counter collection)")
        sys.stdout.flush()
        for cfg in self.configs:
            if cfg.setup_fn:
                cfg.setup_fn()
            cfg.run_fn()
        torch.cuda.synchronize()
        print("[rocm_profiler] JIT pre-warm done")
        sys.stdout.flush()

    def _profile_mode(self) -> None:
        """
        Profile sweep protocol, executed inside each rocprofv3 PMC-pass replay.

        If _ENV_CONFIG_IDX is set, profiles ONLY that one config with zero warmup
        launches (JIT cache was already pre-warmed by _jit_warmup_only()).  This
        single-config mode avoids filling rocprofv3's hardware counter ring buffer,
        which only retains the last ~8 dispatches per process.

        Without _ENV_CONFIG_IDX, falls back to profiling all configs in sequence
        (legacy mode, subject to ring buffer overflow for sweeps larger than ~8).
        """
        cfg_idx_str = os.environ.get(_ENV_CONFIG_IDX)
        if cfg_idx_str is not None:
            # Single-config mode: exactly 1 profiled dispatch, no warmup
            cfg_idx = int(cfg_idx_str)
            cfg = self.configs[cfg_idx]
            if cfg.setup_fn:
                cfg.setup_fn()
            print(
                f"PROFILE name={cfg.name} dispatch=0 "
                f"flops={cfg.theoretical_flops} bytes={cfg.theoretical_bytes}"
            )
            sys.stdout.flush()
            torch.cuda.synchronize()
            cfg.run_fn()
            torch.cuda.synchronize()
            print(f"PROFILE_DONE name={cfg.name}")
            sys.stdout.flush()
            return

        # ── Full-sweep fallback (legacy / debug) ──────────────────────────────
        n = len(self.configs)
        print(
            f"# rocm_profiler profile mode  "
            f"warmup={self.num_warmup}  configs={n}  "
            f"dispatches_per_group={self.num_warmup + 1}"
        )
        print()

        print("## PHASE profile_sweep")
        global_dispatch = 0

        for _, cfg in enumerate(self.configs):
            if cfg.setup_fn:
                cfg.setup_fn()

            for _ in range(self.num_warmup):
                print(f"WARMUP  name={cfg.name} dispatch={global_dispatch}")
                sys.stdout.flush()
                torch.cuda.synchronize()
                cfg.run_fn()
                torch.cuda.synchronize()
                global_dispatch += 1

            # Single profiled launch — this is the measurement row
            print(
                f"PROFILE name={cfg.name} dispatch={global_dispatch} "
                f"flops={cfg.theoretical_flops} bytes={cfg.theoretical_bytes}"
            )
            sys.stdout.flush()
            torch.cuda.synchronize()
            cfg.run_fn()
            torch.cuda.synchronize()
            print(f"PROFILE_DONE name={cfg.name}")
            sys.stdout.flush()
            global_dispatch += 1

        print(f"\n## PHASE profile_sweep: done — {global_dispatch} total dispatches")
        print(
            "## Profiled dispatch indices (0-based, target kernel only): "
            + str(
                [
                    n + cfg_idx * (self.num_warmup + 1) + self.num_warmup
                    for cfg_idx in range(n)
                ]
            )
        )

    # ── rocprofv3 subprocess ──────────────────────────────────────────────────

    def _run_rocprofv3(self) -> Path | None:
        """
        Profile each config individually with rocprofv3 to avoid ring-buffer
        overflow.  rocprofv3 keeps only the last ~8 dispatches per process in
        its hardware counter ring buffer; profiling all configs in one call
        would drop earlier configs for sweeps larger than the buffer.

        Strategy:
          1. Pre-warm JIT cache (one silent subprocess without LD_PRELOAD).
          2. For each config: spawn rocprofv3 with _ENV_CONFIG_IDX set so
             _profile_mode() runs exactly ONE dispatch for that config.
             This guarantees the single dispatch is always captured.
          3. Collect all per-pass CSVs from each invocation's tmpdir and
             merge them: pass CSVs are joined by counter columns (wide-form
             merge), config rows are concatenated.
          4. Write the merged wide-form counter CSV to output_dir.

        Returns the path to the final counter CSV, or None on failure.
        """
        counter_file = self._write_counter_file()
        print(f"[rocm_profiler] Counter spec → {counter_file}")

        # ── Step 1: pre-warm the JIT cache under rocprofv3's LD environment ──
        rocm_lib = _detect_rocm_lib_path()
        prewarm_env = {
            **os.environ,
            _ENV_PREWARM: "1",
        }
        if rocm_lib:
            existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
            prewarm_env["LD_LIBRARY_PATH"] = (
                f"{rocm_lib}:{existing_ld}" if existing_ld else rocm_lib
            )
        print(
            f"[rocm_profiler] JIT pre-warm (LD_LIBRARY_PATH={prewarm_env.get('LD_LIBRARY_PATH', 'unchanged')}) ..."
        )
        sys.stdout.flush()
        pre = subprocess.run(
            [sys.executable, *sys.argv],
            env=prewarm_env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if pre.returncode != 0:
            print(
                f"[rocm_profiler] WARNING: JIT pre-warm exited {pre.returncode}",
                file=sys.stderr,
            )
            if pre.stderr:
                print(pre.stderr[-2000:], file=sys.stderr)
        else:
            print("[rocm_profiler] JIT pre-warm done.\n")
        sys.stdout.flush()

        # ── Step 2: profile each config in a separate rocprofv3 invocation ──
        # rocprofv3 creates per-pass subdirectories: pass_1/, pass_2/, …
        # Each pass CSV has counter columns for that pass only.  We merge all
        # passes for each config into a single wide row.
        all_config_rows: list[dict] = []

        for cfg_idx, cfg in enumerate(self.configs):
            print(
                f"[rocm_profiler] Profiling {cfg_idx + 1}/{len(self.configs)}: {cfg.name}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix="rocm_profiler_") as tmpdir:
                cmd = [
                    "rocprofv3",
                    "--input",
                    str(counter_file),
                    "--output-format",
                    "csv",
                    "--output-directory",
                    tmpdir,
                    "--output-file",
                    self.label,
                ]
                if self.kernel_name_regex:
                    cmd += ["--kernel-include-regex", self.kernel_name_regex]
                cmd += ["--", sys.executable, *sys.argv]

                env = {
                    **os.environ,
                    _ENV_SENTINEL: "1",
                    _ENV_CONFIG_IDX: str(cfg_idx),
                }

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=self._rocprofv3_timeout,
                    )
                except FileNotFoundError:
                    print(
                        "[rocm_profiler] ERROR: 'rocprofv3' not found on PATH. "
                        "Install ROCm and ensure rocprofv3 is available before profiling.",
                        file=sys.stderr,
                    )
                    return None
                except subprocess.TimeoutExpired:
                    print(
                        f"[rocm_profiler] ERROR: rocprofv3 timed out for config "
                        f"'{cfg.name}' after {self._rocprofv3_timeout}s.",
                        file=sys.stderr,
                    )
                    continue

                if result.returncode != 0:
                    print(
                        f"[rocm_profiler] WARNING: rocprofv3 exited "
                        f"{result.returncode} for config '{cfg.name}'",
                        file=sys.stderr,
                    )
                    if result.stderr:
                        print(result.stderr[-1000:], file=sys.stderr)

                # Collect all per-pass CSVs and merge into one wide row
                csv_files = list(Path(tmpdir).rglob("*.csv"))
                if not csv_files:
                    print(
                        f"[rocm_profiler] WARNING: no CSV produced for config "
                        f"'{cfg.name}' — skipping.",
                        file=sys.stderr,
                    )
                    continue

                # Warn if rocprofv3 produced rows but none matched the kernel regex.
                if self.kernel_name_regex:
                    _warn_if_regex_unmatched(
                        csv_files, self.kernel_name_regex, cfg.name
                    )

                row = _merge_pass_csvs(csv_files)
                if row is None:
                    print(
                        f"[rocm_profiler] WARNING: could not extract counter row "
                        f"for config '{cfg.name}' — skipping.",
                        file=sys.stderr,
                    )
                    continue

                row["config_name"] = cfg.name
                all_config_rows.append(row)

        if not all_config_rows:
            print(
                "[rocm_profiler] ERROR: rocprofv3 produced no counter data.",
                file=sys.stderr,
            )
            return None

        # ── Step 3: write merged wide-form CSV ────────────────────────────────
        # Union of all column names; config_name first, then metadata, then counters.
        seen: dict[str, None] = {}
        for row in all_config_rows:
            for k in row:
                seen[k] = None
        all_cols = ["config_name"] + [k for k in seen if k != "config_name"]

        out_csv = self.output_dir / f"{self.label}_counter_collection.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_config_rows)

        print(f"[rocm_profiler] Counter collection → {out_csv}\n")
        return out_csv

    # ── Roofline plot ─────────────────────────────────────────────────────────

    def _plot_roofline(self, timing_csv: Path, counter_csv: Path) -> None:
        """
        Parse timing + counter CSVs, print a metrics table, and generate PNG.
        """
        # Load timing
        timing: dict[str, dict] = {}
        with open(timing_csv, newline="") as f:
            for row in csv.DictReader(f):
                timing[row["name"]] = {
                    "theoretical_flops": int(row["theoretical_flops"]),
                    "theoretical_bytes": int(row["theoretical_bytes"]),
                    "median_ms": float(row["median_ms"]),
                }

        # Load pivoted counter data (wide-form, keyed by config_name)
        counters_by_name: dict[str, dict] = {}
        with open(counter_csv, newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("config_name", "").strip()
                if name:
                    counters_by_name[name] = row

        # Merge and compute derived metrics.
        # Drive iteration from timing CSV so --replot works with configs=[].
        name_to_label = {cfg.name: cfg.label for cfg in self.configs}
        results = []
        for name, t in timing.items():
            c = counters_by_name.get(name)
            if c is None:
                print(
                    f"  WARNING: no counter row for '{name}' — "
                    "check kernel_name_regex and rocprofv3 output",
                    file=sys.stderr,
                )
                continue
            label = name_to_label.get(name, name)

            flops = t["theoretical_flops"]
            theo_bytes = t["theoretical_bytes"]
            median_ms = t["median_ms"]
            duration_s = median_ms * 1e-3

            tflops = flops / duration_s / 1e12
            ai_theory = flops / theo_bytes

            fetch_kb = _float(c.get("FetchSize", 0))
            write_kb = _float(c.get("WriteSize", 0))
            bw_bytes = (fetch_kb + write_kb) * 1024
            ai_measured = flops / bw_bytes if bw_bytes > 0 else float("nan")
            bw_tbs = bw_bytes / duration_s / 1e12

            rdreq = _float(c.get("TCC_EA0_RDREQ_DRAM_sum", 0))
            wrreq = _float(c.get("TCC_EA0_WRREQ_DRAM_sum", 0))
            bytes_hbm = (rdreq + wrreq) * 64
            bw_hbm_tbs = bytes_hbm / duration_s / 1e12 if bytes_hbm > 0 else 0.0
            ai_hbm = flops / bytes_hbm if bytes_hbm > 0 else float("nan")

            mfma_mops = _float(c.get("SQ_INSTS_VALU_MFMA_MOPS_F16", 0))
            actual_flops_mfma = mfma_mops * 512
            mfma_busy_cycles = _float(c.get("SQ_VALU_MFMA_BUSY_CYCLES", 0))

            results.append(
                dict(
                    name=name,
                    label=label,
                    median_ms=median_ms,
                    tflops=tflops,
                    ai_theory=ai_theory,
                    ai_measured=ai_measured,
                    ai_hbm=ai_hbm,
                    bw_tbs=bw_tbs,
                    bw_hbm_tbs=bw_hbm_tbs,
                    bytes_touched=bw_bytes,
                    bytes_hbm=bytes_hbm,
                    bytes_theory=theo_bytes,
                    actual_flops_mfma=actual_flops_mfma,
                    flops_theory=flops,
                    lds_conflict=_float(c.get("SQ_LDS_BANK_CONFLICT", 0)),
                    mfma_busy_cycles=mfma_busy_cycles,
                    mfma_util_pct=(
                        actual_flops_mfma
                        / (self.hw_ceilings.peak_tflops_fp16 * 1e12 * duration_s)
                        * 100
                        if self.hw_ceilings and duration_s > 0 and actual_flops_mfma > 0
                        else 0.0
                    ),
                )
            )

        if not results:
            print("[rocm_profiler] No results to plot.", file=sys.stderr)
            return

        _print_metrics_table(results, self.hw_ceilings, self.label)

        out_png = self.output_dir / f"{self.label}_roofline.png"
        _draw_roofline(results, out_png, self.label, self.hw_ceilings)

    # ── Counter file ──────────────────────────────────────────────────────────

    def _write_counter_file(self) -> Path:
        """
        Write the counter specification to disk and return its path.
        If counters is a preset name, writes YAML to output_dir.
        If counters is a file path, returns it as-is.
        """
        if self.counters in _COUNTER_PRESETS:
            out = self.output_dir / f"{self.label}_counters.yml"
            out.write_text(_COUNTER_PRESETS[self.counters])
            return out
        p = Path(self.counters)
        if not p.exists():
            raise FileNotFoundError(f"Counter file not found: {p}")
        return p


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _detect_rocm_lib_path() -> str | None:
    """Return the ROCm lib directory that rocprofv3 typically injects via
    LD_LIBRARY_PATH (e.g. /opt/rocm-7.2.0/lib).  We detect it by locating
    the rocprofv3 binary and climbing up to <rocm_root>/lib."""
    import shutil

    rp = shutil.which("rocprofv3")
    if rp is None:
        return None
    # rocprofv3 lives at <rocm_root>/bin/rocprofv3
    rocm_root = Path(rp).resolve().parent.parent
    lib_dir = rocm_root / "lib"
    if lib_dir.is_dir():
        return str(lib_dir)
    return None


def _report_ceilings(arch: str) -> HardwareCeilings | None:
    """Announce the ceilings for ``arch``, or warn that there are none."""
    hw = KNOWN_HW.get(arch)
    if hw is None:
        print(
            f"[rocm_profiler] WARNING: GPU arch '{arch}' not in KNOWN_HW. "
            "Roofline will not show hardware ceilings.",
            file=sys.stderr,
        )
        return None
    print(
        f"[rocm_profiler] Detected GPU: {hw.gpu_name} ({arch})  "
        f"peak={hw.peak_tflops_fp16:.0f} TFLOPS  "
        f"BW={hw.peak_bw_tbs:.1f} TB/s  "
        f"ridge≈{hw.ridge_flops_per_byte:.0f} FLOPs/B"
    )
    return hw


def _detect_hw_ceilings() -> HardwareCeilings | None:
    """Identify the GPU arch and return matching hardware ceilings.

    Prefers torch's device properties over rocminfo: it reports the device this
    process will actually run on, whereas the rocminfo scrape takes whichever
    agent is listed first and ignores HIP_VISIBLE_DEVICES.
    """
    try:
        import torch

        from flashinfer.arch_caps import normalize_arch

        arch = normalize_arch(torch.cuda.get_device_properties(0).gcnArchName)
    except Exception:
        arch = None

    if arch in KNOWN_HW:
        return _report_ceilings(arch)

    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        match = re.search(r"(gfx\w+)", result.stdout)
        if match:
            return _report_ceilings(match.group(1))
    except FileNotFoundError:
        print(
            "[rocm_profiler] WARNING: rocminfo not found. "
            "Hardware ceilings unavailable.",
            file=sys.stderr,
        )
    except subprocess.TimeoutExpired:
        print(
            "[rocm_profiler] WARNING: rocminfo timed out. "
            "Hardware ceilings unavailable.",
            file=sys.stderr,
        )
    return None


def _find_col(fieldnames: list[str], candidates: list[str]) -> str | None:
    """Return the first candidate that exists in fieldnames, or None."""
    fset = set(fieldnames)
    for c in candidates:
        if c in fset:
            return c
    return None


def _merge_pass_csvs(csv_files: list[Path]) -> dict | None:
    """
    Merge one or more per-pass rocprofv3 CSV files into a single wide-form dict.

    rocprofv3 writes one CSV per PMC pass into pass_1/, pass_2/, … sub-dirs.
    Each CSV has the same metadata columns (Dispatch_Id, Kernel_Name, …) but
    different counter columns.  We take the last row from each file (the
    profiled dispatch for single-config mode) and union all counter columns.

    Supports both long-form (Counter_Name / Counter_Value columns) and
    wide-form (counter names as column headers) output.
    """
    merged: dict = {}

    for csv_file in sorted(csv_files):  # pass_1 before pass_2
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames: list[str] = reader.fieldnames or []

            counter_name_col = _find_col(
                fieldnames, ["Counter_Name", "counter_name", "CounterName"]
            )
            counter_val_col = _find_col(
                fieldnames, ["Counter_Value", "counter_value", "CounterValue"]
            )
            dispatch_col = _find_col(
                fieldnames, ["Dispatch_Id", "dispatch_id", "DispatchId"]
            )
            is_long_form = counter_name_col is not None and counter_val_col is not None

            rows = list(reader)
            if not rows:
                continue

            if is_long_form:
                # Group by dispatch_id, pick the last (highest) dispatch
                by_dispatch: dict[int, dict] = {}
                for row in rows:
                    try:
                        disp_id = int(row.get(dispatch_col or "", 0))
                    except (ValueError, TypeError):
                        disp_id = 0
                    if disp_id not in by_dispatch:
                        by_dispatch[disp_id] = {
                            k: v
                            for k, v in row.items()
                            if k not in (counter_name_col, counter_val_col)
                        }
                    cname = row.get(counter_name_col, "").strip()
                    cval = row.get(counter_val_col, "").strip()
                    if cname:
                        by_dispatch[disp_id][cname] = cval
                if not by_dispatch:
                    continue
                row_data = by_dispatch[max(by_dispatch)]
            else:
                # Wide-form: take the last row
                row_data = dict(rows[-1])

            # Merge: metadata columns only from the first pass (avoid overwriting),
            # counter columns always take the value from this pass.
            meta_cols = {
                "Dispatch_Id",
                "dispatch_id",
                "DispatchId",
                "Correlation_Id",
                "correlation_id",
                "Agent_Id",
                "agent_id",
                "Queue_Id",
                "queue_id",
                "Process_Id",
                "process_id",
                "Thread_Id",
                "thread_id",
                "Grid_Size",
                "grid_size",
                "Kernel_Id",
                "kernel_id",
                "Kernel_Name",
                "kernel_name",
                "Workgroup_Size",
                "workgroup_size",
                "LDS_Block_Size",
                "lds_block_size",
                "Scratch_Size",
                "scratch_size",
                "VGPR_Count",
                "vgpr_count",
                "Accum_VGPR_Count",
                "accum_vgpr_count",
                "SGPR_Count",
                "sgpr_count",
                "Start_Timestamp",
                "start_timestamp",
                "End_Timestamp",
                "end_timestamp",
            }
            for k, v in row_data.items():
                if k in meta_cols:
                    if k not in merged:  # keep first pass metadata
                        merged[k] = v
                else:
                    merged[k] = v  # counter values: always update

    return merged if merged else None


def _float(v: Any, default: float = 0.0) -> float:
    """Safe float conversion with a default for missing / non-numeric values."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _check_finite(result: Any, name: str) -> None:
    """Warn if the kernel returned NaN or Inf in any output tensor."""
    tensors = result if isinstance(result, (list, tuple)) else (result,)
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            continue
        if not torch.isfinite(t).all():
            print(
                f"[rocm_profiler] WARNING: '{name}' produced NaN/Inf output — "
                "kernel may be incorrectly compiled or have numerical issues.",
                file=sys.stderr,
            )
            return


def _warn_if_regex_unmatched(
    csv_files: list[Path], regex: str, config_name: str
) -> None:
    """
    Scan rocprofv3 CSV rows for any Kernel_Name matching `regex`.
    If none match, print a warning so the user knows their regex is stale.
    """
    try:
        pattern = re.compile(regex)
    except re.error as exc:
        print(
            f"[rocm_profiler] WARNING: kernel_name_regex '{regex}' is invalid: {exc} "
            "— skipping regex check.",
            file=sys.stderr,
        )
        return
    kernel_name_candidates = [
        "Kernel_Name",
        "kernel_name",
        "KernelName",
    ]
    any_kernel_seen = False
    any_match = False
    for csv_file in csv_files:
        try:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                name_col = _find_col(reader.fieldnames or [], kernel_name_candidates)
                if name_col is None:
                    continue
                for row in reader:
                    kname = row.get(name_col, "").strip()
                    if kname:
                        any_kernel_seen = True
                    if kname and pattern.search(kname):
                        any_match = True
                        break
        except Exception:
            continue
        if any_match:
            break

    if any_kernel_seen and not any_match:
        print(
            f"[rocm_profiler] WARNING: kernel_name_regex '{regex}' matched no "
            f"kernels for config '{config_name}'. The kernel may have been "
            "renamed. Check rocprofv3 CSV Kernel_Name column.",
            file=sys.stderr,
        )


def _print_presets() -> None:
    print("Available counter presets:\n")
    for name, content in _COUNTER_PRESETS.items():
        print(f"  '{name}':")
        for line in content.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                print(f"    {s}")
        print()


def _print_metrics_table(
    results: list[dict], hw: HardwareCeilings | None, label: str
) -> None:
    title = f"=== {label} — Roofline Metrics ==="
    if hw:
        title += f"  [{hw.gpu_name}]"
    print(f"\n{title}\n")

    header = (
        f"{'Config':>30} | {'ms':>7} | {'TFLOPS':>7} | {'AI_th':>7} | "
        f"{'AI_meas':>8} | {'BW_TB/s':>8} | {'BW_HBM':>8} | "
        f"{'MFMA_util%':>10} | {'MFMA_busy_cy':>12} | {'LDS_bank_cf':>11}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        mfma_busy = r.get("mfma_busy_cycles", 0)
        print(
            f"{r['label']:>30} | {r['median_ms']:>7.3f} | {r['tflops']:>7.2f} | "
            f"{r['ai_theory']:>7.0f} | {r['ai_measured']:>8.1f} | "
            f"{r['bw_tbs']:>8.3f} | {r['bw_hbm_tbs']:>8.3f} | "
            f"{r['mfma_util_pct']:>10.1f} | {mfma_busy:>12.0f} | "
            f"{r['lds_conflict']:>11.0f}"
        )

    print()
    if hw:
        print(f"  {hw.gpu_name} peak FP16 MFMA : {hw.peak_tflops_fp16:.1f} TFLOPS")
        print(f"  {hw.gpu_name} peak HBM BW    : {hw.peak_bw_tbs:.1f} TB/s")
        print(f"  Ridge point           : {hw.ridge_flops_per_byte:.0f} FLOPs/Byte")
        print("  bytes/th > 1x means actual HBM traffic > cold-cache lower bound")
    print()


def _draw_roofline_backdrop(ax: Any, hw: HardwareCeilings) -> None:
    bw_peak = hw.peak_bw_tbs * 1e12
    fp16_peak = hw.peak_tflops_fp16 * 1e12
    ridge = hw.ridge_flops_per_byte

    ai_range = np.logspace(np.log10(1), np.log10(1e5), 800)
    roof = np.minimum(bw_peak * ai_range, fp16_peak) / 1e12

    ax.loglog(ai_range, roof, "k-", lw=2, zorder=3, label=f"{hw.gpu_name} roofline")
    ax.axvline(
        ridge,
        ls="--",
        color="gray",
        lw=1.2,
        zorder=2,
        label=f"Ridge ≈ {ridge:.0f} FLOPs/B",
    )
    ax.axhline(
        hw.peak_tflops_fp16,
        ls=":",
        color="steelblue",
        lw=1.0,
        alpha=0.6,
        label=f"FP16 peak = {hw.peak_tflops_fp16:.0f} TFLOPS",
    )
    ax.annotate(
        f"{hw.peak_tflops_fp16:.0f} TFLOPS",
        (ai_range[-1], hw.peak_tflops_fp16),
        xytext=(-5, -3),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color="steelblue",
    )
    ax.grid(True, which="both", alpha=0.25, linestyle=":")


def _draw_roofline(
    results: list[dict],
    out_path: Path,
    title_prefix: str,
    hw: HardwareCeilings | None,
    dpi: int = 150,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    cmap = plt.cm.plasma
    colors = [cmap(0.15 + 0.7 * i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    panels = [
        ("ai_measured", "Roofline — Measured AI (FetchSize + WriteSize)"),
        ("ai_theory", "Roofline — Theoretical AI (cold-cache lower bound)"),
    ]

    for ax, (ai_key, panel_title) in zip(axes, panels, strict=True):
        if hw:
            _draw_roofline_backdrop(ax, hw)

        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Arithmetic Intensity  [FLOPs/Byte]", fontsize=10)
        ax.set_ylabel("Performance  [TFLOPS/s]", fontsize=10)

        for r, color in zip(results, colors, strict=True):
            ai = r[ai_key]
            perf = r["tflops"]
            if not (np.isfinite(ai) and ai > 0 and np.isfinite(perf) and perf > 0):
                continue

            ax.loglog(
                ai,
                perf,
                "o",
                color=color,
                ms=9,
                zorder=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            ax.annotate(
                r["label"],
                (ai, perf),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=7,
                color=color,
            )

            if ai_key == "ai_theory" and hw:
                bw_peak = hw.peak_bw_tbs * 1e12
                comp_ceil = hw.peak_tflops_fp16 * 1e12
                roof_at_ai = min(ai * bw_peak, comp_ceil) / 1e12
                if roof_at_ai > 0:
                    eff = perf / roof_at_ai * 100
                    ax.annotate(
                        f"{eff:.0f}%",
                        (ai, perf),
                        textcoords="offset points",
                        xytext=(5, -10),
                        fontsize=6.5,
                        color="dimgray",
                    )
            elif ai_key == "ai_measured":
                ai_th = r.get("ai_theory", 0)
                if (
                    np.isfinite(ai_th)
                    and ai_th > 0
                    and abs(ai_th - ai) / max(ai_th, ai) > 0.05
                ):
                    ax.annotate(
                        "",
                        xy=(ai_th, perf),
                        xytext=(ai, perf),
                        arrowprops=dict(
                            arrowstyle="->", color=color, lw=0.8, linestyle="dashed"
                        ),
                    )

        # Auto-scale axes
        ai_vals = [
            r[ai_key] for r in results if np.isfinite(r[ai_key]) and r[ai_key] > 0
        ]
        perf_vals = [
            r["tflops"] for r in results if np.isfinite(r["tflops"]) and r["tflops"] > 0
        ]
        if ai_vals:
            lo = min(ai_vals) * 0.4
            hi = max(ai_vals) * 2.5
            ax.set_xlim(lo, hi)
        if perf_vals:
            y_hi = hw.peak_tflops_fp16 * 1.5 if hw else max(perf_vals) * 2
            ax.set_ylim(min(perf_vals) * 0.4, y_hi)

        if hw:
            ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle(
        f"{title_prefix} — Roofline Assessment"
        + (
            f"  [{hw.gpu_name}  ridge≈{hw.ridge_flops_per_byte:.0f} FLOPs/B]"
            if hw
            else ""
        ),
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[rocm_profiler] Roofline plot → {out_path}")
    plt.close(fig)
