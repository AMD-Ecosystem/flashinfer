import argparse
import csv
import sys

from routines.attention import parse_attention_args, run_attention_test
from routines.flashinfer_benchmark_utils import (
    benchmark_apis,
    full_output_columns,
    output_column_dict,
)

# The gemm and moe routines pull in CUDA-only modules (flashinfer.autotuner,
# flashinfer.fused_moe), so importing them unconditionally makes the whole runner
# unimportable on ROCm -- including the attention routines, which do work there.
_ROUTINE_IMPORT_ERRORS = {}

try:
    from routines.gemm import parse_gemm_args, run_gemm_test
except Exception as exc:
    parse_gemm_args = run_gemm_test = None
    _ROUTINE_IMPORT_ERRORS["gemm"] = exc

try:
    from routines.moe import parse_moe_args, run_moe_test
except Exception as exc:
    parse_moe_args = run_moe_test = None
    _ROUTINE_IMPORT_ERRORS["moe"] = exc


def require_routine_group(group):
    """Raise with the original import error if this routine group failed to import."""
    exc = _ROUTINE_IMPORT_ERRORS.get(group)
    if exc is not None:
        raise RuntimeError(
            f"The '{group}' benchmark routines are unavailable on this platform: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def run_test(args):
    """
    Route & run a single FlashInfer test case with test routine.

    Args:
        args: Parsed command line arguments containing test configuration
    """

    ## Depending on routine type, route to corresponding test routine
    if args.routine in benchmark_apis["attention"]:
        res = run_attention_test(args)
    elif args.routine in benchmark_apis["gemm"]:
        require_routine_group("gemm")
        res = run_gemm_test(args)
    elif args.routine in benchmark_apis["moe"]:
        require_routine_group("moe")
        res = run_moe_test(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")

    # Write results to output file if specified
    if args.output_path is not None:
        with open(args.output_path, "a", newline="") as fout:
            writer = csv.writer(fout, lineterminator="\n")
            for cur_res in res:
                for key in output_column_dict["general"]:
                    cur_res[key] = getattr(args, key)

                writer.writerow([str(cur_res[col]) for col in full_output_columns])
            fout.flush()
    return


def parse_args(line=sys.argv[1:]):
    """
    Parse command line arguments for test configuration.
    First parse shared arguments, then parse routine-specific arguments.

    Args:
        line: Command line arguments (default: sys.argv[1:])

    Returns:
        Parsed argument namespace
    """

    ## Shared arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routine",
        "-R",
        type=str,
        required=True,
        choices=list(benchmark_apis["attention"])
        + list(benchmark_apis["gemm"])
        + list(benchmark_apis["moe"]),
    )
    args, _ = parser.parse_known_args(line[:])

    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        default=False,
        help="Disable CUDA graph to execute kernels outside of the graph.",
    )
    parser.add_argument(
        "--use_cupti",
        action="store_true",
        default=False,
        help="Use CUPTI for timing GPU kernels when available.",
    )
    parser.add_argument(
        "--refcheck",
        action="store_true",
        default=False,
        help="Run reference check that ensures outputs correct.",
    )
    parser.add_argument(
        "--allow_output_mismatch",
        action="store_true",
        default=False,
        help="Allow output mismatch between backends during reference checks. Error message will be printed but test will continue.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--verbose", "-v", action="count", help="Set verbosity level.", default=0
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="Output path for results. If not specified, results will not be written to a file.",
    )
    parser.add_argument(
        "--num_iters",
        "-n",
        type=int,
        required=False,
        default=30,
        help="Number of iterations to run for measurement.",
    )
    parser.add_argument(
        "--dry_run_iters",
        "-d",
        type=int,
        required=False,
        default=5,
        help="Number of dry runs.",
    )
    parser.add_argument(
        "--dry_run_time_ms",
        type=int,
        required=False,
        default=None,
        help="Warmup budget in ms. Overrides --dry_run_iters. On ROCm prefer this "
        "over an iteration count: clock sampling intervals run to hundreds of ms, "
        "so a handful of iterations warms up well short of steady-state clocks.",
    )
    parser.add_argument(
        "--repeat_time_ms",
        type=int,
        required=False,
        default=None,
        help="Measurement budget in ms. Overrides --num_iters. Leaving this unset "
        "keeps the sample count fixed, which makes std_time comparable run over run.",
    )
    parser.add_argument(
        "--case_tag",
        type=str,
        required=False,
        default=None,
        help="Optional tag for the test case for annotating output.",
    )
    parser.add_argument(
        "--generate_repro_command",
        action="store_true",
        default=False,
        help="If set, will print reproducer command and store it to output csv.",
    )
    parser.add_argument(
        "--repro_command",
        type=str,
        required=False,
        default="",
        help="Placeholder for generated reproducer command for the test case. Not to be used directly.",
    )

    ## Check routine and pass on to routine-specific argument parser
    if args.routine in benchmark_apis["attention"]:
        args = parse_attention_args(line, parser)
    elif args.routine in benchmark_apis["gemm"]:
        require_routine_group("gemm")
        args = parse_gemm_args(line, parser)
    elif args.routine in benchmark_apis["moe"]:
        require_routine_group("moe")
        args = parse_moe_args(line, parser)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")

    if args.generate_repro_command:
        args.repro_command = "python3 flashinfer_benchmark.py " + " ".join(line)
    return args


if __name__ == "__main__":
    # Parse testlist argument first
    testlist_parser = argparse.ArgumentParser(add_help=False)
    testlist_parser.add_argument(
        "--testlist",
        type=str,
        required=False,
        default=None,
        help="Optional testlist file to run multiple cases.",
    )
    testlist_parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="Output path for results csv.",
    )
    testlist_args, _ = testlist_parser.parse_known_args()

    # Setup output file if specified
    if testlist_args.output_path is not None:
        with open(testlist_args.output_path, "w", newline="") as fout:
            csv.writer(fout, lineterminator="\n").writerow(full_output_columns)

    # Process tests either from testlist file or command line arguments
    if testlist_args.testlist is not None:
        # If testlist, run each test in the testlist
        with open(testlist_args.testlist, "r") as f:
            import shlex

            for line in f.readlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    line_args = parse_args(shlex.split(line))
                    line_args.output_path = testlist_args.output_path
                    run_test(line_args)
                except Exception as e:
                    print(f"[ERROR] Error running test: {line}")
                    print(f"[ERROR] Error: {e}")
                    continue
    else:
        # If no testlist, just run the command
        args = parse_args()
        args.output_path = testlist_args.output_path
        run_test(args)
