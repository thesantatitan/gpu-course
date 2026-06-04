#!/usr/bin/env python3
"""Run a course CUDA assignment on Modal with one local command.

Examples:
    uv run run_modal.py
    uv run run_modal.py 01 -- 1048576 256
    uv run run_modal.py assignments/03-memory-performance/starter/main.cu -- 2048 2048
    uv run run_modal.py --gpu L4 02 -- 512
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_GPU = "T4"
DEFAULT_CUDA_IMAGE = "nvidia/cuda:12.4.0-devel-ubuntu22.04"

LAB_ALIASES = {
    "smoke": "tools/cuda-smoke/main.cu",
    "01": "assignments/01-vector-add/starter/main.cu",
    "1": "assignments/01-vector-add/starter/main.cu",
    "vector-add": "assignments/01-vector-add/starter/main.cu",
    "02": "assignments/02-grids-matmul/starter/main.cu",
    "2": "assignments/02-grids-matmul/starter/main.cu",
    "matmul": "assignments/02-grids-matmul/starter/main.cu",
    "03": "assignments/03-memory-performance/starter/main.cu",
    "3": "assignments/03-memory-performance/starter/main.cu",
    "transpose": "assignments/03-memory-performance/starter/main.cu",
    "04": "assignments/04-convolution-stencil/starter/main.cu",
    "4": "assignments/04-convolution-stencil/starter/main.cu",
    "conv": "assignments/04-convolution-stencil/starter/main.cu",
    "convolution": "assignments/04-convolution-stencil/starter/main.cu",
    "05": "assignments/05-histogram-reduction/starter/main.cu",
    "5": "assignments/05-histogram-reduction/starter/main.cu",
    "hist": "assignments/05-histogram-reduction/starter/main.cu",
    "reduction": "assignments/05-histogram-reduction/starter/main.cu",
    "06": "assignments/06-scan-merge/starter/main.cu",
    "6": "assignments/06-scan-merge/starter/main.cu",
    "scan": "assignments/06-scan-merge/starter/main.cu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and run a PMPP course CUDA assignment on Modal."
    )
    parser.add_argument(
        "--gpu",
        default=os.environ.get("PMPP_MODAL_GPU", DEFAULT_GPU),
        help=f"Modal GPU type. Default: {DEFAULT_GPU}, currently Modal's cheapest GPU.",
    )
    parser.add_argument(
        "--cuda-image",
        default=os.environ.get("PMPP_MODAL_CUDA_IMAGE", DEFAULT_CUDA_IMAGE),
        help=f"NVIDIA CUDA devel image with nvcc. Default: {DEFAULT_CUDA_IMAGE}.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("PMPP_MODAL_TIMEOUT", "600")),
        help="Remote timeout in seconds. Default: 600.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="smoke",
        help="Assignment alias such as smoke/01/matmul/scan, or a .cu path. Default: smoke.",
    )
    parser.add_argument(
        "program_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the compiled CUDA program. A leading -- is ignored.",
    )
    return parser.parse_args()


def resolve_source(source: str) -> str:
    source = LAB_ALIASES.get(source, source)
    path = Path(source)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(ROOT))
        except ValueError as exc:
            raise SystemExit(f"Source must be inside this repo: {path}") from exc

    resolved = (ROOT / path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as exc:
        raise SystemExit(f"Source must be inside this repo: {path}") from exc

    if not resolved.exists():
        known = ", ".join(sorted(k for k in LAB_ALIASES if k.startswith("0")))
        raise SystemExit(f"Source not found: {path}\nKnown lab aliases: {known}")
    if resolved.suffix != ".cu":
        raise SystemExit(f"Source must be a .cu file: {path}")
    return str(relative)


def command_works(cmd: list[str]) -> bool:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0


def ensure_modal_command() -> list[str]:
    if command_works([sys.executable, "-c", "import modal"]):
        return [sys.executable, "-m", "modal"]

    if os.environ.get("PMPP_RUNNING_UNDER_UV") == "1":
        raise SystemExit(
            "Modal is not importable even after running through uv.\n"
            "Try: uv sync"
        )

    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "Modal is managed through uv for this repo, but uv was not found.\n"
            "Install uv, then run: uv run run_modal.py 01 -- 1048576 256"
        )

    env = os.environ.copy()
    env["PMPP_RUNNING_UNDER_UV"] = "1"
    os.execve(uv, [uv, "run", "python", str(__file__), *sys.argv[1:]], env)
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    source = resolve_source(args.source)
    program_args = args.program_args
    if program_args and program_args[0] == "--":
        program_args = program_args[1:]

    modal_cmd = ensure_modal_command()
    app_path = ROOT / "tools" / "modal_cuda_app.py"

    env = os.environ.copy()
    env["PMPP_MODAL_GPU"] = args.gpu
    env["PMPP_MODAL_CUDA_IMAGE"] = args.cuda_image
    env["PMPP_MODAL_TIMEOUT"] = str(args.timeout)

    cmd = [
        *modal_cmd,
        "run",
        str(app_path),
        "--source",
        source,
        "--program-args-json",
        json.dumps(program_args),
    ]

    print(f"Modal GPU: {args.gpu}", flush=True)
    print(f"CUDA image: {args.cuda_image}", flush=True)
    print(f"Source: {source}", flush=True)
    print(f"Program args: {program_args or []}", flush=True)
    print("$ " + " ".join(cmd), flush=True)

    completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        print(
            "\nIf this is your first Modal run, authenticate once with:\n"
            "  uv run modal setup",
            file=sys.stderr,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
