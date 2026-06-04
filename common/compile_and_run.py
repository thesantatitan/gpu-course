#!/usr/bin/env python3
"""Compile and run a CUDA starter from the repo root.

Example:
    python3 common/compile_and_run.py assignments/01-vector-add/starter/main.cu -- 1048576 256
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Path to a .cu file, relative to repo root or absolute.")
    parser.add_argument(
        "program_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to the compiled program. Use -- before these.",
    )
    args = parser.parse_args()

    nvcc = shutil.which("nvcc")
    if not nvcc:
        print("nvcc was not found on PATH. Use a CUDA-enabled Colab/Modal/VM runtime.", file=sys.stderr)
        return 127

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        print(f"Source file not found: {source}", file=sys.stderr)
        return 2

    build_dir = ROOT / ".build"
    build_dir.mkdir(exist_ok=True)
    assignment_name = source.parents[1].name if len(source.parents) >= 2 else source.stem
    binary = build_dir / f"{assignment_name}_{source.stem}"

    compile_cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "-I",
        str(ROOT / "common"),
        str(source),
        "-o",
        str(binary),
    ]

    program_args = args.program_args
    if program_args and program_args[0] == "--":
        program_args = program_args[1:]

    print("$ " + " ".join(compile_cmd), flush=True)
    subprocess.run(compile_cmd, check=True, cwd=ROOT)

    run_cmd = [str(binary), *program_args]
    print("$ " + " ".join(run_cmd), flush=True)
    subprocess.run(run_cmd, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
