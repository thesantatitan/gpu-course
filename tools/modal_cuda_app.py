from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/gpu-course")
GPU = os.environ.get("PMPP_MODAL_GPU", "T4")
CUDA_IMAGE = os.environ.get(
    "PMPP_MODAL_CUDA_IMAGE", "nvidia/cuda:12.4.0-devel-ubuntu22.04"
)
TIMEOUT = int(os.environ.get("PMPP_MODAL_TIMEOUT", "600"))
PROGRESS_PREFIX = "PMPP_PROGRESS_JSON="

app = modal.App("pmpp-gpu-course")

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential")
    .add_local_dir(
        LOCAL_ROOT,
        remote_path=str(REMOTE_ROOT),
        ignore=[
            ".git",
            ".git/**",
            ".build",
            ".build/**",
            ".venv",
            ".venv/**",
            "**/__pycache__",
            "**/*.pyc",
            ".ipynb_checkpoints",
            ".ipynb_checkpoints/**",
        ],
    )
)


def _safe_source_path(source: str) -> Path:
    path = (REMOTE_ROOT / source).resolve()
    try:
        path.relative_to(REMOTE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Source must stay inside {REMOTE_ROOT}: {source}") from exc
    if not path.exists():
        raise FileNotFoundError(f"Source not found in Modal container: {source}")
    if path.suffix != ".cu":
        raise ValueError(f"Source must be a .cu file: {source}")
    return path


def _parse_assignment_metrics(stdout: str) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("ok" in value or "lab" in value):
            metrics = value
    return metrics


@app.function(image=image, gpu=GPU, timeout=TIMEOUT)
def compile_and_run(source: str, program_args: list[str]) -> dict[str, object]:
    source_path = _safe_source_path(source)

    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        cwd=REMOTE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    cmd = [
        "python3",
        "common/compile_and_run.py",
        str(source_path.relative_to(REMOTE_ROOT)),
        "--",
        *program_args,
    ]
    result = subprocess.run(
        cmd,
        cwd=REMOTE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "gpu_request": GPU,
        "gpu_info": gpu_info.stdout.strip(),
        "gpu_info_stderr": gpu_info.stderr.strip(),
        "metrics": _parse_assignment_metrics(result.stdout),
        "source": source,
        "program_args": program_args,
        "cuda_image": CUDA_IMAGE,
    }


@app.local_entrypoint()
def main(source: str = "assignments/01-vector-add/starter/main.cu", program_args_json: str = "[]"):
    try:
        program_args = json.loads(program_args_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"program_args_json must be a JSON list: {exc}") from exc

    if not isinstance(program_args, list) or not all(isinstance(x, str) for x in program_args):
        raise SystemExit("program_args_json must decode to a list of strings")

    result = compile_and_run.remote(source, program_args)

    print(f"Requested Modal GPU: {result['gpu_request']}")
    if result["gpu_info"]:
        print(f"Allocated GPU: {result['gpu_info']}")
    elif result["gpu_info_stderr"]:
        print(f"nvidia-smi stderr: {result['gpu_info_stderr']}", file=sys.stderr)

    print("$ " + " ".join(result["command"]))
    if result["stdout"]:
        print(result["stdout"], end="")
    if result["stderr"]:
        print(result["stderr"], end="", file=sys.stderr)

    progress_payload = {
        "assignment_exit_code": result["returncode"],
        "allocated_gpu": result["gpu_info"],
        "cuda_image": result["cuda_image"],
        "metrics": result["metrics"],
        "program_args": result["program_args"],
        "remote_command": result["command"],
        "requested_gpu": result["gpu_request"],
        "source": result["source"],
    }
    print(PROGRESS_PREFIX + json.dumps(progress_payload, sort_keys=True, separators=(",", ":")))
    print(f"Assignment exit code: {result['returncode']}")
