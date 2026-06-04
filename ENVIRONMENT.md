# Environment

The assignments are plain CUDA C++ plus small Python wrappers. Local Python tooling is managed with `uv`; CUDA execution can happen on Modal, Colab, cloud VMs, or any Linux machine with NVIDIA CUDA.

## Minimum Requirements

- `uv` for local Python tooling.
- Python 3.11+ when running the Modal wrapper through `uv`.
- A Modal account for remote GPU runs.
- For local CUDA or cloud VM runs: an NVIDIA GPU and CUDA toolkit with `nvcc`.

Check the runtime:

```bash
uv --version
python3 --version
```

For local CUDA machines, also check `nvidia-smi` and `nvcc --version`.

## uv Workflow

Install/sync Python dependencies:

```bash
uv sync
```

Run the Modal wrapper:

```bash
uv run run_modal.py
```

If you accidentally run `python3 run_modal.py ...`, the script will re-exec itself through `uv run` when `uv` is available.

## Colab Workflow

Use a GPU runtime, then run:

```bash
!nvidia-smi
!nvcc --version
```

For each lab, either upload this repo folder or mount Drive. Compile from the repo root:

```bash
!python3 common/compile_and_run.py assignments/01-vector-add/starter/main.cu -- 1048576 256
```

Colab tips:

- Runtime GPUs change; always record the GPU model in your writeup.
- Keep problem sizes moderate until correctness passes.
- Rerun a benchmark a few times before trusting small timing differences.

## Modal Workflow

Authenticate once:

```bash
uv run modal setup
```

Then run any starter by lab alias:

```bash
uv run run_modal.py
uv run run_modal.py 01 -- 1048576 256
uv run run_modal.py matmul -- 256
uv run run_modal.py transpose -- 2048 2048
```

The wrapper:

- defaults to Modal `T4`, the cheapest listed Modal GPU at the time this was written,
- uses `nvidia/cuda:12.4.0-devel-ubuntu22.04` so `nvcc` is available,
- copies the current repo into the Modal container,
- compiles with [common/compile_and_run.py](common/compile_and_run.py),
- prints the allocated GPU, compile command, and program output,
- updates `progress/runs.jsonl`, `progress/summary.json`, and `progress/summary.md`.

Override the GPU when needed:

```bash
uv run run_modal.py --gpu L4 04 -- 1024 1024 2
```

Skip progress tracking for a throwaway run:

```bash
uv run run_modal.py --no-progress 01
```

## Local Mac Note

Apple Silicon and most Macs cannot run CUDA kernels locally. Use the Mac for editing and use Colab/Modal/remote Linux for execution.

## Benchmarking Rules

- Report GPU model and CUDA version.
- Separate correctness runs from performance runs.
- Time kernels with CUDA events, not wall-clock Python timing.
- Include transfer time only when the assignment asks for end-to-end timing.
- Change one optimization at a time.
