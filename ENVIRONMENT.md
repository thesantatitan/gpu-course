# Environment

The assignments are plain CUDA C++ plus a small Python compile wrapper. That keeps the course portable across Colab, Modal, cloud VMs, and any Linux machine with NVIDIA CUDA.

## Minimum Requirements

- NVIDIA GPU.
- CUDA toolkit with `nvcc`.
- Python 3.9+.
- A shell where you can run `python3 common/compile_and_run.py ...`.

Check the runtime:

```bash
nvidia-smi
nvcc --version
python3 --version
```

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

Use a CUDA-enabled image and run the same commands inside it. The course does not depend on a special Python package beyond the standard library. A good Modal setup should provide:

- CUDA toolkit, not only CUDA runtime.
- `nvcc` on `PATH`.
- this repo mounted or copied into the container.

## Local Mac Note

Apple Silicon and most Macs cannot run CUDA kernels locally. Use the Mac for editing and use Colab/Modal/remote Linux for execution.

## Benchmarking Rules

- Report GPU model and CUDA version.
- Separate correctness runs from performance runs.
- Time kernels with CUDA events, not wall-clock Python timing.
- Include transfer time only when the assignment asks for end-to-end timing.
- Change one optimization at a time.

