# Assignment 02: Grids And Matrix Multiply

## PMPP Chapters

- Chapter 3: multidimensional grids and multidimensional data.
- Chapter 5 preview: shared-memory tiling.

## Goal

Map a 2D grid of CUDA threads onto a square matrix multiplication. First implement the naive version, then implement the tiled version.

## What You Implement

In [starter/main.cu](starter/main.cu), fill in:

- `matmul_basic_kernel`
- `matmul_tiled_kernel`

Keep the CPU reference and checks unchanged until both kernels pass.

## Run

```bash
uv run run_modal.py 02 -- 128
```

Argument:

- `n`: square matrix width/height.

## Experiments

- Correctness sizes: `31`, `64`, `127`, `128`.
- Performance sizes after both kernels pass: `512`, `1024`, `2048` if your runtime can handle them.
- Compare basic and tiled kernel time.

The starter runs untimed warmup launches for both kernels before measuring. It uses full CPU-reference checking for `n <= 512` and deterministic sampled checking for larger sizes, so benchmarking `1024` and `2048` does not get dominated by cubic CPU validation.

## Writeup Prompts

- Which thread computes `C[row, col]`?
- How does your tiled kernel reduce global-memory reads?
- What happens when `n` is not divisible by `TILE`?
