# Assignments

The guided core contains six CUDA programming assignments followed by a final project. Complete assignments 01-06 in order: later labs assume that thread indexing, boundary checks, synchronization, and CUDA error handling from earlier labs are already familiar.

Each assignment directory contains:

- a detailed `README.md` explaining the task and experiments,
- `starter/main.cu` containing the harness and kernel TODOs.

## Guided Core

| Assignment | What You Build | Main Lesson |
| --- | --- | --- |
| [01: Vector Add](01-vector-add/README.md) | A boundary-safe 1D CUDA kernel | Thread indexing and host/device execution |
| [02: Grids And Matrix Multiply](02-grids-matmul/README.md) | Naive and shared-memory tiled matmul | 2D grids and data reuse |
| [03: Memory Performance](03-memory-performance/README.md) | Naive and tiled matrix transpose | Coalescing and shared-memory bank conflicts |
| [04: Convolution And Stencil](04-convolution-stencil/README.md) | Global-memory and constant-memory convolution | Neighborhood computations and read-only data |
| [05: Histogram And Reduction](05-histogram-reduction/README.md) | Atomic histogram and block-level reduction | Contention, atomics, and tree aggregation |
| [06: Scan And Merge](06-scan-merge/README.md) | Block-level exclusive prefix scan | Parallelizing sequential-looking dependencies |
| [07: Final Project](07-final-project/README.md) | A larger GPU application | End-to-end design, verification, and optimization |

## Standard Workflow

For every guided assignment:

1. Read the assignment README and corresponding PMPP chapter.
2. Open `starter/main.cu` and locate the `TODO` comments.
3. Run the untouched starter once so you understand the expected failure.
4. Implement the smallest correct kernel.
5. Test non-divisible and boundary-heavy input sizes.
6. Confirm that the JSON result contains `"ok": true`.
7. Run the listed performance experiments.
8. Write a short explanation of the result, not just a benchmark number.

The starter harness deliberately owns input generation, device allocation, CPU reference checking, warmup, and timing. During the first implementation pass, change only the requested kernel code.

## Correctness Before Performance

An unfinished kernel often reports a very small execution time because it performs no useful work. Treat timing results as meaningful only after:

- the CUDA program exits successfully,
- the result line says `"ok": true`,
- the reported error is within tolerance,
- edge cases and non-divisible sizes pass.

## Running A Starter

From the repository root:

```bash
uv run run_modal.py 01
```

Arguments after `--` are forwarded to the CUDA executable:

```bash
uv run run_modal.py 01 -- 1048576 256
```

Every assignment README lists its argument order and defaults.
