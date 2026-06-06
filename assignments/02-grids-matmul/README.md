# Assignment 02: Multidimensional Grids And Matrix Multiplication

This assignment moves from one-dimensional arrays to two-dimensional data and then introduces shared-memory tiling.

## Why This Assignment Exists

Matrix multiplication is a classic GPU workload because each output element can be calculated independently, but every output requires many input values:

```text
C[row, col] = sum over k of A[row, k] * B[k, col]
```

A direct implementation is easy to parallelize but repeatedly reads the same values from global memory. A tiled implementation lets threads in a block cooperate: they load reusable pieces of `A` and `B` into shared memory, synchronize, and perform several arithmetic operations per global-memory load.

This assignment demonstrates the difference between:

- exposing parallelism,
- and organizing that parallelism to use the memory hierarchy efficiently.

## Reading

PMPP 4th edition:

- Chapter 3: multidimensional grids and data,
- Chapter 4 sections on blocks, warps, and synchronization,
- Chapter 5 sections on shared memory and tiled matrix multiplication.

## Learning Objectives

After completing the assignment, you should be able to:

- map `(blockIdx, threadIdx)` coordinates to matrix row and column indices,
- index row-major matrices stored in flat arrays,
- handle dimensions that are not multiples of the tile size,
- explain why naive matrix multiplication rereads data,
- cooperatively load shared-memory tiles,
- use `__syncthreads()` correctly,
- estimate matrix multiplication throughput in GFLOP/s.

## Data Layout

All matrices are square with shape `n x n` and use row-major storage:

```text
element at (row, col) -> array[row * n + col]
```

The starter uses:

```cpp
#define TILE 16
```

so a block contains `16 x 16 = 256` threads.

## Your Tasks

Open [starter/main.cu](starter/main.cu) and implement both kernels.

### Task 1: Basic Matrix Multiplication

Implement:

```cpp
__global__ void matmul_basic_kernel(const float* a, const float* b, float* c, int n)
```

Each thread should:

1. Determine its output `row` and `col`.
2. Return without writing if the output coordinate is outside the matrix.
3. Loop over `k = 0 ... n-1`.
4. Accumulate `A[row, k] * B[k, col]`.
5. Write one value to `C[row, col]`.

This kernel should use global memory directly. Do not tile it.

### Task 2: Shared-Memory Tiled Matrix Multiplication

Implement:

```cpp
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n)
```

For each tile phase, threads should cooperate to:

1. Load one element of an `A` tile into `a_tile`.
2. Load one element of a `B` tile into `b_tile`.
3. Store zero for loads that fall outside the matrix.
4. Synchronize before reading the tile.
5. Accumulate the products for that tile.
6. Synchronize again before shared memory is reused for the next phase.
7. Write the final result if `row` and `col` are valid.

The two synchronizations serve different purposes. Be prepared to explain both.

## What The Starter Already Does

The harness:

- generates deterministic matrices,
- computes a CPU reference for sizes up to `512`,
- uses deterministic sampled validation for larger matrices,
- allocates and copies device memory,
- launches both kernels once as untimed warmup,
- clears output between kernels,
- times basic and tiled kernels separately,
- reports errors, timings, and GFLOP/s.

Keep the harness unchanged until both kernels pass.

## Run It

Start with a small size:

```bash
uv run run_modal.py 02 -- 31
```

Argument order:

```text
n
```

The default is `n = 128`.

## What Success Looks Like

For `n <= 512`, the result includes:

```json
{
  "lab": "matmul",
  "ok": true,
  "n": 128,
  "check_mode": "full",
  "basic_ms": 0.0,
  "tiled_ms": 0.0,
  "basic_gflops": 0.0,
  "tiled_gflops": 0.0
}
```

For larger sizes, `"check_mode": "sampled"` means 128 deterministic output elements were checked against direct CPU dot products. Timing is meaningful only when `"ok": true`.

## Required Correctness Tests

```bash
uv run run_modal.py 02 -- 1
uv run run_modal.py 02 -- 15
uv run run_modal.py 02 -- 16
uv run run_modal.py 02 -- 17
uv run run_modal.py 02 -- 31
uv run run_modal.py 02 -- 64
uv run run_modal.py 02 -- 127
uv run run_modal.py 02 -- 128
```

The `15`, `17`, `31`, and `127` cases test partial edge tiles.

## Benchmark Experiments

After both kernels pass:

```bash
uv run run_modal.py 02 -- 512
uv run run_modal.py 02 -- 1024
uv run run_modal.py 02 -- 2048
```

For each size:

- record `basic_ms` and `tiled_ms`,
- compute or record tiled speedup,
- compare `basic_gflops` and `tiled_gflops`,
- repeat the run to estimate timing noise.

The harness warms up both kernels before timing so the first kernel does not unfairly pay CUDA context or module-loading overhead.

## What To Submit Or Record

- both completed kernels,
- a table with `n`, basic time, tiled time, speedup, and both GFLOP/s values,
- a short explanation of the global-memory traffic saved by tiling,
- answers to the reflection questions.

## Reflection Questions

- Which output element does thread `(threadIdx.y, threadIdx.x)` own?
- How many times can a loaded tile element be reused by threads in a block?
- Why must out-of-range tile loads write zero instead of doing nothing?
- What race or corruption can occur if either `__syncthreads()` is removed?
- Why is the tiled kernel not guaranteed to be faster for very small `n`?
- What resources could limit occupancy as tile size increases?

## Common Pitfalls

- Reversing row and column coordinates.
- Using `row * n + k` where `k * n + col` is required, or vice versa.
- Checking only the final output bounds but not tile-load bounds.
- Placing `__syncthreads()` inside a conditional taken by only some threads.
- Forgetting the second synchronization before loading the next tile.
- Benchmarking `n = 128` and drawing conclusions from launch-dominated timings.

## Completion Checklist

- [ ] Basic kernel passes non-divisible sizes.
- [ ] Tiled kernel passes non-divisible sizes.
- [ ] Both tile loads are boundary-safe.
- [ ] Synchronization is block-uniform.
- [ ] Default run reports `"ok": true`.
- [ ] Sizes `512`, `1024`, and `2048` are benchmarked.
- [ ] Results explain why performance changed.
