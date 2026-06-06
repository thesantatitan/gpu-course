# Assignment 03: Memory Coalescing And Matrix Transpose

This assignment isolates GPU memory behavior by implementing matrix transpose in two ways.

## Why This Assignment Exists

Matrix transpose performs almost no arithmetic:

```text
output[col, row] = input[row, col]
```

Because the computation is so small, performance is dominated by memory access. This makes transpose a clean experiment for seeing:

- coalesced versus strided global-memory accesses,
- how shared memory can reorganize data,
- why shared-memory bank conflicts matter,
- how padding a shared-memory tile can improve throughput.

Unlike matrix multiplication, there is little computation to hide inefficient memory behavior.

## Reading

PMPP 4th edition:

- Chapter 4: compute architecture and scheduling,
- Chapter 5: memory architecture and data locality,
- Chapter 6: memory coalescing and performance considerations.

## Learning Objectives

After completing the assignment, you should be able to:

- identify coalesced and strided accesses made by a warp,
- transpose a row-major matrix correctly,
- use shared memory to separate global reads from global writes,
- explain why the tile is declared `TILE x (TILE + 1)`,
- calculate effective memory bandwidth,
- reason about rectangular and partial tiles.

## Data Layout

The input has:

- `height` rows,
- `width` columns.

Both input and output are stored as flat row-major arrays. The output shape is `width x height`.

The starter uses:

```cpp
#define TILE 32
```

## Your Tasks

Open [starter/main.cu](starter/main.cu) and implement two kernels.

### Task 1: Naive Transpose

Implement:

```cpp
__global__ void transpose_naive_kernel(
    const float* input,
    float* output,
    int width,
    int height)
```

Each thread should:

1. Calculate input coordinates `(x, y)`.
2. Check that they are inside the input.
3. Read `input[y * width + x]`.
4. Write it to `output[x * height + y]`.

This version should use global memory directly.

### Task 2: Shared-Memory Tiled Transpose

Implement:

```cpp
__global__ void transpose_tiled_kernel(
    const float* input,
    float* output,
    int width,
    int height)
```

Each block should:

1. Load a `TILE x TILE` input region into shared memory using coalesced global reads.
2. Synchronize.
3. Remap block coordinates for the transposed output tile.
4. Write rows from shared memory to the output using coalesced global writes.
5. Handle partial tiles at every boundary.

The shared array is already declared as:

```cpp
__shared__ float tile[TILE][TILE + 1];
```

The extra column is intentional; do not remove it before understanding bank conflicts.

## What The Starter Already Does

The harness:

- creates deterministic input,
- computes a full CPU transpose,
- allocates GPU memory,
- warms up both kernels before timing,
- clears output between implementations,
- reports correctness error,
- calculates effective bandwidth for both kernels.

## Run It

```bash
uv run run_modal.py 03
```

Defaults:

- `width = 1024`,
- `height = 1024`.

Override them:

```bash
uv run run_modal.py 03 -- 1000 777
```

Argument order:

```text
width height
```

## What Success Looks Like

```json
{
  "lab": "transpose",
  "ok": true,
  "width": 1024,
  "height": 1024,
  "naive_ms": 0.0,
  "tiled_ms": 0.0,
  "naive_gbps": 0.0,
  "tiled_gbps": 0.0,
  "naive_err": 0,
  "tiled_err": 0
}
```

Both error fields should be below `1e-6`.

## Required Correctness Tests

```bash
uv run run_modal.py 03 -- 1 1
uv run run_modal.py 03 -- 31 31
uv run run_modal.py 03 -- 32 32
uv run run_modal.py 03 -- 33 33
uv run run_modal.py 03 -- 1000 777
uv run run_modal.py 03 -- 777 1000
```

Rectangular inputs catch coordinate and output-stride mistakes that square matrices can hide.

## Benchmark Experiments

After correctness passes:

1. Compare naive and tiled kernels at `1024 x 1024`, `2048 x 2048`, and `4096 x 4096`.
2. Compare square and rectangular matrices with similar element counts.
3. Record `naive_gbps` and `tiled_gbps`.
4. Temporarily change `TILE` to `16`, rebuild, and compare.
5. As a controlled experiment, remove the `+ 1` padding and observe the tiled result.

Do only one code change at a time and restore the correct version afterward.

## What To Submit Or Record

- both completed kernels,
- a bandwidth table for several shapes,
- one comparison of tile sizes or shared-memory padding,
- answers to the reflection questions.

## Reflection Questions

- In the naive kernel, which operation is coalesced: input reads, output writes, both, or neither?
- How does shared memory allow both global operations to be coalesced?
- Why can a `32 x 32` shared tile suffer bank conflicts during transposed access?
- Why does adding one unused column change bank mapping?
- Why might rectangular matrices reveal bugs that square matrices do not?

## Common Pitfalls

- Allocating or indexing the output as if it had the same logical shape as the input.
- Reusing the input block coordinates for the output tile without swapping them.
- Checking input bounds but not output bounds after coordinate remapping.
- Reading shared memory before all threads have finished loading it.
- Assuming shared memory automatically improves performance without fixing global access patterns.

## Completion Checklist

- [ ] Naive transpose is correct for rectangular inputs.
- [ ] Tiled transpose is correct for partial tiles.
- [ ] Shared-memory synchronization is correct.
- [ ] Default run reports `"ok": true`.
- [ ] Effective bandwidth is recorded.
- [ ] Padding or tile-size experiment is explained.
