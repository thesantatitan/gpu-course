# Assignment 04: Convolution, Constant Memory, And Stencil Thinking

This assignment introduces neighborhood computations: one output depends on several nearby input elements rather than a single input element.

## Why This Assignment Exists

Convolution appears in image processing, signal processing, and neural networks. Stencil computations use a similar neighborhood structure in scientific simulations.

For a mask radius `r`, each output pixel combines a square neighborhood:

```text
mask width = 2 * r + 1
output[y, x] =
    sum over dy and dx of input[y + dy, x + dx] * mask[dy, dx]
```

This creates new GPU-programming questions:

- How should threads handle missing neighbors at image boundaries?
- Which data is shared by many threads?
- Where should a small, read-only filter mask live?
- How do boundary branches and repeated neighborhood reads affect performance?

## Reading

PMPP 4th edition:

- Chapter 7: convolution, constant memory, and caching,
- Chapter 8: stencil sweeps, halo cells, and thread coarsening.

## Learning Objectives

After completing the assignment, you should be able to:

- map a 2D image onto a 2D CUDA grid,
- implement zero-padded neighborhood access,
- flatten two-dimensional mask and image coordinates,
- use CUDA constant memory,
- explain constant-memory broadcast behavior,
- identify halo data and redundant neighborhood reads,
- describe how shared-memory tiling could improve locality.

## Convolution Definition

The starter treats out-of-range image elements as zero. This is called zero padding.

The mask is stored row-major. For offsets `dx` and `dy` in the range `[-radius, radius]`:

```text
mask row = dy + radius
mask col = dx + radius
mask index = mask_row * mask_width + mask_col
```

The basic and constant-memory kernels must produce identical output.

## Your Tasks

Open [starter/main.cu](starter/main.cu) and implement both kernels.

### Task 1: Basic Global-Memory Convolution

Implement:

```cpp
__global__ void conv2d_basic_kernel(
    const float* input,
    const float* mask,
    float* output,
    int width,
    int height,
    int radius)
```

Each thread should:

1. Determine one output coordinate `(x, y)`.
2. Return if the output coordinate is outside the image.
3. Loop over every mask offset.
4. Calculate the corresponding input coordinate.
5. Include the product only when the input coordinate is valid.
6. Write the accumulated value to the output.

Read the mask through the `mask` pointer in this kernel.

### Task 2: Constant-Memory Mask

Implement:

```cpp
__global__ void conv2d_constant_kernel(
    const float* input,
    float* output,
    int width,
    int height,
    int radius)
```

The algorithm and boundary behavior should match Task 1, but mask values must be read from:

```cpp
__constant__ float c_mask[]
```

The host harness already copies the mask with `cudaMemcpyToSymbol`.

## What The Starter Already Does

The harness:

- creates deterministic image data,
- creates an averaging mask,
- computes a full CPU reference,
- copies the mask to global and constant memory,
- warms up both kernels before timing,
- resets output between kernels,
- reports both errors and execution times.

The maximum supported radius is `4`, corresponding to a `9 x 9` mask.

## Run It

```bash
uv run run_modal.py 04
```

Defaults:

- `width = 512`,
- `height = 512`,
- `radius = 1`.

Override them:

```bash
uv run run_modal.py 04 -- 1024 768 2
```

Argument order:

```text
width height radius
```

## What Success Looks Like

```json
{
  "lab": "conv2d",
  "ok": true,
  "width": 512,
  "height": 512,
  "radius": 1,
  "basic_ms": 0.0,
  "constant_ms": 0.0,
  "basic_err": 0,
  "constant_err": 0
}
```

Both errors must be below `1e-4`.

## Required Correctness Tests

```bash
uv run run_modal.py 04 -- 1 1 0
uv run run_modal.py 04 -- 7 5 1
uv run run_modal.py 04 -- 17 19 2
uv run run_modal.py 04 -- 512 512 1
uv run run_modal.py 04 -- 512 512 4
```

Small images and larger radii stress the boundary logic.

## Benchmark Experiments

After both kernels pass:

1. Compare radius `1`, `2`, and `4` at a fixed image size.
2. Compare `512 x 512`, `1024 x 1024`, and `2048 x 2048`.
3. Record `basic_ms`, `constant_ms`, and speedup.
4. Explain whether the benefit of constant memory changes with mask size.

Suggested commands:

```bash
uv run run_modal.py 04 -- 2048 2048 1
uv run run_modal.py 04 -- 2048 2048 2
uv run run_modal.py 04 -- 2048 2048 4
```

## Optional Extension: Shared-Memory Halo Tile

After the required kernels work, add a third implementation that caches image tiles and halo cells in shared memory.

Before coding, specify:

- output tile dimensions,
- input tile dimensions including halo,
- which threads load halo values,
- how out-of-range halo values become zero,
- how many times each cached input may be reused.

This extension connects convolution directly to the stencil techniques in Chapter 8.

## What To Submit Or Record

- both required kernels,
- a timing table across image and mask sizes,
- an explanation of why constant memory is appropriate for the mask,
- optional shared-memory design or implementation,
- answers to the reflection questions.

## Reflection Questions

- Which input region is required to compute one output pixel?
- Why do neighboring output threads read many of the same input values?
- Why is the mask a better constant-memory candidate than the image?
- When does constant-memory broadcast work especially well?
- How do image boundaries cause warp divergence?
- What is a halo cell, and why does a tiled implementation need it?

## Common Pitfalls

- Mixing up `dx`/`dy` or `x`/`y`.
- Using image width as mask width.
- Reading an out-of-range input before checking its coordinates.
- Applying a different boundary rule in the two kernels.
- Forgetting that radius `r` produces mask width `2r + 1`.
- Assuming constant memory removes repeated image reads; it only changes mask access.

## Completion Checklist

- [ ] Basic kernel matches zero-padded CPU convolution.
- [ ] Constant-memory kernel produces the same output.
- [ ] Non-square and small images pass.
- [ ] Radius `4` passes.
- [ ] Default run reports `"ok": true`.
- [ ] Constant-memory benchmark results are explained.
