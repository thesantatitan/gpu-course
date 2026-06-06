# Assignment 05: Parallel Histogram And Reduction

This assignment contains two related many-to-few parallel patterns:

- histogram: many input elements update a smaller set of bins,
- reduction: many input elements combine into a single value.

## Why This Assignment Exists

Earlier assignments gave each thread an independent output. Histogram and reduction require threads to cooperate or contend for shared results.

A histogram exposes concurrent updates:

```text
bins[data[i]] += 1
```

Many threads may target the same bin, so an ordinary increment is incorrect.

A reduction exposes a different dependency:

```text
sum = input[0] + input[1] + ... + input[n - 1]
```

A sequential loop works, but a GPU needs a tree-shaped algorithm that combines partial results in parallel.

Together, these tasks introduce atomics, shared memory, synchronization, contention, and hierarchical aggregation.

## Reading

PMPP 4th edition:

- Chapter 9: atomic operations, histogram privatization, coarsening, and aggregation,
- Chapter 10: reduction trees, divergence, shared memory, and hierarchical reduction.

## Learning Objectives

After completing the assignment, you should be able to:

- explain why a histogram increment has a data race,
- use `atomicAdd` for concurrent updates,
- describe contention and privatization,
- load values into dynamic shared memory,
- perform an in-block tree reduction,
- place `__syncthreads()` correctly,
- distinguish block-level partial sums from a complete global reduction.

## Your Tasks

Open [starter/main.cu](starter/main.cu) and implement both kernels.

### Task 1: Global Atomic Histogram

Implement:

```cpp
__global__ void histogram_atomic_kernel(
    const unsigned int* data,
    unsigned int* bins,
    int n,
    int num_bins)
```

Each thread should:

1. Calculate its global input index.
2. Check whether that index is smaller than `n`.
3. Read `data[i]`.
4. Atomically increment the corresponding global bin.

The generated input values are always in `[0, num_bins)`.

This first implementation intentionally uses global atomics. Privatization is an optimization experiment, not a requirement for initial correctness.

### Task 2: Per-Block Sum Reduction

Implement:

```cpp
__global__ void reduce_sum_kernel(
    const float* input,
    float* block_sums,
    int n)
```

The kernel receives dynamic shared memory through:

```cpp
extern __shared__ float scratch[];
```

Each block should:

1. Have each thread load one input value into shared memory.
2. Load zero when the global index is outside `n`.
3. Synchronize after loading.
4. Repeatedly combine pairs of values using a reduction tree.
5. Synchronize between reduction steps.
6. Have thread `0` write one partial sum to `block_sums[blockIdx.x]`.

The harness copies block sums to the CPU and performs the final small sum there. The required kernel is therefore a block-level reduction, not a full recursive GPU reduction.

## What The Starter Already Does

The harness:

- generates deterministic histogram inputs and floating-point values,
- computes CPU histogram and sum references,
- allocates bins and block sums,
- provides dynamic shared-memory size at kernel launch,
- warms up both kernels,
- clears accumulation buffers before measured launches,
- checks every histogram bin,
- reports histogram and reduction timing separately.

## Run It

```bash
uv run run_modal.py 05
```

Defaults:

- `n = 1,048,576`,
- `num_bins = 256`,
- `block_size = 256`.

Override them:

```bash
uv run run_modal.py 05 -- 1000000 64 256
```

Argument order:

```text
n num_bins block_size
```

## What Success Looks Like

```json
{
  "lab": "hist_reduce",
  "ok": true,
  "n": 1048576,
  "num_bins": 256,
  "block_size": 256,
  "hist_ms": 0.0,
  "reduce_ms": 0.0,
  "hist_ok": true,
  "reduce_ok": true,
  "sum_error": 0.0
}
```

Both `hist_ok` and `reduce_ok` must be true.

## Required Correctness Tests

```bash
uv run run_modal.py 05 -- 1 1 128
uv run run_modal.py 05 -- 1000 16 128
uv run run_modal.py 05 -- 1025 256 256
uv run run_modal.py 05 -- 1048576 256 256
uv run run_modal.py 05 -- 1048577 4096 512
```

The `1025` and `1048577` cases test partial final blocks.

## Benchmark Experiments

### Histogram Experiments

At a fixed large `n`, compare:

- `num_bins = 16`,
- `num_bins = 256`,
- `num_bins = 4096`.

Explain how the number of bins changes atomic contention.

### Reduction Experiments

Compare block sizes:

- `128`,
- `256`,
- `512`,
- `1024`.

Record both timing and number of output block sums.

## Optional Extension: Privatized Histogram

Implement a second histogram kernel in which each block accumulates into a private shared-memory histogram and later merges into global bins.

Measure:

- reduced global atomic contention,
- extra shared-memory usage,
- initialization and merge overhead,
- behavior when `num_bins` becomes too large for efficient privatization.

## Optional Extension: Fully GPU Hierarchical Reduction

Instead of completing the final sum on the CPU, repeatedly reduce `block_sums` on the GPU until one value remains.

Your design must handle:

- arbitrary input length,
- changing grid size each pass,
- synchronization between kernel launches,
- avoiding unnecessary host/device copies.

## What To Submit Or Record

- both required kernels,
- histogram timings across bin counts,
- reduction timings across block sizes,
- an explanation of contention and reduction-tree synchronization,
- optional optimized implementations,
- answers to the reflection questions.

## Reflection Questions

- Why is `bins[data[i]]++` incorrect without an atomic operation?
- Why does a small number of popular bins increase contention?
- What does histogram privatization gain, and what resources does it consume?
- Why should out-of-range reduction threads load zero?
- How many active values remain after each reduction step?
- Why can thread `0` safely write the block sum only after the tree completes?

## Common Pitfalls

- Forgetting to clear bins before an accumulation launch.
- Using a normal increment instead of `atomicAdd`.
- Returning early from some reduction threads before a later `__syncthreads()`.
- Reading shared memory before every thread has initialized its slot.
- Reducing only powers-of-two test sizes and missing boundary bugs.
- Expecting one block-level reduction launch to produce a single global result.

## Completion Checklist

- [ ] Histogram passes all-bin comparison.
- [ ] Reduction handles partial blocks by loading zero.
- [ ] Shared-memory synchronization is correct.
- [ ] Default run reports `"ok": true`.
- [ ] Bin-count contention experiment is recorded.
- [ ] Reduction block-size experiment is recorded.
