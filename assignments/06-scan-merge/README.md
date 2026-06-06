# Assignment 06: Prefix Scan And Parallel Merge

This assignment studies prefix scan, a foundational parallel primitive, and connects it to the partitioning ideas used in parallel merge.

## Why This Assignment Exists

An exclusive prefix sum transforms:

```text
input:  [a, b, c, d]
output: [0, a, a+b, a+b+c]
```

Each output appears to depend on all previous inputs, which looks sequential. Parallel scan reorganizes those dependencies into a tree.

Scan is used inside:

- stream compaction and filtering,
- radix sort,
- histogram prefix offsets,
- sparse data-structure construction,
- graph frontier generation,
- parallel allocation and work queues.

Merge introduces another important idea: input positions may not be known until runtime, so threads must determine which slices of sorted inputs belong to their output range.

## Reading

PMPP 4th edition:

- Chapter 11: Kogge-Stone scan, Brent-Kung scan, work efficiency, segmented and single-pass scan,
- Chapter 12: co-rank partitioning and parallel merge.

Chapter 11 is dense. It is reasonable to study it over two sessions.

## Learning Objectives

After completing the assignment, you should be able to:

- distinguish inclusive and exclusive scan,
- implement a synchronized block-level scan,
- reason about depth and total work,
- handle inactive threads using identity values,
- explain how block sums extend scan to arbitrary lengths,
- describe co-rank partitioning for parallel merge,
- identify applications that can be built from scan.

## Required Task: Block-Level Exclusive Scan

Open [starter/main.cu](starter/main.cu) and implement:

```cpp
__global__ void block_exclusive_scan_kernel(
    const float* input,
    float* output,
    float* block_sums,
    int n)
```

The starter provides dynamic shared memory:

```cpp
extern __shared__ float scratch[];
```

### Milestone 1: One Block

First make the kernel correct for:

```text
n <= block_size
grid_size == 1
```

Your algorithm should:

1. Load one input per thread, or zero when out of range.
2. Synchronize.
3. Perform a parallel scan in shared memory.
4. Convert the result to an exclusive scan if your internal algorithm is inclusive.
5. Write valid output elements.
6. Store the total block sum in `block_sums[0]`.

You may implement Kogge-Stone or Brent-Kung. State which one you chose.

### Milestone 2: Multiple Blocks

Extend the design for arbitrary `n`:

1. Each block scans its local chunk and writes a block total.
2. The block totals are themselves scanned.
3. Each block adds its scanned block offset to its local outputs.

This usually requires additional kernel launches or helper kernels. You may modify the harness after the single-block implementation passes.

## Merge Component

The starter does not contain a required merge kernel. For the Chapter 12 portion, produce either:

- clear pseudocode and a worked example of co-rank partitioning, or
- an optional CUDA merge implementation.

Your merge design should explain:

1. How the merged output is divided into contiguous chunks.
2. How a chunk boundary determines positions in sorted arrays `A` and `B`.
3. How a co-rank search finds those positions.
4. How a thread or block merges its assigned input slices.
5. Why independently produced output chunks remain globally sorted.

## What The Starter Already Does

The supplied scan harness:

- creates deterministic positive input values,
- computes a CPU exclusive scan,
- allocates output and block-sum buffers,
- launches with dynamic shared memory,
- performs an untimed warmup,
- resets output buffers,
- reports maximum absolute error and scan time.

The default input uses one full block so you can finish Milestone 1 before changing the harness.

## Run It

```bash
uv run run_modal.py 06
```

Defaults:

- `n = 1024`,
- `block_size = 1024`.

Override them:

```bash
uv run run_modal.py 06 -- 512 512
```

Argument order:

```text
n block_size
```

## What Success Looks Like

```json
{
  "lab": "scan",
  "ok": true,
  "n": 1024,
  "block_size": 1024,
  "grid_size": 1,
  "scan_ms": 0.0,
  "max_abs_error": 0
}
```

For the original starter, begin by requiring `"ok": true` when `grid_size` is `1`.

## Required Correctness Tests For Milestone 1

```bash
uv run run_modal.py 06 -- 1 128
uv run run_modal.py 06 -- 31 32
uv run run_modal.py 06 -- 32 32
uv run run_modal.py 06 -- 127 128
uv run run_modal.py 06 -- 512 512
uv run run_modal.py 06 -- 1024 1024
```

Pay special attention to the first output element: an exclusive scan must begin with zero.

## Correctness Tests For Milestone 2

After extending the harness:

```bash
uv run run_modal.py 06 -- 1025 256
uv run run_modal.py 06 -- 10000 256
uv run run_modal.py 06 -- 1048576 256
```

These should produce `grid_size > 1`.

## Benchmark And Analysis

For your chosen scan algorithm:

- report block size and `scan_ms`,
- state the number of synchronization rounds,
- estimate the number of additions,
- compare Kogge-Stone and Brent-Kung work counts on paper,
- explain whether your implementation prioritizes depth or work efficiency.

Optional application experiment:

1. Create a boolean keep/discard flag for each input.
2. Exclusive-scan the flags.
3. Use scanned positions to compact kept elements into an output array.

## What To Submit Or Record

- working block-level exclusive scan,
- multi-block extension if completed,
- algorithm diagram or explanation,
- work/depth comparison,
- merge co-rank pseudocode or optional implementation,
- answers to the reflection questions.

## Reflection Questions

- What is the difference between inclusive and exclusive scan?
- What identity value should an inactive addition thread contribute?
- Why is synchronization required between scan steps?
- How much work does your algorithm perform for a block of size `B`?
- Why can blocks not independently finish a global scan without offsets?
- How does co-rank convert an output boundary into two input boundaries?
- How does scan enable stream compaction?

## Common Pitfalls

- Returning early before a block-wide `__syncthreads()`.
- Accidentally producing an inclusive result.
- Updating shared memory in place without preserving values required by other threads.
- Assuming block synchronization works across different blocks.
- Writing local scans correctly but forgetting block offsets.
- Testing only powers of two without testing a partially filled block.

## Completion Checklist

- [ ] Single-block exclusive scan begins with zero.
- [ ] Partial blocks use zero as the identity value.
- [ ] Shared-memory synchronization is correct.
- [ ] Default run reports `"ok": true`.
- [ ] Work and synchronization depth are explained.
- [ ] Multi-block design is documented or implemented.
- [ ] Merge co-rank partitioning is explained.
