# Assignment 01: Vector Add

Your first CUDA program maps a simple array operation onto thousands of GPU threads.

## Why This Assignment Exists

Vector addition is mathematically trivial:

```text
C[i] = A[i] + B[i]
```

That simplicity is useful. You can focus entirely on the mechanics that every later CUDA program uses:

- deciding which output a thread owns,
- converting CUDA thread coordinates into an array index,
- preventing out-of-bounds memory access,
- launching enough blocks to cover arbitrary input sizes,
- checking a GPU result against a CPU reference,
- measuring warmed-up kernel execution time.

Once this pattern is comfortable, the later assignments add multidimensional data, data reuse, synchronization, and communication between threads.

## Reading

PMPP 4th edition:

- Chapter 2, especially the vector-add example,
- device global memory and data transfer,
- kernel functions, thread indexing, and kernel launch configuration.

## Learning Objectives

After completing the assignment, you should be able to:

- explain the roles of host code and device code,
- calculate a global 1D thread index,
- choose a grid size using ceiling division,
- write a boundary-safe CUDA kernel,
- interpret CUDA block size and problem size independently,
- distinguish correctness results from performance results.

## Your Task

Open [starter/main.cu](starter/main.cu) and implement:

```cpp
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n)
```

Each CUDA thread should:

1. Calculate the single global element index assigned to it.
2. Check whether that index is smaller than `n`.
3. If it is valid, read one value from `a` and one from `b`.
4. Add them and store the result in `c`.
5. Do nothing if the index is outside the vector.

Do not add a loop during the first pass. The intended mapping is one thread per output element.

## What The Starter Already Does

The supplied harness:

- creates deterministic input vectors,
- computes the expected result on the CPU,
- allocates `d_a`, `d_b`, and `d_c`,
- copies inputs to the GPU,
- computes grid and block dimensions,
- performs an untimed warmup launch,
- times the measured launch with CUDA events,
- copies the output back,
- reports maximum absolute error and pass/fail status.

Your first implementation should change only `vector_add_kernel`.

## Run It

From the repository root:

```bash
uv run run_modal.py 01
```

The defaults are:

- `n = 1,048,576` elements,
- `block_size = 256` threads.

Override them after `--`:

```bash
uv run run_modal.py 01 -- 1000 128
```

Argument order:

```text
n block_size
```

## What Success Looks Like

A correct run ends with a result similar to:

```json
{"lab":"vector_add","ok":true,"n":1048576,"block_size":256,"gpu_ms":0.1,"max_abs_error":0}
```

The exact time depends on the GPU. The important correctness fields are:

- `"ok": true`,
- `max_abs_error` below `1e-5`.

Before implementing the TODO, `"ok": false` is expected.

## Required Correctness Tests

Use sizes that exercise both ordinary and boundary cases:

```bash
uv run run_modal.py 01 -- 1 256
uv run run_modal.py 01 -- 1000 256
uv run run_modal.py 01 -- 1024 256
uv run run_modal.py 01 -- 1025 256
uv run run_modal.py 01 -- 1048576 256
```

The `1000` and `1025` cases are important because `n` is not exactly covered by complete blocks.

## Benchmark Experiments

After all correctness tests pass:

1. Fix `n = 16,777,216`.
2. Try block sizes `64`, `128`, `256`, `512`, and `1024`.
3. Run each configuration at least three times.
4. Record the median `gpu_ms`.
5. Repeat with several vector lengths and observe how time scales.

Suggested commands:

```bash
uv run run_modal.py 01 -- 16777216 64
uv run run_modal.py 01 -- 16777216 128
uv run run_modal.py 01 -- 16777216 256
uv run run_modal.py 01 -- 16777216 512
uv run run_modal.py 01 -- 16777216 1024
```

## What To Submit Or Record

- your completed `vector_add_kernel`,
- a table containing `n`, block size, and median `gpu_ms`,
- answers to the reflection questions below.

## Reflection Questions

- Why can the grid contain more threads than there are vector elements?
- What failure can occur if the `i < n` check is omitted?
- Why does changing block size not change the mathematical answer?
- Is the kernel primarily limited by arithmetic or memory traffic?
- Does runtime scale approximately linearly with vector length?

## Common Pitfalls

- Using only `threadIdx.x` and forgetting `blockIdx.x`.
- Multiplying by `gridDim.x` instead of `blockDim.x`.
- Writing to `c[i]` before checking that `i < n`.
- Changing the CPU reference or tolerance to hide an incorrect result.
- Treating a very small `gpu_ms` from an empty kernel as a valid speed result.

## Completion Checklist

- [ ] Kernel uses a global 1D index.
- [ ] Kernel contains a boundary check.
- [ ] Non-divisible sizes pass.
- [ ] Default run reports `"ok": true`.
- [ ] Block-size benchmark table is recorded.
- [ ] Reflection questions are answered.
