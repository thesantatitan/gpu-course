# Assignment 01: Vector Add

## PMPP Chapters

- Chapter 2: heterogeneous data-parallel computing.

## Goal

Implement the smallest useful CUDA program: one thread computes one output element.

## What You Implement

In [starter/main.cu](starter/main.cu), fill in:

- `vector_add_kernel`

Do not rewrite the harness on your first pass. The point is to make the thread indexing and boundary check automatic in your hands.

## Run

```bash
python3 common/compile_and_run.py assignments/01-vector-add/starter/main.cu -- 1048576 256
```

Arguments:

- `n`: vector length.
- `block_size`: threads per block.

## Experiments

- Try `n = 1000`, `1048576`, and `16777216`.
- Try block sizes `64`, `128`, `256`, `512`, `1024`.
- Record GPU time and max absolute error.

## Writeup Prompts

- Why is the `if (i < n)` check needed?
- Which block size was fastest on your GPU?
- Does the timing scale linearly with `n`?
