# Assignment 04: Convolution And Stencil

## PMPP Chapters

- Chapter 7: convolution, constant memory, caching.
- Chapter 8: stencil, halo regions, coarsening.

## Goal

Implement a correct 2D convolution first, then improve mask access using constant memory. Treat this as the bridge from "one output per thread" to "one output per thread with neighborhood data."

## What You Implement

In [starter/main.cu](starter/main.cu), fill in:

- `conv2d_basic_kernel`
- `conv2d_constant_kernel`

The starter uses zero-padding at image boundaries.

## Run

```bash
python3 common/compile_and_run.py assignments/04-convolution-stencil/starter/main.cu -- 512 512 1
```

Arguments:

- `width`
- `height`
- `radius`

## Experiments

- Try radius `1`, `2`, and `4`.
- Compare basic mask reads with constant-memory mask reads.
- Optional: add a shared-memory tile with halo cells.

## Writeup Prompts

- Which input pixels does one output depend on?
- How do boundary checks affect divergence?
- Why is the mask a good fit for constant memory?
