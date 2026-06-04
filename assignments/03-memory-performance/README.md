# Assignment 03: Memory Performance

## PMPP Chapters

- Chapter 4: compute architecture and scheduling.
- Chapter 5: memory architecture and data locality.
- Chapter 6: performance considerations.

## Goal

See memory coalescing and shared-memory tiling directly by implementing matrix transpose two ways.

This is not the only way to study Chapters 5-6, but transpose makes the memory pattern painfully visible: one direction is naturally coalesced and the other is not unless you tile.

## What You Implement

In [starter/main.cu](starter/main.cu), fill in:

- `transpose_naive_kernel`
- `transpose_tiled_kernel`

## Run

```bash
python3 common/compile_and_run.py assignments/03-memory-performance/starter/main.cu -- 1024 1024
```

Arguments:

- `width`
- `height`

## Experiments

- Try square and rectangular matrices.
- Compute effective bandwidth: `2 * width * height * sizeof(float) / time`.
- Change `TILE` from 16 to 32 and compare.

## Writeup Prompts

- Which loads/stores are coalesced in the naive kernel?
- Why does the tiled kernel use `TILE + 1` in shared memory?
- Is the best tile size the same on every GPU you try?
