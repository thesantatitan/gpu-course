# Assignment 05: Histogram And Reduction

## PMPP Chapters

- Chapter 9: parallel histogram.
- Chapter 10: reduction and minimizing divergence.

## Goal

Implement two patterns where many threads combine information into fewer outputs:

- histogram: many inputs update shared bins,
- reduction: many inputs combine into one sum.

## What You Implement

In [starter/main.cu](starter/main.cu), fill in:

- `histogram_atomic_kernel`
- `reduce_sum_kernel`

Start with simple global atomics and a per-block tree reduction. Then optimize.

## Run

```bash
python3 common/compile_and_run.py assignments/05-histogram-reduction/starter/main.cu -- 1048576 256 256
```

Arguments:

- `n`
- `num_bins`
- `block_size`

## Experiments

- Try `num_bins = 16`, `256`, `4096`.
- Compare global atomic histogram with a privatized per-block histogram.
- Try reduction block sizes `128`, `256`, `512`, `1024`.

## Writeup Prompts

- What causes contention in the histogram?
- What does privatization trade for less contention?
- Why does a reduction tree reduce divergence compared with a naive pattern?
