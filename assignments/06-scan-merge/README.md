# Assignment 06: Scan And Merge

## PMPP Chapters

- Chapter 11: prefix sum / scan.
- Chapter 12: merge.

## Goal

Build the parallel primitive that makes many "sequential-looking" algorithms parallel: prefix sum. Then use the same partitioning mindset to reason about merge.

## What You Implement

In [starter/main.cu](starter/main.cu), fill in:

- `block_exclusive_scan_kernel`

The starter checks a single-block exclusive scan first. After that passes, extend it to multiple blocks by scanning block sums and adding offsets.

For merge, write pseudocode or an implementation using the co-rank idea:

- Given sorted arrays `A` and `B`, partition the merged output range into chunks.
- For each chunk boundary, find how many elements should come from `A` and `B`.
- Have one block/thread group merge its chunk.

## Run

```bash
python3 common/compile_and_run.py assignments/06-scan-merge/starter/main.cu -- 1024 1024
```

Arguments:

- `n`
- `block_size`

## Experiments

- First use `n <= block_size`.
- Then extend to arbitrary `n`.
- Compare Kogge-Stone style and Brent-Kung style work counts on paper, even if you implement only one.

## Writeup Prompts

- Is your scan inclusive or exclusive?
- How many additions does your scan perform for one block?
- Where does synchronization happen?
- How would scan help implement stream compaction?
