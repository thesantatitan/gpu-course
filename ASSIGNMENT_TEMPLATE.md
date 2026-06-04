# Assignment Template

Each assignment follows the same contract so you can spend attention on CUDA decisions instead of scaffolding.

## Starter Layout

```text
assignments/XX-name/
  README.md
  starter/
    main.cu
```

The starter should include:

- deterministic input generation,
- CPU reference implementation,
- CUDA allocation/copy/launch boilerplate,
- correctness comparison,
- untimed warmup launches before timed kernel measurements,
- CUDA-event timing,
- a compact JSON-like result line.

## Your Submission

For each lab, submit:

- modified `.cu` files,
- benchmark table,
- short writeup.

The writeup should answer:

- What data-parallel mapping did you use?
- What are the main global-memory access patterns?
- What bottleneck do you think dominates?
- Which optimization helped most?
- What result surprised you?

## Rubric

- Correctness: 50%.
- Performance reasoning: 25%.
- Code clarity: 15%.
- Writeup: 10%.

For self-study, use the rubric as a checklist rather than a grade. If correctness fails, performance numbers do not matter yet.
