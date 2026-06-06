# Assignment 07: Final GPU Project

The final project asks you to apply the course’s patterns to a larger computation that requires design choices, profiling, and multiple optimization iterations.

## Why This Project Exists

The guided assignments isolate one CUDA idea at a time. Real GPU work is less tidy. You must decide:

- where the parallelism is,
- how work maps to threads and blocks,
- which data layout supports efficient access,
- where synchronization or atomics are required,
- what should remain on the CPU,
- how correctness will be established,
- which bottleneck is worth optimizing.

The final project is where those decisions become the main task rather than supplied boilerplate.

## Learning Objectives

By the end of the project, you should be able to:

- turn an application-level problem into one or more GPU kernels,
- create a trustworthy CPU baseline or reference,
- define measurable correctness and performance goals,
- profile before optimizing,
- compare alternative kernel or data-layout designs,
- explain performance using GPU architecture concepts,
- communicate results in a concise technical report and demo.

## Choosing A Project

A good project is:

- narrow enough to finish in six or seven weeks,
- large enough to require more than one trivial kernel,
- measurable with a CPU baseline and clear inputs,
- verifiable with deterministic tests or numerical tolerances,
- rich enough to support at least two meaningful optimization ideas.

Good scope:

```text
Implement CSR sparse matrix-vector multiplication and measure how row-length
distribution affects load balance and bandwidth.
```

Scope that is too broad:

```text
Do graph algorithms on a GPU.
```

Suggested tracks are listed in [PROJECT_IDEAS.md](../../PROJECT_IDEAS.md).

## Required Deliverables

### 1. Proposal

Write approximately one page covering:

- the problem,
- why GPU acceleration is relevant,
- the exact input and output,
- the CPU baseline,
- the correctness metric,
- the initial GPU approach,
- the expected bottleneck,
- a realistic minimum result and one stretch goal.

### 2. Design Document

Complete [DESIGN_DOC.md](DESIGN_DOC.md) before heavy implementation.

The document should specify:

- thread and block mapping,
- data structures and memory layout,
- kernel boundaries,
- expected synchronization or atomic operations,
- verification tests,
- benchmark sizes,
- project milestones.

The design may change after profiling. The purpose is to make assumptions explicit.

### 3. Correct CPU Baseline

Provide a serial or existing trusted implementation that:

- produces the expected answer,
- handles the same input format as the GPU version,
- is simple enough to trust,
- provides a meaningful runtime baseline when practical.

For library comparisons, state library version and configuration.

### 4. First Correct GPU Version

Build the simplest GPU implementation that passes correctness tests.

Do not optimize before this milestone. Record:

- kernel time,
- transfer time,
- end-to-end time,
- speedup relative to the chosen baseline,
- GPU and input configuration.

### 5. Profiling And Optimization

Form at least two concrete hypotheses, such as:

- accesses are uncoalesced,
- atomic contention dominates,
- blocks have uneven work,
- shared-memory reuse can reduce traffic,
- transfer time dominates small inputs,
- occupancy is limited by registers or shared memory.

Change one major factor at a time and keep an ablation table showing the effect.

### 6. Final Code And Tests

The final repository should include:

- reproducible input generation or included small fixtures,
- CPU reference or baseline,
- GPU implementation,
- correctness tests,
- benchmark command or script,
- instructions for reproducing the main result.

### 7. Final Report

Use [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md).

The report should include:

- problem and motivation,
- algorithm and GPU mapping,
- verification approach,
- hardware/software environment,
- benchmark methodology,
- results and plots or tables,
- optimization ablations,
- bottleneck analysis,
- limitations and future work.

### 8. Demo

Prepare an 8-10 minute demonstration:

1. Explain the problem.
2. Show the thread/data mapping.
3. Demonstrate correctness.
4. Present the most important benchmark.
5. Explain the optimization that mattered most.
6. State one limitation honestly.

## Suggested Timeline

### Week 8: Proposal

- select the problem,
- identify reference material,
- define input sizes and correctness criteria.

### Week 9: Baseline

- implement or validate CPU baseline,
- build input generator,
- create small deterministic tests,
- implement first GPU kernel.

### Week 10: First Correct GPU Version

- pass correctness tests,
- collect initial timing breakdown,
- identify likely bottlenecks.

### Week 11: Profiling

- gather profiler or timing evidence,
- write two optimization hypotheses,
- choose the highest-value change.

### Week 12: Optimization And Stress Tests

- implement optimizations one at a time,
- test larger and irregular inputs,
- record ablations.

### Week 13: Report Draft

- produce final tables and figures,
- write design and analysis sections,
- verify reproducibility from a clean environment.

### Week 14: Final Submission And Demo

- finish report,
- clean code and instructions,
- give demo,
- write retrospective.

## Benchmark Requirements

Every reported benchmark should state:

- GPU model,
- CUDA version or container image,
- input size and shape,
- warmup procedure,
- number of measured runs,
- whether timing is kernel-only or end-to-end,
- correctness status,
- baseline used for speedup.

Use medians or another clearly stated summary when timings vary. Never compare a warmed-up optimized kernel against a cold baseline launch.

## Minimum Acceptance Criteria

A complete project must have:

- a defined, nontrivial computation,
- a correct CPU reference or trusted baseline,
- at least one correct GPU implementation,
- tests covering ordinary and edge cases,
- warmed-up benchmark measurements,
- at least two investigated optimization ideas,
- a report explaining results rather than only listing them,
- reproducible run instructions.

The project does not need to beat a highly tuned vendor library. A careful explanation of why it does not can be an excellent result.

## Evaluation Rubric

- Correctness and verification: 30%.
- GPU design and implementation: 25%.
- Performance methodology and reasoning: 25%.
- Reproducibility and code clarity: 10%.
- Report and demo: 10%.

## Common Failure Modes

- Choosing a project with no reliable correctness oracle.
- Starting with a problem too large to debug.
- Reporting speedup without defining the baseline.
- Timing CUDA context initialization or data generation as kernel work.
- Optimizing several things simultaneously and losing causal evidence.
- Comparing different precision, algorithms, or output quality without disclosure.
- Spending the entire project building infrastructure and never reaching a correct GPU kernel.

## Final Retrospective Questions

- Which original performance hypothesis was wrong?
- Which optimization produced the largest improvement, and why?
- Where does the final implementation still waste work or bandwidth?
- Which result depends strongly on the GPU or input distribution?
- What would you change with one additional week?
