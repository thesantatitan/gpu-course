# Syllabus

This plan is built around PMPP 4e. It avoids trying to cover every advanced chapter equally; the core goal is to make you fluent enough to read and implement GPU kernels independently.

## Weekly Rhythm

- Lecture block: 75 minutes per chapter or major chapter pair.
- Code block: 2-4 hours implementing the starter.
- Performance block: 1-2 hours profiling, changing one thing at a time, and recording results.
- Reflection: 15 minutes answering "what parallel pattern did I use, where is the bottleneck, what would I try next?"

## Phase 0: Setup

### Week 0: Tooling And Baseline

- Reading: skim Chapter 1.
- Lab: verify CUDA runtime, `nvcc`, `nvidia-smi`, and the starter harness.
- Deliverable: one successful compile/run and a note listing GPU model, CUDA version, and platform.

## Phase 1: Guided CUDA Core

### Week 1: Heterogeneous Data Parallel Computing

- PMPP: Chapter 2.
- Concepts: host/device split, global memory allocation, copies, kernel launch syntax, thread indexing.
- Lab: [Assignment 01: Vector Add](assignments/01-vector-add/README.md).
- Exercise set:
  - Explain why the boundary check is needed.
  - Try block sizes 64, 128, 256, 512, 1024.
  - Measure how runtime changes with `n`.

### Week 2: Multidimensional Grids And Data

- PMPP: Chapter 3.
- Concepts: 2D/3D block and grid indexing, mapping data layout to thread layout, matrix multiplication basics.
- Lab: [Assignment 02: Grids And Matrix Multiply](assignments/02-grids-matmul/README.md).
- Exercise set:
  - Draw the mapping from `(blockIdx, threadIdx)` to matrix row/column.
  - Implement boundary-safe kernels for matrix sizes not divisible by tile size.

### Week 3: Compute Architecture And Scheduling

- PMPP: Chapter 4.
- Concepts: SMs, blocks, warps, divergence, latency hiding, occupancy.
- Lab continuation: improve Assignment 02 and add a short occupancy discussion.
- Exercise set:
  - Identify where your kernel has control divergence.
  - Estimate launched blocks, threads, and warps for three problem sizes.

### Week 4: Memory Architecture And Performance

- PMPP: Chapters 5-6.
- Concepts: memory hierarchy, shared memory, tiling, coalescing, occupancy tradeoffs, bottleneck diagnosis.
- Lab: [Assignment 03: Memory Performance](assignments/03-memory-performance/README.md).
- Exercise set:
  - Compare naive global-memory access with tiled/shared-memory access.
  - Record achieved bandwidth or effective GFLOP/s.

### Week 5: Convolution And Stencil

- PMPP: Chapters 7-8.
- Concepts: convolution, stencil sweeps, halo cells, constant memory, cache behavior, thread coarsening.
- Lab: [Assignment 04: Convolution And Stencil](assignments/04-convolution-stencil/README.md).
- Exercise set:
  - Implement zero-padding correctly at the image boundary.
  - Compare basic global-memory convolution with constant-memory mask access.

### Week 6: Histogram And Reduction

- PMPP: Chapters 9-10.
- Concepts: atomics, privatization, aggregation, reduction trees, divergence minimization.
- Lab: [Assignment 05: Histogram And Reduction](assignments/05-histogram-reduction/README.md).
- Exercise set:
  - Compare one global histogram with per-block privatized histograms.
  - Implement a reduction that handles arbitrary input lengths.

### Week 7: Scan And Merge

- PMPP: Chapters 11-12. Treat Chapter 11 as two lecture blocks if needed.
- Concepts: work efficiency, Kogge-Stone, Brent-Kung, segmented scan, dynamic input partitioning, merge co-rank.
- Lab: [Assignment 06: Scan And Merge](assignments/06-scan-merge/README.md).
- Exercise set:
  - Implement block-level exclusive scan first.
  - Extend it to arbitrary length with block sums.
  - Use scan as a building block in a compact/filter operation.

## Phase 2: Advanced Patterns And Project

### Week 8: Project Selection And Advanced Pattern Survey

- PMPP: Chapter 13 plus one of Chapters 14-18.
- Concepts: sorting, sparse formats, BFS, CNN convolution, MRI reconstruction, electrostatic potential maps.
- Deliverable: final project proposal from [Assignment 07](assignments/07-final-project/README.md).

### Week 9: Sparse, Graph, Or Deep Learning Track

- PMPP: choose one primary chapter from 14, 15, or 16.
- Deliverable: CPU baseline, data generator, correctness metric, and first GPU kernel.

### Week 10: Application-Specific Optimization

- PMPP: choose one secondary chapter from 13-18.
- Deliverable: performance profile and two optimization hypotheses.

### Week 11: Streams, Transfers, And System Behavior

- PMPP: Chapter 20 sections on overlap, plus Chapter 22 sections relevant to profiling and host/device interaction.
- Deliverable: add timing breakdown for transfers, kernel time, and end-to-end time.

### Week 12: Project Clinic

- PMPP: Chapter 19.
- Deliverable: optimized kernel, ablation table, and correctness stress tests.

### Week 13: Final Report Draft

- Deliverable: draft report with problem statement, design, results, and next steps.
- Optional topic: Chapter 21 if your project has nested/adaptive parallelism.

### Week 14: Final Demo And Retrospective

- Deliverable: final code, report, and 8-10 minute presentation.
- Final question: "What changed in how I think about decomposing computation?"
