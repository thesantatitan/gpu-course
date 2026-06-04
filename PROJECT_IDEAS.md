# Final Project Ideas

Pick one project that is small enough to finish but rich enough to optimize. The best projects have a CPU baseline, a clear correctness metric, and at least three meaningful GPU design choices.

## Track A: Sorting Pipeline

- PMPP anchor: Chapter 13.
- Build: radix sort for unsigned integers or key-value pairs.
- Stretch: compare against merge sort or a library baseline.
- Good metrics: throughput, passes over memory, effect of radix width.

## Track B: Sparse Matrix-Vector Multiply

- PMPP anchor: Chapter 14.
- Build: COO and CSR SpMV.
- Stretch: add ELL or hybrid ELL-COO for structured sparse matrices.
- Good metrics: effective bandwidth, sensitivity to row length distribution.

## Track C: Graph BFS

- PMPP anchor: Chapter 15.
- Build: vertex-centric BFS on generated graphs.
- Stretch: add frontier-based traversal or privatized frontier construction.
- Good metrics: traversed edges per second, contention, graph structure sensitivity.

## Track D: CNN Convolution Layer

- PMPP anchor: Chapter 16.
- Build: direct convolution inference for a small CNN layer.
- Stretch: im2col plus GEMM-style formulation.
- Good metrics: GFLOP/s, memory traffic, batch-size sensitivity.

## Track E: Image Or Volume Stencil

- PMPP anchor: Chapters 7-8.
- Build: 2D blur/sharpen or 3D heat diffusion.
- Stretch: shared-memory tiling with halo cells and thread coarsening.
- Good metrics: effective bandwidth, halo overhead, tile shape sensitivity.

## Track F: Streaming Pipeline

- PMPP anchor: Chapters 20 and 22.
- Build: process chunks through copy-in, kernel, copy-out.
- Stretch: overlap transfers and compute with streams.
- Good metrics: end-to-end throughput, overlap efficiency, chunk-size sensitivity.

## Milestones

- Proposal: problem, baseline, data, correctness metric, expected bottleneck.
- Design doc: kernel mapping, memory layout, verification plan, profiling plan.
- Alpha: CPU baseline plus first correct GPU version.
- Beta: at least two optimizations and a benchmark table.
- Final: report, code, demo, and honest retrospective.

