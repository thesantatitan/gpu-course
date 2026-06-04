# PMPP Self-Study GPU Course

This repo turns *Programming Massively Parallel Processors, 4th ed.* into a self-study course with guided programming assignments. The design follows the book's own two-phase teaching shape:

- Phase 1: fundamentals and reusable parallel patterns from Parts I-II, with weekly starter-code labs.
- Phase 2: advanced patterns/applications from Parts III-IV, tied together by a final project.

The target experience is like a good computer graphics course assignment: the harness, input generation, correctness checks, timing, and compile commands are already there, so your main work is implementing the kernel and explaining the performance.

## Course Shape

- Primary pace: 14 weeks, about 8-12 hours per week.
- Faster pace: combine two adjacent weeks and finish the guided core in 7-8 weeks.
- Each week has one or two 75-minute "lecture blocks": read the assigned chapter sections actively, work through the code/data mapping by hand, then do the lab.
- Each programming assignment produces three artifacts:
  - implemented CUDA code,
  - a small benchmark table,
  - a short writeup explaining correctness and the bottleneck.

## Files

- [SYLLABUS.md](SYLLABUS.md): week-by-week course plan.
- [ENVIRONMENT.md](ENVIRONMENT.md): Colab/Modal/local CUDA workflow.
- [ASSIGNMENT_TEMPLATE.md](ASSIGNMENT_TEMPLATE.md): standard lab contract and grading rubric.
- [PROJECT_IDEAS.md](PROJECT_IDEAS.md): final project tracks and milestones.
- [assignments](assignments): guided labs and starter code.
- [common](common): reusable CUDA helpers and compile script.

## How To Use A Lab

From the repo root, run CUDA on Modal with one command:

```bash
uv run run_modal.py
```

That default command runs a tiny CUDA smoke test. To run a course lab, pass a lab alias:

```bash
uv run run_modal.py 01 -- 1048576 256
```

The default GPU is Modal `T4`, because it is currently the cheapest Modal GPU that can compile and run these CUDA labs. The Modal image is an NVIDIA CUDA `devel` image so `nvcc` is available in the container.

For local CUDA machines or Colab, compile and run directly:

```bash
python3 common/compile_and_run.py assignments/01-vector-add/starter/main.cu -- 1048576 256
```

At first, most starters intentionally fail because the kernels contain TODOs. Your loop is:

1. Read the chapter sections listed in the assignment.
2. Implement only the TODO kernels unless the assignment asks for more.
3. Run the correctness check.
4. Benchmark a few sizes/block shapes.
5. Write down what changed and why.

On Colab, copy the same folder structure into the runtime or mount this repo from Drive. On Modal, use `uv run run_modal.py ...`; it ships the repo into a CUDA container, compiles the selected CUDA file, and prints the results.

Modal runs also update [progress/summary.md](progress/summary.md) and append raw history to `progress/runs.jsonl`, so you can track which assignments are done and compare benchmark timings over time.
