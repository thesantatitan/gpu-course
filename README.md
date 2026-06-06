# PMPP Self-Study GPU Course

A hands-on CUDA course built around *Programming Massively Parallel Processors: A Hands-on Approach, 4th Edition* by Hwu, Kirk, and El Hajj.

The book explains GPU programming and parallel algorithms. This repository supplies the course structure around it: a weekly syllabus, guided assignments, CUDA starter code, correctness checks, benchmark harnesses, one-command Modal execution, and automatic progress tracking.

You write the important GPU code. The repository handles most of the setup and boilerplate.

## What You Will Learn

By completing the guided assignments, you will learn how to:

- map data onto CUDA threads, blocks, and grids,
- allocate device memory and move data between CPU and GPU,
- reason about warps, divergence, occupancy, and latency hiding,
- use coalesced memory access and shared-memory tiling,
- implement common parallel patterns such as convolution, histogram, reduction, and scan,
- check GPU results against CPU reference implementations,
- benchmark kernels without including first-launch warmup overhead,
- explain why one kernel is faster or slower than another,
- design and evaluate a larger GPU application as a final project.

This is not a collection of completed solutions. Most assignment kernels intentionally contain `TODO` sections and initially return `"ok": false`.

## Who This Is For

The course assumes that you:

- can read and write basic C or C++,
- understand arrays, loops, functions, and pointers,
- are comfortable running commands in a terminal,
- have access to the PMPP 4th edition book.

Prior CUDA experience is not required. A local NVIDIA GPU is also not required: the default workflow runs CUDA remotely on Modal.

## How The Course Works

The course follows the two-phase structure suggested by the PMPP authors:

1. **Guided CUDA core:** Chapters 2-12, with six assignments that build reusable GPU-programming skills.
2. **Advanced application project:** Chapters 13-22 as needed, culminating in a larger optimization project.

The full schedule is in [SYLLABUS.md](SYLLABUS.md). The default pace is 14 weeks at roughly 8-12 hours per week, but the guided core can be completed faster.

Each guided assignment follows the same loop:

1. Read the listed PMPP chapter sections.
2. Read the assignment README before opening the code.
3. Inspect the supplied CPU reference and CUDA harness.
4. Implement only the requested kernel TODOs.
5. Run small correctness cases until the JSON output says `"ok": true`.
6. Run larger benchmark cases.
7. Record observations and answer the writeup questions.

## Quick Start

### 1. Clone The Repository

```bash
git clone https://github.com/thesantatitan/gpu-course.git
cd gpu-course
```

### 2. Install The Python Environment

The repository uses [uv](https://docs.astral.sh/uv/) for Python dependencies:

```bash
uv sync
```

### 3. Authenticate With Modal

Create a Modal account if needed, then authenticate once:

```bash
uv run modal setup
```

Modal authentication is stored locally. You do not need to repeat this for every assignment.

### 4. Verify CUDA Execution

```bash
uv run run_modal.py
```

This sends a small completed CUDA smoke test to Modal. A successful run ends with output similar to:

```json
{"lab":"cuda_smoke","ok":true,"n":1024,"block_size":256,"gpu_ms":0.01,"max_abs_error":0}
```

### 5. Start Assignment 01

Read [Assignment 01: Vector Add](assignments/01-vector-add/README.md), then edit:

```text
assignments/01-vector-add/starter/main.cu
```

Run it with its default input size and block size:

```bash
uv run run_modal.py 01
```

Before you implement the TODO, `"ok": false` is expected. That means the harness compiled and ran correctly, but your CUDA output did not match the CPU reference.

## Running Assignments

Use a short assignment alias:

```bash
uv run run_modal.py 01
uv run run_modal.py 02 -- 128
uv run run_modal.py 03 -- 2048 2048
```

The `--` separates options for `run_modal.py` from arguments passed to the compiled CUDA program.

For example:

```bash
uv run run_modal.py 01 -- 1048576 256
```

means:

- run assignment `01`,
- use vectors containing `1,048,576` elements,
- launch `256` CUDA threads per block.

Arguments are optional. Every starter provides reasonable defaults.

### Assignment Aliases

| Alias | Assignment | Main CUDA file |
| --- | --- | --- |
| `01`, `vector-add` | Vector addition and basic CUDA indexing | `assignments/01-vector-add/starter/main.cu` |
| `02`, `matmul` | 2D grids and tiled matrix multiplication | `assignments/02-grids-matmul/starter/main.cu` |
| `03`, `transpose` | Coalescing and shared-memory transpose | `assignments/03-memory-performance/starter/main.cu` |
| `04`, `conv` | Convolution and constant memory | `assignments/04-convolution-stencil/starter/main.cu` |
| `05`, `hist` | Histogram atomics and reduction | `assignments/05-histogram-reduction/starter/main.cu` |
| `06`, `scan` | Exclusive prefix scan | `assignments/06-scan-merge/starter/main.cu` |

Modal uses a T4 by default because it is an inexpensive GPU that supports these labs. Override it when needed:

```bash
uv run run_modal.py --gpu L4 04 -- 1024 1024 2
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for Modal, Colab, local CUDA, and benchmarking details.

## Understanding The Output

Each starter prints one JSON result line. For example:

```json
{
  "lab": "vector_add",
  "ok": true,
  "n": 1048576,
  "block_size": 256,
  "gpu_ms": 0.12,
  "max_abs_error": 0
}
```

Important fields:

- `ok`: whether the GPU result passed the supplied correctness check.
- `gpu_ms`, `basic_ms`, and similar fields: warmed-up kernel execution times measured with CUDA events.
- `max_abs_error` or related error fields: difference from the CPU reference.
- problem-shape fields such as `n`, `width`, `height`, and `block_size`: the exact configuration tested.

Do not interpret benchmark speed until correctness passes. A kernel that does no work can be extremely fast and completely wrong.

## Progress Tracking

Normal Modal runs automatically update:

- [progress/summary.md](progress/summary.md): readable assignment dashboard,
- `progress/summary.json`: machine-readable summary,
- `progress/runs.jsonl`: append-only history of every tracked attempt.

The tracker records pass/fail status, arguments, benchmark metrics, GPU details, Modal run URL, and current git state. An assignment is marked done after at least one run returns exit code `0` and `"ok": true`.

Skip tracking for a throwaway test:

```bash
uv run run_modal.py --no-progress 02 -- 16
```

Read [progress/README.md](progress/README.md) for the full schema and rebuild command.

## Repository Layout

```text
.
├── README.md                    # Start here
├── SYLLABUS.md                  # Week-by-week course plan
├── ENVIRONMENT.md               # Modal, Colab, and local CUDA setup
├── ASSIGNMENT_TEMPLATE.md       # Common assignment contract and rubric
├── PROJECT_IDEAS.md             # Suggested final-project directions
├── assignments/
│   ├── 01-vector-add/
│   ├── 02-grids-matmul/
│   ├── 03-memory-performance/
│   ├── 04-convolution-stencil/
│   ├── 05-histogram-reduction/
│   ├── 06-scan-merge/
│   └── 07-final-project/
├── common/
│   ├── cuda_utils.h             # CUDA error checking and event timer
│   ├── compile_and_run.py       # nvcc compile/run harness
│   └── progress.py              # Progress-history and summary generator
├── progress/                    # Automatically generated course progress
├── tools/modal_cuda_app.py      # Remote Modal CUDA application
└── run_modal.py                 # One-command local runner
```

## Course Assignments

| Assignment | Main idea | Core implementation |
| --- | --- | --- |
| [01: Vector Add](assignments/01-vector-add/README.md) | One thread per output element | Boundary-safe 1D kernel |
| [02: Grids And Matmul](assignments/02-grids-matmul/README.md) | 2D indexing and data reuse | Naive and tiled matmul |
| [03: Memory Performance](assignments/03-memory-performance/README.md) | Coalescing and bank conflicts | Naive and tiled transpose |
| [04: Convolution And Stencil](assignments/04-convolution-stencil/README.md) | Neighborhood access and constant memory | Two convolution kernels |
| [05: Histogram And Reduction](assignments/05-histogram-reduction/README.md) | Contention and tree aggregation | Atomic histogram and block reduction |
| [06: Scan And Merge](assignments/06-scan-merge/README.md) | Work-efficient parallel dependencies | Block-level exclusive scan |
| [07: Final Project](assignments/07-final-project/README.md) | End-to-end GPU optimization | Application chosen by you |

## Getting Help From The Harness

The starter code already provides:

- deterministic input generation,
- CPU reference implementations,
- CUDA allocation and copy boilerplate,
- untimed warmup launches,
- CUDA-event timing,
- correctness checks,
- compact machine-readable results.

When debugging, first reduce the problem size and inspect indexing. Keep the supplied harness unchanged until your core kernel passes; changing both the kernel and its test at the same time makes errors much harder to locate.

## License And Book

This repository contains original course scaffolding and starter code. It does not include the PMPP textbook. Obtain the book separately and use the chapter references in each assignment.
