# GPU and SubIso environment variables

MatrixGraph reads a small set of **optional** environment variables to choose CUDA devices, kernel launch shapes, and WOJ (SubIso) tuning. **You do not need to set any of them** for normal runs: if a variable is missing or invalid, the code falls back to built-in defaults (see below).

This document covers the variables used by the **C++/CUDA** library and executables. Automation scripts (e.g. `scripts/run_autoconfig_gpu_exp.py`) may **inject** `MG_*` / `MATRIXGRAPH_CUDA_*` per YAML config for reproducible sweeps; that does not change the fact that **manual runs work with an empty environment** (aside from normal OS/CUDA requirements).

---

## CUDA device selection

**Precedence** (`MatrixGraphCudaDeviceList()` in `core/util/cuda_device.cuh`):

1. **`MATRIXGRAPH_CUDA_DEVICES`** — If set and parses to **at least one** valid index in `[0, device_count)`, **use only that list** (even if **`MATRIXGRAPH_CUDA_ALL_DEVICES=1`** is also set).
2. Else **`MATRIXGRAPH_CUDA_ALL_DEVICES=1`** — use every visible device `0 … device_count-1`.
3. Else **single device** — **`MATRIXGRAPH_CUDA_DEVICE`** if set (clamped), otherwise **0**.

| Variable | Role |
|----------|------|
| **`MATRIXGRAPH_CUDA_DEVICES`** | Comma-separated logical indices, e.g. `0,1,2` or a single GPU `2`. |
| **`MATRIXGRAPH_CUDA_ALL_DEVICES`** | When set exactly to **`1`**, only applies if step (1) did not produce a list. |
| **`MATRIXGRAPH_CUDA_DEVICE`** | Default / single-GPU index when (1)–(2) do not apply; also used by `core/go_api/matrixgraph_go_api.cu` (`cudaSetDevice`). |
| **`MATRIXGRAPH_CUDA_STREAMS`** | Streams per GPU for overlap (`MatrixGraphCudaStreamsPerGpu()`); default **2**, minimum **1**. |

**Physical GPUs vs logical indices**: **`CUDA_VISIBLE_DEVICES`** (or the container) remaps hardware to logical `0, 1, …` before MatrixGraph sees them.

---

## Kernel launch dimensions (WCC / BFS / PageRank)

Read in `core/util/cuda_launch_dims.cuh` via `MatrixGraphEnvLaunchGridDim` / `MatrixGraphEnvLaunchBlockDim`. Call sites pass app-specific compile-time defaults when the env is missing or not a positive integer.

| Variable | Role |
|----------|------|
| **`MG_GPU_GRID`** | Optional grid dim (must be **> 0** to override). |
| **`MG_GPU_BLOCK`** | Optional block dim (must be **> 0** to override). |

If unset or invalid, each app keeps its **own** default passed into these helpers.

---

## SubIso WOJ launch and host-side tuning

Defined in `core/task/gpu_task/kernel/kernel_woj_subsio.cu` (and related host code).

| Variable | Role | Default if unset |
|----------|------|-------------------|
| **`MG_SUBISO_GRID`** / **`MG_SUBISO_BLOCK`** | `gridDim` / `blockDim.x` for WOJ Filter/Join (must be **> 0**). | From **`kGridDim` / `kBlockDim`** in `core/common/consts.h` (often **512** × **128**) when unset—manual `subiso_exec` logs then show `cudaGrid=(512,1,1) block=(128,1,1)`. |
| **`MG_WOJ_STRIPE_THREADS`** | Host worker threads for WOJ Filter/Join striping. | **4** |
| **`MG_WOJ_JOIN_STRIPES_PER_GPU`** | Join row partitions per GPU. | **1** |
| **`MG_WOJ_JOIN_MAX_THREADS`** | Cap on host join workers. | `hardware_concurrency` / stripe policy (see source). |
| **`MG_WOJ_PREFETCH_JOIN`**, **`MG_WOJ_BUSHY_JOIN`** | WOJ scheduling toggles read in WOJ kernels/wrapper (see `.cu` for semantics). | Off / default branches in code. |

Benchmark scripts (`run_autoconfig_gpu_exp.py`) typically set **`MG_SUBISO_GRID`** / **`MG_SUBISO_BLOCK`** (and **`MG_GPU_GRID`** / **`MG_GPU_BLOCK`**) from each YAML **`grid_size`** / **`block_size`** together with **`MATRIXGRAPH_CUDA_DEVICES`** derived from **`num_gpus`**. That overrides any pre-set values **for those runs only** inside the subprocess environment.

---

## Autoconfig sweeps vs manual runs

- **Manual**: `./bin/subiso_exec -p … -g …` (etc.) relies on defaults above when you export nothing MatrixGraph-specific.
- **`scripts/run_autoconfig_gpu_exp.py`**: For each YAML with valid `num_gpus`, `grid_size`, and `block_size`, the script assigns **`MATRIXGRAPH_CUDA_DEVICES`**, **`MG_GPU_GRID`**, **`MG_GPU_BLOCK`**, **`MG_SUBISO_GRID`**, **`MG_SUBISO_BLOCK`** so the run matches that configuration. Inherited variables such as **`CUDA_VISIBLE_DEVICES`** are still inherited from `os.environ` unless your shell/session changes them separately.

---

## Quick reference

- **Minimal setup**: usable GPU drivers + CUDA runtime; optional **`CUDA_VISIBLE_DEVICES`** to hide busy cards.
- **No MatrixGraph-specific variables are required** for executables to start; without an explicit **`MATRIXGRAPH_CUDA_DEVICES`** list, SubIso uses **all visible GPUs** if **`MATRIXGRAPH_CUDA_ALL_DEVICES=1`**, otherwise **one** logical device (**`MATRIXGRAPH_CUDA_DEVICE`** or **0**).
- **Tuning**: set **`MG_SUBISO_*`** / **`MG_GPU_*`** only when you want to override built-in launch dimensions or host stripe counts.
