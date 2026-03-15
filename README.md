# Overview

MatrixGraph is a C++/CUDA library for parallel graph computing. It provides **GPU-accelerated** graph algorithms (PageRank, BFS, WCC, subgraph isomorphism, matrix operations), format conversion tools, and partitioning utilities. Graphs are stored in a binary CSR format; edge-list and other formats can be converted via the included tools.

**GPU support**: Core algorithms run on NVIDIA GPUs via CUDA. The library uses cuBLAS for matrix operations and custom kernels for graph traversals. A compatible CUDA toolkit and GPU (compute capability 7.0+) are required for GPU workloads.

# Building MatrixGraph

## Dependencies

- **C++20** and **CUDA** toolchain
- **GCC 14** (g++-14) — required as the host compiler for nvcc
- **CMake** 3.20+
- **gflags**, **yaml-cpp** (built from `third_party/`)
- **TBB** (oneTBB; prefers bundled build, see below)
- **CUDA Toolkit** with cuBLAS

Install dependencies using the provided script:

```bash
./dependencies.sh
```

This installs system packages (e.g. g++-14, cmake, build-essential) and initializes git submodules. On Ubuntu, if g++-14 is not available, add the Toolchain PPA first:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
```

**TBB**: The project prefers the bundled oneTBB (`third_party/oneTBB`) to avoid libstdc++ version mismatch. If the submodule is not initialized, CMake falls back to the system TBB.

## Source-Tree Organization

```
MatrixGraph/
├── core/                 # Core library (matrixgraph_core)
│   ├── data_structures/  # Graph representations, buffers
│   ├── task/             # GPU/CPU algorithms
│   │   ├── gpu_task/     # CUDA kernels: gemm, pagerank, bfs, wcc, subiso
│   │   └── cpu_task/     # CPU algorithms (e.g. subgraph isomorphism)
│   ├── util/             # Utilities (execution policy, format converter)
│   └── components/      # Scheduler, etc.
├── tools/                # Standalone tools
│   ├── graph_converter/  # Format conversion (CSV, CSR, tiled matrix)
│   └── graph_partitioner/# Graph partitioning (GridCut, etc.)
├── apps/                  # Example applications (gemm, pagerank, bfs, wcc, ...)
├── docs/                  # Documentation
└── third_party/           # gflags, yaml-cpp, oneTBB, googletest
```

**GPU components**: The `core/task/gpu_task/` directory contains CUDA implementations. Algorithms such as WCC, PageRank, BFS, and GEMM (graph walks) run on the GPU. The library targets NVIDIA GPUs with compute capability 7.0 (Volta) or newer.

## Compiling and Testing MatrixGraph

We use CMake for building, testing, and installing. Common commands:

```bash
# Configure (Debug by default)
cmake -B build -S .

# Build
cmake --build build

# Run tests (if TEST=ON)
ctest --test-dir build
```

### CMake Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | string | `Debug` | `Debug` or `Release` |
| `ENABLE_AVX` | bool | `ON` | Enable AVX/AVX2 (C++ only; CUDA uses scalar fallback) |
| `USE_JEMALLOC` | bool | `OFF` | Enable jemalloc |
| `TEST` | bool | `OFF` | Enable GoogleTest |
| `CUDA_ARCHITECTURES` | string | `sm_70` | CUDA target (e.g. sm_60, sm_70, sm_80) |
| `CMAKE_CUDA_HOST_COMPILER` | string | `g++-14` | Host compiler for nvcc (g++-14 required) |

Examples:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake -B build -S . -DENABLE_AVX=OFF
```

# Running MatrixGraph Applications

## Graph Format

Graphs are stored in a binary CSR format. Converting from edge-list CSV, partitioning, and producing tiled matrices require the tools in `tools/`. For detailed instructions on:

- **Format conversion** (CSV ↔ binary edgelist ↔ CSR): see `docs/tools/GraphConverter.md`
- **Partitioning** (GridCut, hash-based cuts): see `docs/tools/GraphPartitioner.md`
- **Other tools** (embedding, random graph generation, etc.): see `docs/tools/`

## Running

Minimal example: run **WCC** (Weakly Connected Components) on a small graph. No partitioning required.

```bash
# 1. Install dependencies and build (from project root)
./dependencies.sh
cmake -B build -S .
cmake --build build

# 2. Create a small edge-list CSV
mkdir -p data
echo -e "0,1\n1,2\n2,0\n3,4\n4,3" > data/graph.csv

# 3. Convert CSV to binary edgelist
./bin/tools/graph_converter -i data/graph.csv -o data/edgelist/ \
  -convert_mode edgelistcsv2edgelistbin -sep ","

# 4. Convert binary edgelist to CSR (required by WCC)
./bin/tools/graph_converter -i data/edgelist/ -o data/csr/ \
  -convert_mode edgelistbin2csrbin

# 5. Run WCC on GPU
./bin/wcc_exec -g data/csr/
```

For graph walks (GEMM), PPR, or other apps that need partitioned graphs in CSR tiled matrix format, see `docs/tools/GraphConverter.md` and `docs/tools/GraphPartitioner.md`.

# Using MatrixGraph as a Library

Add MatrixGraph as a submodule or copy it into your project, then in your `CMakeLists.txt`:

```cmake
set(THIRD_PARTY_ROOT ${PROJECT_ROOT_DIR}/third_party)
add_subdirectory(${THIRD_PARTY_ROOT}/MatrixGraph)

add_executable(my_app ...)
target_link_libraries(my_app PRIVATE matrixgraph_core)
target_include_directories(my_app PRIVATE
  ${THIRD_PARTY_ROOT}/MatrixGraph
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
```

# Documentation

See [docs/README.md](docs/README.md) for the full index. Key documents:

- **Tools**: [GraphConverter](docs/tools/GraphConverter.md), [GraphPartitioner](docs/tools/GraphPartitioner.md), [SubIsoTraining](docs/tools/SubIsoTraining.md), [Preprocessing4MatrixFilter](docs/tools/Preprocessing4MatrixFilter.md)
- **GPU**: [Matrix Operations](docs/gpu_task/matrix_ops.md)
- **CPU**: [SubIso](docs/cpu_task/subiso.md)
