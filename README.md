# Overview
MatrixGraph is a C++/CUDA library designed to ease parallel programming for graph computing. 

# Build Options

All CMake options are passed via `-D` during configuration:

## Option Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | string | `Debug` | Build type: `Debug` or `Release` |
| `ENABLE_AVX` | bool | `ON` | Enable AVX/AVX2 instructions (C++ only; CUDA uses scalar fallback) |
| `USE_JEMALLOC` | bool | `OFF` | Enable jemalloc memory allocator |
| `TEST` | bool | `OFF` | Enable GoogleTest unit tests |
| `CUDA_ARCHITECTURES` | string | `sm_70` | CUDA target architecture (e.g. sm_60, sm_70, sm_80) |
| `CMAKE_CUDA_HOST_COMPILER` | string | `g++-13` | Host compiler used by nvcc |

## Configuration Examples

```bash
# Basic configuration (Debug)
cmake -B build -S .

# Release build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Disable AVX (for older CPUs or to avoid nvcc conflicts)
cmake -B build -S . -DENABLE_AVX=OFF

# Enable jemalloc
cmake -B build -S . -DUSE_JEMALLOC=ON

# Enable unit tests
cmake -B build -S . -DTEST=ON

# Specify CUDA architecture (e.g. Tesla V100)
cmake -B build -S . -DCUDA_ARCHITECTURES=sm_70

# Combined example
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_AVX=ON \
  -DCUDA_ARCHITECTURES=sm_70
```

## Dependencies

- **TBB**: Prefers bundled oneTBB (`third_party/oneTBB`), built with the same compiler as the project to avoid libstdc++ version mismatch. Falls back to system TBB if the submodule is not initialized.
- **Initialize oneTBB submodule**: `git submodule update --init third_party/oneTBB`

# Running MatrixGraph Applications

## Build

### Clone the repository and install dependencies in your environment:
```shell
# Clone the project using SSH. Ensure your public key is uploaded to GitHub.
git clone git@github.com:SICS-Fundamental-Research-Center/MatrixGraph.git
```

### Install dependencies.
```shell
SRC_DIR="MatrixGraph"  # Top-level MatrixGraph source directory
cd $SRC_DIR
./dependencies.sh
```

### Build the project:
```shell
BUILD_DIR=<path-to-your-build-dir>
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake ..
make
```

## Preparation: Partition & Convert Graph

Graphs are stored in a binary CSR format. You can convert an edge-list in CSV format to CSR using the graph_convert_exec tool provided in tools.

Follow these steps:

### Convert edge-list CSV to binary format:
```shell
./bin/tools/graph_converter_exec -i [path-to-edgelist.csv] -sep [separator] -o [output-path] -convert_mode edgelistcsv2edgelistbin
```
### Convert binary edge-list to CSR binary format:
```shell
./bin/tools/graph_converter_exec -i [path-to-edgelist-bin] -o [output-path] -convert_mode edgelistbin2csrbin
```
### Sort vertices by outdegree and compressed ID, then convert CSR back to binary edge-list:
```shell
./bin/tools/graph_converter_exec -i [path-to-csr-bin] -o [output-path] -convert_mode csrbin2edgelistbin -compressed
```
### Partition the graph using GridCut:
```shell
./bin/tools/graph_partitioner_exec -i [path-to-edgelist-bin] -o [output-path] -partitioner gridcut -n_partitions [number-of-partitions]
```
### Convert graph partitions to CSR tiled matrix:
```shell
./bin/tools/graph_converter_exec -i [path-to-partitions] -o [output-path] -convert_mode gridedgelistbin2csrtiledmatrix -tile_size [tile size]
```
### Run graph walks:
```shell
./bin/gemm_exec -i [path-to-csr-tiled-matrix] -it [path-to-csr-tiled-matrix] -o [output-path]
```

# Using MatrixGraph as a library
To use MatrixGraph as a library, one way is to copy this repository into your own CMake project, typically using a git submodule. Then you can put the following in your CMakeLists.txt:
```cmake
# Directory structure
set(THIRD_PARTY_ROOT ${PROJECT_ROOT_DIR}/third_party)

# Include MatrixGraph library
add_subdirectory(${THIRD_PARTY_ROOT}/MatrixGraph)

include_directories(SYSTEM
        ${THIRD_PARTY_ROOT}/MatrixGraph
        ${THIRD_PARTY_ROOT}/
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )

add_executable(app ...)
target_link_libraries(app PRIVATE matrixgraph_core)
```


