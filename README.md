# Overview
MatrixGraph is a C++/CUDA library designed to ease parallel programming for graph computing. 

# Build Options

所有 CMake 编译选项如下，在配置时通过 `-D` 传入：

## 完整选项列表

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `CMAKE_BUILD_TYPE` | string | `Debug` | 构建类型：`Debug` 或 `Release` |
| `ENABLE_AVX` | bool | `ON` | 启用 AVX/AVX2 指令（仅 C++ 代码，CUDA 不启用） |
| `USE_JEMALLOC` | bool | `OFF` | 启用 jemalloc 内存分配器 |
| `TEST` | bool | `OFF` | 启用 GoogleTest 单元测试 |
| `CUDA_ARCHITECTURES` | string | `sm_70` | CUDA 目标架构（如 sm_60, sm_70, sm_80） |
| `CMAKE_CUDA_HOST_COMPILER` | string | `g++-13` | nvcc 使用的 Host 编译器 |

## 配置示例

```bash
# 基础配置（Debug）
cmake -B build -S .

# Release 构建
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# 禁用 AVX（兼容老 CPU 或避免与 nvcc 冲突）
cmake -B build -S . -DENABLE_AVX=OFF

# 启用 jemalloc
cmake -B build -S . -DUSE_JEMALLOC=ON

# 启用单元测试
cmake -B build -S . -DTEST=ON

# 指定 CUDA 架构（如 Tesla V100）
cmake -B build -S . -DCUDA_ARCHITECTURES=sm_70

# 指定 nvcc Host 编译器（若系统无 g++-13）
cmake -B build -S . -DCMAKE_CUDA_HOST_COMPILER=g++-11

# 组合示例
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_AVX=ON \
  -DCUDA_ARCHITECTURES=sm_70
```

## 依赖说明

- **TBB**：优先使用 bundled oneTBB（`third_party/oneTBB`），与项目同编译器构建，避免 libstdc++ 版本不匹配。若子模块未初始化，会回退到系统 TBB。
- **初始化 oneTBB 子模块**：`git submodule update --init third_party/oneTBB`

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


