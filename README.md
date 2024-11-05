# Build
Clone the repository and install dependencies in your environment:
shell
Copy code
## Clone the project using SSH. Ensure your public key is uploaded to GitHub.
git clone git@github.com:SICS-Fundamental-Research-Center/MatrixGraph.git

## Install dependencies.
SRC_DIR="MatrixGraph"  # Top-level MatrixGraph source directory
cd $SRC_DIR
./dependencies.sh
Build the project:
shell
Copy code
BUILD_DIR=<path-to-your-build-dir>
mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake ..
make

# Running MatrixGraph Applications
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
