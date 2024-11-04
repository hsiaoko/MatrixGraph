### Build

First, clone the project and install dependencies on your environment.

```shell
# Clone the project (SSH).
# Make sure you have your public key has been uploaded to GitHub!
git clone git@github.com:SICS-Fundamental-Research-Center/MatrixGraph.git
# Install dependencies.
$SRC_DIR=`MatrixGraph` # top-level MatrixGraph source dir
$cd $SRC_DIR
$./dependencies.sh
```

Build the project.
```shell
$BUILD_DIR=<path-to-your-build-dir>
$mkdir -p $BUILD_DIR
$cd $BUILD_DIR
$cmake ..
$make
```
### Running MiniGraph Applications


#### Preparation: Partition & convert graph.
We store graphs in a binary CSR format. Edge-list in CSV format 
can be converted to our CSR format with graph_convert tool provided in tools.
You can use graph_convert_exec as follows:
```shell
$./bin/graph_partition_exec -t csr_bin -p -n  [the number of fragments] -i [graph in csv format] -sep [seperator, e.g. ","] -o [workspace]  -cores [degree of parallelism] -tobin -partitioner ["vertexcut" or "edgecut"]
// Step 1. Convert edgelist csv to edgelist bin format.
$./bin/tools/graph_converter_exec -i [path of edgelist in csv format] -sep [separator e.g. ","] -o [output path] -convert_mode edgelistcsv2edgelistbin
// Step 2. Convert edgelist bin to csr bin.
$./bin/tools/graph_converter_exec -i [path of edgelist in binary format] -o [output path] -convert_mode edgelistbin2csrbin
// Step 3. Sort vertices by outdegree, and then compressed ID of vertices, convert csr back to edgelist bin.
$./bin/tools/graph_converter_exec -i [path of csr in binary format] -o [output path] -convert_mode csrbin2edgelistbin -compressed
// Step 4. GridCut partitioning
$./bin/tools/graph_partitioner_exec -i [path of edgelist in binary format] -o [output path] -partitioner gridcut -n_partitions 1
// Step 5. Convert partitions of graph to csr tiled matrix.
$./bin/tools/graph_converter_exec -i  [path of partitions] -o [output path] -convert_mode gridedgelistbin2csrtiledmatrix -tile_size 2
// Step 6. Walks.
$./bin/gemm_exec -i /data/zhuxiaoke/workspace/MatrixGraph/csr_tiled_matrix/test_4x4_2/ -it /data/zhuxiaoke/workspace/MatrixGraph/csr_tiled_matrix/test_4x4_2/ -o /data/zhuxiaoke/workspace/MatrixGraph/csr_tiled_matrix/test_4x4_2-2-hop/
```
