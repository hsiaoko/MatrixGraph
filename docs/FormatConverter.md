# graph_converter for MatrixGraph
-------------
MatrixGraph represents graphs using a **binary Compressed Sparse Row (CSR) format**. This tool (`graph_converter`) provides conversion between various graph formats and MatrixGraphs binary CSR, as well as reverse conversions for benchmarking against other tools.


## Usage
The binary will be located at ./bin/tools/graph_converter_exec.

### edge-list CSV to binary edgelist:
```shell
./bin/tools/graph_converter_exec -i [path-to-edgelist.csv] -sep [separator] -o [output-path] -convert_mode edgelistcsv2edgelistbin -sep [seperator]
```

### edgelist CSV to binary CSR:
```shell
./bin/tools/graph_converter_exec -i [path-to-edgelist-bin] -o [output-path] -convert_mode edgelistcsv2csrbin -sep [seperator] 
```

### binary edgelist to binary CSR:
```shell
./bin/tools/graph_converter_exec -i [path-to-edgelist-bin] -o [output-path] -convert_mode edgelistbin2csrbin
```

### binary CSR to Rapids format:
the following baselines use this format 
* https://github.com/RapidsAtHKUST/EGSM
* https://github.com/RapidsAtHKUST/RapidMatch
* https://github.com/RapidsAtHKUST/SubgraphMatching
```shell
./bin/tools/graph_converter_exec -i [path-to-csr-bin] -o [output-path] -convert_mode csrbin2egsm
```

### binary CSR to Rapids format:
```shell
./bin/tools/graph_converter_exec -i [path-to-rapids-format] -o [output-path] -convert_mode egsm2csrbin
```

### binary Rapids format to binary edgelist:
```shell
./bin/tools/graph_converter_exec -i [path-to-rapid-format] -o [output-path] -convert_mode egsm2edgelistbin
```

### binary CSR to vf3lib format:
the following baselines use this format 
* https://github.com/MiviaLab/vf3lib.git
```shell
./bin/tools/graph_converter_exec -i [path-to-csr-bin] -o [output-path] -convert_mode csrbin2vf3
```

### Convert CSR to GNNPE format:
the following baselines use this format 
* https://github.com/JamesWhiteSnow/GNN-PE
```shell
./bin/tools/graph_converter_exec -i [path-to-csr-bin] -o [output-path] -convert_mode csrbin2gnnpe
```
