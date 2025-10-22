# Preprocessing for Matrix Filter
**Binary Path**: ` $PROJECT_ROOT_DIR/bin/tools/python/data.py`

-------------
To use Matrix Filter, one should do the following things for preprocessing:
* convert Rapids's format to torch's binary torch format (.pt).
* convert binary' torch format to binary embedding of MatrixGraph.
* generating ground truth for training.
* training similarity models (the process also include GNN model but).

## Usage

### convert Rapids's format to torch's binary torch format (.pt):
```shell
python  $PROJECT_ROOT_DIR/tools/python/graph_reader.py [path-to-rapids-graph] [output-path]
```

### convert binary' torch format to binary embedding of MatrixGraph.
```shell
python  $PROJECT_ROOT_DIR/tools/python/data.py [path-to-binary-torch] [output-path]
```

### generating groud truth.
Any method is support.
The only require is that the ground truch is one dimention array of uint64_t which store true candidate vertices id (vid) for the first query vertices of pqttern

### training similarity model and GNN
```shell
python  $PROJECT_ROOT_DIR/tools/python/train.py
```

