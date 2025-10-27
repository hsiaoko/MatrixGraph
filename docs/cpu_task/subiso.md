# SubIso Application User Guide

**Binary Path**: `bin/cpu_subiso_exec`

**Source**: ` $PROJECT_ROOT_DIR/core/task/cpu_task/cpu_subiso.cu`

SubIso is a subgraph isomorphism application that supports both traditional VF3 algorithm and ML-enhanced filtering
approaches.

## Basic Command Format

### VF3 SubIso

Basic VF3 SubIso

```bash
$PROJECT_ROOT_DIR/bin/cpu_subiso_exec \
  -p <pattern_graph_csr_directory> \
  -g <data_graph_csr_directory> 
```

### Using ML model as filter

For ML model usage, refer to:

* Model training: $PROJECT_ROOT_DIR/docs/tools/SubIsoTraining.md
* Graph format conversion: $PROJECT_ROOT_DIR/docs/tools/GraphConverter.md

```bash
$PROJECT_ROOT_DIR/bin/cpu_subiso_exec \
  -p <pattern_graph_csr_directory> \
  -g <data_graph_csr_directory> \
  -m1 <pattern_embedding_directory> \
  -m2 <data_graph_embedding_directory> \
  -m3 <mlp_layer1_weight_directory> \
  -m4 <mlp_layer1_bias_directory> \
  -m5 <mlp_layer2_weight_directory> \
  -m6 <mlp_layer2_bias_directory>
```

## Parameter Details

-p : Pattern graph CSR directory
-g : Data graph CSR directory
-m1 : Pattern embedding directory
-m2 : Data graph embedding directory
-m3 : MLP first layer weights directory
-m4 : MLP first layer bias directory
-m5 : MLP second layer weights directory
-m6 : MLP second layer bias directory

## Requirements

* All directory paths must exist and contain valid files.
* Model files should be pre-trained.
* Ensure sufficient memory for graph processing.
* Graph data must be in proper CSR format


