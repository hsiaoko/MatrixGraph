#ifndef MATRIXGRAPH_PUBLIC_DATA_H_
#define MATRIXGRAPH_PUBLIC_DATA_H_

// Host-side graph data types for external C++ consumers: read a CSR or an
// edge list, inspect metadata, and hand the buffers to a task or engine.
// Device upload views stay internal; everything here compiles without the
// CUDA toolkit.

#include "core/data_structures/edgelist.h"
#include "core/data_structures/immutable_csr.cuh"
#include "core/data_structures/metadata.h"

#endif  // MATRIXGRAPH_PUBLIC_DATA_H_