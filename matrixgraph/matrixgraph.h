#ifndef MATRIXGRAPH_PUBLIC_MATRIXGRAPH_H_
#define MATRIXGRAPH_PUBLIC_MATRIXGRAPH_H_

// Single stable entry point for code outside this repository.
//
// Everything reachable from this header is guaranteed to compile as plain
// C++ (no CUDA toolkit required) and to stay free of TBB; both rules are
// enforced by scripts/check-headers.sh. Internal headers under core/ may
// move without notice; these four umbrellas and the headers they include
// (see data.h) are the compatibility surface.

#include "matrixgraph/data.h"
#include "matrixgraph/engine.h"
#include "matrixgraph/error.h"

#endif  // MATRIXGRAPH_PUBLIC_MATRIXGRAPH_H_