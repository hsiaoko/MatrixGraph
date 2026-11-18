# Public C++ API

External code includes one header:

```cpp
#include "matrixgraph/matrixgraph.h"
```

It pulls three umbrellas. Each is a one-line include of the real header, so
there is no shadow type to learn.

- `matrixgraph/error.h` exposes `util::Error` and the append-only
  `ErrorCode`. Library calls throw; your entry point translates.
- `matrixgraph/engine.h` exposes `Backend`, `EngineOptions`, `Engine`, and
  `MakeEngine`. Run a task with `engine->Run(task)`; a task handed to the
  wrong engine kind throws `Error(kUnsupported)`.
- `matrixgraph/data.h` exposes the host graph types: `ImmutableCSR`,
  `Edges`, and the metadata structs. `ImmutableCSR::Read` loads a CSR
  directory written by `graph_converter`; `GetGraphBufferSize` reports the
  contiguous buffer size, parsed-pointer derived.

## Typical use

```cpp
using namespace sics::matrixgraph::core;

data_structures::ImmutableCSR g;
g.Read("/path/to/csr/");

auto engine = engine::MakeEngine(engine::Backend::kGpu);
task::BFS bfs("/path/to/csr/", 0);
engine->Run(bfs);
```

## Guarantees and how they are checked

- Everything reachable from the umbrella compiles as plain C++ with no CUDA
  toolkit include paths, so data-only consumers never need nvcc.
- The same headers compile inside an nvcc translation unit, which also
  proves they carry no TBB includes (nvcc cannot parse TBB headers).
- Both rules are enforced by `scripts/check-headers.sh`; adding a header to
  the umbrella without passing the gate is a build break.

Host-only dependencies such as yaml-cpp are allowed on this surface. The
rules are specifically about CUDA and TBB.

## Which backend runs which task

There are two task contracts and one engine per execution space:

- `task::CPUTaskBase` — host algorithms (`SubIsoCPU`, `LFTJSubIso`, `Skew`,
  `Diameter`). `Backend::kCpuSerial` and `Backend::kCpuParallel` both run them;
  the two differ only in the worker cap (`EngineOptions::max_threads`, and
  `kCpuSerial` forces one worker).
- `task::TaskBase` — device algorithms (`BFS`, `WCC`, `PageRank`, `SubIso`,
  `GARMatch`, `GEMM`, `LFTJSubIsoGpu`, `ComputeFeaturesTask`). Only
  `Backend::kGpu` runs them.

A mismatch is a programming error and throws `util::Error(kUnsupported)`
("CPU engine cannot run a GPU task", "GPU engine cannot run a CPU task"), so a
wrong `-backend` fails at the call instead of silently doing something else.

`ComputeFeaturesTask`, `ExecuteAggPrim` and `GraphAggregate` are service-style:
their value comes from typed `Compute()` calls on buffers the caller supplies,
and they are used directly rather than through `Engine::Run`. Their `Run`
stays empty on purpose.

Per executable, `-backend` defaults to the family the algorithm belongs to:

| Executables | Default | Accepted |
|-------------|---------|----------|
| `bfs`, `wcc`, `pagerank`, `subiso`, `gar_match`, `gemm`, `lftj_subiso_gpu` | `gpu` | `gpu` only |
| `diameter`, `skew`, `subiso_cpu`, `lftj_subiso` | `cpu` | `cpu`, `cpu_serial` |

`apps/compute_features.cpp`, `apps/execute_agg_prim.cpp`,
`apps/graph_aggregate.cpp` and the `graphrag_*` wrappers take no `-backend`:
the first three drive service tasks directly, the last group calls kernels.

## What is not public yet

Headers under `core/` outside the umbrellas are internal and move without
notice: task classes, kernels, IO formats, and device upload views
(`UnifiedOwnedBuffer`, `DeviceOwnedBuffer`). `ErrorCode` numbering is
append-only. Promoting a type to the public surface means adding it to the
umbrella and to the gate's header list in the same change.