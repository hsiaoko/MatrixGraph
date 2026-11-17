#ifndef MATRIXGRAPH_CORE_ENGINE_ENGINE_INTERNAL_H_
#define MATRIXGRAPH_CORE_ENGINE_ENGINE_INTERNAL_H_

#include "core/engine/engine.h"

#include <memory>

namespace sics {
namespace matrixgraph {
namespace core {
namespace engine {
namespace detail {

// Per-backend constructors live in their own translation units so TBB stays
// out of CUDA files and CUDA stays out of TBB files (nvcc cannot parse TBB
// headers; probe-verified in .audit/engine-grounding.md).
std::unique_ptr<Engine> MakeCpuEngine(const EngineOptions& options);

std::unique_ptr<Engine> MakeGpuEngine(const EngineOptions& options);

}  // namespace detail
}  // namespace engine
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_ENGINE_ENGINE_INTERNAL_H_