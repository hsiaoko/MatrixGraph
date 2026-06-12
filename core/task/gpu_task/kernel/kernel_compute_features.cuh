#ifndef MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_COMPUTE_FEATURES_CUH_
#define MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_COMPUTE_FEATURES_CUH_

#include <cuda_runtime.h>
#include <cstdint>

#include "core/common/types.h"
#include "core/data_structures/attributes.h"
#include "core/task/gpu_task/compute_features_types.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

/**
 * @file kernel_compute_features.cuh
 * @brief Device-side declarations for the ComputeFeatures interpreter.
 *
 * This header exposes EvalExpr() and ComputeFeaturesKernel() so that the host
 * task can launch the kernel.  All interpretation state lives in the kernel
 * arguments and in the per-block workspace; no device-side globals are used.
 */

using VertexID = sics::matrixgraph::core::common::VertexID;
using EdgeIndex = sics::matrixgraph::core::common::EdgeIndex;
using Attributes = sics::matrixgraph::core::data_structures::Attributes;

using MGExprType = MatrixGraphExprType;
using MGNavType = MatrixGraphNavType;
using MGCondType = MatrixGraphCondType;
using MGPlanNode = MatrixGraphPlanNode;
using MGFeatureValue = MatrixGraphFeatureValue;
using MGCondNode = MatrixGraphCondNode;

/**
 * @brief Maximum number of bindings collected per pivot.
 *
 * This is also the block size used by ComputeFeaturesKernel.  It must be a
 * power of two and large enough for the block-level bitonic sort used by the
 * order-statistic aggregation primitives.
 */
constexpr uint32_t kComputeFeaturesMaxNeighbors = 256;

/**
 * @brief Recursively evaluate expression @p expr_idx for @p pivot_vid.
 *
 * @param plan               Flat expression plan (device memory).
 * @param navs               Flat navigator plan (device memory).
 * @param conds              Flat condition array (device memory).
 * @param vertex_attrs       Per-vertex attribute maps (device memory).
 * @param graph_data         Contiguous ImmutableCSR buffer (device memory).
 * @param n_vertices         Number of vertices in the graph.
 * @param n_in_edges         Number of incoming edges.
 * @param n_out_edges        Number of outgoing edges.
 * @param labels             Optional per-vertex labels (may be nullptr).
 * @param pivot_vid          Vertex id to use as the expression pivot.
 * @param expr_idx           Index into @p plan to evaluate.
 * @param workspace          Per-thread scratch workspace.
 * @param workspace_capacity Number of MGFeatureValue slots in one workspace
 *                           slice (equal to blockDim.x).
 * @return The typed feature value produced by the expression.
 */
__device__ MGFeatureValue EvalExpr(
    const MGPlanNode* plan,
    const MGPlanNode* navs,
    const MGCondNode* conds,
    const Attributes* vertex_attrs,
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const uint32_t* labels,
    uint32_t pivot_vid,
    int32_t expr_idx,
    MGFeatureValue* workspace,
    uint32_t workspace_capacity);

/**
 * @brief Main kernel: one CUDA block per pivot, one thread per slot.
 *
 * Each block evaluates every expression in @p output_expr_indices for its
 * assigned pivot and writes the results to @p outputs in row-major order.
 *
 * @param per_pivot_workspace Total number of MGFeatureValue slots reserved for
 *                            each pivot.  Must be at least
 *                            workspace_capacity * max_workspace_depth.
 */
__global__ void ComputeFeaturesKernel(
    const uint8_t* graph_data,
    uint32_t n_vertices,
    uint32_t n_in_edges,
    uint32_t n_out_edges,
    const uint32_t* labels,
    const Attributes* vertex_attrs,
    const uint32_t* pivot_vertex_ids,
    uint32_t n_pivots,
    const MGPlanNode* plan,
    uint32_t n_plan_nodes,
    const MGPlanNode* navs,
    uint32_t n_navs,
    const MGCondNode* conds,
    uint32_t n_conds,
    const int32_t* output_expr_indices,
    uint32_t n_outputs,
    MGFeatureValue* workspace,
    uint32_t workspace_capacity,
    uint32_t per_pivot_workspace,
    MGFeatureValue* outputs);

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_TASK_GPU_TASK_KERNEL_COMPUTE_FEATURES_CUH_
