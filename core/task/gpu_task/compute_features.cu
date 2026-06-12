#include "core/task/gpu_task/compute_features.cuh"

#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <vector>

#include "core/util/cuda_check.cuh"

/**
 * @file compute_features.cu
 * @brief Host implementation of ComputeFeaturesTask.
 *
 * This file handles graph/attribute/label upload to the GPU, builds the
 * per-vertex attribute maps used by the device interpreter, derives a safe
 * workspace size from the expression plan, and launches ComputeFeaturesKernel.
 */

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

using VertexID = sics::matrixgraph::core::common::VertexID;
using ValueType = sics::matrixgraph::core::data_structures::ValueType;
using Attribute = sics::matrixgraph::core::data_structures::Attribute;
using AttributeName = sics::matrixgraph::core::data_structures::AttributeName;
using DeviceAttributes =
    sics::matrixgraph::core::data_structures::DeviceAttributes;

// ---------------------------------------------------------------------------
// Value type helpers.
// ---------------------------------------------------------------------------
__host__ ValueType MGToInternalValueType(MatrixGraphValueType type) {
  switch (type) {
    case MG_VALUE_INT:     return ValueType::kInt;
    case MG_VALUE_FLOAT64: return ValueType::kFloat64;
    case MG_VALUE_BOOL:    return ValueType::kBool;
    case MG_VALUE_TIME:    return ValueType::kTime;
    case MG_VALUE_FLOAT32: return ValueType::kFloat32;
    case MG_VALUE_STRING:  return ValueType::kString;
    default:               return ValueType::kInvalid;
  }
}

__host__ size_t ComputeFeaturesTask::ValueTypeSize(MatrixGraphValueType type) {
  switch (type) {
    case MG_VALUE_INT:
    case MG_VALUE_FLOAT64:
    case MG_VALUE_TIME:
      return 8;
    case MG_VALUE_BOOL:
      return 1;
    case MG_VALUE_FLOAT32:
      return 4;
    case MG_VALUE_STRING:
      return sizeof(sics::matrixgraph::core::data_structures::StringView);
    default:
      return 0;
  }
}

__host__ MatrixGraphValueType ComputeFeaturesTask::ValueTypeFromMG(int32_t type) {
  return static_cast<MatrixGraphValueType>(type);
}

// ---------------------------------------------------------------------------
// Graph loading / device transfer.
// ---------------------------------------------------------------------------

/**
 * @brief Read a MatrixGraph CSR graph and upload its buffer to the GPU.
 *
 * The upload covers everything from globalid up to and including localid_by_
 * globalid, which is the contiguous region consumed by ParseCSR() on device.
 */
__host__ void ComputeFeaturesTask::LoadGraph(const std::string& graph_path) {
  graph_path_ = graph_path;
  graph_ = std::make_unique<ImmutableCSR>();
  graph_->Read(graph_path);
  std::cout << "[ComputeFeaturesTask] Loaded graph: "
            << graph_->get_num_vertices() << " vertices, "
            << graph_->get_num_outgoing_edges() << " outgoing edges"
            << std::endl;
  TransferGraphDataToDevice();
}

/**
 * @brief Upload the contiguous ImmutableCSR buffer to the GPU.
 *
 * The size is computed from the start of the graph buffer to the end of the
 * localid_by_globalid table so that the device CSR parser sees all arrays.
 */
__host__ void ComputeFeaturesTask::TransferGraphDataToDevice() {
  if (graph_ == nullptr) return;

  const uint8_t* h_buf = graph_->GetGraphBuffer();
  const uint8_t* h_end =
      reinterpret_cast<const uint8_t*>(graph_->GetLocalIDBasePointer()) +
      sizeof(VertexID) * (graph_->get_max_vid() + 1);
  d_graph_data_size_ = h_end - h_buf;

  CUDA_CHECK(cudaMalloc(&d_graph_data_, d_graph_data_size_));
  CUDA_CHECK(
      cudaMemcpy(d_graph_data_, h_buf, d_graph_data_size_, cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// Attribute loading.
// ---------------------------------------------------------------------------

/**
 * @brief Upload attribute columns and rebuild per-vertex attribute maps.
 *
 * For each column we allocate a device buffer, copy the host data, and then
 * rebuild per_vertex_attrs_ so that every vertex can look up any loaded
 * attribute by name.  Previously loaded columns are preserved; this makes it
 * convenient to load attributes in batches (e.g. scores first, flags later).
 */
__host__ void ComputeFeaturesTask::LoadAttributes(
    uint32_t n_columns, const ComputeFeaturesAttributeColumn* columns) {
  if (graph_ == nullptr) {
    std::cerr << "[ComputeFeaturesTask::LoadAttributes] Graph not loaded"
              << std::endl;
    return;
  }

  const uint32_t n_vertices = graph_->get_num_vertices();

  // Upload each new column to the device and append to the existing column set.
  size_t prev_n_columns = d_columns_.size();
  for (uint32_t c = 0; c < n_columns; ++c) {
    const auto& col = columns[c];
    if (col.n_values != n_vertices) {
      std::cerr << "[ComputeFeaturesTask::LoadAttributes] Column '" << col.key
                << "' has " << col.n_values << " values, expected "
                << n_vertices << std::endl;
      return;
    }
    MatrixGraphValueType mg_type = ValueTypeFromMG(col.value_type);
    size_t elem_size = ValueTypeSize(mg_type);
    if (elem_size == 0 || col.values == nullptr) {
      std::cerr << "[ComputeFeaturesTask::LoadAttributes] Unsupported or empty"
                   " column type for '"
                << col.key << "'" << std::endl;
      return;
    }

    DeviceColumnBuffer buf;
    std::strncpy(buf.key, col.key, sizeof(buf.key) - 1);
    buf.key[sizeof(buf.key) - 1] = '\0';
    buf.type = mg_type;
    buf.bytes = elem_size * col.n_values;
    CUDA_CHECK(cudaMalloc(&buf.d_values, buf.bytes));
    CUDA_CHECK(cudaMemcpy(buf.d_values, col.values, buf.bytes,
                          cudaMemcpyHostToDevice));
    d_columns_.push_back(buf);
  }

  // Rebuild per-vertex Attributes hash maps from all loaded columns.
  per_vertex_attrs_.clear();
  per_vertex_attrs_.reserve(n_vertices);

  const size_t total_n_columns = d_columns_.size();
  std::vector<AttributeName> names(total_n_columns);
  std::vector<Attribute> attrs(total_n_columns);
  for (size_t c = 0; c < total_n_columns; ++c) {
    names[c] = AttributeName(d_columns_[c].key);
  }

  for (uint32_t v = 0; v < n_vertices; ++v) {
    for (size_t c = 0; c < total_n_columns; ++c) {
      MatrixGraphValueType mg_type = d_columns_[c].type;
      size_t elem_size = ValueTypeSize(mg_type);
      std::memset(&attrs[c], 0, sizeof(Attribute));
      std::strncpy(attrs[c].name, d_columns_[c].key,
                   sizeof(attrs[c].name) - 1);
      attrs[c].type = MGToInternalValueType(mg_type);
      attrs[c].n_rows = 1;
      attrs[c].n_elements = 1;
      attrs[c].data = static_cast<const uint8_t*>(d_columns_[c].d_values) +
                      v * elem_size;
      attrs[c].offsets = nullptr;
    }
    per_vertex_attrs_.emplace_back(v, names.data(), attrs.data(),
                                   static_cast<uint32_t>(total_n_columns));
  }

  // Build a contiguous device array of Attributes views.
  if (d_per_vertex_attrs_) cudaFree(d_per_vertex_attrs_);
  CUDA_CHECK(cudaMalloc(&d_per_vertex_attrs_, sizeof(Attributes) * n_vertices));
  std::vector<Attributes> h_attr_views(n_vertices);
  for (uint32_t v = 0; v < n_vertices; ++v) {
    h_attr_views[v] = per_vertex_attrs_[v].View();
  }
  CUDA_CHECK(cudaMemcpy(d_per_vertex_attrs_, h_attr_views.data(),
                        sizeof(Attributes) * n_vertices,
                        cudaMemcpyHostToDevice));

  std::cout << "[ComputeFeaturesTask] Loaded " << n_columns
            << " attribute column(s) for " << n_vertices << " vertices"
            << std::endl;
}

// ---------------------------------------------------------------------------
// Labels.
// ---------------------------------------------------------------------------

/**
 * @brief Upload per-vertex labels to the GPU.
 *
 * Labels are optional.  When present they enable label filtering in
 * NeighborNav (nav.target_label) and will be required by PatternNav later.
 */
__host__ void ComputeFeaturesTask::LoadLabels(const uint32_t* labels,
                                              uint32_t n) {
  if (graph_ == nullptr) {
    std::cerr << "[ComputeFeaturesTask::LoadLabels] Graph not loaded"
              << std::endl;
    return;
  }
  if (n != graph_->get_num_vertices()) {
    std::cerr << "[ComputeFeaturesTask::LoadLabels] Label count " << n
              << " != vertex count " << graph_->get_num_vertices()
              << std::endl;
    return;
  }
  if (d_labels_) cudaFree(d_labels_);
  CUDA_CHECK(cudaMalloc(&d_labels_, sizeof(uint32_t) * n));
  CUDA_CHECK(
      cudaMemcpy(d_labels_, labels, sizeof(uint32_t) * n, cudaMemcpyHostToDevice));
  n_labels_ = n;
  std::cout << "[ComputeFeaturesTask] Loaded " << n << " label(s)"
            << std::endl;
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/**
 * @brief Maximum recursion depth of expression @p expr_idx.
 *
 * Used to reserve enough workspace for nested AggExpr / TransExpr evaluations.
 */
static uint32_t ComputeExprMaxDepth(const std::vector<MatrixGraphPlanNode>& plan,
                                    int32_t expr_idx) {
  if (expr_idx < 0 || static_cast<size_t>(expr_idx) >= plan.size()) return 0;
  const MatrixGraphPlanNode& node = plan[expr_idx];
  const MatrixGraphExprType type = static_cast<MatrixGraphExprType>(node.type);

  if (type == MG_EXPR_ATTR || type == MG_EXPR_CONST ||
      type == MG_EXPR_PATTERN_ATTR) {
    return 1;
  }

  if (type == MG_EXPR_AGG) {
    return 1 + ComputeExprMaxDepth(plan, node.src_idx);
  }

  if (type == MG_EXPR_TRANS) {
    uint32_t da = ComputeExprMaxDepth(plan, node.child_a);
    uint32_t db = ComputeExprMaxDepth(plan, node.child_b);
    return 1 + std::max(da, db);
  }

  return 1;
}

/** @brief Maximum expression depth among all requested outputs. */
static uint32_t ComputePlanMaxExprDepth(
    const std::vector<MatrixGraphPlanNode>& plan,
    const std::vector<int32_t>& output_expr_indices) {
  uint32_t max_depth = 1;
  for (int32_t idx : output_expr_indices) {
    max_depth = std::max(max_depth, ComputeExprMaxDepth(plan, idx));
  }
  return max_depth;
}

/**
 * @brief Collect every navigator index referenced by an expression tree.
 *
 * The result is used to compute the navigator nesting depth, which in turn
 * determines the per-pivot workspace size.
 */
static void CollectNavIndices(const std::vector<MatrixGraphPlanNode>& plan,
                              int32_t expr_idx,
                              std::vector<int32_t>& out) {
  if (expr_idx < 0 || static_cast<size_t>(expr_idx) >= plan.size()) return;
  const MatrixGraphPlanNode& node = plan[expr_idx];
  const MatrixGraphExprType type = static_cast<MatrixGraphExprType>(node.type);

  if (type == MG_EXPR_AGG) {
    out.push_back(node.nav_idx);
    CollectNavIndices(plan, node.src_idx, out);
  } else if (type == MG_EXPR_TRANS) {
    CollectNavIndices(plan, node.child_a, out);
    CollectNavIndices(plan, node.child_b, out);
  }
}

/**
 * @brief Maximum nesting depth of navigator @p nav_idx.
 *
 * A FilterNav adds one level on top of its inner navigator.
 */
static uint32_t ComputeNavDepth(
    const std::vector<MatrixGraphPlanNode>& navs,
    int32_t nav_idx) {
  if (nav_idx < 0 || static_cast<size_t>(nav_idx) >= navs.size()) return 0;
  const MatrixGraphPlanNode& nav = navs[nav_idx];
  const MatrixGraphNavType type = static_cast<MatrixGraphNavType>(nav.type);

  if (type == MG_NAV_SELF || type == MG_NAV_NEIGHBOR) {
    return 1;
  }
  if (type == MG_NAV_FILTER) {
    return 1 + ComputeNavDepth(navs, nav.inner_nav_idx);
  }
  return 1;
}

/**
 * @brief Total workspace depth needed for a plan.
 *
 * The per-pivot workspace is allocated as
 *   kComputeFeaturesMaxNeighbors * (expr_depth + nav_depth + 1)
 * where the extra +1 provides scratch space for condition evaluation.
 */
static uint32_t ComputePlanMaxWorkspaceDepth(
    const std::vector<MatrixGraphPlanNode>& plan,
    const std::vector<MatrixGraphPlanNode>& navs,
    const std::vector<int32_t>& output_expr_indices) {
  uint32_t expr_depth = ComputePlanMaxExprDepth(plan, output_expr_indices);

  std::vector<int32_t> nav_indices;
  for (int32_t idx : output_expr_indices) {
    CollectNavIndices(plan, idx, nav_indices);
  }
  uint32_t nav_depth = 1;
  for (int32_t nav_idx : nav_indices) {
    nav_depth = std::max(nav_depth, ComputeNavDepth(navs, nav_idx));
  }

  // Add a small safety margin for condition evaluation scratch.
  return expr_depth + nav_depth + 1;
}

// ---------------------------------------------------------------------------
// Compute.
// ---------------------------------------------------------------------------

/**
 * @brief Launch the feature evaluation kernel and return host-side results.
 *
 * All plan/navigator/condition/pivot data are copied to temporary device
 * buffers for this launch and freed before returning.  The graph, attributes
 * and labels loaded earlier are reused across Compute() calls.
 */
__host__ std::vector<MatrixGraphFeatureValue> ComputeFeaturesTask::Compute(
    const std::vector<uint32_t>& pivot_vertex_ids,
    const std::vector<MatrixGraphPlanNode>& plan,
    const std::vector<MatrixGraphPlanNode>& navs,
    const std::vector<MatrixGraphCondNode>& conds,
    const std::vector<int32_t>& output_expr_indices) {
  std::vector<MatrixGraphFeatureValue> result;
  if (graph_ == nullptr || d_graph_data_ == nullptr) {
    std::cerr << "[ComputeFeaturesTask::Compute] Graph not loaded"
              << std::endl;
    return result;
  }
  if (per_vertex_attrs_.empty()) {
    std::cerr << "[ComputeFeaturesTask::Compute] Attributes not loaded"
              << std::endl;
    return result;
  }
  if (pivot_vertex_ids.empty() || output_expr_indices.empty() ||
      plan.empty()) {
    return result;
  }

  const uint32_t n_pivots = static_cast<uint32_t>(pivot_vertex_ids.size());
  const uint32_t n_outputs = static_cast<uint32_t>(output_expr_indices.size());

  // Upload plan, navigators, output indices, and pivots.
  MatrixGraphPlanNode* d_plan = nullptr;
  MatrixGraphPlanNode* d_navs = nullptr;
  MatrixGraphCondNode* d_conds = nullptr;
  int32_t* d_output_expr_indices = nullptr;
  uint32_t* d_pivot_vertex_ids = nullptr;

  CUDA_CHECK(cudaMalloc(&d_plan, sizeof(MatrixGraphPlanNode) * plan.size()));
  CUDA_CHECK(cudaMemcpy(d_plan, plan.data(),
                        sizeof(MatrixGraphPlanNode) * plan.size(),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_navs, sizeof(MatrixGraphPlanNode) * navs.size()));
  CUDA_CHECK(cudaMemcpy(d_navs, navs.data(),
                        sizeof(MatrixGraphPlanNode) * navs.size(),
                        cudaMemcpyHostToDevice));

  if (!conds.empty()) {
    CUDA_CHECK(cudaMalloc(&d_conds, sizeof(MatrixGraphCondNode) * conds.size()));
    CUDA_CHECK(cudaMemcpy(d_conds, conds.data(),
                          sizeof(MatrixGraphCondNode) * conds.size(),
                          cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaMalloc(&d_output_expr_indices,
                        sizeof(int32_t) * output_expr_indices.size()));
  CUDA_CHECK(cudaMemcpy(d_output_expr_indices, output_expr_indices.data(),
                        sizeof(int32_t) * output_expr_indices.size(),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_pivot_vertex_ids, sizeof(uint32_t) * n_pivots));
  CUDA_CHECK(cudaMemcpy(d_pivot_vertex_ids, pivot_vertex_ids.data(),
                        sizeof(uint32_t) * n_pivots,
                        cudaMemcpyHostToDevice));

  // Workspace and outputs.
  constexpr uint32_t kWorkspaceCapacity =
      sics::matrixgraph::core::task::kernel::kComputeFeaturesMaxNeighbors;
  const uint32_t per_pivot_workspace =
      kWorkspaceCapacity *
      ComputePlanMaxWorkspaceDepth(plan, navs, output_expr_indices);
  MatrixGraphFeatureValue* d_workspace = nullptr;
  MatrixGraphFeatureValue* d_outputs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_workspace,
                        sizeof(MatrixGraphFeatureValue) * n_pivots *
                            per_pivot_workspace));
  CUDA_CHECK(cudaMalloc(&d_outputs,
                        sizeof(MatrixGraphFeatureValue) * n_pivots * n_outputs));

  // Launch one block per pivot.
  kernel::ComputeFeaturesKernel<<<n_pivots, kWorkspaceCapacity>>>(
      d_graph_data_,
      graph_->get_num_vertices(),
      graph_->get_num_incoming_edges(),
      graph_->get_num_outgoing_edges(),
      d_labels_,
      d_per_vertex_attrs_,
      d_pivot_vertex_ids,
      n_pivots,
      d_plan,
      static_cast<uint32_t>(plan.size()),
      d_navs,
      static_cast<uint32_t>(navs.size()),
      d_conds,
      static_cast<uint32_t>(conds.size()),
      d_output_expr_indices,
      n_outputs,
      d_workspace,
      kWorkspaceCapacity,
      per_pivot_workspace,
      d_outputs);
  CUDA_CHECK(cudaDeviceSynchronize());

  result.resize(n_pivots * n_outputs);
  CUDA_CHECK(cudaMemcpy(result.data(), d_outputs,
                        sizeof(MatrixGraphFeatureValue) * n_pivots * n_outputs,
                        cudaMemcpyDeviceToHost));

  cudaFree(d_plan);
  cudaFree(d_navs);
  cudaFree(d_conds);
  cudaFree(d_output_expr_indices);
  cudaFree(d_pivot_vertex_ids);
  cudaFree(d_workspace);
  cudaFree(d_outputs);

  return result;
}

/** @brief TaskBase hook; loads the graph if it has not been loaded yet. */
__host__ void ComputeFeaturesTask::Run() {
  std::cout << "[ComputeFeaturesTask] Run()" << std::endl;
  if (graph_ == nullptr) {
    LoadGraph(graph_path_);
  }
}

/** @brief Free all persistent device buffers owned by this task. */
__host__ void ComputeFeaturesTask::FreeDeviceBuffers() {
  if (d_graph_data_) {
    cudaFree(d_graph_data_);
    d_graph_data_ = nullptr;
    d_graph_data_size_ = 0;
  }
  if (d_per_vertex_attrs_) {
    cudaFree(d_per_vertex_attrs_);
    d_per_vertex_attrs_ = nullptr;
  }
  if (d_labels_) {
    cudaFree(d_labels_);
    d_labels_ = nullptr;
    n_labels_ = 0;
  }
  for (auto& col : d_columns_) {
    if (col.d_values) cudaFree(col.d_values);
  }
  d_columns_.clear();
  per_vertex_attrs_.clear();
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
