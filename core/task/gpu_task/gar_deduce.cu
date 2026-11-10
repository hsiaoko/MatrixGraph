#include "core/task/gpu_task/gar_deduce.cuh"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include <cub/cub.cuh>

#include "core/task/gpu_task/kernel/kernel_gar_deduce.cuh"
#include "core/util/cuda_check.cuh"

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {

namespace dk = sics::matrixgraph::core::task::kernel;

void GARDeduce::SetGraph(const GARGraphArrays& g) {
  n_vertices_ = g.n_vertices;
  n_edges_ = g.n_edges;
  has_v_label_ = g.v_label_idx != nullptr;
  if (g.n_edges > 0) {
    d_e_src_.Init(sics::matrixgraph::core::data_structures::Buffer<uint32_t>{
        const_cast<uint32_t*>(g.e_src),
        sizeof(uint32_t) * static_cast<size_t>(g.n_edges)});
    d_e_dst_.Init(sics::matrixgraph::core::data_structures::Buffer<uint32_t>{
        const_cast<uint32_t*>(g.e_dst),
        sizeof(uint32_t) * static_cast<size_t>(g.n_edges)});
    d_e_label_.Init(sics::matrixgraph::core::data_structures::Buffer<int32_t>{
        const_cast<int32_t*>(g.e_label_idx),
        sizeof(int32_t) * static_cast<size_t>(g.n_edges)});
  }
  if (has_v_label_ && g.n_vertices > 0) {
    d_v_label_.Init(sics::matrixgraph::core::data_structures::Buffer<int32_t>{
        const_cast<int32_t*>(g.v_label_idx),
        sizeof(int32_t) * static_cast<size_t>(g.n_vertices)});
  }
}

int GARDeduce::Run(const GARPatternArrays& p, const GARPredicate* preds,
                   int n_preds, const GARAttrColumnSet& attrs, int head_u,
                   int head_v, int frontier_cap, int eval_cap,
                   GARDeduceArrays* out) {
  status_ = 0;
  if (out == nullptr || out->num_embeddings == nullptr ||
      out->num_x_satisfied == nullptr || out->match_capped == nullptr) {
    status_ = 1;
    return status_;
  }
  *out->num_embeddings = 0;
  *out->num_x_satisfied = 0;
  *out->match_capped = 0;
  if (out->head_size != nullptr) *out->head_size = 0;
  if (out->emb_rows != nullptr) *out->emb_rows = 0;
  if (p.n_nodes <= 0 || p.n_nodes > dk::kGARDeduceMaxNodes ||
      p.n_edges <= 0 || n_preds > dk::kGARDeduceMaxPreds ||
      attrs.n_cols > GARAttrColumnSet::kMaxCols || frontier_cap <= 0 ||
      eval_cap <= 0) {
    status_ = 2;  // unsupported rule shape; caller treats as skipped
    return status_;
  }
  eval_cap = std::min(eval_cap, frontier_cap);

  using sics::matrixgraph::core::data_structures::Buffer;

  // ---- pattern upload ----
  DeviceOwnedBuffer<int32_t> d_node_label;
  DeviceOwnedBuffer<int32_t> d_p_src;
  DeviceOwnedBuffer<int32_t> d_p_dst;
  DeviceOwnedBuffer<int32_t> d_p_label;
  d_node_label.Init(Buffer<int32_t>{
      const_cast<int32_t*>(p.node_label_idx),
      sizeof(int32_t) * static_cast<size_t>(p.n_nodes)});
  d_p_src.Init(Buffer<int32_t>{const_cast<int32_t*>(p.edge_src),
                               sizeof(int32_t) * static_cast<size_t>(p.n_edges)});
  d_p_dst.Init(Buffer<int32_t>{const_cast<int32_t*>(p.edge_dst),
                               sizeof(int32_t) * static_cast<size_t>(p.n_edges)});
  d_p_label.Init(Buffer<int32_t>{
      const_cast<int32_t*>(p.edge_label_idx),
      sizeof(int32_t) * static_cast<size_t>(p.n_edges)});

  // ---- candidate extraction (count pass, prefix sum, fill pass) ----
  DeviceOwnedBuffer<int> d_counts(sizeof(int) * static_cast<size_t>(p.n_edges));
  dk::GARDeduceFilterParams fp{};
  fp.g_e_src = d_e_src_.GetPtr();
  fp.g_e_dst = d_e_dst_.GetPtr();
  fp.g_e_label = d_e_label_.GetPtr();
  fp.g_v_label = has_v_label_ ? d_v_label_.GetPtr() : nullptr;
  fp.g_n_edges = n_edges_;
  fp.p = dk::GARDeducePatternDev{d_node_label.GetPtr(), d_p_src.GetPtr(),
                                 d_p_dst.GetPtr(),      d_p_label.GetPtr(),
                                 p.n_nodes,             p.n_edges};
  fp.cand_counts = d_counts.GetPtr();
  dk::LaunchGARDeduceCountCandidates(fp);

  std::vector<int> h_counts(static_cast<size_t>(p.n_edges), 0);
  Buffer<int> h_counts_buf{h_counts.data(), sizeof(int) * h_counts.size()};
  d_counts.Device2Host(&h_counts_buf);
  std::vector<int> h_offsets(static_cast<size_t>(p.n_edges) + 1, 0);
  for (int pe = 0; pe < p.n_edges; ++pe) {
    h_offsets[static_cast<size_t>(pe) + 1] =
        h_offsets[static_cast<size_t>(pe)] + h_counts[static_cast<size_t>(pe)];
  }
  const int total_cand = h_offsets.back();
  if (total_cand == 0) return 0;  // no candidates anywhere: zero matches

  DeviceOwnedBuffer<int> d_offsets;
  Buffer<int> h_offsets_buf{h_offsets.data(), sizeof(int) * h_offsets.size()};
  d_offsets.Init(h_offsets_buf);
  std::vector<int> h_cursors(h_offsets.begin(), h_offsets.end() - 1);
  DeviceOwnedBuffer<int> d_cursors;
  Buffer<int> h_cursors_buf{h_cursors.data(), sizeof(int) * h_cursors.size()};
  d_cursors.Init(h_cursors_buf);
  EnsureCapacity(d_cand_src_, sizeof(uint32_t) *
                                 static_cast<size_t>(total_cand));
  EnsureCapacity(d_cand_dst_, sizeof(uint32_t) *
                                 static_cast<size_t>(total_cand));
  fp.cand_offsets = d_offsets.GetPtr();
  fp.cand_src = d_cand_src_.GetPtr();
  fp.cand_dst = d_cand_dst_.GetPtr();
  dk::LaunchGARDeduceFillCandidates(fp, d_cursors.GetPtr());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- pack candidates into sortable keys and sort per pattern-edge
  // segment (fwd by (src,dst), bwd by (dst,src)); the join's binary
  // searches run over these — the device analogue of the CPU matcher's
  // sorted rel-pair arrays ----
  const size_t cand_bytes =
      sizeof(uint64_t) * static_cast<size_t>(total_cand);
  EnsureCapacity(d_fwd_in_, cand_bytes);
  EnsureCapacity(d_bwd_in_, cand_bytes);
  dk::LaunchGARDeducePackCandidates(d_cand_src_.GetPtr(),
                                    d_cand_dst_.GetPtr(), total_cand,
                                    d_fwd_in_.GetPtr(), d_bwd_in_.GetPtr());
  EnsureCapacity(d_fwd_, cand_bytes);
  EnsureCapacity(d_bwd_, cand_bytes);
  {
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, temp_bytes, d_fwd_in_.GetPtr(), d_fwd_.GetPtr(), total_cand,
        p.n_edges, d_offsets.GetPtr(), d_offsets.GetPtr() + 1);
    EnsureCapacity(d_sort_temp_, temp_bytes);
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_sort_temp_.GetPtr(), temp_bytes, d_fwd_in_.GetPtr(),
        d_fwd_.GetPtr(), total_cand, p.n_edges, d_offsets.GetPtr(),
        d_offsets.GetPtr() + 1);
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_sort_temp_.GetPtr(), temp_bytes, d_bwd_in_.GetPtr(),
        d_bwd_.GetPtr(), total_cand, p.n_edges, d_offsets.GetPtr(),
        d_offsets.GetPtr() + 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- join order: init from the edge with the fewest candidates, then
  // BFS over shared pattern nodes so every expansion edge touches the
  // bound set (no cartesian blow-up) ----
  int e0 = 0;
  for (int pe = 1; pe < p.n_edges; ++pe) {
    if (h_counts[static_cast<size_t>(pe)] <
        h_counts[static_cast<size_t>(e0)]) {
      e0 = pe;
    }
  }
  std::vector<int> order;
  order.reserve(static_cast<size_t>(p.n_edges));
  std::vector<char> used(static_cast<size_t>(p.n_edges), 0);
  std::vector<char> bound(static_cast<size_t>(p.n_nodes), 0);
  order.push_back(e0);
  used[static_cast<size_t>(e0)] = 1;
  bound[p.edge_src[e0]] = 1;
  bound[p.edge_dst[e0]] = 1;
  while (order.size() < static_cast<size_t>(p.n_edges)) {
    bool progressed = false;
    for (int pe = 0; pe < p.n_edges; ++pe) {
      if (used[static_cast<size_t>(pe)]) continue;
      if (bound[p.edge_src[pe]] || bound[p.edge_dst[pe]]) {
        order.push_back(pe);
        used[static_cast<size_t>(pe)] = 1;
        bound[p.edge_src[pe]] = 1;
        bound[p.edge_dst[pe]] = 1;
        progressed = true;
      }
    }
    if (!progressed) {
      status_ = 3;  // disconnected pattern
      return status_;
    }
  }
  if (h_counts[static_cast<size_t>(order.front())] == 0) return 0;

  // ---- embedding join ----
  const size_t emb_rows = static_cast<size_t>(frontier_cap);
  const size_t emb_buf_bytes =
      sizeof(int32_t) * emb_rows * static_cast<size_t>(p.n_nodes);
  EnsureCapacity(d_frontier_a_, emb_buf_bytes);
  EnsureCapacity(d_frontier_b_, emb_buf_bytes);
  DeviceOwnedBuffer<int> d_count(sizeof(int));

  const int c0_begin = h_offsets[static_cast<size_t>(e0)];
  dk::LaunchGARDeduceInitEmbeddings(
      d_fwd_.GetPtr() + c0_begin, h_counts[static_cast<size_t>(e0)],
      p.edge_src[e0], p.edge_dst[e0], p.n_nodes, d_frontier_a_.GetPtr(),
      d_count.GetPtr(), frontier_cap);

  int h_count = 0;
  Buffer<int> h_count_buf{&h_count, sizeof(int)};
  d_count.Device2Host(&h_count_buf);
  int capped = h_count > frontier_cap ? 1 : 0;
  int n_curr = std::min(h_count, frontier_cap);
  bool a_is_curr = true;

  for (size_t oi = 1; oi < order.size() && n_curr > 0; ++oi) {
    const int pe = order[oi];
    const int n_cand = h_counts[static_cast<size_t>(pe)];
    const int c_begin = h_offsets[static_cast<size_t>(pe)];
    DeviceOwnedBuffer<int> d_next_count(sizeof(int));
    int32_t* curr = a_is_curr ? d_frontier_a_.GetPtr() : d_frontier_b_.GetPtr();
    int32_t* next = a_is_curr ? d_frontier_b_.GetPtr() : d_frontier_a_.GetPtr();
    dk::LaunchGARDeduceExpandEmbeddings(
        curr, n_curr, p.n_nodes, p.edge_src[pe], p.edge_dst[pe],
        d_fwd_.GetPtr() + c_begin, d_bwd_.GetPtr() + c_begin, n_cand, next,
        d_next_count.GetPtr(), frontier_cap);
    d_next_count.Device2Host(&h_count_buf);
    if (h_count > frontier_cap) capped = 1;
    n_curr = std::min(h_count, frontier_cap);
    a_is_curr = !a_is_curr;
  }
  int32_t* d_final = a_is_curr ? d_frontier_a_.GetPtr() : d_frontier_b_.GetPtr();

  // The eval cap mirrors the CPU chase's per-rule match cap: only the
  // first eval_cap embeddings are counted / X-evaluated / head-emitted.
  const int eval_rows = std::min(n_curr, eval_cap);
  *out->num_embeddings = static_cast<uint64_t>(eval_rows);
  *out->match_capped = (capped || n_curr > eval_cap) ? 1 : 0;
  if (eval_rows == 0) return 0;

  // ---- X evaluation + head emission ----
  dk::GARDeduceEvalParams ep{};
  ep.emb = d_final;
  ep.n_emb = eval_rows;
  ep.n_nodes = p.n_nodes;
  ep.g_e_src = d_e_src_.GetPtr();
  ep.g_e_dst = d_e_dst_.GetPtr();
  ep.g_e_label = d_e_label_.GetPtr();
  ep.g_n_edges = n_edges_;
  ep.n_preds = n_preds;
  for (int i = 0; i < n_preds; ++i) ep.preds[i] = preds[i];
  ep.n_cols = attrs.n_cols;
  for (int c = 0; c < attrs.n_cols; ++c) {
    const int32_t cid = attrs.col_ids[c];
    if (attr_vals_cache_.find(cid) == attr_vals_cache_.end()) {
      auto vals = std::make_unique<DeviceOwnedBuffer<double>>();
      Buffer<double> h_vals{const_cast<double*>(attrs.values[c]),
                            sizeof(double) * static_cast<size_t>(n_vertices_)};
      vals->Init(h_vals);
      attr_vals_cache_.emplace(cid, std::move(vals));
      if (attrs.valid[c] != nullptr) {
        auto valid = std::make_unique<DeviceOwnedBuffer<uint8_t>>();
        Buffer<uint8_t> h_valid{
            const_cast<uint8_t*>(attrs.valid[c]),
            sizeof(uint8_t) * static_cast<size_t>(n_vertices_)};
        valid->Init(h_valid);
        attr_valid_cache_.emplace(cid, std::move(valid));
      }
    }
    ep.attr_vals[c] = attr_vals_cache_.at(cid)->GetPtr();
    const auto it = attr_valid_cache_.find(cid);
    ep.attr_valid[c] = it != attr_valid_cache_.end() ? it->second->GetPtr()
                                                     : nullptr;
  }
  ep.has_head = (head_u >= 0 && head_v >= 0) ? 1 : 0;
  ep.head_u = head_u;
  ep.head_v = head_v;

  DeviceOwnedBuffer<int> d_head_count(sizeof(int));
  if (ep.has_head && out->head_src != nullptr && out->head_capacity > 0) {
    EnsureCapacity(d_head_src_,
                   sizeof(uint32_t) * static_cast<size_t>(out->head_capacity));
    EnsureCapacity(d_head_dst_,
                   sizeof(uint32_t) * static_cast<size_t>(out->head_capacity));
    ep.head_src = d_head_src_.GetPtr();
    ep.head_dst = d_head_dst_.GetPtr();
    ep.head_count = d_head_count.GetPtr();
    ep.head_cap = out->head_capacity;
  } else {
    ep.has_head = 0;
  }

  DeviceOwnedBuffer<unsigned long long> d_x_count(
      sizeof(unsigned long long));
  ep.x_count = d_x_count.GetPtr();

  if (out->emb_x_ok != nullptr) {
    EnsureCapacity(d_x_ok_, sizeof(uint8_t) * static_cast<size_t>(eval_rows));
    ep.x_ok = d_x_ok_.GetPtr();
  }

  dk::LaunchGARDeduceEvalEmit(ep);

  unsigned long long h_x_count = 0;
  Buffer<unsigned long long> h_x_count_buf{&h_x_count, sizeof(h_x_count)};
  d_x_count.Device2Host(&h_x_count_buf);
  *out->num_x_satisfied = h_x_count;

  if (ep.has_head) {
    int h_head_count = 0;
    Buffer<int> h_head_count_buf{&h_head_count, sizeof(int)};
    d_head_count.Device2Host(&h_head_count_buf);
    *out->head_size = h_head_count;
    const int n_back = std::min(h_head_count, out->head_capacity);
    if (n_back > 0) {
      CUDA_CHECK(cudaMemcpy(out->head_src, d_head_src_.GetPtr(),
                            sizeof(uint32_t) * static_cast<size_t>(n_back),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(out->head_dst, d_head_dst_.GetPtr(),
                            sizeof(uint32_t) * static_cast<size_t>(n_back),
                            cudaMemcpyDeviceToHost));
    }
  }

  if (out->embeddings != nullptr && out->emb_rows != nullptr) {
    const int n_back = std::min(eval_rows, out->emb_capacity);
    CUDA_CHECK(cudaMemcpy(out->embeddings, d_final,
                          sizeof(int32_t) * static_cast<size_t>(n_back) *
                              static_cast<size_t>(p.n_nodes),
                          cudaMemcpyDeviceToHost));
    if (out->emb_x_ok != nullptr) {
      CUDA_CHECK(cudaMemcpy(out->emb_x_ok, d_x_ok_.GetPtr(),
                            sizeof(uint8_t) * static_cast<size_t>(n_back),
                            cudaMemcpyDeviceToHost));
    }
    *out->emb_rows = n_back;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}

}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
