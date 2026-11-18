# Interface and abstraction-boundary audit

Seam: public interfaces and abstraction boundaries. Read-only. Duplication and style are covered by other auditors and are not repeated here. Every claim carries file:line evidence. Counts below are raw includer counts across apps/, core/, tools/ (build/ and third_party/ excluded); a header's own .cu counts as one includer.

Principles applied (leaf skills read this session): boundary-discipline, minimize-reader-load, laziness-protocol, type-system-discipline.

## (a) Public-surface map

- core/matrixgraph.cuh — C++ facade class MatrixGraph. 15 includers. Heavy: pulls all 4 schedulers, gpu_task_manager, task_base, cuda_check, <iostream>.
- core/matrixgraph.cu — facade impl. Dispatches Run via enum + reinterpret_cast.
- core/go_api/matrixgraph_go_api.h — C ABI for Go CGO. Consumed by go_api.cu (relative include) and external Go; 0 core-path includers. Lean: stdint + compute_features_types.h only.
- core/go_api/matrixgraph_go_api.cu — C ABI impl. Opaque handle create/destroy + return-code wrappers.
- core/task/gpu_task/compute_features_types.h — C/Go wire types (MatrixGraphFeatureValue, plan nodes, enums). 4 includers. Self-contained, extern "C". Lives inside the GPU task dir though it is the boundary type.
- core/task/gpu_task/task_base.cuh — GPU task base, owns CUDA streams. 29 includers. No virtual Run.
- core/task/cpu_task/cpu_task_base.h — CPU task base. 5 includers. Empty class, heavy includes.
- core/task/gpu_task/execute_agg_prim.cuh — ExecuteAggPrim task + task::FeatureValue/AggPrim/AllFeatures. 7 includers.
- core/task/gpu_task/graph_aggregate.cuh — GraphAggregate task. 5 includers.
- core/task/gpu_task/compute_features.cuh — ComputeFeaturesTask. 4 includers.
- core/task/gpu_task/graph_filter_aggregate.cuh — filter+aggregate task. 1 includer (own .cu).
- core/task/gpu_task/block_agg_prims.cuh — block-level agg primitives. 3 includers.
- core/task/gpu_task/kernel/kernel_graph_aggregate.cuh — re-exports task::FeatureValue/AggPrim/AllFeatures into namespace kernel (lines 31-37). Apps and go_api alias through kernel::.
- apps/wrapper.h — vestigial. Only includes cuda_runtime.h + stdio.h, no declarations. 0 includers.
- core/common/types.h — typedefs, macros, TaskType/StoreStrategy enums. 87 includers. Macro pollution.
- core/common/consts.h — tuning constants. 35 includers. static const in header.
- core/common/yaml_config.h — YAML::convert adapters for metadata. 11 includers. Heavy yaml-cpp include in common.
- core/common/host_algorithms.cuh — host algorithm helpers. 8 includers.
- core/util/cuda_check.cuh — CUDA_CHECK macro. 33 includers. exit() on error.
- core/util/execution_policy.h — ParForEach + MATRIXGRAPH_EXEC_POLICY. 19 includers. Global namespace, backend chosen by __CUDACC__.
- core/util/cpu_parallel_scope.h — TBB global_control RAII. 2 includers. Hard oneTBB include.
- core/util/logging.h — folly logging wrappers. 1 includer (tools/graph_partitioner). Non-inline function in header.
- core/util/matrix.h — cuBLAS/cuRAND demo helpers. 0 includers. Dead.
- core/util/cuda_device.cuh — device-list env helpers. 11 includers.
- core/util/cuda_prefetch.cuh — managed-memory prefetch. 7 includers.
- core/util/format_converter.cuh — format conversion. 10 includers.
- core/util/bitmap.h / bitmap_ownership.h / bitmap_no_ownership.h — bitmap variants. 16 / 15 / 16 includers.
- core/util/atomic.h — atomics. 16 includers.
- core/util/set_operations.h — set ops. 1 includer.
- core/util/gpu_bitmap.cuh — GPU bitmap. 4 includers.
- core/util/cuda_launch_dims.cuh — launch dims. 3 includers.
- core/components/scheduler/scheduler.h — Scheduler base + SchedulerType. 18 includers. Non-virtual dtor.
- core/components/gpu_task_manager.cuh — GPUTaskManager. 2 includers (facade + own .cu). Never instantiated.
- core/data_structures/immutable_csr.cuh — CSR graph. 43 includers.
- core/data_structures/attributes.h — attribute storage + ValueType. 9 includers.

## (b) Issues ranked by severity

### High

- H1 (ABI-SENSITIVE). GraphAggregate C output conversion clobbers the discriminated union. core/go_api/matrixgraph_go_api.cu:328-331 and :337-343 write dst->i64, then dst->f64, then dst->b sequentially into MatrixGraphFeatureValue, whose i64/f64/b share one 8-byte slot (core/task/gpu_task/compute_features_types.h:201-207). The last write (b) wins, so every int/float result returned by matrixgraph_graph_aggregate_compute_features and compute_all is corrupted. The ExecuteAggPrim path converts correctly with a type switch (matrixgraph_go_api.cu:465-485). Direction: convert through one type-switched helper shared by both paths; never assign multiple union members.
- H2 (ABI-SENSITIVE). A shared library terminates the host process on CUDA error. core/util/cuda_check.cuh:12 calls exit(EXIT_FAILURE) inside CUDA_CHECK, used by ExecuteAggPrim::Compute (core/task/gpu_task/execute_agg_prim.cu:575-577) which the Go C API calls. core/data_structures/edgelist.cu:169,178,191 also exit() on file errors. The C ABI promises return codes (matrixgraph_go_api.h:13) and go_api.cu wraps calls in catch(...) returning 1, but exit() bypasses all of it and kills the Go process. Direction: library code propagates errors (return code or exception caught at the C boundary); exit/abort only in app main.
- H3 (ABI-SENSITIVE). Two AggPrim numbering schemes cross the ABI, with a false compatibility comment. ComputeFeatures plan ops use MatrixGraphAggPrim (MG_AGG_SUM=3, compute_features_types.h:96-113) mirrored by constexpr kAgg* (kernel_compute_features_primitives.cuh:30-46). GraphAggregate FeatureRequest.prim is static_cast straight to task::AggPrim (kSum=1, execute_agg_prim.cuh:102-119) at matrixgraph_go_api.cu:311 with no remap. compute_features_types.h:93-95 claims the two orderings "intentionally match" so callers can reuse constants; they do not. A caller reusing MG_AGG_SUM=3 in a FeatureRequest gets kMedian. Direction: one primitive numbering at the boundary, explicit remap tables per task, correct the comment.
- H4. Facade dispatch is type-unsafe and silently no-ops. core/matrixgraph.cu:38-130 and :132-160 switch on TaskType and reinterpret_cast TaskBase*/CPUTaskBase* to a concrete task. Nothing enforces that the enum matches the pointer type; a mismatch is UB. TaskType values kMatrixAnalysis, kSSSP, kGEMV, kExecuteAggPrim (core/common/types.h:42-62) hit default: break and do nothing silently. TaskBase has no virtual Run (task_base.cuh:16-29), so the enum parameter is redundant. Direction: a virtual Run() on the task base removes the enum, the cast switch, and the silent default.
- H5 (ABI-SENSITIVE). Documented Go mirror of MatrixGraphAllFeatures is one field short. docs/go_api/README.md:383-400 defines AllFeatures with 16 fields (Count..Skew, no DFeat). The C struct has 17 fields incl dfeat (core/go_api/matrixgraph_go_api.h:57-75) and compute_all writes 17 per pivot. A Go caller copying the README mirror undersizes its buffer by 16 bytes per pivot and the C call writes out of bounds. Direction: regenerate the Go mirror from the C struct; add DFeat.

### Medium

- M1. Facade owns a scheduler it never uses and leaks it. core/matrixgraph.cu:25-34 news CHBLScheduler only for kCHBL (other SchedulerType values leave scheduler_ indeterminate) and ~MatrixGraph()=default (matrixgraph.cuh:38) never deletes it. scheduler_ is never read in Run. Members streams_, p_streams_mtx_, p_hr_start_mtx_, p_hr_start_lck_, p_hr_start_cv_, p_hr_terminable_ (matrixgraph.cuh:48-60) are never referenced in the .cu. Direction: delete the dead scheduler/member state or wire it; own the scheduler with unique_ptr if kept.
- M2. Stream ownership contract is unclear and not uniform. TaskBase owns streams_ but ~TaskBase()=default does not call DestroyStreams (task_base.cuh:16-29, task_base.cu:19-25); release depends on each derived destructor calling it (execute_agg_prim.cu:475 does). GraphAggregate inherits TaskBase and sets n_streams_ but keeps its own per_gpu_states_[].streams and never calls TaskBase::EnsureStreams/DestroyStreams (graph_aggregate.cuh PerGpuState; graph_aggregate.cu:517-530). Direction: base destructor releases what the base owns; one stream-ownership path for all GPU tasks.
- M3. Eight kernel singletons are dead global mutable state. GetInstance() plus inline static XKernelWrapper* ptr_ exist in kernel_bfs.cuh:34,44; kernel_wcc.cuh:54,78; kernel_pagerank.cuh:33,44; kernel_gar_match.cuh:25,34; kernel_matrix_ops.cuh:34,73; kernel_subiso.cuh:32,57; kernel_woj_subiso.cuh:43,60; matrix_operations.cuh:27,123. GetInstance() has zero call sites; the real interface is the static methods (e.g. BFSKernelWrapper::BFS). GetInstance news without delete (kernel_bfs.cu:45-49). Direction: delete the singleton scaffolding (private ctor, deleted copy, ptr_, GetInstance); keep static entry points or free functions.
- M4. Scheduler base is polymorphic with a non-virtual destructor and silent setters. core/components/scheduler/scheduler.h:28 ~Scheduler()=default despite virtual GetBinID (:30); deleting a derived via Scheduler* is UB. RecordConsume (:47) and set_bin_id_by_ball_id (:68) use map::insert, which silently keeps the first value on repeat keys. get_bin_id_by_ball_id (:62) returns INT_MAX but the header includes only <unordered_map>,<utility> (non-self-contained). Direction: virtual dtor; setters that overwrite; include <climits>.
- M5. CPUTaskBase is an empty abstraction with heavy includes. core/task/cpu_task/cpu_task_base.h:14-21 defines a class with only a non-virtual default destructor yet includes <future>,<mutex>,<thread>,<unordered_map>. It provides no behavior; the CPU facade still reinterpret_casts to concrete types. Direction: give it a virtual Run and drop unused includes, or remove it.
- M6. GraphRAG apps bypass the task layer and reimplement buffer setup. apps/graphrag_bfs.cpp:80 comments "Replicate BFS::ExecuteBFS buffer setup" and calls BFSKernelWrapper::BFS directly (:116); graphrag_pagerank.cpp:82,133 and graphrag_wcc.cpp:78,117 do the same for PageRank/WCC. All three construct MatrixGraph system (graphrag_bfs.cpp:78) but never call system.Run, instantiating the facade (and its leaked scheduler) pointlessly. Direction: expose the needed buffer setup through the task class and call it; drop the unused facade object.
- M7. util/matrix.h is dead and unsound. core/util/matrix.h has 0 includers, defines non-inline free functions PrintMatrix/cuRANDFill/float2int8 (ODR hazard if ever included twice), and uses std::cerr, std::chrono, INT_MAX, std::is_same without including <iostream>,<chrono>,<climits>,<type_traits>. Heavy cublas/curand/fp16/profiler includes. Direction: delete.
- M8. logging.h is a heavy folly dependency for one consumer with an ODR hazard. core/util/logging.h:27 defines non-inline DefaultConfigWithLogFile in a header; only tools/graph_partitioner/graph_partitioner.cpp:6 includes it. Generic macros LOG_INFO/LOG_ERROR/LOG_FATAL (logging.h:60-86) collide with common logging libraries. Direction: move folly logging into the one tool that uses it, or make the function inline and namespace the macros.
- M9. Error strategy is inconsistent across layers. io/attributes_io.cu:37-153, data_structures/immutable_csr.cu:193-197, matrix.cu:29, matches.cuh:259 and task/gpu_task/gar_match.cu throw std::runtime_error; edgelist.cu and cuda_check.cuh exit(); go_api.cu returns codes. The same load path can throw (caught) or exit (fatal) depending on which helper fails. Direction: one error contract per layer; exceptions or codes internally, translated to return codes only at the C boundary.
- M10. Serialization adapter sits in common and propagates yaml-cpp widely. core/common/yaml_config.h:5-13 includes <yaml-cpp/yaml.h> and <iostream>, opens namespace YAML, and is included by 11 files. GraphMetadata::decode (:104-118) accesses node["GraphMetadata"] without a size guard, unlike the other decoders. Direction: move YAML adapters to the io boundary; validate at decode.

### Low

- L1. common/types.h pollutes the global macro namespace and leaves semantic primitives unbranded. core/common/types.h:8-22 defines CHECK, CHECK_POINT, EQUALITIES, SIM, MAX_VERTEX_ID etc. as macros seen by 87 includers; CHECK is highly collision-prone. VertexID/VertexIndex/VertexLabel/EdgeIndex (:27-34) are all uint32_t and interchangeable. Direction: replace macros with named constexpr in namespace; brand distinct id types.
- L2. Parallel backend leaks into util headers and is compiler-selected. core/util/execution_policy.h:14-26 defines ParForEach and MATRIXGRAPH_EXEC_POLICY at global namespace, serial under __CUDACC__ and std::execution::par otherwise. core/util/cpu_parallel_scope.h:8 includes oneapi/tbb/global_control.h and stores a TBB type as a member. The algorithm layer is coupled to TBB and to the compiler macro rather than to a backend interface. Direction: a single named parallel-for seam in namespace, backend hidden behind it.
- L3. consts.h uses static const in a header. core/common/consts.h:12-66 gives each TU its own copy of every constant; the closing comment names the wrong guard (:68 MATRIXGRAPH_CORE_COMMON_TYPES_H_). Direction: inline constexpr; fix the comment.
- L4. C ABI handle and doc conventions are loose. core/go_api/matrixgraph_go_api.h:13 says functions "generally" return 0; opaque void* handles carry no type tag, so passing a handle from the wrong family is UB after static_cast (matrixgraph_go_api.cu:260). The matrixgraph_subiso CSR-layout doc block (go_api.h:243-262) is orphaned above the ComputeFeatures section while the declaration sits at :355. Direction: state the exact error contract; move the subiso doc to its declaration.
- L5. Plan-node discriminator doc contradicts itself. core/task/gpu_task/compute_features_types.h:167 says type is "MatrixGraphExprType for expressions; negative nav tag" but the struct doc (:161-165) and MatrixGraphNavType (:51-58) use non-negative MG_NAV_* values. Direction: document one actual convention matching the interpreter.
- L6 (ABI-SENSITIVE). Documented union member type differs from the header. docs/go_api/README.md:186 shows uint8_t b in the GraphAggregate MatrixGraphFeatureValue, but the actual member is int32_t b (compute_features_types.h:205); the ExecuteAggPrim doc section correctly says int32. The 8-byte slot is unchanged so the Go raw[8]byte mirror still works. Direction: correct the doc to int32_t.
- L7. Include-guard closing comments are wrong. core/util/logging.h:88 names MINIGRAPH_UTILITY_LOGGING_H_; core/common/consts.h:68 names TYPES. Direction: match the guard.

## (c) ABI-sensitive items (one-way door; explicit owner approval required before any change)

These touch the C/Go ABI or the task::FeatureValue vs MatrixGraphFeatureValue boundary. Do not change layout, enum values, field order, or conversion semantics without owner sign-off.

- H1. Union-clobbering conversion at the GraphAggregate C boundary (matrixgraph_go_api.cu:328-343). Fixing it changes the bytes Go receives.
- H2. exit() reachable from the C ABI (cuda_check.cuh:12 via execute_agg_prim.cu:575-577). Changing to return codes alters the failure contract Go relies on.
- H3. Two AggPrim orderings and the false match comment (compute_features_types.h:93-113; execute_agg_prim.cuh:102-119; matrixgraph_go_api.cu:311). Renumbering breaks serialized plans and Go constants.
- H5. MatrixGraphAllFeatures 17-field layout vs 16-field Go mirror (go_api.h:57-75; README.md:383-400).
- L6. MatrixGraphFeatureValue union member type doc drift (compute_features_types.h:205; README.md:186).
- Boundary placement. The wire types live in core/task/gpu_task/compute_features_types.h and the C++ value type is task::FeatureValue (execute_agg_prim.cuh:22-40, union i64/f64/f32/b/str). The C union has no f32 or str member, so FLOAT32 is read as f64 (matrixgraph_go_api.cu:474-476) and STRING has no C representation (ToMgValueType returns MG_VALUE_INVALID, matrixgraph_go_api.cu:424-439). Unifying task::FeatureValue and MatrixGraphFeatureValue is the known pending design decision; any unification changes the ABI and needs owner approval. The kernel:: aliases (kernel_graph_aggregate.cuh:31-37) are re-exports of task:: types, not a third value type.

## (d) Unused or near-unused exports

- apps/wrapper.h — 0 includers, no declarations. Delete.
- core/util/matrix.h — 0 includers, dead demo helpers. Delete (M7).
- core/components/gpu_task_manager.cuh — GPUTaskManager never instantiated; included only by the facade (matrixgraph.cuh:4) and its own .cu. Remove the include and the component if confirmed dead.
- GetInstance() + inline static ptr_ in 8 kernel wrappers — zero call sites (M3). Delete the singleton scaffolding.
- MatrixGraph facade members streams_, p_streams_mtx_, p_hr_start_mtx_, p_hr_start_lck_, p_hr_start_cv_, p_hr_terminable_ and the unused scheduler_ — never read (M1). Delete or wire.
- TaskType values kMatrixAnalysis, kSSSP, kGEMV, kExecuteAggPrim — exported but never dispatched by the facade (H4). Remove or implement.
- matrixgraph_gar_match — documented stub returning empty output (go_api.h:208-215; routes to GARMatch::Run). ABI-sensitive; confirm with owner before removal.
- core/util/logging.h — 1 consumer; core/util/set_operations.h — 1 consumer; core/task/gpu_task/graph_filter_aggregate.cuh — 1 consumer (own .cu). Candidates to fold into their single caller.
- Facade MatrixGraph::Run itself is bypassed by compute_features.cpp, execute_agg_prim.cpp, graph_aggregate.cpp and graphrag_expand.cpp, which use task classes directly. Two entry conventions coexist; converge on one.
