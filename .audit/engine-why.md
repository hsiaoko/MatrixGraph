# Engine why audit. Rationale from the record, for the engine/API redesign.

Date 2026-11-17. HEAD 4cec1d5. Read-only. History is grafted at 5fec213 (2026-07-17); 7 commits visible, the 6 later ones all 2026-11-10. Every questioned file already exists at the graft, so pre-graft rationale is unrecoverable. Sources are comments, docs/, commit messages, and the probe log .audit/engine-grounding.md (2026-11-17). Inferences are labeled.

## Q1. execution_policy.h. Serial under __CUDACC__, compiler-selected backend.

Record. execution_policy.h:5-11 comments "nvcc 与 GCC 12+ 的 avx512bf16/amxtileintrin 不兼容，<execution> 会间接包含这些头文件导致编译失败" and serial is "与历史行为一致"; pure CXX TUs get std::execution::par. core/CMakeLists.txt:26 passes -Xcompiler=-mno-avx512f,-mno-avx512bf16,-mno-amx-tile; :117-120 says "nvcc cannot compile <execution> reliably, so that logic stays out of .cu files" and links TBB for libstdc++ par algorithms. tools/graph_converter/CMakeLists.txt:3 says "纯主机 CXX；ParForEach 等在 __CUDACC__ 外走 std::execution::par". Probe .audit/engine-grounding.md:16,19 confirms <execution> and every TBB header fail under nvcc 12.6 with GCC 12.4/13 even with the -mno flags, because the EDG frontend parses intrinsic bodies.
Inference. ParForEach exists to give one call-site spelling across both TU kinds while nvcc TUs stay serial. Backend is compiler-selected because the failure is a frontend parse problem that no flag or runtime check can fix; the only lever is which compiler sees the TU.
Unanswered. When the workaround was introduced (pre-graft). Whether any nvcc TU ever ran parallel host code. Why host-parallel logic was only partially moved to .cpp (edgelist_khop.cpp is the cited example).

## Q2. gpu_task/task_base.cuh. Streams only, no task contract.

Record. At graft TaskBase was empty except commented-out GetStream(task_id)/ReleaseStream/IsTaskFinished and a streams_by_task_id_ map. The matching async engine GPUTaskManager::SubmitTask/GetStream/IsTaskFinished is implemented (gpu_task_manager.cu:8-36) but has zero callers repo-wide. 5c9d97a deleted the dead code with comment "Kept minimal: derived tasks own their CUDA streams and scratch buffers, so there is no shared state to manage here." 4cec1d5 hoisted EnsureStreams/DestroyStreams/n_streams_/streams_ out of ExecuteAggPrim and GraphFilterAggregate into TaskBase and made the destructor virtual; GraphFilterAggregate keeps its old default via n_streams_=2 in its constructor (graph_filter_aggregate.cu:684). Current comment task_base.cuh:13-14 is "Owns the CUDA streams used for per-stream parallelism; derived tasks own their own device/pinned scratch buffers." Scratch stays per-task, "Persistent scratch reused across ComputeByPrim calls ... Removes per-call cudaMalloc/Free" (execute_agg_prim.cuh:285-286).
Inference. The base owns exactly the state that was duplicated in practice (streams), nothing more. There is no virtual Run contract because task entry points are heterogeneous (Run, ComputeAll, LoadAttributes+Compute) and the async manager design they would have served was abandoned before the graft.
Unanswered. Why the manager/scheduler design was abandoned. Whether per-task-id stream pooling is still a goal.

## Q3. matrixgraph.cu facade. Enum + reinterpret_cast, unused scheduler.

Record. Both the enum/reinterpret_cast dispatch and the constructed-but-never-read scheduler_ exist verbatim at graft (git show 5fec213:core/matrixgraph.cu). No comment or commit justifies either; 5c9d97a only renamed locals. Apps do use the facade, e.g. apps/wcc.cpp:73-76 constructs MatrixGraph system(scheduler_type) then system.Run(kWCC, task). Run() only casts, calls task->Run(), prints timing; prepare_end_time is unused. The scheduler's would-be consumer GPUTaskManager has no callers.
Inference. reinterpret_cast is safe only because each app constructs the concrete task and passes the matching enum itself; it is a stand-in for the virtual contract TaskBase never got. The scheduler is a vestige of the abandoned async multi-device engine, kept because the constructor signature is public API used by every app's -scheduler flag.
Unanswered. Any intended scheduling semantics. Whether the dispatch shape was a deliberate choice at all; the record is silent.

## Q4. cpu_task/ vs gpu_task/ split, LFTJ duplication, subiso_cpu.cu.

Record. docs/apps/lftj_subiso.md:8-12 states "Both reuse the same host-side preprocessing ... The GPU version moves the enumeration ... to a CUDA kernel" and calls the GPU side a "count-only MVP". lftj_subiso_gpu.cuh:18-22 claims "Host side reuses the CPU LFTJ preprocessing". The implementation does not share it: both TUs reimplement BuildUndirectedAdjacency/BuildCandidateSets/ComputeMatchingOrder and the GPU TU never references LFTJSubIso. Real sharing is min_wise_filter.h, made the single templated CollectEllHop implementation by 4cec1d5 (commit message). GPU filter defaults differ (ldf/bloom off, lftj_subiso_gpu.cuh:38-41). subiso_cpu.cu has zero __global__/__device__/<<< (verified in .audit/engine-grounding.md:17), includes CUDA-annotated data structures and MatrixOpsKernelWrapper, and keeps a disabled GPUMatrixFilter whose body starts with return true; (subiso_cpu.cu:794-798, identical at graft :901-905).
Inference. The directory split tracks the compiler split: .cpp TUs may use TBB/par, .cu TUs may not (Q1). GPU LFTJ preprocessing was copied to ship the MVP fast, and the doc's "reuse" claim describes intent plus the shared min-wise filter, not the adjacency/order code. subiso_cpu.cu is .cu by inheritance from GPU SubIso (shared WOJ structures, the dead GPU matrix-filter path), and that extension silently serializes its ParForEach through nvcc (grounding:17).
Unanswered. Why preprocessing was copied instead of extracted. Whether subiso_cpu must remain a .cu.

## Q5. C wire types and the two AggPrim numberings.

Record. compute_features_types.h:10-27 documents the C/Go boundary, flat plan arrays, host ownership ("must keep it alive until LoadAttributes() returns"), and the union clobber hazard; :33-35 says "New expression kinds must be added at the end to keep existing serialized plans stable"; :196-199 repeats the union hazard. docs/go_api/compute_features.md:7-9 says "The API is intentionally low-level"; docs/go_api/README.md:192 says Go wrappers must mirror the 8-byte union slot exactly. Internal task::FeatureValue (execute_agg_prim.cuh:22-42) carries a StringView pointer, f32, bool and __host__ __device__ methods; its ValueType "mirrors the Go enum used by igeenum" (attributes.h:17-30) and is numbered differently from MG_VALUE_*; explicit switches bridge them (kernel_compute_features.cu:59-64, go_api ToCppValueType).
Numberings. MG_AGG_* (compute_features_types.h:96-113) equals the graft-time GraphAggregate local enum (git show 5fec213:core/task/gpu_task/kernel/kernel_graph_aggregate.cuh:72-89) and the kAgg* constants (kernel_compute_features_primitives.cuh:29-45). task::AggPrim (execute_agg_prim.cuh:102-120) is the MG_EXEC_AGG_*/README ordering. git blame shows every line of compute_features_types.h is ^5fec213, so both enums predate visible history. 5c9d97a added "AggPrim and AllFeatures below are intentionally LOCAL to GraphAggregate ... the Go C API depends on this ordering, so they must not be aliased"; 4cec1d5 deleted that comment and aliased GraphAggregate to task::AggPrim. That aliasing made the :93 "intentionally match" claim false; it was true at the graft. go_api.h:94 ("different from the ordering used by GraphAggregate/ComputeFeatures") is now half stale. ComputeFeatures kept the old numbering: its interpreter dispatches on it (ApplyAggPrim, kernel_compute_features_primitives.cuh:472-509) and apps pass MG_AGG_SUM=3 (apps/compute_features.cpp:244). The C AllFeatures field order is a wire contract copied by field name (matrixgraph_go_api.cu:345-364), decoupled from C++ declaration order.
Inference. Separate C types exist because the internal value type is pointer-bearing, CUDA-annotated, and numbered after an external Go enum, while the C ABI needs a stable pointer-free layout Go can mirror. ComputeFeatures numbering was left untouched to keep serialized plans and existing callers stable.
Unanswered. Where the Go consumers live (none found in the GARAgent tree). Whether MG_AGG_* will be bridged or renumbered. Why 4cec1d5 left the stale comments.

## Q6. docs/ intent.

Record. No engine architecture doc exists. docs/README.md is an index. docs/MATRIXGRAPH_ENV.md documents the device/stream/launch env contract with defaults (streams default 2, device-list precedence). docs/graph_filter_aggregate.md:1-12 states the fusion intent (navigation+filter+gather+reduce in one launch, graph uploaded once and reused). docs/go_api/* state the low-level flat-plan intent. docs/apps/lftj_subiso.md states shared-preprocessing intent. .audit/engine-grounding.md (2026-11-17) records the probe constraints and the redesign goal of explicit backend selection and a unified contract.
Inference. Design intent lives in per-task docs and comments, not one design doc; the graft makes the original engine rationale unrecoverable.
Unanswered. All pre-graft architecture decisions.

## Constraints for the redesign.

Must survive (recorded rationale or measured constraint).
1. nvcc TUs cannot include <execution> or any TBB header; probe-verified frontend failure, not flag-fixable. CPU-parallel code lives in pure CXX TUs behind TBB-free interface headers.
2. C wire layout is a contract: 8-byte union slot, MatrixGraphAllFeatures field order, flat plan arrays, append-only enum growth for serialized-plan stability.
3. Two live prim numberings: task::AggPrim (GraphAggregate/ExecuteAggPrim/Go) and MG_AGG/kAgg (ComputeFeatures plans). Unification needs an explicit bridge or a versioned plan break.
4. Ownership rules: host plan/attribute pointers valid until LoadAttributes returns; handle owns GPU resources; persistent scratch avoids per-call cudaMalloc/Free.
5. Stream-count configurability (MATRIXGRAPH_CUDA_STREAMS default 2, set_num_streams before first compute) and per-stream parallelism.
6. ValueType mirrors the external igeenum Go numbering; boundary conversions stay explicit.
Accidents that can be discarded (no recorded rationale).
1. __CUDACC__ as the backend selector; replace with explicit engine choice once parallel code is in .cpp.
2. Enum+reinterpret_cast facade dispatch, unused scheduler_, caller-less GPUTaskManager async vestiges.
3. Duplicated LFTJ host preprocessing; the doc already claims sharing.
4. Device-free .cu files (subiso_cpu.cu, hom_match.cu) that force nvcc and silently serialize ParForEach.
5. Stale comments (compute_features_types.h:93, go_api.h:94, primitives.cuh:29) and the third primitive copy in kernel_compute_features_primitives.cuh.
6. Disabled GPUMatrixFilter dead code.