# Style, Naming, Comments Audit

Scope: 194 C++ files (34 .h, 80 .cuh, 59 .cu, 23 .cpp, 0 .cc), ~41.8k lines in core/, apps/, tools/. third_party/ excluded. Read-only audit; no builds run (one in parallel), no files modified except this report.

Method: grep sweeps over all 194 files plus full reads of ~25 representative files. camelCase measured by token-frequency sweep with external APIs filtered (CUDA, cuRAND, cuBLAS, folly, TBB, yaml-cpp, gflags, gtest). Counts are declaration-level estimates, not use-site counts. Struct data fields are treated as exempt from the trailing-underscore rule per Google style (class members only); this assumption is listed as an open decision. The prior local-variable sweep is verified complete: zero remnants of `dimBlock`/`loadRow` patterns, and all residual camelCase tokens are string literals (ArangoDB JSON keys at tools/graph_converter/converter/to_arangodb_json.cuh:284-288, AQL queries at core/task/gpu_task/gar_match.cu:302-327) or external API names.

Tooling gap: clang-format binary is NOT installed. Formatting deviation counts below are grep-based estimates, not formatter output. .clang-format exists but is itself misconfigured (see open decisions).

## (a) Per-directory violation estimates

Naming = project-owned identifier violations (functions, types, constants, members, namespaces, filenames). Guards = include-guard defects. Format = line-level deviations (long lines, pointer alignment, access-specifier indent, braceless multiline bodies). Comments = files lacking responsibility headers / dead-comment lines.

| Directory | Files/lines | Naming | Guards | Format (lines) | Comment issues |
|---|---|---|---|---|---|
| core/common | 4 / 279 | 15 | 1 stale endif | ~10 | 3 of 4 headers no header; 7 dead-comment lines |
| core/components | 6 / 344 | 4 | 1 broken | ~15 | 5 of 6 files no/weak header; 3 dead lines; 5 WHAT comments |
| core/data_structures | 41 / 5889 | 46 | 4 stale endif | ~120 | 18 headers no header; 14 zero-comment files; 21 dead lines |
| core/go_api | 3 / 1371 | 0 in scope (C/Go ABI) | 0 | ~25 | 1 orphaned doc block; well documented otherwise |
| core/io | 10 / 1017 | 0 | 0 | ~30 | 3 headers no class header; 4 files legacy @DESCRIPTION |
| core/task | 76 / 25692 | 23 | 0 | ~330 | 16 headers no header; 14 zero-comment files; 118 dead lines |
| core/util | 16 / 1958 | 17 | 0 | ~40 | 3 headers no header; 3 zero-comment files; 2 dead lines |
| apps | 19 / 2935 | 0 | 1 nonstandard | ~60 | 5 zero-comment files; wrapper.h empty shell |
| tools | 19 / 2285 | 6 | 4 | ~45 | 10 headers no header; 3 zero-comment files; 6 dead lines |
| Total | 194 / 41770 | ~111 | ~11 | ~670 | 57 headers; <=155 dead lines |

Naming breakdown (verified locations):
- snake_case/lowercase methods: 51 declarations in 12 files. Worst: matches.cuh:303-318 (7), immutable_csr.cuh:140-154 (6), immutable_csr_gpu.cuh:95-109 (5), woj_exec_plan.cuh:75-87 (6), exec_plan.cuh:205-226 (4+2), edgelist.h:206-216 (4+1), hash_map.cuh:129-158 (5, incl. lowercase find/contains), scheduler.h:32,60,69 (3), matrix.cuh:27,29 (2), hom_match.cuh:45-46 (2), heap.cuh:65,69 (2), bitmap.h:132,134 (2, STL-style size()/data(), gray zone). Call-site blast radius: 533 (427 dot + 106 arrow accesses, grep count).
- snake_case kernels: 11 of 67 kernels. matrix_operations.cu:256,282,298,318,368,494,505 (7), gpu_bitmap.cu:13, hash_map_test.cu:22, attributes_hashmap_test.cu:26, plus ppr_query.cu:57 kernel literally named `add`.
- camelCase/mixed function names: cuBLASMatmult and cuBLASRelu (core/task/gpu_task/matrix_ops.cuh:38,149), InitializeMatrix_kernel, cuRANDFill, float2int8, cuBLASGemmExWrapper, TestcuBLASGemm (core/util/matrix.h:20,106,117,141,169), hash_function (core/common/host_algorithms.cuh:9), swap (core/task/gpu_task/kernel/algorithms/sort.cuh:21, shadows std::swap).
- Type names: BitmapNoOwnerShip (core/util/bitmap_no_ownership.h:11) violates PascalCase; aliased in exec_plan.cuh:30, bit_tiled_matrix.cuh:23, csr_tiled_matrix.cuh:26, ppr_query.cu:51. No lowercase type names found (struct timeval at core/util/matrix.h:174 is POSIX, excluded).
- Constants: 10 object-like macros in core/common/types.h:7-35 (EQUALITIES, SIM, MAX_CANDIDATE_COUNT, MAX_EID_COL_SIZE, CHECK_POINT, CHECK_POINT_CHAR, MAX_HASH_TIMES, MAX_VERTEX_ID, DEFAULT_TILE_NUM, CHECK) should be constexpr kPascalCase. 4 typo constants kDefalut* (core/common/consts.h:15,18,19,20), 17 total occurrences. Function-like bit-offset macros defined in 6 files under 3 names (WORD_OFFSET/BIT_OFFSET at bitmap.h:14-15, bit_tiled_matrix.cu:23-24, matrix_operations.cu:26-27; KERNEL_WORD_OFFSET/BIT_OFFSET at kernel_bitmap.cuh:13-14 and kernel_bitmap_no_ownership.cuh:13-14; GPU_WORD_OFFSET/GPU_BIT_OFFSET at gpu_bitmap.cu:10-11); duplication auditor owns the dedup, style owns the macro-to-constexpr conversion. LOG_*/CUDA_CHECK macros are idiomatic and not counted.
- Members: A_, B_, C_ uppercase members in gemm.cuh:67-69 and ppr_query.cuh:50-53 (6). Trailing-underscore convention otherwise broadly followed; all other candidates verified as struct fields (exempt) or locals.
- Namespaces: sics::graph::tools::partitioner in hash_based_edgecut.h:7 and hash_based_vertexcut.h:7 vs sics::matrixgraph everywhere else; graph_partitioner.cpp:12-18 references sics::graph::tools::common and sics::graph::core::common which are defined nowhere (tools/common/types.h:4-12 defines an empty sics::matrixgraph::tools::common). All other namespaces snake_case and mirror directories.
- Filenames: CHBL_scheduler.h (uppercase), kernel_subsio.cu and kernel_woj_subsio.cu (typo; their headers are spelled kernel_subiso.cuh / kernel_woj_subiso.cuh, confirmed pair mismatch at kernel_subsio.cu:10 and kernel_woj_subsio.cu:30). Stray backup file kernel_woj_subsio.cu.bak in the source tree.
- Modernization (not strict naming): 8 typedefs (types.h:23-30) vs using; 3 project-owned unscoped enums (types.h:37,45, scheduler.h:13) vs enum class. The 7 unscoped enums in compute_features_types.h and matrixgraph_go_api.h are C ABI and excluded.

Guard defects (11): gpu_task_manager.cuh:1 has #ifndef with no #define (guard inert); to_bit_tiled_matrix.cuh:1-2 duplicates the guard of grid_edgelist_2_bit_tiled_matrix.cuh (latent silent-skip collision); 6 stale INC_51_11 endif comments (kernel_subiso.cuh:66, device_tiled_matrix.cuh:4, kernel_bitmap.cuh:210, immutable_csr.cuh:267, kernel_bitmap_no_ownership.cuh:108, tools/common/types.h:16); 2 legacy SICS_GRAPH guards (hash_based_edgecut.h:1-2, hash_based_vertexcut.h:1-2); consts.h:68 endif comment names TYPES_H_ (top guard verified correct via cat -A); edgelist_subgraphs_io.cuh:2 guard says WRITER vs filename io; apps/wrapper.h:1 guard lacks trailing underscore.

Formatting (grep estimates, clang-format unavailable): 334 lines over 80 columns in 88 files (Google limit is 80; some are unsplittable log strings, exempt per guide). ~159 right-aligned pointer/reference declarations (63 `Type *name`, 96 `Type &name`) in ~40 files vs PointerAlignment: Left. 50 unindented access specifiers in 21 files. 122 multiline braceless control bodies (worst: kernel_bitmap.cuh 10, csr_tiled_matrix.cu 7). 8 trailing-whitespace lines, 0 tabs. 867 single-line braceless control statements are allowed-but-discouraged by Google, not counted as violations. 42 files place project includes before system includes, which matches the current .clang-format but not standard Google order (config is google-cloud-cpp derived, see open decisions).

Self-containment (owner rule, verified by include grep): 12 headers use std::cout/cerr without <iostream> (round_robin_scheduler.h, even_split_scheduler.h, exec_plan.cuh, matches.cuh, format_converter.cuh, matrix.h, set_operations.h, to_bit_tiled_matrix.cuh, to_egsm_graph.cuh, to_tiled_matrix.cuh, to_immutable_csr.cuh, to_edgelist.cuh). 3 use INT_MAX/INT_MIN without <climits> (scheduler.h:63, CHBL_scheduler.h, matrix.h). gpu_task_manager.cuh uses std::unordered_map without <unordered_map>. These compile today only via transitive includes.

## (b) Worst 10 files

1. core/task/gpu_task/kernel/kernel_woj_subsio.cu (1985 lines). 35 commented-out code lines, largest block at 409-451 (dead min-wise filter loops). Filename typo subsio vs header subiso. No file header. Live code.
2. core/components/gpu_task_manager.cuh. Broken include guard (line 1, no #define). Stale @param docs at 36-40 name kernel_wrap/host_input/host_output parameters that do not exist in SubmitTask(size_t, TaskType, TaskBase*). WHAT comments "// Constructor" (29), "// Destructor" (32). Missing <unordered_map>.
3. core/components/scheduler/scheduler.h. snake_case methods get_n_device/get_bin_id_by_ball_id/set_bin_id_by_ball_id (32,60,69). INT_MAX without <climits> (63). Unexplained magic numbers 80*2048 (23) and 1024*256 (38). Zero comments. Member ball_id_by_bin_id_ stores ball->bin, name reads backwards.
4. core/components/scheduler/CHBL_scheduler.h. Uppercase filename. Commented-out include (6) and debug print (57). Unexplained magic seed 174321 (65). Unindented public/private. INT_MIN without <climits>. Debug cout in ctor/dtor (21,24).
5. core/common/yaml_config.h. Injects project using-declarations into external namespace YAML (9-16). WHAT comment "// template is needed for this function" (41,74). Commented-out code (87-90). Braceless if (60-61), right-aligned references throughout.
6. core/common/types.h. 10 constant macros (7-35), 8 typedefs (23-30), 2 unscoped enums (37,45) in the foundational 70-line header. No responsibility comment.
7. core/task/gpu_task/gemm.cu + gemm.cuh. Commented-out debug prints and dead code (gemm.cu:87-93, 759-769). Phase narration "Step 1"/"Step 3" (171,233). Uppercase members A_/B_/C_ (gemm.cuh:67-69).
8. core/task/gpu_task/kernel/matrix_operations.cu. 7 snake_case kernels (256,282,298,318,368,494,505). WORD_OFFSET/BIT_OFFSET macros (26-27). 15 commented-out code lines. Phase narration (436).
9. core/task/gpu_task/ppr_query.cu + ppr_query.cuh. Kernel named `add` (ppr_query.cu:57). Uppercase members A_/B_/C_ (ppr_query.cuh:50-53). Phase narration (148,219). Commented-out prints (68-69).
10. core/util/matrix.h. Densest naming cluster: 5 bad function names (20,106,117,141,169), PascalCase locals AType/BType/CType/ComputeType (145), nr_rows_A/nr_cols_A params (88,106). Misleading comment "// Generate arbitrary elements" (30) above code that writes constant 1 (35). Missing <iostream>/<chrono>/<type_traits>/<climits>. Zero includers found (grep); deletion likely beats repair, coordinate with duplication auditor.

Runners-up: kernel_matrix_ops.cu (16 dead-comment lines incl. Chinese AVX block 83-170), hash_map.cuh (5 lowercase methods), core/matrixgraph.cuh (public facade class, zero comments), tools/common/types.h (empty namespace shell, legacy guard, no newline at EOF).

## (c) Comment quality

- Responsibility headers: 57 of 114 headers have no comment in the first 30 lines; 51 have no comment above the first class/struct/template declaration. 44 of 194 files contain zero comment lines. Worst directories: core/data_structures (18 headers), core/task (16), tools/graph_converter (7).
- Good exemplars exist and define the target: task_base.cuh:13-14 states ownership and non-ownership ("Owns the CUDA streams... derived tasks own their own... buffers"); attributes.h:17 documents the cross-module invariant "ValueType mirrors the Go enum used by igeenum"; compute_features_types.h:10-27 documents memory ownership and the discriminated-union clobber hazard; consts.h:55-61 explains the chunking invariant; gar_match_arrays.h:13-27 documents row semantics. These are recent refactors.
- Commented-out code: <=155 pattern-matched lines (upper bound). Verified genuine clusters in ~20 files, concentrated in core/task (118 lines): kernel_woj_subsio.cu 35, kernel_matrix_ops.cu 16, matrix_operations.cu 15, gemm.cu 15. False positives exist: gar_match_arrays.h:41-49 is a documentation example, not dead code. Block-comment dead code: none found beyond doc blocks.
- Phase narration: ~15 lines. subiso_cpu.cu:1319-1443 ("Step 1..4", names real algorithm phases, borderline acceptable), gemm.cu:171,233, ppr_query.cu:148,219, matrix_operations.cu:436, format_converter.cuh:469,521 (mechanical narration, should go).
- WHAT-not-WHY: gpu_task_manager.cuh:29,32 and unified_buffer.cuh:17, device_buffer.cuh:17 ("// Constructor"/"// Default constructor"); gpu_task_manager.cu:25,31,38,41,43 (5 narration comments in a 50-line file); yaml_config.h:41,74; sort.cuh:142-143,157-158.
- Stale/misleading comments: gpu_task_manager.cuh:36-40 (@param for nonexistent parameters), sort.cuh:161 (references quickSort(), function is QuickSort), matrix.h:30-35 (comment claims arbitrary values, code writes 1).
- Comment language: Chinese comments in 3 files (subiso_cpu.cu:1128,1136 WHAT comments; kernel_matrix_ops.cu:83-170 inside dead AVX block plus live WHY comments at 435-459; execution_policy.h:15 inside a macro definition). Language policy is an open decision.
- Tag style mixed: 14 files use legacy @DESCRIPTION (all 4 io matrix headers, util/bitmap*.h, util/atomic.h, util/gpu_bitmap.cuh), 15 files use @file/@brief Doxygen. No project standard documented.
- Debug noise in lieu of logging: scheduler constructors/destructor print to std::cout (CHBL_scheduler.h:21,24, round_robin_scheduler.h:16, even_split_scheduler.h:19) while core/util/logging.h exists.

## (d) Mechanical vs manual split

Mechanical (regex/codemod + compiler verification, or clang-format once installed):
- All formatting: ~670 line-level deviations across ~120 files. Blocked on clang-format installation. Include order depends on the config decision first.
- Guard repairs: ~10 files, scriptable (add missing #define, deduplicate to_bit_tiled_matrix.cuh guard, normalize 6 stale endif comments, 2 legacy guards, wrapper.h).
- kDefalut->kDefault typo rename: 4 constants, 17 occurrences, compiler-verified.
- snake->Pascal method renames: 51 declarations, 533 call sites, ~25 files touched. Regex per identifier + compiler. Requires the accessor-naming policy decision first (one-time).
- Kernel renames: 12, mostly file-local static kernels, low risk.
- A_/B_/C_ -> snake_case members: 6 members, 2 headers + gemm.cu/ppr_query.cu call sites.
- BitmapNoOwnerShip->BitmapNoOwnership: 1 type + 4 alias sites + call sites, compiler-verified.
- Filename renames: CHBL_scheduler.h (3 includers), kernel_subsio.cu/kernel_woj_subsio.cu (match their .cuh spelling), update include paths + CMake if listed. Delete kernel_woj_subsio.cu.bak.
- Namespace sics::graph->sics::matrixgraph: 3 files, but the headers reference nonexistent namespaces and may be dead; coordinate with duplication auditor before renaming.
- types.h macros->constexpr: 10, mechanical (CHECK_POINT needs <climits> added).
- Dead-comment removal: <=155 lines, ~20 files; mechanical deletion but review each block against the duplication auditor's dead-code list.

Manual judgment:
- Responsibility headers for 57 files: must state what the module owns and does NOT own; requires reading each module. Largest single work item.
- WHAT->WHY rewrites and stale-comment fixes: ~15 files (gpu_task_manager.*, yaml_config.h, sort.cuh, matrix.h, scheduler magic numbers need rationale).
- Semantic renames: Scheduler2Enum (apps, digit-for-To), localid2globalid vs localid_to_globalid inconsistency (edgelist.h:150-160), ball_id_by_bin_id_ direction (scheduler.h:76), subsio typo (mechanical rename but confirm no external references).
- Policy decisions (one-time, then mechanical): accessor convention, struct-field exemption, brace policy for 867 single-line statements (clang-tidy readability-braces-around-statements can automate if adopted), comment language, extensions, .clang-format rewrite.

Rough split: of ~111 naming violations, ~90 (81%) are mechanically fixable after one-time policy decisions. Formatting is 100% mechanical but tool-blocked. Comment work is ~70% manual by effort (57 headers dominate). File counts: ~120 files need mechanical touches, ~65 need manual judgment, overlap ~50. Estimated total distinct files touched by the style seam: ~140 of 194.

## (e) Open decisions

1. File extensions. Owner rule mentions .cc/.h; repo has 0 .cc, 23 .cpp, 59 .cu, 80 .cuh, 34 .h. Renaming .cpp->.cc touches CMake and every include; .cu/.cuh are CUDA conventions (nvcc needs .cu for device compilation). Also core/task/cpu_task/hom_match.cu and subiso_cpu.cu contain no __global__/__device__ code (grep verified) and could be .cpp. Recommend deciding .cpp vs .cc once; keep .cu/.cuh. Do not assume.
2. .clang-format is google-cloud-cpp derived. IncludeCategories reference google/cloud/ and grpc/ headers, RawStringFormats configures TextProto, CommentPragmas is @copydoc. Its include priorities sort project headers before system headers, opposite of standard Google. Decide: rewrite to standard Google categories (then 42 files re-sort) or keep. Pin a clang-format version when installing (Google style drifts across versions; 14+ is current). Binary absent today, so all formatting counts here are grep estimates.
3. Accessor naming conflict. Owner rule says functions PascalCase; Google style itself permits snake_case accessors matching member names (num_vertices()). The 51 snake methods and bitmap.h size()/data() sit in this conflict. Decide GetNumVertices() vs NumVertices() vs keep-snake-accessors once, then rename mechanically.
4. Struct-field underscore exemption. This audit assumed Google's exemption for plain struct data members (HashMap, MinWiseFilterCache, ArangoExportOptions, VidCountPair, DeviceColumnBuffer). Owner rule says "class members"; confirm structs are exempt, otherwise ~60 more fields need underscores.
5. Brace policy. 867 single-line braceless statements are legal Google but discouraged; 122 multiline braceless bodies are violations. Decide whether to run clang-tidy braces-around-statements repo-wide (large diff) or fix only the 122.
6. Comment language. Chinese comments in 3 files; decide translate-to-English vs bilingual-allowed.
7. Comment tag standard. @DESCRIPTION (14 files) vs @file/@brief (15 files) vs plain //; pick one.
8. C/Go ABI surface is frozen. matrixgraph_go_api.h and compute_features_types.h snake_case fields/functions are ABI-bound (Go passes prim ids by value); excluded from all rename counts. Any change there needs owner approval per the refactor initiative's one-way-door rule.
9. Stray files. kernel_woj_subsio.cu.bak should be deleted. tools/python/ (16 .py/.sh files) is outside C++ style scope; confirm it stays out of the refactor.