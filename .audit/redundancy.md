# MatrixGraph 冗余与死代码审计（redundancy seam）

只读审计，HEAD 4cec1d5。核对的构建文件：根 CMakeLists.txt、core/CMakeLists.txt、core/tests/CMakeLists.txt、apps/CMakeLists.txt、tools/CMakeLists.txt、tools/graph_converter/CMakeLists.txt、tools/graph_partitioner/CMakeLists.txt。third_party/ 不在范围内。LOC 数字来源：wc -l 与逐函数 diff（diff 行数已注明），均为估计值而非精确可删行数。core 库源码由 core/CMakeLists.txt:36-59 的目录白名单 GLOB 收集（*.cu/*.cpp，排除 go_api 与 *_test），apps 由 apps/CMakeLists.txt:24-28 全量 GLOB，因此「未引用」判定以符号级 grep 为准而非仅看 CMake。

## (a) 已知项现状核实

1. BlockSumAll 统一：属实。block_agg_prims.cuh:63 唯一定义，execute_agg_prim.cu:241-304 与 graph_filter_aggregate.cu:531-604 均调用它。
2. stream 管理上提 TaskBase：部分属实。task_base.cuh:16-30 提供 EnsureStreams/DestroyStreams，execute_agg_prim.cu:629,747,829,943 与 graph_filter_aggregate.cu:753-755 在用；但 bfs.cu:103、pagerank.cu:122、wcc.cu:148、subiso.cu:180、matrix_ops.cu:82-125 仍自建局部 stream（这些类都继承 TaskBase）。gemm.cu:166、ppr_query.cu:143、kernel_woj_subsio.cu:1206 的多 stream 矩阵是另一套，未走 TaskBase。
3. CollectEllHop 模板化复用：属实。min_wise_filter.h:117,166 按 Bitmap 模板化，subiso_cpu.cu:273,295 调用。但 subiso_cpu.cu 仍保留过滤器栈的其余私有副本（见簇 C4），上次的去重只覆盖了 CollectEllHop 本身。
4. kernel::AggPrim/AllFeatures 统一到 task::：属实。AggPrim 唯一定义在 execute_agg_prim.cuh:102（featurelib 序），kernel_graph_aggregate.cuh:31-37 用 using 声明重导出，无第二份枚举。
5. 第三份聚合原语：属实。kernel_compute_features_primitives.cuh（517 行）仍有自己的 BlockSum:120、BlockMinVal:140、BlockMaxVal:160、16 个 Agg*:222-470、ApplyAggPrim:472，值类型 MGFeatureValue = MatrixGraphFeatureValue（compute_features_types.h:200,224）。合并必须先统一双值类型，跨 C/Go ABI（docs/go_api/README.md）。新证据：两侧枚举编号不同（ValueType attributes.h:18-29 为 kFloat64=1/kInt=3/kBool=9；MatrixGraphValueType compute_features_types.h:80-90 为 INT=1/FLOAT64=2/BOOL=3），go_api 的转换 switch（matrixgraph_go_api.cu:406-486）是必需的而非冗余。另发现 compute_features_types.h:97-99 注释声称 MG 序「与 GraphAggregate 的 AggPrim 序一致」已过时失真（MG_AGG_SUM=3 而 AggPrim::kSum=1）；primitives.cuh:30-46 的 kAgg* 常量又是 MatrixGraphAggPrim（compute_features_types.h:96-113）的第三份手写副本，这两处可在 MG 侧内部机械清理。

## (b) 重复簇

难度：mechanical=同类型无 ABI 影响；needs-design=需改接口/抽象；ABI-sensitive=跨 C/Go 边界。LOC 为估计的重复量（可删上限）。

| # | 位置 | 重复LOC | 难度 | 行为风险 |
|---|------|--------|------|---------|
| C1 | graphrag 三 app 复制 task 的 Execute 缓冲装配：graphrag_bfs.cpp:80-124 ≡ bfs.cu:58-110；graphrag_pagerank.cpp:82-135 ≡ pagerank.cu:64-128；graphrag_wcc.cpp:78-120 ≡ wcc.cu:98-154 | ~140 | needs-design | 中。task 类需暴露内存结果；pagerank 有刻意差异：pagerank.cu:105 按 max_vid 分配（off-by-one），graphrag_pagerank.cpp:104-106 按 num_vertices 修正，合并须保留修正 |
| C2 | ImmutableCSR 缓冲尺寸 8 项公式 ×12：bfs.cu:78-85、pagerank.cu:88-95、wcc.cu:122-129、subiso.cu:126-133,152-159、kernel_woj_subsio.cu:1242-1249,1264-1271、graph_aggregate.cu:552-559、compute_features.cu:93-100、graphrag_bfs.cpp:81-87、graphrag_wcc.cpp:81-88、graphrag_pagerank.cpp:85-92 | ~90 | mechanical | 低。纯算术，bfs 写法代数等价；加 ImmutableCSR::GetGraphBufferSize() 一处收敛 |
| C3 | Scheduler2Enum ×12 app：bfs.cpp:28、wcc.cpp:27、pagerank.cpp:36、gemm.cpp:26、skew.cpp:30、diameter.cpp:31、subiso.cu:45、subiso_cpu.cpp:59、gar_match.cu:22、graphrag_bfs.cpp:37、graphrag_pagerank.cpp:38、graphrag_wcc.cpp:36 | ~100 | mechanical | 无。且 MatrixGraph 构造只处理 kCHBL（matrixgraph.cu:27-32），EvenSplit/RoundRobin 选项本就是空转 |
| C4 | subiso_cpu.cu 私有过滤器栈 ≡ min_wise_filter.h：VertexFilterCache:164-178 ≡ MinWiseFilterCache:86-100；BuildFilterCache:251-322 ≡ BuildMinWiseFilterCache:216-295；KMinBloomFilter:340-369 ≡ KMinBloomIPFilter:297-327；KMinWiseIPFilter:489-580 ≡ min_wise_filter.h:329-431 | ~300 | mechanical | 低-中。差异仅全局 cache vs 参数、MiniKernelBitmap vs MinWiseBitmap、thread_local visited；须核对 bitmap 语义（subiso_cpu.cu:314 直接写 data_） |
| C5 | LFTJ CPU/GPU 宿主预处理：lftj_subiso.cpp vs lftj_subiso_gpu.cu 的 BuildUndirectedAdjacency（119-158 vs 32-68，diff 5 行）、BuildMinWiseFilterCaches（160-201 vs 70-110，diff 3 行）、ComputeMatchingOrder（343-421 vs 250-317，diff 17 行）、Run 内 LCF 预检（33-60 vs 419-443） | ~180 | needs-design | 低。纯宿主代码，提取共享基类/自由函数；BuildCandidateSets 已分叉（diff 186 行）不并 |
| C6 | GEMM::InitC ≡ PPRQuery::InitResultMatrix：gemm.cu:96-450 vs ppr_query.cu:73-458，355 行中 diff 仅 51 行，差异主要是 GetTiledMatrixPtrByIdx vs GetBitTileMatrixPtrByIdx 与计时打印 | ~330 | needs-design | 中。多 stream 设备代码，需按矩阵类型模板化；PPRQuery 本身不可达（D14），若删则簇消失 |
| C7 | GPU KMinWiseIPFilter 两份：kernel_woj_subsio.cu:316-470（CSR、仅 out-edge，in-edge 段被注释 409-460）vs kernel_gar_match.cu:120-205（GAR 边表、O(E) 扫描） | ~100 | needs-design | 中。同一 k-min 支配算法不同数据表示；WOJ 版刻意不做 in-edge 过滤，属行为差异 |
| C8 | 哈希/归约助手：HashTable（algorithms/hash.cuh:31-62）≡ MinWiseHashTable（min_wise_filter.h:64-82）同一 16 置换；HashMix（execute_agg_prim.cu:343-348）≡ HashMix64（graph_filter_aggregate.cu:401-406）逐字节相同 splitmix64；kHashEmpty+开放寻址探测环 execute_agg_prim.cu:385-420 vs graph_filter_aggregate.cu:460-530；BlockArgReduce（execute_agg_prim.cu:197-225）≡ graph_filter_aggregate.cu:538-575 内联 argmin/argmax（同 tie-break） | ~100 | mechanical | 低。注意 common::hash_function（host_algorithms.cuh:9-16）与 kernel::Hash（hash.cuh:21-28）不是副本：常量不同（0x85ebca6b vs 0x853bca6b）且用途不同 |
| C9 | 第三份聚合原语（已知项 5）：kernel_compute_features_primitives.cuh 517 行 vs block_agg_prims.cuh 597 行，16 原语+排序+归约+分发器全量重复，值类型不同 | ~450 | ABI-sensitive | 高。MG 序 vs featurelib 序不同，Go 侧按数值传 prim（matrixgraph_go_api.h:96-113）；统一值类型是单向门，须先方案对比再经用户批准 |
| C10 | CollectEllHopOutLabels ≡ CollectEllHopInLabels：min_wise_filter.h:117-164 vs 166-213，仅 Out/In 访问器不同 | ~50 | mechanical | 无。方向参数化即可 |
| C11 | app 脚手架：18 个 app 各自 ValidateParameters/PrintConfig/try-catch main；graphrag 三 app 另有 device 选择/prefetch/stream/memcpy/summary 同构段（graphrag_bfs.cpp:100-124 等） | ~200 | needs-design | 低。共享 app harness，收益分散，优先级低 |
| C12 | 两套 bitmap 家族并存：util/bitmap.h(165)+bitmap_no_ownership.h(34)+bitmap_ownership.h(83)+gpu_bitmap.cuh(37) vs data_structures/kernel_bitmap.cuh(209)+kernel_bitmap_no_ownership.cuh(107)+mini_kernel_bitmap.cuh(130)+MinWiseBitmap(min_wise_filter.h:29-59) | ~300 | needs-design | 中。全部在用（KernelBitmap 用于 matrix_operations/woj/subsio；MiniKernelBitmap 用于 subiso_cpu/gar_match/woj），统一需设计，非快赢 |
| C13 | 属性列装载：GraphFilterAggregate::BuildAttributesFromColumns（graph_filter_aggregate.cu:852-955）vs GraphAggregate::LoadAttributes（graph_aggregate.cu:625-726），同用途已分叉（diff 176 行）；列类型 GraphAggregateAttributeColumn（graph_aggregate.cuh:22-28，多 valid 指针）vs ComputeFeaturesAttributeColumn（compute_features_types.h:215-220，C ABI） | ~100 | needs-design | 中。若 D3 删除则前半消失；列类型合并跨 ABI |
| C14 | k-hop BFS 形状重复：graphrag_expand.cpp:75-120 Expand vs Edges::BuildKHopOutSubgraphs（edgelist_khop.cpp:24-127），同队列+深度+visited 形状，契约不同（多种子+度上限+边索引 vs 逐中心子图） | ~40 | needs-design | 低。优先级低 |

## (c) 死代码

零风险（无任何引用，可直接删）：
- D1 kernel_woj_subsio.cu.bak（1459 行）：.bak 后缀不进任何 GLOB（core/CMakeLists.txt:53-54 只收 *.cu/*.cpp），全库零引用；内容是 kernel_woj_subsio.cu 的旧版（diff 确认）。git 历史已保存。
- D2 tools/graph_partitioner/graph_partitioner.cpp（101 行）：tools/graph_partitioner/CMakeLists.txt:14-17 只 GLOB *.cu，本文件不被编译；且 include 三个不存在的头（core/common/multithreading/thread_pool.h、tools/common/yaml_config.h、partitioner/csr_based_planar_vertexcut.h，ls 确认缺失），命名空间还是旧项目 sics::graph。不可编译的遗留物。
- D3 util/matrix.h（199 行）：全库零 includer（grep util/matrix.h 与 "matrix.h" 均无）。cuBLAS/cuRAND 矩阵初始化助手。
- D4 util/logging.h（87 行）：唯一 includer 是 D2 的死文件，传递性死亡。
- D5 tools/common/types.h（16 行）：空命名空间壳；被 7 个 tool 文件 include（edgelist_subgraphs_io.cuh、graph_partitioner.cu、grid_cut.cu/.cuh、graph_converter.cpp、to_grid_csr_tiled_matrix.cuh、to_edgelist.cuh），删文件需同步删 include。
- D6 apps/wrapper.h（7 行）：零 includer，内容仅两个系统头。
- D7 MatrixAnalyzer（matrix_analyzer.cuh，49 行）：全库仅自引用；传递性杀死 util/set_operations.h（106 行，唯一 includer 是它）。kMatrixAnalysis 枚举值（types.h:47）无分派无调用。
- D8 GPUTaskManager（gpu_task_manager.cu 50 + .cuh 66 = 116 行）：仅被 matrixgraph.cuh:4 include，从未实例化。MatrixGraph 的 streams_/p_streams_mtx_/p_hr_start_mtx_/p_hr_start_lck_/p_hr_start_cv_/p_hr_terminable_ 成员（matrixgraph.cuh:48-58）在 matrixgraph.cu 中零使用（grep 确认）。scheduler_ 在 matrixgraph.cu:29 构造后从未调用，EvenSplit/RoundRobin 从未构造（matrixgraph.cu:31 default 空分支）；scheduler/*.h 共 228 行属死重。
- D9 空 stub 与死局部：GEMM::FillTiles/Count 空函数体（gemm.cu:917-919）；WOJExecutionPlan::GenerateJoinPlan 空体（woj_exec_plan.cuh:53）；BFS::ExecuteBFS 的 parallelism/worker/mtx/step（bfs.cu:59-64）与 SubIso::Matching 同款（subiso.cu:109-114）声明后未用，是 warning 源。
- D10 死枚举值：kSSSP、kGEMV、kExecuteAggPrim（types.h:54,59,62）无分派 case 无调用；kMatrixAnalysis 见 D7。
- D11 注释代码块（无 #if 0，全库 grep 为零）：kernel_woj_subsio.cu:409-460 in-edge 过滤段 ~55 行；kernel_matrix_ops.cu:78-100,123-136,158-170 等 AVX 段 ~70 行；subiso_cpu.cu:1145-1148,1625-1630,1825-1827,1902-1905,1944 ~30 行；gemm.cu:84-93 注释打印。

需业主确认（不可达但可能是有意保留）：
- D12 旧版 GPU weft SubIso 路径 ~815 行：SubIso::Matching（subiso.cu:107-195）唯一调用点是注释行 subiso.cu:266，Run 走 WOJMatching（subiso.cu:258-265）；SubIsoKernelWrapper::SubIso 仅被 subiso.cu:182 调用；kernel_subsio.cu（658 行，内核全 static）+ kernel_subiso.cuh（65 行）随之死亡。注意 Matches/ExecutionPlan 不能删：subiso_cpu.cu:1202,1311,1759 仍在用。
- D13 GraphFilterAggregate（graph_filter_aggregate.cu 1134 + .cuh 200 = 1334 行）：零外部调用；不在 TaskType 枚举（types.h:45-64）、无分派（matrixgraph.cu:40-121）、go_api 不 include（matrixgraph_go_api.cu:1-11）、无 app。有完整文档 docs/graph_filter_aggregate.md，consts.h:51,62-63 的 kFilterAggBlockDim/kFilterMaxChunk* 专为它存在。接线或删，业主定。
- D14 PPRQuery（ppr_query.cu 993 + .cuh 66 = 1059 行）：matrixgraph.cu:59-63 有分派，但全库无任何调用方传 kPPRQuery（grep 仅定义+分派两处），无 app 无 go_api。
- D15 EnumerateOnHost（lftj_subiso_gpu.cu:319-408 ~90 行）：头文件注明 for validation（lftj_subiso_gpu.cuh:65），零调用。
- D16 go_api 测试 execute_agg_prim_go_api_test.cpp（137 行）：被 core/CMakeLists.txt:59-61 的 go_api 与 *_test 过滤排除，core/tests/CMakeLists.txt:12-16 的 GLOB 只覆盖 data_structures 与 gpu_task，任何目标都不构建它。
- D17 CMake TEST=ON 目标冲突：core/CMakeLists.txt:94-95 先 add_subdirectory(tests)，:170,:189 再建 graph_aggregate_test/execute_agg_prim_test；core/tests/CMakeLists.txt:12-16 的 GLOB 也含这两个文件，TEST=ON 时同名目标重复定义必然 configure 失败。（gtest 套件本身因 third_party/googletest 缺失从未构建，见决策日志 2026-11-17。）

WIP 勿删（近期 GAR 研究功能，零外部调用但属在建工程）：
- D18 GARDeduce 簇（gar_deduce.cu 344 + .cuh 117 + kernel_gar_deduce.cu 355 + .cuh 125 + gar_deduce_arrays.h 83 = 1024 行，commit d65415b）、HomMatch（hom_match.cu 268 + .cuh 87 = 355 行，commit 267b86a）、AttributesIO（attributes_io.cu 233 + .cuh 73 = 306 行，d65415b 引入但 gar_deduce.cu 的 include 列表未含它）。合计 ~1685 行。与用户 Graph Association Rules 研究方向一致，删除前必须问业主。

## (d) Top 5 快赢（按 可删LOC/风险 排序）

1. 删零引用死文件：D1 .bak(1459) + D2 遗留 partitioner.cpp(101) + D3 util/matrix.h(199) + D4 logging.h(87) + D5 空 types.h(16) + D6 wrapper.h(7) ≈ 1869 行。风险≈0，全部无引用，git 历史兜底。
2. 删死类/死成员/死枚举/stub/死局部：D7(155) + D8(116+成员) + D9 + D10 ≈ 350 行。风险≈0，顺带消一批 unused warning。
3. C2 缓冲尺寸公式收敛为 ImmutableCSR::GetGraphBufferSize()：12 处 → 1 处，~90 行。机械，纯算术，低风险。
4. C3 Scheduler2Enum 收敛到共享头：12 处 → 1 处，~100 行。机械，零行为影响（调度选项本就空转）。
5. C4 subiso_cpu.cu 过滤器栈改用 min_wise_filter.h：~300 行。机械偏设计（全局 cache 改参数），需 bitmap 语义核对 + subiso_cpu 实测；上次 CollectEllHop 合并已铺好路。

次级候选：D12 旧 weft 路径 ~815 行（需业主点头）；C5 LFTJ 宿主预处理 ~180 行；C8 哈希/归约助手 ~100 行。ABI-sensitive 仅 C9 一簇（~450 行），是单向门，不在快赢之列。

合计：零风险可删 ~2200 行；业主确认后可再删 ~3400 行（D12-D16）；机械去重 ~820 行；WIP ~1685 行不动。