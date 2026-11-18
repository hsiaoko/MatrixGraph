# MatrixGraph 清理重构计划（Phase B 设计，待业主 checkpoint）

日期 2026-11-17。HEAD 4cec1d5。依据：三份已核实审计（.audit/{redundancy,style,interfaces}.md）+ 双构建基线（.audit/warnings-baseline*.txt、tests-baseline*.txt）。决策日志 .audit/matrixgraph-refactor.tsv。

## 完成谓词（Phase A 定稿）

1. 行为不变：基线通过的测试仍通过；两个红测试诊断修复后转绿；GPU 实测（3× RTX 3090）。
2. 零 warning：core/apps/tools 在 Debug+Release `-Wall -Wextra`（CUDA `-Xcompiler=-Wall,-Wextra`）下 0 条项目 warning（基线 173 处去重位置）；第三方 14 条豁免列理由。
3. 无冗余：审计重复簇合并或给出保留理由；死代码按分层删除；LOC 净减以审计计数为准（零风险 ~2200 + 机械去重 ~820 + 业主批准 ~3400）。
4. 风格达标：项目自有标识符符合 Google/项目规则；clang-format（ pinned 版本）全量通过；57 个缺责任头文件的头文件补齐。
5. Go C ABI 隔离到 go-api 分支（分支上修 H1+H5/L6 后冻结，push 须业主批准）；main 为纯 C++ 库，公共面按 architect 合成设计；ComputeFeatures 保留为内部 C++ 任务（wire 类型头改分类为内部类型，MG 编号变内部约定）。

## 全局护栏

- WIP 勿动：GARDeduce/HomMatch/AttributesIO ~1685 行（D18），全程不删不改名。
- ABI 冻结：matrixgraph_go_api.h、compute_features_types.h 的布局/枚举值/字段序在 Phase 6 批准前不动。
- 每单元 = 一个 commit + Debug/Release 双构建 + ctest 对基线 + warning 闸门（只降不升）；红则回退。
- 机械单元可移交便宜模型执行；设计单元保留完整推理。
- 外部 API 标识符（folly/CUDA/cuRAND/gflags/TBB/yaml-cpp/gtest）不改名。

## Phase 0 — 验证 harness（一切的前提，rigor 高）

- U0.1 gtest 套件修复：SSH 拉取 googletest（pin release tag）；修 TEST=ON 目标冲突（core/CMakeLists.txt:94 vs :170/:189，重命名独立 test 目标或从 tests GLOB 排除）；处理 TEST_DATA_DIR 指向不存在的 testfile/；TEST=ON 构建 + ctest 全量跑，记录基线 .audit/ctest-baseline.txt。验证：configure 成功、ctest 出结果。
- U0.2 sum=0 运行时探针（scratch worktree，main 不动）：给 CollectNeighborValuesBlock/ReadAttributeValue 加 printf，测 deg、attr find、n_collected、首值；核对 GetFloat64 在 offsets=nullptr 的语义。静态已排除 5 假设（见日志），不重复。验证：探针输出定位失败阶段，根因带证据。
- U0.3 sum=0 修复：按 U0.2 根因最小修复。验证：graph_aggregate_test 双构建转绿。
- U0.4 Debug launch 健壮性：ComputeAggPrimKernel 加 __launch_bounds__ 或 block-dim 策略（consts.h:42 kBlockDim=1024，:44 有注释掉的 64）。验证：execute_agg_prim_test Debug 转绿，Release 不回退。
- U0.5 warning 闸门杠杆：scripts/check-warnings.sh，双构建提取项目 warning 对基线 diff，升则报错。验证：脚本对当前 HEAD 输出 173 处基线吻合。
- U0.6 接线 go_api 测试：execute_agg_prim_go_api_test.cpp（D16，137 行，从未被任何目标构建）纳入 core/tests，给 Phase 6 ABI 工作提供回归覆盖。验证：构建并运行。

## Phase 1 — 零风险减法（~2200 LOC + Go 隔离，rigor 中）

- U1.0 Go ABI 隔离（业主已批准）：main 删 core/go_api/（3 文件 1371 行）、docs/go_api/（2 文件）、matrixgraph_goapi CMake 目标（core/CMakeLists.txt:150-165）及 :59 过滤行；compute_features_types.h 保留但头文档从「C/Go wire」改分类为 ComputeFeatures 内部 plan 类型（MG 编号内部化，去假「intentionally match」注释）；ComputeFeatures task/app 保留。验证：双构建 + ctest + warning 闸门（go_api 相关 warning 随之消失）。分支侧 H1/H5/L6 修复见「分支 track」。
- U1.1 死文件 D1-D6（~1869 行）：kernel_woj_subsio.cu.bak(1459)、遗留 graph_partitioner.cpp(101，include 三个不存在头)、util/matrix.h(199)、util/logging.h(87，唯一 includer 是死文件)、tools/common/types.h(16，空壳，同步删 7 处 include)、apps/wrapper.h(7)。新增已核实：hash_based_edgecut.h/hash_based_vertexcut.h 唯一 includer 也是死文件 graph_partitioner.cpp，传递性死亡，并入本单元（partitioner_base.h 若仅被它们引用则一并核删）。验证：双构建 + ctest + warning 闸门。
- U1.2 死类/死成员/死枚举/stub D7-D10（~350 行）：MatrixAnalyzer+set_operations.h、GPUTaskManager+facade 死成员（streams_/p_hr_*/scheduler_ 及 CHBL 构造）、死 TaskType 值（kMatrixAnalysis/kSSSP/kGEMV/kExecuteAggPrim）、空 stub（GEMM::FillTiles/Count、WOJExecutionPlan::GenerateJoinPlan）、死局部变量。注意：scheduler.h 有 18 个 includer，调度器集群本体留到 U2.2/U5.1 处理，本单元只删 facade 侧。验证：同上，unused warning 应下降。

## Phase 2 — 机械去重（~820 LOC，rigor 中）

- U2.1 C2 缓冲尺寸公式 ×12 → ImmutableCSR::GetGraphBufferSize()（~90 行，纯算术）。
- U2.2 C3 Scheduler2Enum ×12 → 共享头（~100 行）；顺带评估 EvenSplit/RoundRobin 死调度器删除（facade 只构造 CHBL，选项本就空转）。
- U2.3 C8 哈希/归约助手（~100 行）：HashMix≡HashMix64、kHashEmpty 探测环、BlockArgReduce 复用、HashTable≡MinWiseHashTable。陷阱：common::hash_function 与 kernel::Hash 常量不同（0x85ebca6b vs 0x853bca6b），不是副本，不并。
- U2.4 C10 CollectEllHop Out/In 方向参数化（~50 行）。
- U2.5 C4 subiso_cpu 私有过滤器栈 → min_wise_filter.h（~300 行）：核对 bitmap 语义（subiso_cpu.cu:314 直写 data_）、全局 cache 改参数。验证：subiso_cpu_exec 在 toy 图上前后输出一致 + 双构建。
- U2.6 stream 所有权补完（M2）：TaskBase 析构调 DestroyStreams；bfs/pagerank/wcc/subiso/matrix_ops 局部 stream 上提；GraphAggregate 每 GPU stream 是另一形态，保留并注释。验证：GPU 测试 + app 冒烟。

## Phase 3 — 零 warning 清扫（173 处，rigor 中-高，可按目录 fan-out）

- U3.1 正确性相关优先：-Wreturn-type(2)、-Wclass-memaccess(4)、-Wnarrowing(51)。逐处语义审查，禁止盲目加 cast。
- U3.2 -Wreorder(183 行)：成员初始化顺序，机械。
- U3.3 -Wsign-compare(84)：逐处定 cast 还是改类型。
- U3.4 残余 -Wunused-*、-Wmissing-field-initializers(53)、-Wtype-limits/-Wparentheses/-Wunused-value。
- fan-out 仅按不相交目录分缝，每 worker 独立 worktree；warning 闸门验证单调降至 0。第三方 14 条写豁免清单。

## Phase 4 — 风格/命名/注释（rigor 中，政策先行）

- U4.0 政策定案（checkpoint 问题）+ 安装 pinned clang-format + 重写 .clang-format（现配置是 google-cloud-cpp 衍生，include 优先级与标准 Google 相反）。
- U4.1 include guard 修复（11 处，含 gpu_task_manager.cuh 失效 guard、to_bit_tiled_matrix.cuh 与 grid_edgelist_2_bit_tiled_matrix.cuh 重复 guard 碰撞——已核实，属正确性相关）+ 头文件自包含（12 缺 <iostream>、3 缺 <climits>、1 缺 <unordered_map>）。
- U4.2 机械改名（codemod+编译器验证）：kDefalut→kDefault(17 处)、51 个 snake_case 方法(533 调用点)、12 个 kernel（含名为 add 的）、A_/B_/C_ 成员(6)、BitmapNoOwnerShip→BitmapNoOwnership、types.h 10 宏→constexpr kPascalCase、8 typedef→using、3 项目枚举→enum class。
- U4.3 文件名/命名空间：CHBL_scheduler.h→chbl_scheduler.h、kernel_subsio→kernel_subiso、kernel_woj_subsio→kernel_woj_subiso（同步 include/CMake）；sics::graph 命名空间文件已核实传递性死亡，走 U1.1 删除而非改名。
- U4.4 clang-format 全量（单 commit，双构建+ctest 闸门）。
- U4.5 死注释删除（≤155 行，对冗余清单交叉核对；gar_match_arrays.h:41-49 是文档示例不是死代码）+ WHAT→WHY + 过时注释修复（~15 文件）。
- U4.6 责任头文件补齐（57 个头文件，手工，按目录分单元；范本 task_base.cuh/attributes.h/compute_features_types.h）。最大手工项，可与 Phase 5 并行（不相交文件）。

## Phase 5 — 接口/设计（needs-design，rigor 高）

- U5.1 H4 facade 虚 Run：TaskBase/CPUTaskBase 加 virtual Run，删枚举+reinterpret_cast 分发+静默 no-op default（M5 空壳基类一并处理）。
- U5.2 M6/C1 graphrag 三 app 回归 task 层（~140 行）。陷阱：保留 graphrag_pagerank 的 off-by-one 修正（task 版按 max_vid 分配是错的）。
- U5.3 M9 错误契约统一：内部异常/返回码，C 边界翻译；H2 exit() 移除属 ABI 门，留 Phase 6。
- U5.4 C5 LFTJ CPU/GPU 宿主预处理共享（~180 行；BuildCandidateSets 已分叉 186 行，不并）。
- U5.5 C12 两套 bitmap 家族统一（设计，争议则 architect/arena）。
- U5.6 L2 并行后端接缝：算法层与 TBB/GPU 解耦（业主原则），命名空间内单一 ParForEach 接缝。
- U5.7 M10 yaml_config 迁 io 边界 + decode 守卫；C11 app 脚手架（低优先，可选）。
- C6(GEMM/PPR InitC ~330)/C13(属性装载 ~100) 依赖 D13/D14 业主决定后再定。

## Phase 6 — 业主 gated（大删除 + 内部统一，rigor 高；ABI 门已随 Go 隔离消失）

- U6.1 大删除决定 D12-D16（~3400 行）：旧 weft SubIso(815，唯一调用点是注释行；Matches/ExecutionPlan 不能删，subiso_cpu 在用)、GraphFilterAggregate(1334，接线或删)、PPRQuery(1059，有分派无调用)、EnumerateOnHost(90，文档注明 validation 用，建议保留)。
- U6.2 C9 值类型统一（内部重构，无 ABI 门）：task::FeatureValue 与 MatrixGraphFeatureValue（现 ComputeFeatures 内部类型）统一 + 第三份原语副本合并；MG↔featurelib 双编号内部桥接或统一（main 无外部序列化 plan 消费方）；architect/arena 合成设计覆盖此项。H2 exit()→错误传播、H3 编号文档/显式转换并入 U5.3 内部错误契约单元。
- U6.3 可选未来 C API v2（若 Go 库要迁回 main）：在干净 C++ 公共面之上薄封装，单独 gated 设计，本轮不展开。

## 分支 track — go-api（独立 worktree /tmp/mg-goapi，业主已批准）

- B1 H1 修复：CopyFeatureValueToC 与 ComputeFeatures 转换循环改 type-switch（复用 ToMgValueType 映射 ValueType→MG_VALUE_*，注意两套 ValueType 编号不同，static_cast type 也是错的），union 只写对应成员。验证：构建 matrixgraph_goapi + 新增 GraphAggregate C API 数值探针测试（float/int/bool 结果逐类型断言）。
- B2 H5/L6 文档修复：README GraphAggregate AllFeatures Go 镜像补 DFeat（17 字段）；union 成员 b 文档 uint8→int32。
- B3 冻结：分支 commit，push 须业主明确批准；外部 Go 库 pin 该分支。

## 规模与成本

~35 个单元。Phase 0-2 价值/风险比最高；Phase 3-4 机械量大（便宜模型候选）；Phase 5-6 设计+ gated。增量构建使每单元验证成本低。全量双构建仅在单元触碰公共头文件时需要。

## 待业主决定（checkpoint）

1. 计划批准与起步范围。
2. D12-D16 删除/保留/接线。
3. ABI bug 修复组（H1/H5/L6）是否批准。
4. 风格政策包默认值：保留 .cpp（0 个 .cc，改名纯 churn）；注释英文；访问器 GetXxx PascalCase（业主规则优先于 Google snake 访问器）；struct 数据字段豁免尾下划线；只修 122 处多行无括号（不做 867 处全库括号 churn）；Doxygen @file/@brief；clang-format 18 pinned + 标准 Google categories；tools/python/ 不在范围。
