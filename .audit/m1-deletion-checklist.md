# M1 删除清单（执行手册，includer 已实测 2026-11-17）

每项执行后：Debug+Release 双构建 + ctest 6/6 + `scripts/check-warnings.sh`。按单元分 commit。

## U1.0 Go ABI 隔离（业主已批准）

- 删 `core/go_api/`（3 文件；探针 cpp 只在 go-api 分支）。
- 删 `docs/go_api/`（2 文件）。
- `core/CMakeLists.txt`：删 :62 的 `list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/go_api/.*")`（go_api 本就不在 CORE_SOURCE_DIRS 白名单里，此行随之无意义）、:150-165 matrixgraph_goapi 目标块。
- 已核实：`matrixgraph_go_api.h` 在 core/apps/tools 零 includer，删除无涟漪。

## U1.1 死文件（零 includer 已核实）

- `core/task/gpu_task/kernel/kernel_woj_subsio.cu.bak`（1459 行）。
- `tools/graph_partitioner/graph_partitioner.cpp`（101 行；include 三个不存在头，不在 *.cu GLOB）。
- `tools/graph_partitioner/partitioner/hash_based_edgecut.h`、`hash_based_vertexcut.h`（唯一 includer 是上一条死文件；sics::graph 错误命名空间随之消失）。
  - **陷阱**：`partitioner_base.h` 被活的 `grid_cut.cuh` include，**保留**。
- `core/util/matrix.h`（199 行，零 includer）。
- `core/util/logging.h`（87 行，唯一 includer 是死文件 graph_partitioner.cpp）。
- `apps/wrapper.h`（7 行空壳，零 includer）。
- `tools/common/types.h`（16 行空壳，**精确复核：零 includer**——此前「6 处 include」是误匹配 core/common/types.h（活文件，88 处引用），直接删即可，无需摘 include）。

## U1.2 死类/死成员/死枚举/stub

- `core/task/gpu_task/matrix_analyzer.cuh/.cu?` + `set_operations.h`（互相引用，外部零 includer，已核实）。
- `core/components/gpu_task_manager.cuh/.cu` + facade 侧：`core/matrixgraph.cuh` 的 include、`scheduler_` 成员与 CHBL 构造、streams_/p_hr_* 死成员（interfaces 审计 M1/L 项）。
- 8 个 kernel 单例 `GetInstance()`+`ptr_`（interfaces 审计清单）零调用，逐个删。
- 死 TaskType 枚举值：kMatrixAnalysis/kSSSP/kGEMV/kExecuteAggPrim（core/common/types.h 或 consts.h）+ facade switch 对应 case（H4 静默 no-op 随之消失一半；facade 本体退役在 M3）。
- 空 stub：GEMM::FillTiles/Count、WOJExecutionPlan::GenerateJoinPlan；死局部变量（style 审计清单）。

## U1.3 业主批准的大删除（D12-D14，~3200 行）

- **D12 旧 weft SubIso 路径（~815 行，范围已按 redundancy.md:50 精确化）**：SubIso::Matching（subiso.cu:107-195，唯一调用点是注释行 :266）+ SubIsoKernelWrapper::SubIso（仅被 :182 调用）+ `kernel_subsio.cu`（658 行，内核全 static）+ `kernel_subiso.cuh`（65 行）。**Matches/ExecutionPlan 保留**（subiso_cpu.cu:1202,1311,1759 在用）。**kernel_woj_subsio.cu（1985 行）是活的 WOJ 路径**（在 core GLOB 内编译，SubIso::Run→WOJMatching 使用），不删；死的只有 .bak（D1）。
- **D13 GraphFilterAggregate（1334 行）**：graph_filter_aggregate.cu/.cuh + kernel 文件；唯一外部引用是 facade matrixgraph.cu（include、TaskType case、成员/分派）；CMake GLOB 自动收敛。注意：它用的是已合并的 BlockSumAll/stream 上提成果，删除不影响 block_agg_prims。
- **D14 PPRQuery（1059 行）**：ppr_query.cu/.cuh + facade 分派（唯一引用 matrixgraph.cu）；C6 重复簇随之消失。
- **保留**：EnumerateOnHost（validation 文档用途）、go_api 测试（在分支）。

## 顺序与验证

U1.0 → U1.1 → U1.2 → U1.3，各自独立 commit；每单元后跑双构建 + ctest + warning 闸门（预期 unused warning 大幅下降，闸门 cap 随实测下调）。WIP（GARDeduce/HomMatch/AttributesIO）与 `-scheduler` 旗标（M3/M4 处理）不碰。
