# Engine 架构 grounding（architect Phase A 产物，arena  runners 的共同输入）

日期 2026-11-17。HEAD 4cec1d5。仓库 /home/zhuxk/project/graph/GARAgent/third_party/MatrixGraph。
所有论断带 file:line 或实测证据。设计目标见 §7。硬约束见 §6。

## 1. 系统与消费者

MatrixGraph 是 C++20/CUDA 图计算库（~196 文件 / 42k 行，core/apps/tools）。三类消费者：
- apps/ 18 个示例/工具程序（bfs、pagerank、wcc、subiso、gemm、graphrag_* 等），部分绕过 task 层直调 kernel（.audit/interfaces.md M6）。
- 外部 C++ 复用者：基于 core 复用数据结构（ImmutableCSR、bitmap、attributes）、算法或整个 engine（业主明确需求）。
- 外部 Go 库：经 core/go_api 的 C ABI（opaque handle + 返回码 + wire types），Go 只是传输外壳（业主明确说明）。

## 2. 当前并行接缝（核心病灶）

- core/util/execution_policy.h:14-26：ParForEach 由编译宏 __CUDACC__ 选择后端——nvcc TU 串行 std::for_each，纯 CXX TU 用 std::execution::par。全局命名空间，无运行时选择。
- 实测探针（2026-11-17，/tmp/probe_execution.cu）：nvcc 12.6 + GCC 12.4 下，.cu 包含 <execution> 即使带项目现有 -Xcompiler=-mno-avx512f,-mno-avx512bf16,-mno-amx-tile 仍编译失败（EDG 前端解析 avx512bf16vlintrin.h 报 __builtin_ia32_cvtne2ps2bf16 未定义）。**串行 workaround 是真约束，不是历史遗留**（execution_policy.h:5-11 注释准确）。
- 推论（已核实文件形态）：subiso_cpu.cu、hom_match.cu 是 .cu 但无 __global__/__device__ 代码（.audit/style.md §e1 grep 验证）→ 走 nvcc → ParForEach 静默串行。「CPU 并行任务」实际串行，后端选择是编译产物的意外而非意图。
- core/util/cpu_parallel_scope.h：唯一直接 TBB 使用点，oneapi::tbb::global_control RAII 限线程数（libstdc++ 的 std::execution::par 路由到 TBB）。includer 仅 diameter.cpp、skew.cpp（纯 CXX TU）。
- **TBB 头文件在 nvcc TU 不可用（2026-11-17 探针实测，三次）**：oneapi/tbb/detail/_machine.h:44 无条件 #include <immintrin.h>（为 _mm_pause），EDG 前端解析 GCC 的 avx512bf16vlintrin.h/amxtileintrin.h 时报 __builtin_ia32_* 未定义。nvcc 12.6 + GCC 12.4 失败（parallel_for.h 与 global_control.h 均失败）；-ccbin g++-13 失败（amxtileintrin.h）；g++-13 + 项目 -mno-avx512f,-mno-avx512bf16,-mno-amx-tile 仍失败（-mno 是后端 target 开关，EDG 照样解析头文件函数体）。**结论：任何含 TBB include 的代码必须住在纯 CXX TU（.cpp）；被 .cu 包含的公共头必须 TBB-free。** 这是 engine 分层设计的硬约束，不是风格偏好。
- 直接 tbb:: 调用全库仅 cpu_parallel_scope.h 一处；ParForEach/std::execution 使用点 12 文件（subiso_cpu.cu、diameter.cpp、min_wise_filter.h、skew.cpp、gemm.cu、kernel_woj_subsio.cu、ppr_query.cu、edgelist.cu、immutable_csr.cu、edgelist_khop.cpp 等）。
- GPU 并行：每 task 自建 stream（TaskBase 提供 EnsureStreams/DestroyStreams 但只有 execute_agg_prim/graph_filter_aggregate 在用；bfs.cu:102、pagerank.cu:122、wcc.cu:148、subiso.cu:180、matrix_ops.cu:82 仍自建）；GraphAggregate 另有每 GPU×每 stream 二维并行（graph_aggregate.cu:975-1010）；gemm/ppr_query 有多 stream 矩阵 tiled 并行另一套。

## 3. Task 层现状

- core/task/gpu_task/task_base.cuh:16-30：TaskBase 只管 stream 生命周期，无 virtual Run，无任务契约（输入/输出/错误/资源）。
- core/task/cpu_task/cpu_task_base.h:14-21：CPUTaskBase 空壳，非虚析构，重 include（future/mutex/thread/unordered_map）。
- core/matrixgraph.cu:38-160：facade 按 TaskType 枚举 switch + reinterpret_cast<TaskBase*/CPUTaskBase*> 到具体 task；4 个枚举值静默 no-op（H4）；scheduler_ 构造后从未使用且泄漏（M1）。
- 任务目录：cpu_task/（diameter、hom_match、lftj_subiso、min_wise_filter、pattern_vertex_order、skew、subiso_cpu）；gpu_task/（bfs、compute_features、execute_agg_prim、gar_deduce、gar_match、gemm、graph_aggregate、graph_filter_aggregate、lftj_subiso_gpu、matrix_ops、pagerank、ppr_query、subiso）。
- 已有的「一任务双引擎」雏形是 ad-hoc 的：lftj_subiso.cpp(CPU) ↔ lftj_subiso_gpu.cu(GPU) 共享宿主预处理但各抄一份（C5，diff 3-17 行）；subiso_cpu ↔ subiso(GPU WOJ) 是不同算法（CPU LFTJ vs GPU WOJ），不是同算法双后端。
- 数据装配重复：ImmutableCSR 缓冲尺寸 8 项公式 ×12 处（C2）；graphrag 三 app 复制 task 的 Execute 缓冲装配（C1，pagerank 有刻意 off-by-one 修正）。

## 4. 数据结构层

- ImmutableCSR（core/data_structures/immutable_csr.cuh，43 includers）：host 结构 + 单块 device buffer blob，ParseBasePtr 按固定偏移切 8 段（globalid/indegree/outdegree/in_offset/out_offset/incoming/outgoing/edges_globalid/localid）；kernel 侧手工重解析同一布局（kernel_graph_aggregate.cu:88-99）。布局契约散落在 host 填充与 device 解析两处。
- bitmap 两家族并存（C12）：util/bitmap*.h + gpu_bitmap.cuh vs data_structures/kernel_bitmap*.cuh + mini_kernel_bitmap.cuh + MinWiseBitmap，全部在用。
- attributes：host Attribute 描述符 + DeviceAttributes/DevicePerVertexAttributes 设备哈希表（attributes.h、graph_aggregate.cu:124-224 Build 深拷贝连续池）；ValueType 与 Go 枚举镜像（attributes.h:17 注释）。
- 双值类型：task::FeatureValue（execute_agg_prim.cuh，union i64/f64/f32/b/str）vs MatrixGraphFeatureValue（compute_features_types.h:200-207，C wire，union i64/f64/b，无 f32/str）。第三份聚合原语 kernel_compute_features_primitives.cuh（517 行）用 MG 值类型（C9，ABI 单向门）。

## 5. Go C ABI 现状

- core/go_api/matrixgraph_go_api.h：opaque void* handle（无类型标签，错族 handle 是 UB，L4）、返回码契约（"generally" 0，L4）、wire types 在 compute_features_types.h（住在 gpu_task 目录里）。
- 已核实 bug：H1 union 顺序写覆盖数值（go_api.cu:326-341）；H3 双 prim 编号 MG_AGG_SUM=3 vs kSum=1 + 假「intentionally match」注释 + 无重映射（go_api.cu:311）；H5 README Go AllFeatures 镜像少 dfeat（16 vs 17 字段）；H2 CUDA_CHECK exit() 杀死 Go 宿主进程（cuda_check.cuh:12）。
- 错误策略三层不一致（M9）：io/data_structures 抛异常、edgelist/cuda_check exit()、go_api 返回码。
- go_api 测试 execute_agg_prim_go_api_test.cpp 从未被任何目标构建（D16）。

## 6. 硬约束（设计必须尊重）

1. nvcc TU 不能用 <execution>/std::execution::par，也不能用任何 TBB 头（§2 三次探针实测，GCC 12.4/13 均失败）。CPU 并行实现必须住纯 CXX TU（.cpp），经 TBB-free 的接口头被 .cu 调用。工具链升级（更新 nvcc/EDG）可能解除此约束，但分层设计不应依赖它。
2. 工具链：GCC 12.4 / CUDA 12.6 / cmake 3.28 / C++20；3× RTX 3090（sm_86）；bundled oneTBB/gflags/yaml-cpp；googletest 缺失待补。
3. C/Go ABI 冻结：布局/枚举值/字段序/转换语义变更须业主批准（单向门）。设计可以提议变更但必须单独标记 gated。
4. WIP 勿动：GARDeduce/HomMatch/AttributesIO ~1685 行（gar_deduce.*、hom_match.*、attributes_io.*、kernel_gar_deduce.*、gar_deduce_arrays.h）。
5. 外部 API 标识符不改名：folly/CUDA/cuRAND/cuBLAS/gflags/TBB/yaml-cpp/gtest。
6. 业主代码原则：算法层与后端解耦（切后端不改算法代码）、高内聚低耦合、窄接口、无跨层全局可变状态、注释写为什么、Google C++ 风格（类型/函数 PascalCase、变量 snake_case、成员尾下划线、常量 kPascalCase）。
7. 行为保持：现有测试基线（.audit/tests-baseline*.txt）；sum=0 真 bug 待修（U0.2/U0.3）；Debug -G 下 1024 线程 kernel launch 失败（U0.4）。

## 7. 设计目标（业主原话意图）

A. **Engine 管理**：仓库本质是并行库，含 TBB CPU 并行、CUDA GPU 并行、CPU 串行三种执行引擎。一个 task 希望同时有 CPU 并行与 GPU 实现。要求 engine 整齐规整：统一契约、统一资源/错误/生命周期管理、后端选择是显式意图而非编译意外。
B. **对外接口对齐**：外部用户基于 core 写自己的程序——复用数据结构、复用算法、或直接整个 engine 拿走用。另有 Go C ABI（服务外部 Go 库）。要求对外接口整齐规整：分层清晰、命名一致、契约明确、复用路径不止一条野路子。

## 8. 留给设计回答的开放问题

- 引擎选择：运行时参数 vs 编译期策略 vs 混合？同一 task 的 CPU/GPU 实现如何注册与发现？
- 算法共享的诚实边界：CPU LFTJ 与 GPU WOJ 是不同算法，强行「一份算法代码两个后端」是不是假抽象？哪些层真能共享（宿主预处理、数据装配、验证），哪些必须 per-backend？
- 资源所有权：stream、device buffer、TBB arena、pinned memory 归谁？析构契约？
- 错误契约：内部异常/返回码如何统一，C 边界如何翻译，exit() 如何退出库代码？
- 公共头文件分层与稳定性：哪些头是 public API（承诺稳定），哪些是 internals（可随改）？复用者如何只拿数据结构不拖进整个 task 层？
- wire types 与 domain types 的边界：C ABI 类型如何不泄漏进 C++ 公共面？
- 迁移路径：从现状到目标如何分阶段、每阶段行为保持、不 big-bang？
