# Engine 设计 arena 合成决定（Phase C/D）

日期 2026-11-17。裁判说明：跨厂商裁判不可用（claude CLI 未登录、无 codex/claude-agent-acp），按 arena Phase C 回退规则由父 agent 独自评分选 base，**此局限如实披露**。两个候选同为父模型（generation-bound 工作，规则允许）；rubric 未下发给候选。

## Rubric 评分（1-10，带依据）

| 准则 | C1 | C2 | 依据 |
|---|---|---|---|
| 1 引擎统一性 | 8 | 9 | C2 的 protected Execute + friend Engine 使 Engine::Run 成为唯一执行入口（类型系统保证）；C1 保留 facade+engines 两个整机概念 |
| 2 诚实共享边界 | 9 | 9 | 平手：两者都拒绝「一份算法体两后端」假抽象，LFTJ/WOJ 以显式 algorithm 参数区分，仅共享宿主预处理 |
| 3 工具链约束合规 | 9 | 10 | 两者都有 TBB-free 接缝 + .cpp 实现；C2 多一个 build 期 header-hygiene gate（每个公共头分别按 CXX 和 .cu 编译，违规即构建失败）= 约束结构化 |
| 4 公共面深度/对齐 | 8 | 9 | C2 分层更细（platform/types/status 独立层，19 公共头 + module map）；C1 四层 + umbrella 也深，但 facade 是浅包装风险 |
| 5 ABI 兼容与迁移 | 8 | 9 | 两者都正确吸收隔离决定；C2 的 prim_bridge.h（迁移期映射 + static_assert + 逐值测试）比 C1 的「一波迁移」更安全；C2 阶段与既有 plan Phase 0-5 对齐 |
| 6 简洁/红旗 | 8 | 8.5 | C1 保留 MatrixGraph facade（第二个整机概念）；C2 明确拒绝转发头树/注册表/浅包装，代价（MG_HOST_DEVICE 宏、friend 耦合）已如实披露 |
| **合计** | **50/60** | **54.5/60** | |

## 决定：Base = Candidate 2

核心形状：运行时 `Engine` 句柄（`MakeEngine(Backend)`）是唯一整机概念；`Engine::Run(Task&)` 是唯一执行入口；分层 0-3 对纯 C++ 编译器 CUDA-free/TBB-free（MG_HOST_DEVICE + hygiene gate）；域类型统一（一个 FeatureValue、一个 featurelib AggPrim、一套 block_agg_prims）；facade 退役；一次性任务与有状态服务两种形态共享同一错误/资源契约。

## 从 Candidate 1 嫁接

- **G1 引擎生命周期类型化**：task 以 `std::shared_ptr<Engine>` 持有引擎（C1 D2），替代 C2 的裸引用，把「引擎活得比任务久」编进类型而非调用纪律。
- **G2 AllFeatures 字段数 static_assert**：`static_assert` 把 AllFeatures 字段数绑到 kAggPrimCount（C1 红旗筛查产物），进统一后的 feature_value.h。
- **G3 Service 家族的 typed load/compute API 命名规范**（C1 D4 的 service 描述更完整），并入 C2 的 AggregateService/ComputeFeaturesService 头文件文档。
- **G4 迁移兜底**：C1 的「目标拆分前先 header 纪律 + 保留单目标回退构建」写进 M1/M4 执行注记（C2 的 target split 风险缓解）。

## 合成后仍开放的业主决定

1. `MatrixGraph` 类名去留：C2 退役该类（项目名保留）；C1 保留为整机门面。默认按 C2（退役 + apps 迁到 MakeEngine），若要保留名字可做 thin alias。
2. `-scheduler` 旗标：M3 临时 shim 映射到 Backend::kGpu，M4 apps 一波迁到 `-backend` 后删除（默认），或永久保留兼容。
3. AggregateService 是否本轮就加 CPU 并行实现：默认不加（GPU-only + 契约就绪，laziness）。
4. hom_match.cu 是 WIP：TU 种类（.cu→.cpp）也不动；subiso_cpu.cu 非 WIP，M2 迁移。
5. TBB 并发隔离：默认先 scoped global_control（与现状等价），per-engine task_arena 列为后续。
6. 大删除 D12-D16（旧 weft SubIso 815 / GraphFilterAggregate 1334 / PPRQuery 1059 / EnumerateOnHost 90）：仍待业主决定。
7. 风格政策包（.clang-format 重写 + clang-format 18 pinned、访问器 PascalCase、struct 字段豁免、括号只修 122 处多行、注释英文、保留 .cpp）：仍待业主确认。

## 产物

- 候选 1：/tmp/arena-engine/candidate-1/（design.md + sketch/ 21 头 + module map）
- 候选 2：/tmp/arena-engine/candidate-2/（design.md + sketch/ 19 公共头 + 9 内部头 + module_map.md）
- 迁移路径以 C2 的 M0-M5 为骨架（与 .audit/plan.md Phase 0-5 对齐），Phase 6 ABI 门已消失，C9/H2/H3 内化进 M3/M5。
