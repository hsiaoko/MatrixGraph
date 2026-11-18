# 交接快照（2026-11-18）

**状态**：M0–M4、M5.1、M5.2（A 值容器统一 + B plan 类型化、第三份原语删除）、U5.2 后端能力边界、Phase 3、Phase 4 全部完成；`main` 与 `go-api` 均已 push（`origin/main` @ `95f61fe`、`origin/go-api` @ `d3f54ab`）；最新 `95f61fe`（`b76b13d` A、`7869526` B、`1734ec0` parity 基线）。ctest 7/7、warning 闸门 cap 0、header 闸门 12 头全绿。

剩余（均为可选/待决定）：`AggregateService` 的 CPU 实现（原设计默认不加，属新增 API）；Release 下 sanitizer 的既有 AttributeName 展宽读（Debug/app 均无报告）；`go-api` 分支已于 2026-11-18 push 到 `origin/go-api`（`d3f54ab`）。

## 已完成

- **M0 harness**：googletest v1.14.0 submodule、TEST=ON 冲突修复、6 测试入 ctest、warning 闸门 `scripts/check-warnings.sh`。
- **真 bug 修复**：sum=0（launch_bounds + 落 launch 后查错）、Debug -G 可 launch（kFeatureRequestBlockDim=512、ExecuteAggPrim launch_bounds、debug 专用 32KB 设备栈）。报告 `.audit/sum0-diagnosis.md`。
- **M1 减法**：Go ABI 隔离到 `go-api` 分支（`d3f54ab`，2026-11-18 已 push 为 `origin/go-api`）；8 死文件；死类/facade 残骸/7 单例/3 stub；D12 weft SubIso + D13 GraphFilterAggregate + D14 PPRQuery（业主批准）。净删约 5900 行。
- **M2**：C2 缓冲公式 11 处收敛（修两处欠分配）；TBB-free 并行接缝（57 调用迁移，删 execution_policy.h）；stream 单一所有者（~TaskBase 真销毁 + PrimaryStream）。
- **M3**：`util::Error` + 抛错（库内 exit 归零）；virtual Run + facade 枚举分发删除；`core/engine/`（Backend/EngineOptions/Engine/MakeEngine，CPU TBB global_control，GPU 设备上下文）。
- **M4**：U4.1 14 个 app 迁 `-backend`/`MakeEngine`（`669e6f6`，错后端干净拒绝）；U4.2 删 facade/`TaskType`/scheduler 集群 + 8 文档同步（`1b5cc92`）；U4.3 header-hygiene gate `scripts/check-headers.sh`（`afc9106`）；U4.4 GpuEngine 持有 stream 池、任务借用（`727fa6e`）。
- **M5.1**：公共 C++ 面 `matrixgraph/matrixgraph.h` + 三个伞头，闸门扩到 12 头，`docs/API.md`（`3caec81`）。
- **Phase 3 零 warning**：158 唯一位置 → **0**（干净 Release **与** Debug 实测均为 0，ctest 双 6/6），cap=0；提交 W1 正确性组 `ad03b60`、unused+默认值 `4a3b69d`、reorder+参数 `500ef15`、sign-compare+残余 `277be97`、宏/重载 `ff6cacc`、cap 归零 `900ae26`。第三方 14 行豁免在案。
- **Phase 4（全部完成）**：clang-format 18.1.8 pinned + `.clang-format` 重写，163 文件格式化（`5b98cd3`）；拼写/宏/typedef（`f075407`）；9 个 kernel、52 个访问器（734 处）、12 个自由函数 PascalCase，`StoreStrategy` → enum class（`37aa26b`）；死注释清理 −149 行 / 22 文件（`c88ff67`）；责任头注释 76/103（`4bdbde5` 18 个 + `d687959` 58 个；1 个空头、10 个有理由跳过），2026-11-18 补齐最后 3 个（`b2f3b26`：`format_converter.cuh` 真缺口 + 2 个空占位注明），现 102 个头全部有文件级注释、扫描 missing=0；全树格式复归干净。教训：`kDefalut*` 前缀型标识符不能用 `\b` 词边界；`binary_search` 改名会误伤 `std::binary_search`。

## 验证协议（每个单元必做）

Release 构建 + `ctest -j`（应 6/6）+ `scripts/check-warnings.sh /tmp/mg-m0`（cap 已归零：第一条新 warning 即失败；该闸门只计当次构建新增，全量重测用干净 build dir）+ `scripts/check-headers.sh`。行为变更（并行/生命周期）另加运行时对拍：converter 双跑 diff、subiso -t1 vs -t8、wcc/bfs/pagerank 冒烟。构建目录 `/tmp/mg-m0`（Release）、`/tmp/mg-debug`（Debug）——/tmp 若被清，重配即可（命令见决策日志）。

- **M5.2 开工**：设计方案 `.audit/m52-design.md`（A 值容器统一+桥 → B plan 类型化；锚 = block_agg_prims）；parity harness `compute_features_parity_test.cu`（17 case × 8 pivot，进 ctest 现 7/7）；基线抓到两处真缺陷——num_unique 输出写竞争（已修，线程 0 单写者）、MG 聚合枚举缺 DFEAT。

## M5.2 完成（2026-11-18）

- **A 值容器统一**（已提交）：解释器与第三份原语改用 `task::FeatureValue`；`Compute()` 返回 `std::vector<FeatureValue>`；属性列内部类型为 `ValueType`。wire 结构体只留在 C 边界（`compute_features_value.h` 的 ToFeatureValue/ToWireValue，A 之后 B 连这层也删了）。
- **B plan 类型化**：`plan_types.h` 迁入 `core/task/gpu_task/compute_features/`；`op` 拆为 `AggPrim agg_op` + `MatrixGraphTransOp trans_op`，`const_type` 为 `ValueType`，条件 `op` 为 `MatrixGraphCondType`；删除 `MatrixGraphValueType` / `MatrixGraphAggPrim` / `MatrixGraphFeatureValue`；删除第三份原语（457 行），改调 `BlockApplyAggPrim`；`kDFeat` 现在可由 plan 表达并进入 parity（19 case）。
- **两套编号确已分叉**：`AggPrim` 是 featurelib 序（sum=1），被删的 MG 枚举是 sum=3——第三份原语按 MG 序写，`BlockApplyAggPrim` 按 featurelib 序，删掉正好消除分歧。

### M5.2 实施中新增的运行时发现

1. **`BlockExclusiveScan` 不适用于 256 线程块**：percent_true / count_gt_mean 的 scan 缓冲按 `common::kBlockDim`（1024）线程设计；本解释器每 pivot 一块 256 线程，实测每线程谓词计数正确（local=1）而 scan 汇总为 0。两个计数原语改为在解释器 `ApplyAggPrim` 包装里用 `BlockSum` 实现（就地注释说明），其余仍走 `BlockApplyAggPrim`。
2. **设备栈**：递归解释器加上 workspace 参数后 Release 也会栈溢出（compute-sanitizer: CollectBindings Stack overflow → launch failure）。`cudaLimitStackSize` 现在所有构建都抬到 32 KB，不再只在 NDEBUG 之外。
3. **既有缺陷（非本单元引入）**：compute-sanitizer 在 Release 下报 `HashMap<AttributeName,...>::find` 里 4 字节读落在分配间隙（ReadAttribute 路径），每条线程一条、线程 0 不报。证据链指向优化器把 `AttributeName`（64 字节结构）的逐字节 hash/equal 循环展宽成 4 字节读：Debug `-G` 构建同一代码 0 报告、数值全对、app 0 报告。同一现象在 HEAD `1734ec0` 复现，故非本单元引入；按「已知噪声 + 真有后果再查」处理，已记入决策日志。

## 下一步 M5.2（历史记录，已完成）

1. 设计级开工：C9 值类型统一（`task::FeatureValue` 与 ComputeFeatures 的 MG 值类型合并、`compute_features_types.h` 内部化、prim_bridge 迁移映射 + 逐值测试）。两个枚举编号不同且 plan 用 int32 字段，须先设计再动。
2. 可选：AggregateService 的 CPU 实现；`-backend` 能力边界文档化。
3. 收尾项已全部完成（Phase 3 零 warning、Phase 4 风格与注释）。

## U5.2 后端能力边界（2026-11-18，commit `95f61fe`）

- `Skew` / `Diameter` 原本放在 `core/task/cpu_task/` 却继承 GPU 的 `TaskBase`：它们自己的默认 `-backend cpu` 直接抛 "CPU engine cannot run a GPU task"（改动前已在 diameter 上复现），而 `-backend gpu` 反而「能跑」——把纯 CPU 算法挂在 GPU 引擎下。现已改为 `CPUTaskBase`，顺带让两个纯主机头不再包含 `cuda_runtime.h`。
- `docs/API.md` 新增「Which backend runs which task」：两套 task 契约、错配抛 `kUnsupported`、service-style 任务（ComputeFeatures / ExecuteAggPrim / GraphAggregate）、以及每个可执行文件的 `-backend` 默认值。app 文档同步纠偏：GPU-only 任务不再宣称支持 `cpu`/`cpu_serial`，CPU-only 任务不再宣称支持 `gpu`。
- 运行时证据：diameter 与 skew 在 `kCpuSerial` / `kCpuParallel` 结果一致（parity ring 上 diameter=2、skew=0.333333）；bfs/wcc 在 `kGpu` 正常；两个方向的错配都被干净拒绝。

## 待业主决定/注意

- `go-api` 分支已 push（`origin/go-api` @ `d3f54ab`）；后续在该分支上的改动仍按「业主批准才 push」处理。
- M4 完成后可切回便宜模型做 Phase 3 warning 清扫与 Phase 4 改名 codemod（风格政策默认包已批准：保留 .cpp、英文注释、GetXxx PascalCase、struct 字段豁免、clang-format 18 pinned、tools/python 不在范围）。
- WIP 勿动：GARDeduce/HomMatch/AttributesIO。`gemm.cu` 是 CRLF（脚本改它须 `newline=''`）。`cuda_check.cuh` 内引用 `Error` 须全限定名。
- 决策日志：`.audit/matrixgraph-refactor.tsv`（每单元一行，含证据指针）。