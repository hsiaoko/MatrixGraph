# M5.2 设计方案：值类型统一与第三份原语回收

2026-11-18。只读调研（HEAD `d687959`）。无代码改动。

## 现状证据

**值容器两套**
- `task::FeatureValue`（`core/task/gpu_task/execute_agg_prim.cuh:24-45`）：标签 `ValueType`（`attributes.h:20`，`enum class ... : uint8_t`，镜像 igeenum 编号），union 有 `i64/f64/f32/b/StringView`，配套 `Make*Value` 工厂与 `ToDouble/Compare`。
- `MatrixGraphFeatureValue`（`compute_features_types.h`）：标签 `int32_t`（MG_VALUE_* 编号），union 只有 `i64/f64/int32 b`，**没有 f32 与字符串成员**，而 MG_VALUE_STRING=4、MG_VALUE_FLOAT32=6 两个枚举值存在——这是潜在的不一致，B 阶段应确认字符串值是否可达，否则删除死枚举。

**编号三套并存**
- `ValueType`（igeenum 序：kFloat64…kInt=3…kBool=9，uint8）；`MatrixGraphValueType`（INT=1/FLOAT64=2/BOOL=3/STRING=4/TIME=5/FLOAT32=6）。
- `task::AggPrim`（featurelib 序，kCount=0…kSum=1…kDFeat=16，`enum class : uint8_t`）；`MatrixGraphAggPrim`（第三套：COUNT=0/COUNT_GT_MEAN=1/NUM_UNIQUE=2/SUM=3/MEAN=4/…）。
- plan 节点用 int32 存量：`MatrixGraphPlanNode.op`（MG_AGG 序）、`const_type`（MG_VALUE 序）；表达式/导航/条件/变换各有一套 MG 枚举，解释器直接 switch 这些 int32。

**原语三份 → 两份仍在**
- 第三份 `kernel_compute_features_primitives.cuh`：533 行，一整套 `Make*/ToDouble/Compare/BlockSum/BlockMin/Max/BlockBitonicSort/AggCount…AggQuartile3`，全部按 `MGFeatureValue` 编写。
- 规范层 `block_agg_prims.cuh`：`BlockSum/BlockSumAll/BlockMin/BlockMax/BlockBitonicSort` + `BlockAggCount/Sum/Mean/Min/Max/Variance/Std/Median/Mode/NumUnique/Entropy/Quarter/Quartile3`（及其后文件内可能还有 PercentTrue/CountGreaterThanMean）——覆盖面与第三份重合，签名是「块内共享数组 + 工作区」形态。

**MG 层的消费者仅 5 个**：`compute_features.cu/.cuh`、`kernel/kernel_compute_features.{cu,cuh}`、`kernel_compute_features_primitives.cuh`、`compute_features_test.cu`、`apps/compute_features.cpp`（18 处 MG_ 引用）。没有任何其他模块依赖 MG 类型。

**测试覆盖是缺口**：`compute_features_test.cu` 只用过 `MG_AGG_SUM` 与 `MG_AGG_MEAN`。16 个原语的数值一致性目前无测试保护。

## 方案

**推荐 A → B 两段式，C 否决。**

- **A（值容器统一 + 编号桥，plan 结构不动）**：解释器与第三份原语改用 `task::FeatureValue`；`MatrixGraphFeatureValue` 变成别名；第三份的 `Agg*` 函数改为调用 `block_agg_prims` 的 `BlockAgg*`（需要收集缓冲 + 工作区，解释器已有逐绑定收集逻辑可复用），删掉与规范层重复的 `BlockSum/BlockMin/BlockMax/Sort` 等；在 `compute_features_types.h` 增加 `constexpr` 桥 `ToAggPrim(int32_t)` 与 `ToValueType(int32_t)`（显式 switch，未知值抛 `util::Error(kInvalidArgument)`），只在两处分派点使用。app/test 源码不动。
- **B（plan 类型化，桥删除）**：plan 节点的 `op`/`const_type` 改为 `AggPrim`/`ValueType` 类型字段；app/test 改用类型化枚举；删除 MG_VALUE_*/MG_AGG_* 与桥函数；`compute_features_types.h` 迁到内部路径（如 `core/task/gpu_task/compute_features/plan_types.h`）并从 `docs/API.md` 的内部边界中说明。布局变化仅内部可见（指定初始化器由编译期检查）。
- **C 否决**：只共享原语、保留双值类型与三套编号，等于把最贵的维护面留在原地。

## 行为保护（A 和 B 都要过）

新增 `core/task/gpu_task/compute_features_parity_test.cu`，走现有 tests GLOB 自动进 ctest（无需改 CMake）：合成环图 + 每类属性（int/f64/f32/bool/time），对 16 个原语逐一跑 plan 解释器，与宿主参考实现（std 算法，浮点聚集用显式容差）对比；同时断言返回值的类型标签与 `kDFeat` 透传。A 之前先记录当前基线（同一张表跑在 HEAD），A、B 之后各跑一次——三次数值必须一致，迁移即被证明为行为保持。

## 风险与处置

1. **两份原语语义可能不同**（中位数定义、熵的底、skew 公式、NumUnique 哈希）。不能假设等价：先跑 parity 基线，若差异出现，以 **block_agg_prims 为语义之锚**（它是已被 ExecuteAggPrim 与 GraphAggregate 生产路径验证的实现），差异处修第三份侧。此项需业主确认（见下）。
2. plan 节点布局在 B 中变化（int32→uint8 字段影响 padding），仅内部可见，指定初始化器编译期兜底。
3. 字符串/f32 值在 MG 容器里无存储成员：若解释器实际产生字符串，A 阶段 `FeatureValue` 会自动获得能力；若不可达则 B 阶段删死枚举。
4. `ValueType` 编号是 igeenum 镜像，**冻结不改**；桥只做显式映射。

## 待业主决定

1. 范围：A 单独（plan 结构不动、风险最低）还是 A+B（类型化 plan、app/test 有 API 改动）？建议 A+B。
2. 语义之锚：两份原语行为冲突时，以 `block_agg_prims`（生产验证过的实现）为准，还是逐原语对比后由你裁决？
3. 内部头落点：`compute_features_types.h` 迁到 `core/task/gpu_task/compute_features/plan_types.h`（推荐）还是原地保留只改文档？
## 实施中发现（2026-11-18，parity 基线）

1. **num_unique 恒为 0 的数据竞争**：`ComputeFeaturesKernel` 让每个线程都执行 `outputs[...] = v`，而顺序统计原语（median/mode/num_unique/entropy/quarter/quartile3）只有线程 0 计算出真值、其余线程返回占位值——占位值覆盖了真值。已改为线程 0 单写者，parity 全绿。A 阶段应保留这个修法（或改由 block_agg_prims 的广播语义自然覆盖）。
2. **MG_AGG_DFEAT 不存在**：MG 聚合枚举到 SKEW=15 为止，而 `AggPrim` 有 `kDFeat=16`——plan 无法表达 dfeat。B 阶段换成 `AggPrim` 后该缺口自动消失；A 阶段 parity 表不含 dfeat（无 plan 表达方式），但 DeleteThirdCopy 后可用 C++ 直调覆盖。
3. 尝试过的错误修法（记录以免重犯）：在 `ApplyAggPrim` 内部加 `__syncthreads()` 广播会导致非法访问——该函数处在逐 binding 的循环里，各线程的调用次数未必一致，函数内同步点破坏收敛性。同步点必须放在 kernel 层、共享控制流处。
