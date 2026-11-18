# sum=0 根因诊断报告（U0.2 完成）

日期 2026-11-17。诊断于 scratch worktree /tmp/mg-sumprobe（main 未动）。状态：**根因已定位，修复已在 worktree 验证转绿**。

## 现象

graph_aggregate_test 在 Debug 与 Release 均 FAIL：`sum at v=0 expected=3 got=0`。C API 探针另示：compute_features 对全部 prim 返回 type=0（kInvalid）、值全 0；compute_all 完全正确。

## 根因（运行时证据链）

1. `ComputeFeaturesKernel`（kernel_graph_aggregate.cu）无 `__launch_bounds__`，Release 下编译出 **REG:130**（cuobjdump 实测）。
2. 以 `kBlockDim=1024` 线程 launch → 130×1024 = 133,120 寄存器 > sm_86 每 SM 65,536 上限 → **cudaErrorLaunchOutOfResources (701)**，kernel 从未入队（host 探针实测 `post-launch err=701`）。
3. **静默机制**：launch 后无 `cudaGetLastError()` 检查；失败的 launch 不产生流上错误，后续 `cudaMemcpyAsync`（拷回全 0 的 d_outputs）与 `cudaStreamSynchronize` 的 CUDA_CHECK 全部通过 → 错误答案而非错误码。
4. ComputeAll 不受影响的原因：`ComputeAllFeaturesKernel` 带 `__launch_bounds__(kAllFeaturesBlockDim=512)`，REG:128×512 = 65,536 恰好达标（consts.h:49-50 注释表明它是被专门调优过的，ComputeFeaturesKernel 漏掉了同款处理）。

排除过程：静态排除 5 假设（见决策日志 diagnosis 行）；运行时探针证明收集/属性/哈希链路健康（修复后 n_collected=3、sum f64=3）。ExecuteAggPrim Release 通过是因其 kernel 寄存器恰好 ≤64/线程（其 Debug 失败是 -G 膨胀，U0.4 同类修复）。

## 已验证修复（2 处，worktree 实测三段测试全 PASSED）

kernel_graph_aggregate.cu，ComputeFeaturesKernel 前加：

```cpp
// 1024 threads/block caps registers at 64/thread; without this bound the
// compiler allocates ~130 and every launch dies with error 701 (out of
// resources), which used to pass silently because no post-launch check
// existed. Spills are the accepted cost; retune block dim in a perf unit.
__launch_bounds__(common::kBlockDim)
__global__ void ComputeFeaturesKernel(
```

graph_aggregate.cu，ComputeFeaturesKernel launch 之后加：

```cpp
      CUDA_CHECK(cudaGetLastError());
```

验证输出（/tmp/sumprobe-fixed-run.txt）：Synthetic Test PASSED (100 vertices, 10 requests)；ComputeAll PASSED；Edge Attribute PASSED。探针显示 n_collected=3、req0 prim=1 f64=3。

## 遗留事项

- launch_bounds(1024) 强制 64 寄存器会有 local spill，性能影响待测；正解可能是给该 kernel 单独调 block dim（如 512，对齐 ComputeAll），列入 perf 单元（解释数字原则：先测后调）。
- 「launch 后必查 cudaGetLastError」应全库 sweep（所有 kernel launch 点），并入 Phase 5 错误契约单元（M9/H2）。
- go-api 分支同样含此 bug：compute_features C API 在分支上返回全 0。H1 转换修复已验证（compute_all 路径），端到端 compute_features 需同步 cherry-pick 本修复。
