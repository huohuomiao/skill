# Config-Tuner：BANG C 配置微调

## 职责

依据真实 MLU590 correctness、notifier、CNPerf 与 CNCC/MLISA 证据，在最多十个候选内微调一个参数族。候选可涉及片上 tile、每 Task 工作量、任务规模/function type、向量处理长度、buffer 复用或流水级数；同轮不得同时改变多个族。

平台限制读取 `.claude/skills/share/mlu/references/platform-rules.md`，不得硬编码 NRAM/SRAM/WRAM 容量、Core/Cluster 数、架构 flag 或 function type 约束。

## Step 1：提取配置与轴信息

读取：

- 当前 `input.mlu` 与 `kernel-info/strategy.md` 生成的 `kernel_info.json`。
- [get-tensor-axis-info.md](references/get-tensor-axis-info.md)。
- baseline correctness/notifier、最新 CNPerf 原始报告和可用的 MLISA/中间产物。

汇总每个逻辑轴：shape、stride、访问连续性、并行/归约角色、每次处理元素/字节、片上 buffer、循环次数和 tail。优先优化连续访问且在热点循环中的轴，但最终由实测决定。

## Step 2：选择一个参数族

| 证据 | 仅可选择的参数族 |
| --- | --- |
| 片上利用低、搬运批次过多 | tile elements/bytes |
| 编译器明确资源超限 | 较小 tile 或缩短 buffer 生命周期 |
| Task 负载不均、尾批明显 | 每 Task 工作量或任务规模 |
| 可用 Core 未充分利用 | task dimension/function type（先验证合法性） |
| IO/MV/Compute 流存在可掩盖空隙 | ping-pong/流水级数 |
| 向量指令处理长度低效 | vector/intrinsic length 与对齐策略 |
| 重复 GDRAM 访问 | buffer 复用/片上 staging |

没有真实证据时只能做小范围、保守的 OOB 搜索，并标记原因。

## Step 3：生成有限候选

候选从当前配置的安全邻域产生，不做大笛卡尔积。例如当前 tile 的较小值、当前值和较大值；实际候选必须先通过：

- 所有片上 buffer 字节布局与生命周期可证明。
- 编译器未报告资源超限。
- 搬运长度、对齐和尾块安全。
- task ID 映射完整、无重复冲突。
- function type/任务规模由当前 SDK/设备支持。
- runtime Shape 不被错误冻结为某个测试常量。

配置使用集中宏、模板参数或 policy 函数表达，确保每个候选 diff 只改变一个族。

## Step 4：外部编译运行搜索

每个候选独立保存源码、binary、编译日志、正确性和 notifier 样本。使用与 baseline 相同的 `cncc` 命令；只有当前配置值不同。

候选选择顺序：

1. 编译和 CNRT/launch 通过。
2. 全部 correctness 通过。
3. 所有关键 Shape 无回退。
4. MLU590 device median 最低且收益超过噪声。
5. 性能区间重叠时选择改动更小、资源更稳的配置。

最多十个候选；不存在可证明收益时逐字回退 `input.mlu`。把获胜配置冻结进 `bangc_optimized.mlu`，报告完整候选表，不只报告赢家。
