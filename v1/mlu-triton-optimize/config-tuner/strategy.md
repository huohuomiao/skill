# Config-Tuner

CUDA 的 shared memory、寄存器、occupancy、Grid、`num_warps` 和 `num_stages` 约束统一读取 `.claude/skills/share/gpu/references/platform-rules.md`；本文件只保留通用调参流程。

## 职责概述

自动化微调**当前 launch 架构内**的 Triton Kernel 配置。通过优先级策略在 **10次** 尝试内改进 `BLOCK_SIZE`、`num_warps` 和 `num_stages`；它不是普通/persistent/split-K 的架构搜索器。

## Step 1：提取与汇总轴信息

从 kernel 定义、wrapper 函数中提取 kernel 中输入输出 tensor 的轴信息，供后续所有步骤使用。

**加载参考**：阅读 [`get-tensor-axis-info.md`](references/get-tensor-axis-info.md) 查看如何获取输入 tensor 轴信息。


**示例**：
```python
{
    "a":
    {
        "axis": ["B", "M", "N"],
        "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
        "stride": [4096, 256, 1],
        "block_size": ["BLOCK_B", "BLOCK_M", "BLOCK_N"],
        "has_loop"：[True, True, True],
    },
    "b":
    {
        "axis": ["B", "N", "M"],
        "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
        "stride": [4096, 16, 1],
        "block_size": ["BLOCK_B", "BLOCK_N", "BLOCK_M"],
        "has_loop"：[True, True, True],
    }
}
```

汇总结果如下：

```python
{
    "axis": ["B", "M", "N"],
    "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
    "stride": [4096, 1, 1],
    "block_size": ["BLOCK_B", "BLOCK_M", "BLOCK_N"],
    "has_loop"：[True, True, True],
    "priority": [1, 0, 0],
}
```
**汇总过程分析**：两个tensor总共有B、M和N三个轴，首先分析名字为'B'的轴，提取a和b tensor中'B'轴的信息，axis_type，stride，block_size，has_loop均相同；其次提取a和b tensor中'M'轴的信息，'M'轴在a tensor中为第1维度，在b tensor中为第2维，按位置提取a和b tensor中的轴信息，汇总axis_type，block_size，has_loop均相同，但stride在a tensor中为256，在b tensor中为1，所以'M'轴最终的stride为1；'N'轴在a tensor中为第2维度，在b tensor中为第1维，按位置提取a和b tensor中的轴信息，汇总axis_type，block_size，has_loop均相同，但stride在a tensor中为1，在b tensor中为16，所以'N'轴最终的stride为1；根据优先级判断原则，'B'，'M'，'N'的axis_type均相同，判断stride，stride_b > stride_m == stride_n，所以B轴低于M轴和N轴，且M轴和N轴的优先级相同，优先级最高的priority设为0，B轴的优先级次之，priority值为1。

**Priority 排序**：`REDUCE` 轴优先于 `PARALLEL`；同类型下 `Stride` 越小优先级越高（Priority 值越小）。

## Step 2：确定 num_warps & num_stages

**说明**：persistent loop 是 persistent kernel 的具体实现形态，可以理解为：在 kernel 内部用一个循环，让每个 pid 反复处理多个数据块，从而“常驻执行”。它本质上是把“调度层的循环”搬进了 kernel 内。

分析 kernel以及根据 step2 提取到的 tensor 轴信息，获取以下信息：

- 计算瓶颈还是 io 瓶颈：分析 kernel 中是计算密集型还是 io 密集型，判断依据是 kernel 中如果包含**矩阵乘法**、**卷积**等运算为计算密集，其它类别如简单的 **elementwise**、**归约**等都属于 io 密集型
- 是否存在 persistent loop：对于 step1 中的汇总结果，**只对于 axis type 为 "PARALLEL" 的轴，has_loop 为 True 则说明存在 persistent loop**；否在不存在
- 是否有对输入tensor做升位宽操作（如 float16 升位 float32）或者 transpose 操作

在 RTX 3090（sm_86）上以 `num_warps ∈ {2,4,8}` 为常见候选；包含 `tl.dot`/软件流水的 kernel 实测 `num_stages ∈ {2,3,4}`，简单 pointwise/reduce 先以 `num_stages=2` 为基线，再按编译结果裁剪。不要根据“是否有 loop”直接断言流水收益。

| 场景 | `num_stages` 候选 | `num_warps` 候选 | 裁剪依据 |
|:---|:---:|:---:|:---|
| `tl.dot` / 多级加载 | `[2,3,4]` | `[4,8]` | 编译后的寄存器、shared memory 与 occupancy |
| pointwise / 常规 reduce | `[2,3]` | `[2,4,8]` | NCU 吞吐、延迟、occupancy 与真实耗时 |
| persistent 候选 | `[2,3,4]` | `[2,4,8]` | 必须先编译，再由每个 SM 的可驻留 block 数限制 grid |

禁止为 RTX 3090 生成 FP8、TMA、thread-block cluster 或 Hopper 专属配置。候选是否保留只由同输入、同计时方法的精度与实测性能决定。

## Step 3：Try 出最优 BLOCK_SIZE

**BLOCK_SIZE 微调方向**：

- 寄存器或 shared memory 压力低且并行度充足时，可尝试调大 block size；存在多个调优轴时，高优先级轴优先
- 遇到 `OutOfResources`、寄存器 spilling、shared-memory 超限或 occupancy 过低时，先调小低优先级轴的 block size，再评估降低 `num_stages` 或 `num_warps`

**BLOCK_SIZE 约束说明**：

- 仅可调整 `@triton.autotune` 中的 block_size 参数，不可改变 `triton.heuristic` 中固定的 block_size
- block size 一般对齐 2 的幂次或 32 的倍数，小于 8 时无需对齐
- 对于 `stride=1` 的轴，其 `BLOCK_SIZE * dtype_size` 建议不低于128 Bytes，以确保基本的访存效率

按照以上说明，try 出最优配置，要求如下：

- `num_warps`、`num_stages` 与 BLOCK_SIZE 是耦合参数；若编译后资源占用超限或 occupancy 明显降低，可在 10 次预算内联合调整，不得只靠静态内存估算锁死参数
- 精度测试通过是底线，任何精度测试不通过的尝试都为无效尝试
- 最大尝试次数为 10 次
- 清理try最优配置过程中产生的临时文件

### 架构逃逸门禁

若当前代码是 persistent matmul，且真实编译/NCU 同时显示 spilling 或接近寄存器上限，并由 register limiter 导致约 1 active block/SM（或理论 occupancy 约不高于 25%），本策略最多用少量尝试确认较小 tile/stages 能否解除限制。仍未解除或性能更差时：

1. 保留本轮输入，不把“10 次内无更优 config”写成优化空间已穷尽；
2. 在报告设置 `architecture_reselect_required=true`、`force_non_persistent=true`，并列出 registers、spill、limiter 与 occupancy 证据；
3. 建议下一策略为 `gen-autotune-config`，强制独立调优普通完整 Grid；K 足够大时再把 split-K 作为候选。

`maxnreg` 只能作为实编译、实测候选；它可能把寄存器压力转成更多 spill，禁止把设置寄存器上限当作修复成功。

## Step 4：结果输出

将当前架构内性能最优 config 更新到 Triton 配置并输出。若触发架构逃逸门禁且没有稳定提升，则逐字回退输入代码，同时在报告发出上述机器可读路由标记。
