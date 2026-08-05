# Config-Tuner 

## 职责概述

自动化微调 Triton Kernel 配置。通过优先级策略在 **10次** 尝试内锁定最优 `BLOCK_SIZE`、`num_warps` 和 `num_stages`。

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

根据上述结果，然后按照下表确定 num_stages 和 num_warps 的候选值：
| 瓶颈 | 是否有persistent loop | 输入是否有升位宽/trans 操作 | [`num_stages`,`num_warps`] 候选 | 说明 |
|:---:|:--------------:|:------------------------:|:----------------:|-----|
| 计算 |      有        |           有             |   `[3, 1]、[4, 4]` | 并行轴上开启流水选项，计算瓶颈下，开启 num_warps=4 选项可以使输入的升位宽/transpose 可以走mv流|
| 计算 |      有        |           无             |   `[3, 1]`         | 并行轴上开启流水选项 |
| 计算 |      无        |           有             |   `[1, 1]`         | 并行轴上没有循环，不开启流水选项 |
| 计算 |      无        |           无             |   `[1, 1]`         | 并行轴上没有循环，不开启流水选项 |
| io  |      有        |           有             |   `[3, 1]`         | 并行轴上开启流水选项 |
| io  |      有        |           无             |   `[3, 1]`         | 并行轴上开启流水选项 |
| io  |      无        |           有             |   `[1, 1]`         | 并行轴上没有循环，不开启流水选项 |
| io  |      无        |           无             |   `[1, 1]`         | 并行轴上没有循环，不开启流水选项 |

## Step 3：Try 出最优 BLOCK_SIZE

**BLOCK_SIZE 微调方向**：

- NRAM 占用率低时，尝试调大 block size，当存在多个调优轴时，高优先级轴优先
- NRAM 超限时（即遇到 `OutOfResources: out of resource: NRAM, Required: XX, Hardware limit: XX` 错误），适当调小 block size，当存在多个调优轴时，低优先级轴优先

**BLOCK_SIZE 约束说明**：

- 仅可调整 `@triton.autotune` 中的 block_size 参数，不可改变 `triton.heuristic` 中固定的 block_size
- block size 一般对齐 2 的幂次或 32 的倍数，小于 8 时无需对齐
- 对于 `stride=1` 的轴，其 `BLOCK_SIZE * dtype_size` 建议不低于128 Bytes，以确保基本的访存效率

按照以上说明，try 出最优配置，要求如下：

- num_warps 与 num_stages 已确定，原则上不能再更改，若 Step 2 更改 num_warps 与 num_stages 造成了 NRAM 超限问题，则适当调低 BLOCK_SIZE 大小
- 精度测试通过是底线，任何精度测试不通过的尝试都为无效尝试
- 最大尝试次数为 10 次
- 清理try最优配置过程中产生的临时文件

## Step 4：结果输出

将性能最优 config 更新到 triton autotune 配置项中，并输出为最终代码。

