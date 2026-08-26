# Triton 算子 autotune config 自动生成（MLU）

先读取 `.claude/skills/share/mlu/references/platform-rules.md` 获取 MLU 硬件与运行时约束；本文件保留可复用的 autotune 装饰器生成和参数回写流程。

## 职责概述

**为 Triton kernel 生成标准的 @triton.autotune 装饰器配置，且强制 autotune 中只有单一最优配置项。**

## 工作流程

1. 进入 Step 1
2. 根据 triton kernel 中的 autotune 配置情况，跳转 Step：
  - triton kernel 无 autotune 配置 → Step 2
  - triton kernel 的 autotune 配置中包含 block size，但不包含 num_stages 或者 num_warps → 跳转至 Step 2.2
  - triton kernel 的 autotune 配置完整 → 跳转至 Step 4
3. 顺序执行后续所有步骤

## 步骤

### Step 1: 提取必要信息

#### 提取 tensor 轴信息

从 kernel 定义、wrapper 函数中提取 kernel 中输入输出 tensor 的轴信息，供后续所有步骤使用。

**加载参考**：阅读 [`get-tensor-axis-info.md`](references/get-tensor-axis-info.md) 查看如何获取输入 tensor 轴信息。

#### 汇总 tensor 轴信息

同一轴（即相同名字的轴）可能出现在不同的tensor中，所以需要汇总轴的信息，汇总其中某个轴的时候，汇总原则如下：

- **抽取所有tensor中同名轴的列信息**，包括对应索引位置的"axis"，"axis_type"，"stride"，"block_size"，"has_loop"
- 将该轴上不同tensor的"axis"，"axis_type"，"stride"，"block_size"，"has_loop"等信息进行汇总，生成该轴的最终信息，其中：
  - **stride 取所有tensor中该轴对应stride的最小值**
  - axis_type 取 REDUCE 优先于 PARALLEL
  - has_loop 取`或`操作

生成结果是一个字典，包含"axis"，"axis_type"，"stride"，"block_size"，"has_loop"，以及新增的"priority"字段，说明如下：
- 抹去原始的 tensor 名称，只保留汇总的其它信息
- 新增 `priority` 字段，axis type 为 REDUCE 的轴优先级高于 PARALLEL 轴，相同 axis type 时，stride 越小优先级越高，axis type 与 stride 都相同的轴优先级相同，数字越小代表优先级越高，priority 值从 0 开始递增

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

**⚠️ 注意：必须严格保证 `priority` 的正确性，因为后续 block size 的选取会依赖于 `priority` 来确定优先级。**

#### 提取测试数据规格

从给定的测试数据规格中提取输入 tensor 形状、数据类型等信息，供后续步骤使用。

### Step 2: 生成 @triton.autotune 装饰器配置

#### 2.1 确定 blocksize

首先根据 kernel 内容分析 nram 占用情况，生成一个 NRAM 预估函数 `estimate_nram`，输入为所有的 BLOCK_SIZE 加输入的 dtype 信息，输出是 NRAM 占用大小，要求：

- **充分考虑到内存复用**，分析 kernel 中 tensor 的生命周期，避免简单地将所有 tensor 占用的内存加总
- 不用考虑标量运算
- 不考虑 mask 以及地址相关的计算

生成 block size 组合，原则如下：
**必须严格遵守的原则**

1. 当某个block size已经在 @triton.heuristics 被设置，那么在 auto tune 配置项中无需考虑此 block size。
2. 所有 stride=1 的轴 的 block size 不能低于 128 Byte
3. block size 设置为 2 的幂次或 32 的倍数，以更好地适配 MLU 硬件
4. reduce 轴的 block size 配置选项要包含等于该轴的实测数据大小

**block size 选取规则参考**

1. **通过 `estimate_nram` 预估占用的 nram，生成的 block size 尽可能最大化利用 nram 空间（接近512KB）**
2. 在 1 的基础上依据 step2 中提取的轴信息（`priority` 字段体现了 block size 的优先级，priority 越小，优先级越高），**优先级越高的轴越适合设置较大的 block size 来充分利用内存带宽，所以高优先级轴的block size可以调大，优先级低的轴可以适当调小**
3. 限制 block size 组合总数在合理范围内（不超过 30 个），以避免过长的编译时间，其中优先级高的轴配置选项可以少一些，优先级低的轴配置项可以多一些

#### 2.2 确定 num_warps 和 num_stages

**说明**：persistent loop 是 persistent kernel 的具体实现形态，可以理解为：在 kernel 内部用一个循环，让每个 pid 反复处理多个数据块，从而“常驻执行”。它本质上是把“调度层的循环”搬进了 kernel 内。

分析 kernel以及根据 step2 提取到的 tensor 轴信息，获取以下信息：

- 计算瓶颈还是 io 瓶颈：分析 kernel 中是计算密集型还是 io 密集型，判断依据是 kernel 中如果包含矩阵乘法、卷积等运算为计算密集，其它类别如简单的 elementwise、归约等都属于 io 密集型
- 是否存在 persistent loop：对于 step2 中的结果，**只对于 axis type 为 "PARALLEL" 的轴，has_loop 为 True 则说明存在 persistent loop**；否在不存在
- 是否有对输入tensor做升位宽操作（如 float16 升位 float32）或者 transpose 操作

根据上述结果，然后按照下表确定 num_stages 和 num_warps 的候选值：

| 瓶颈 | 是否有persistent loop | 输入是否有升位宽/trans 操作 | `num_stages` 候选 | `num_warps` 候选 | 说明 |
|:---:|:--------------:|:------------------------:|:----------------:|:----------------:|-----|
| 计算 |      有        |           有             |   `[1, 3, 4]`    |     `[1, 4]`     | 并行轴上开启流水选项，计算瓶颈下，开启 num_warps=4 选项可以使输入的升位宽/transpose 可以走mv流|
| 计算 |      有        |           无             |   `[1, 3]`       |     `[1]`        | 并行轴上开启流水选项 |
| 计算 |      无        |           有             |     `[1]`        |     `[1]`        | 并行轴上没有循环，不开启流水选项 |
| 计算 |      无        |           无             |     `[1]`        |     `[1]`        | 并行轴上没有循环，不开启流水选项 |
| io  |      有        |           有             |   `[1, 3]`       |     `[1]`        | 并行轴上开启流水选项 |
| io  |      有        |           无             |   `[1, 3]`       |     `[1]`        | 并行轴上开启流水选项 |
| io  |      无        |           有             |     `[1]`        |     `[1]`        | 并行轴上没有循环，不开启流水选项 |
| io  |      无        |           无             |     `[1]`        |     `[1]`        | 并行轴上没有循环，不开启流水选项 |

#### 2.3 整合配置项

整合上述步骤获取的信息，输出 `@triton.autotune` 装饰器配置以及原始代码，生成原则如下：

- configs 使用 for 循环推导式，block size, num_warps, num_stages 备选项组成列表被 for 循环依次遍历，每个备选项列表元素从小到大排序
- 包含 key 列表：使用 Step 1 中获取结果中的 `original_axis` 列表
- 生成的 `@triton.autotune` 装饰器配置置于 kernel 定义之上，且紧挨着 kernel 定义
- **若 kernel 中存在 in-place 操作，即对同一个输入指针既有 load 又有 store 操作，需要生成 restore_value 来保证每次编译时输入数据的一致性，不然精度一定会受到影响**。restore_value 为 @triton.autotune 的参数，值为列表，包含所有需要 restore 的输入指针的名字。

#### 示例

  ```python
  @triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': bm, 'BLOCK_N': bn}, num_stages=s, num_warps=w)
        for bm in [128, 64, 32]
        for bn in [256, 128, 64]
        for s in [1, 3]
        for w in [1]
    ],
    key=['M', 'N'],
    restore_value=[...],
  )
  def triton_kernel(...):
    ...
  ```
  ### Step 3: 修改 wrapper 函数

加入 autotune 之后，block size，num_warps，num_stages 的值将不再由 wrapper 函数固定传入，而是由 autotune 在运行时动态确定。因此，wrapper 函数中与 block size，num_warps，num_stages 直接或间接相关的参数都需要修改，主要包括以下几点：

1. 删除 wrapper 函数中 block size，num_warps，num_stages 的赋值和传参
2. 若 grid size 的计算依赖于 block_size 或 num_warps 或 num_stages，需要追溯从 block_size/num_warps/num_stages 到 grid size 的计算流，使用 lambda 函数替换原有的 grid size 计算方式，并且通过 meta 获取 block size，num_warps 及 num_stages
3. triton kernel 传入的参数中，除了 block size，num_warps，num_stages 之外，若有其它参数的传入数值跟 block_size/num_warps/num_stages 有关，则追溯 block_size/num_warps/num_stages 到输入参数的计算流，改为在 kernel 内部以相同的计算流得到，特殊计算流需特殊处理，情况如下：
  - triton.cdiv(M, BLOCK_M) 函数只能在 host 侧使用，无法再 kernel 内部使用，映射到 kernel 内通过手动计算： (M + BLOCK_M - 1) // BLOCK_M
  - 输入参数中存在 grid[n]，映射到 kernel 内通过 tl.num_programs(n) 替换
4. 删除 kernel 所有与 block_size/num_warps/num_stages 相关的参数

#### 示例

  ```python
  # 修改前的 kernel 及 wrapper 函数
  @triton.autotune(
    ...
  )
  @triton.jit
  def triton_kernel(
    ...,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    num_blocks_m: tl.constexpr,
    num_blocks_n: tl.constexpr,
    num_programs: tl.constexpr,
  ):
    ...
  def wrapper(...):
    ...
    BLOCK_M = 128
    BLOCK_N = 256
    num_warps = 1
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    core_num = torch.mlu.get_device_properties(0).multi_processor_count
    MAX_GRID_SIZE = core_num // num_warps
    grid = (min(num_blocks_m * num_blocks_n, MAX_GRID_SIZE), )
    triton_kernel[grid](..., BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_blocks_m=num_blocks_m, num_blocks_n=num_blocks_n, num_programs=grid[0],num_warps=num_warps,)

  # 修改后的 kernel 及 wrapper 函数
  @triton.autotune(
    ...
  )
  @triton.jit
  def triton_kernel(
    ...,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
  ):
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    num_programs = tl.num_programs(0)
    ...
  def wrapper(...):
    ...
    core_num = torch.mlu.get_device_properties(0).multi_processor_count
    grid = lambda META: (min(triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), core_num // META['num_warps']), )
    triton_kernel[grid](...)
  ```

**改动**：
1. 在 wrapper 函数中删除了 BLOCK_M, BLOCK_N 的赋值和传参
2. grid 参数 "num_blocks_m * num_blocks_n" 依赖于 BLOCK_M, BLOCK_N 的数值，所以改为通过 lambda 函数动态计算；"MAX_GRID_SIZE" 依赖 num_warps 的数值，也通过 lambda 动态获取
3. num_blocks_m, num_blocks_n, num_programs 的传参依赖 BLOCK_M, BLOCK_N，num_warps，也删除，改为在 kernel 内部以相同的计算流计算。

### Step 4: 生成单一最优配置

1. 开启环境变量 `TRITON_PRINT_AUTOTUNING` 并运行生成的代码
2. 若精度有错误，请根据错误信息进行调试修改；若精度无误，提取 best_config 信息，并将 @triton.autotune 中的配置项组合缩减为单一最优配置
3. 再次运行生成代码，若精度有错误，请根据错误信息进行调试修改；若精度无误，输出最终代码

## 关键约束与边界条件

| 约束项 | 规则 |
|--------|------|
| NRAM 上限 | 512 KB（524288 bytes）参考值；运行时通过 `max_nram_size` 动态获取实际值 |
| block size 上限 | 65536（`MAX_BLOCK_SIZE`） |
