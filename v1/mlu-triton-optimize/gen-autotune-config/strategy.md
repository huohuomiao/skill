# Triton 算子 autotune config 自动生成（CUDA GPU）

先读取 `.claude/skills/share/gpu/references/platform-rules.md` 获取 CUDA 硬件、运行时和 RTX 3090（sm_86）特性约束；本文件保留可复用的 autotune 装饰器生成和参数回写流程。

## 职责概述

**联合搜索 Triton kernel 的 launch 架构与 config；最终只输出全局胜出架构及其单一冻结配置。** 普通、persistent 与 split-K 不能共用一个先冻结的 config。

## 工作流程

1. 进入 Step 1
2. 根据 Triton kernel 中的 autotune 配置情况和 launch 架构，跳转 Step：
  - triton kernel 无 autotune 配置 → Step 2
  - triton kernel 的 autotune 配置中包含 block size，但不包含 num_stages 或者 num_warps → 跳转至 Step 2.2
  - triton kernel 的 autotune 配置完整 → 仍进入 Step 4；已有 best config 只属于当前架构，不能视为全局最优
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

先根据 kernel 内容生成保守的 tile 规模候选，但不使用静态张量大小冒充 CUDA 资源占用。对每个“架构 + config”实际编译，记录 registers/thread、shared-memory/block、spilling、理论 occupancy 与 occupancy limiter；这些结果才是资源裁剪依据。设备能力与采集接口统一从 `share/gpu` 读取。

先保存一个未 persistent 化的正确根版本，再建立互不污染的架构家族：

| 家族 | 准入与搜索要求 |
|---|---|
| `ordinary` | 必选。启动全部逻辑 tile；matmul 优先比较 grouped 1D 与原正确 Grid，不按 SM 数封顶 |
| `persistent` | 仅当 `modify-grid` 准入成立；使用本家族 config 的编译资源计算 resident Grid |
| `split_k` | 仅限可拆 K 的 matmul，且 K 并行可能弥补不足；必须包含完整合并开销和独立精度门禁 |

三个家族必须各自从同一正确语义根版本生成临时代码、各自调参。不得先把 ordinary 的 best config 冻结后套给 persistent，也不得让一次 `@triton.autotune` 在同一个 kernel 函数中切换不同控制流架构。

生成 block size 组合，原则如下：
**必须严格遵守的原则**

1. 当某个block size已经在 @triton.heuristics 被设置，那么在 auto tune 配置项中无需考虑此 block size。
2. stride=1 轴优先让连续访问覆盖至少一个合并事务，但不得把 128 Byte 当作所有 dtype/算子的硬下限
3. `tl.arange` 范围使用 Triton 支持的 2 的幂；其它 tile 维度优先从 16/32 的倍数中取值，并以编译和实测为准
4. reduce 轴的 block size 配置选项要包含等于该轴的实测数据大小

**block size 选取规则参考**

1. 从小到大生成 BLOCK 候选并逐项编译，剔除资源超限组合；spilling/低 occupancy 标记为高风险并减少基准预算，但不得仅凭 occupancy 删除一个实测最快候选
2. 依据轴信息（`priority` 越小优先级越高），高优先级连续轴可尝试较大 block，低优先级轴保持较小以控制寄存器和 shared-memory 压力
3. 每个架构家族限制在不超过 30 个有依据的配置，避免无约束笛卡尔积；先编译裁剪，再对剩余候选计时
4. sm_86 matmul 必须包含小 accumulator tile（如 `32x64`、`64x64`、`64x128`、`128x64`、`128x128` 的适用子集）以及 `BLOCK_K`、warps、stages 的联合变化。`128x256`/`256x128` 是寄存器高风险候选，只有编译与实测支持时保留，不能因它减少 persistent 循环次数就直接胜出

#### 2.2 确定 num_warps 和 num_stages

**说明**：persistent loop 是 persistent kernel 的具体实现形态，可以理解为：在 kernel 内部用一个循环，让每个 pid 反复处理多个数据块，从而“常驻执行”。它本质上是把“调度层的循环”搬进了 kernel 内。

分析 kernel以及根据 step2 提取到的 tensor 轴信息，获取以下信息：

- 计算瓶颈还是 io 瓶颈：分析 kernel 中是计算密集型还是 io 密集型，判断依据是 kernel 中如果包含矩阵乘法、卷积等运算为计算密集，其它类别如简单的 elementwise、归约等都属于 io 密集型
- 是否存在 persistent loop：对于 step2 中的结果，**只对于 axis type 为 "PARALLEL" 的轴，has_loop 为 True 则说明存在 persistent loop**；否在不存在
- 是否有对输入tensor做升位宽操作（如 float16 升位 float32）或者 transpose 操作

根据上述结果生成 RTX 3090（sm_86）候选。loop 只用于描述结构，不直接决定流水收益：

| 瓶颈 | 是否有persistent loop | 输入是否有升位宽/trans 操作 | `num_stages` 候选 | `num_warps` 候选 | 说明 |
|:---:|:--------------:|:------------------------:|:----------------:|:----------------:|-----|
| 计算 |      有        |           有             |   `[1,2,3,4]`    |     `[4,8]`      | 高寄存器压力必须包含 stage 1/2 与小 tile |
| 计算 |      有        |           无             |   `[1,2,3,4]`    |     `[4,8]`      | `tl.dot` 流水候选必须实测 |
| 计算 |      无        |           有             |   `[1,2,3,4]`    |     `[4,8]`      | 不因无 loop 禁用 CUDA 软件流水候选 |
| 计算 |      无        |           无             |   `[1,2,3,4]`    |     `[4,8]`      | 以编译资源与耗时选优 |
| io  |      有        |           有             |   `[2,3]`        |     `[2,4,8]`    | 检查访存吞吐与 occupancy |
| io  |      有        |           无             |   `[2,3]`        |     `[2,4,8]`    | persistent 仅作为实测候选 |
| io  |      无        |           有             |   `[2,3]`        |     `[2,4,8]`    | 常规 CUDA launch，不限制到 SM 数 |
| io  |      无        |           无             |   `[2,3]`        |     `[2,4,8]`    | 常规 CUDA launch，不限制到 SM 数 |

RTX 3090 不支持 FP8 Tensor Core 路径、TMA、thread-block cluster 或 Hopper 专属配置；禁止生成这些候选。`num_warps=1` 不是本目标平台的默认值，仅在已有代码实测证明时作为兼容候选保留。

#### 2.3 整合配置项

整合上述步骤获取的信息，输出 `@triton.autotune` 装饰器配置以及原始代码，生成原则如下：

- 同一家族内可用显式列表或有条件的推导式生成 configs；不得生成无约束笛卡尔积，也不得把不同架构控制流塞进同一个装饰器
- 包含 key 列表：使用 Step 1 中获取结果中的 `original_axis` 列表
- 生成的 `@triton.autotune` 装饰器配置置于 kernel 定义之上，且紧挨着 kernel 定义
- **若 kernel 中存在 in-place 操作，即对同一个输入指针既有 load 又有 store 操作，需要生成 restore_value 来保证每次编译时输入数据的一致性，不然精度一定会受到影响**。restore_value 为 @triton.autotune 的参数，值为列表，包含所有需要 restore 的输入指针的名字。

#### 示例

  ```python
  @triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
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
    BLOCK_N: tl.constexpr,
    num_blocks_m: tl.constexpr,
    num_blocks_n: tl.constexpr,
    num_programs: tl.constexpr,
  ):
    ...
  def wrapper(...):
    ...
    BLOCK_M = 128
    BLOCK_N = 256
    num_warps = 4
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    grid = (num_blocks_m * num_blocks_n, )
    triton_kernel[grid](..., BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_blocks_m=num_blocks_m, num_blocks_n=num_blocks_n, num_programs=grid[0], num_warps=num_warps)

  # 修改后的 kernel 及 wrapper 函数
  @triton.autotune(
    ...
  )
  @triton.jit
  def triton_kernel(
    ...,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
  ):
    num_blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    flat_pid = tl.program_id(0)
    ...
  def wrapper(...):
    ...
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    triton_kernel[grid](...)
  ```

**改动**：
1. 在 wrapper 函数中删除了 BLOCK_M, BLOCK_N 的赋值和传参
2. grid 参数 `num_blocks_m * num_blocks_n` 依赖 BLOCK_M、BLOCK_N，所以改为 lambda 动态计算；普通 CUDA kernel 保留全部逻辑 program，不按 SM 数封顶
3. num_blocks_m、num_blocks_n 的传参依赖 BLOCK_M、BLOCK_N，删除后在 kernel 内以同一计算流恢复。只有明确生成 persistent 候选时才使用 `tl.num_programs(0)` 循环覆盖任务，并在候选编译后按寄存器/shared-memory occupancy 计算驻留 grid

### Step 4: 架构级调优并生成单一全局最优配置

#### 4.1 各家族独立选 config

1. 开启 `TRITON_PRINT_AUTOTUNING`，分别运行 `ordinary`，以及满足准入条件的 `persistent`/`split_k` 临时候选。每个家族都从同一正确输入重新生成，不能串行继承另一家族的代码。
2. 对每个 config 先做编译与精度检查，再记录资源和 CUDA Event 耗时。输入、warmup、repeat、同步与统计量必须一致；禁止用 NCU replay 时间选 winner。
3. 每个家族只保留本家族 best config。persistent 必须用这个 config 的真实资源重新计算 `active_blocks_per_sm` 与 Grid；若 Grid 改变，重新验证和计时。
4. 若某个 persistent matmul 出现接近 255 registers/thread、local-memory spilling、register limiter 只允许 1 block/SM 或理论 occupancy 不高于约 25%，必须强制完成小 tile ordinary 家族的独立重调；`config-tuner` 在原 persistent 架构内失败不算完成该门禁。

#### 4.2 Split-K 候选

仅当输入是 matmul、K 足够大且普通 Grid 并行/延迟隐藏不足，或 persistent 因串行遍历与资源压力受限时尝试 `SPLIT_K in {2,4,8}` 的适用子集。split-K 不是必然更快，也不天然降低 accumulator 寄存器；必须与较小 M/N tile 联合搜索。

- 每个 `pid_k` 只遍历自己的 K 分片并正确 mask 尾部。
- FP16/BF16 输入保持 FP32 累加。优先让各 split 写入互不重叠的 FP32 workspace，再用 finalize kernel 求和并转换输出；若用 FP32 atomic，必须在每次 autotune/benchmark 前清零，并处理原地/别名语义。
- benchmark 必须包含清零、workspace 写回和 finalize 等原调用新增的全部 GPU 开销；同时报告额外 workspace 字节数。输出为 FP16/BF16 时，禁止直接用低精度 atomic 冒充 FP32 累加语义。
- 含 atomic 或可变状态时配置 `reset_to_zero`、`restore_value` 或等价 pre-hook，保证每个 config 输入一致；无法证明精度与复现性则删除 split-K 候选。

#### 4.3 全局选择与冻结

1. 将各家族自己的 best config 在目标 RTX 3090 上重新做精度与统一性能测试，并比较完整 wrapper 路径。
2. 只保留稳定更快的全局 winner；若 ordinary 获胜，必须彻底移除 persistent loop/SM 上限；若 persistent 获胜，仍需保留完整 grid-stride 覆盖证明。split-K 获胜则同时保留其 workspace/finalize 路径。
3. 将 winner 的 `@triton.autotune` 缩减为一个配置，或等价冻结 launch meta；再次运行精度与性能测试后输出最终代码。其它家族只能留在报告，不留死代码。
4. 报告逐家族记录 best config、registers/thread、spill、shared memory、active blocks/SM、occupancy、kernel/完整路径耗时、精度与回退理由；未采集项写 `N/A`，不得把其它 GPU 的结果推导为 3090 收益。

## 关键约束与边界条件

| 约束项 | 规则 |
|--------|------|
| CUDA 资源 | 逐候选编译并读取 registers、shared memory、spilling、occupancy；硬件属性转读 `share/gpu` |
| 架构公平性 | ordinary/persistent/split-K 各用自己的 best config，再比较完整路径；禁止架构先锁定 |
| RTX 3090 特性 | 禁止 FP8、TMA、cluster、Hopper 专属路径；warps 常用 2/4/8，dot stages 在资源压力下包含 1/2 |
