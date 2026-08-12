# Triton Kernel Grid 优化 (grid_modify_optimization)

Grid 解析与展平属于通用 Triton 变换；CUDA 设备属性、资源上限、occupancy 与 RTX 3090 特性门控统一读取 `.claude/skills/share/gpu/references/platform-rules.md`。

## 职责概述

分析每个 Triton kernel 的 launch Grid，在不改变逻辑任务覆盖和输出语义的前提下：

1. 保留正确的普通 CUDA Grid；多维 Grid 在 CUDA 上合法，不需要仅为适配硬件而压成一维。
2. 仅在索引更简单或实测更快时，把多维 Grid 等价展平为一维；展平后仍启动全部逻辑 program，不限制到 SM 数。
3. 仅对明确的 persistent 候选生成 grid-stride loop。候选必须先编译，再根据 registers/thread、shared-memory/block、threads/block 和设备上限得到 active blocks/SM，最后计算驻留 Grid。

严禁把普通 kernel 的 `grid` 强制限制为 SM 数。`sm_count // num_warps` 不是 CUDA occupancy 公式，也不得用作 Grid 上限。

## 步骤

### Step 1：提取所有 Grid 结构

#### 1.1 定位定义与调用

1. 找到所有 `@triton.jit` kernel 及其 wrapper launch。
2. 对每个 `kernel_name[grid_expr](...)` 独立分析，不得只取第一个 kernel。
3. 回溯 `grid` 变量、lambda 和参与计算的 BLOCK/meta 参数。
4. 若只有 kernel 定义、没有可解析调用，则无法证明 launch 语义，保持代码不变并在报告中说明。参考 `references/modify_grid_without_grid.md`。

#### 1.2 识别写法

| 写法 | 示例 | 处理 |
|---|---|---|
| 内联元组 | `kernel[(M, N)](...)` | 读取各维表达式 |
| 变量引用 | `kernel[grid](...)` | 回溯 `grid = ...` |
| lambda | `grid = lambda meta: (...)` | 保留 meta 依赖 |
| 单个表达式 | `kernel[triton.cdiv(N, B)](...)` | 视为一维 |

按顶层逗号拆解 tuple，忽略函数调用内部逗号。例如 `(triton.cdiv(M, BM), triton.cdiv(N, BN))` 必须拆成两个维度。

记录：

- `dims`：每个 Grid 维度表达式；
- `has_lambda`：是否依赖 meta；
- `logical_grid`：各维乘积，即逻辑任务总数；
- PID 到输出区域的映射，以及任务间是否独立；
- kernel 内是否已经存在 `tl.num_programs(0)` 驱动的 grid-stride loop。

对普通 Grid 建立覆盖证明：设各维逻辑块数为 `G0...Gn`，launch program 总数应为其乘积；PID 解码后每个维度都落在对应半开区间 `[0, Gi)`，且线性化/反线性化互为逆映射。若 mask 仅保护 tensor 边界而不保护非法 PID，则 Grid 绝不能多发 program。只有 kernel 明确检查逻辑 PID 边界时，才允许向上取整后的额外 program 安全退出。

同时检查 wrapper 中 Grid 维顺序与 kernel 的 `tl.program_id(axis)` 用法。CUDA 多维 Grid 的 axis 0/1/2 顺序不能根据变量名猜测；必须从原 launch 和地址表达式共同确认。若一个 kernel 被多个 wrapper 以不同 Grid 调用，应分别保留或分别生成候选，不能用首个调用的结论覆盖其它调用。

### Step 2：分类并选择候选

#### 情况 A：无法提取 Grid

保持原代码。不得虚构 `n_elements`、BLOCK_SIZE、Grid 或 wrapper；缺失调用上下文时无法证明任务覆盖。加载 `references/modify_grid_without_grid.md`。

#### 情况 B：普通一维 Grid

若 Grid 已覆盖全部逻辑块，默认保持不变：

```python
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
```

不要添加 `min(..., sm_count)`。只有满足 Step 3 的 persistent 准入条件时，才另建候选而不直接覆盖基线。加载 `references/modify_grid_1d.md`。

#### 情况 C：普通多维 Grid

CUDA 支持多维 launch，默认可保留：

```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
```

若展平能简化统一调参或索引，可生成不封顶的一维候选：

```python
blocks_m = triton.cdiv(M, BLOCK_M)
blocks_n = triton.cdiv(N, BLOCK_N)
grid = (blocks_m * blocks_n,)
```

kernel 内用 `flat_pid` 恢复维度。展平只是索引映射，不等于 persistent。加载 `references/modify_grid_3d.md`；规约场景另读 `references/modify_grid_reduce.md`。

#### 情况 D：Grid 全为 1

先判断单 program 是否因完整 tile 造成编译超限或并行度不足。可生成普通 tiled Grid 候选，直接启动 `total_blocks` 个 program；不得默认压到 SM 数：

```python
grid = lambda meta: (triton.cdiv(M * N, meta["BLOCK_SIZE"]),)
```

仅在普通 tiled 基线之后，且 Step 3 全部通过时再生成 persistent 版本。加载 `references/modify_grid_constexpr.md`。

### Step 3：Persistent 候选准入与 Grid 计算

#### 3.1 必须同时满足

1. 普通 Grid 基线已能正确运行并通过精度测试。
2. `logical_grid` 明显大于可同时驻留的 program 数，且减少调度/重复加载存在可解释收益。
3. 每次循环迭代处理独立逻辑任务；不存在跨 program barrier、危险别名或未处理的输出竞争。
4. 所有任务可由 `tl.num_programs(0)` 固定步长循环完整覆盖。
5. 已编译目标候选并取得 registers/thread、shared-memory/block、threads/block 等真实资源数据。
6. 最终通过相同输入、相同计时方法的 A/B 实测；无稳定提升则回退普通 Grid。

规约写同一输出时，必须证明原有原子/分阶段合并语义仍正确。不得仅因展平或 persistent 化而擅自把普通 store 改成 atomic，也不得把所有输出都改成零初始化。

#### 3.2 动态设备属性

设备探测优先调用 `.claude/skills/share/gpu/runtime/get_device_info.py`，或使用稳定的 PyTorch CUDA 属性：

```python
device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
sm_count = props.multi_processor_count
```

不要硬编码 RTX 3090 的 SM 数、寄存器数或 shared-memory 数。不要导入非 CUDA backend，也不要依赖 Triton 的私有 NVIDIA driver API。

#### 3.3 编译后 occupancy

先固定一个具体 config 并编译，再从编译元数据/`share/gpu/perf-analyzer/analyzer_ncu.py` 获取资源占用。按照 `platform-rules.md` 计算或读取 `active_blocks_per_sm`：

```text
resident_programs = sm_count * active_blocks_per_sm
persistent_grid = min(logical_grid, resident_programs)
```

`active_blocks_per_sm` 必须同时受 threads、registers、shared memory 和架构 block 上限约束，不能用 `num_warps` 单独推导。若无法取得编译后资源信息，跳过 persistent 候选。

autotune 中不同 config 的资源占用可能不同。由于主流程里的 `gen-autotune-config` 在本策略之后执行，本轮不得生成需要冻结 config 的 persistent 改写；只记录候选。待第 5 个策略冻结最终 config 后，`gen-autotune-config` 必须按本节重新计算 occupancy、生成 persistent 候选并做 A/B；不得沿用早期 Grid 上限，也不得对所有 config 共用猜测值。

RTX 3090 为 sm_86：禁止 FP8、TMA、thread-block cluster 和 Hopper 专属 persistent 路径。

#### 3.4 Occupancy 计算记录

为每个 persistent 候选保存一条可复核记录，不允许只写“按核心数优化”：

| 字段 | 来源 | 用途 |
|---|---|---|
| GPU 名称与 compute capability | `share/gpu/runtime/get_device_info.py` 或 PyTorch CUDA 属性 | 确认实际目标为 sm_86 或记录偏差 |
| `num_warps` / threads per program | 冻结后的 Triton config | 计算 thread 约束 |
| registers per thread/program | 编译元数据或 NCU | 计算 register 约束并识别 spilling |
| static/dynamic shared memory per program | 编译元数据或 NCU | 计算 shared-memory 约束 |
| blocks limited by threads/registers/shared memory | CUDA occupancy 数据 | 解释 `active_blocks_per_sm` 的瓶颈来源 |
| `logical_grid` / `resident_programs` / final grid | wrapper 与 occupancy 结果 | 证明 launch 与循环覆盖关系 |

概念上，active blocks/SM 是多个上限的最小值：thread 上限、register 上限、shared-memory 上限和架构 block 上限。寄存器与 shared memory 还受分配粒度影响，因此优先采用 CUDA occupancy API、编译器元数据或 NCU 已计算结果；只有 `.claude/skills/share/gpu/references/platform-rules.md` 明确允许时才手算，且所有设备上限都必须动态读取。

若 NCU 的 achieved occupancy 低于理论值，不要直接扩大 Grid。先检查 kernel 是否太短、是否受内存延迟、依赖链、寄存器 spilling 或 launch 数不足影响；occupancy 只是诊断指标，不是越高越快的目标函数。

#### 3.5 Persistent 收益信号

下列信号只用于决定“是否值得生成候选”，不能替代实测：

- 单个逻辑任务很短、logical Grid 很大，launch/调度开销占比可见；
- 每个 program 可在多次迭代间复用只读参数、索引或常驻状态；
- 工作分布均匀，grid-stride loop 不会让少数 program 承担长尾；
- 冻结 config 后仍能维持足够 active blocks/SM，且没有新增 spilling；
- 普通 Grid 的 NCU 时间线显示调度或利用率空洞，而不是纯带宽饱和。

下列情况默认跳过 persistent：任务数接近或小于驻留 program 数、单任务耗时差异大、输出存在复杂竞争、需要跨 program 同步、autotune config 尚未冻结、或资源元数据不可得。

### Step 4：代码改写

#### 4.1 普通展平候选

wrapper 只计算全部逻辑块：

```python
grid = lambda meta: (
    triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
)
```

kernel 使用唯一 PID 恢复二维索引：

```python
flat_pid = tl.program_id(0)
blocks_n = (N + BLOCK_N - 1) // BLOCK_N
pid_m = flat_pid // blocks_n
pid_n = flat_pid % blocks_n
```

三维 `(A, B, C)` 按同一线性化顺序恢复：

```python
c_i = flat_pid % C
b_i = (flat_pid // C) % B
a_i = flat_pid // (B * C)
```

host 侧可用 `triton.cdiv`；kernel 内使用 `tl.cdiv` 或等价整数式 `(a + b - 1) // b`，不要调用 host-only API。

#### 4.2 Persistent 候选

wrapper 使用 Step 3 得到的编译后驻留 Grid：

```python
grid = (min(total_blocks, resident_programs),)
```

kernel 必须通过实际 launch 数取步长并覆盖所有逻辑块：

```python
pid = tl.program_id(0)
num_programs = tl.num_programs(0)
for flat_pid in range(pid, total_blocks, num_programs):
    # 恢复逻辑索引并处理一个完整任务块
    ...
```

不得把 `resident_programs` 当成 kernel 内步长；实际 Grid 可能因 `logical_grid` 较小而缩短，步长必须来自 `tl.num_programs(0)`。

#### 4.3 边界与类型

- `tl.arange` 的范围必须是编译时常量并符合 Triton 约束。
- 每个 load/store 保留正确 mask；展平后不得遗漏尾块。
- `tl.zeros`、`tl.ones`、`tl.full` 的 shape 使用 tuple。
- 保留原 kernel/wrapper 名称和输入输出契约。
- launch 直接传 tensor，不使用 `.data_ptr()`。
- host tensor、同步和设备检查统一使用 CUDA。

### Step 5：验证与保留

对每个改写依次验证：

1. 编译与运行成功；目标设备确认为 CUDA，且 GPU capability 与 `share/gpu` 记录一致。
2. 与原 PyTorch reference 和普通 Grid 基线比较精度，沿用用户给定阈值。
3. 证明每个逻辑块恰好被覆盖，特别检查尾块、展平解码和规约写冲突。
4. 用相同输入、warmup、repetition 和同步方式比较普通 Grid、展平候选、persistent 候选。
5. 仅保留实测稳定更快的候选；性能持平、回退或资源指标恶化时恢复普通 Grid。

测试至少覆盖三类规模：小规模（检查 launch 开销与空 Grid）、代表性规模（用于最终选优）和非整除规模（检查尾块）。若用户只提供一个固定规格，不得自行声称其它 shape 也有收益；报告要明确最终代码的适用 shape/key。

对规约/atomic kernel 额外检查重复运行稳定性。浮点 atomic 的执行顺序可能变化，必须沿用用户阈值并记录最大误差；不得为了让候选通过而放宽容差。对 in-place 或别名输入，分别验证普通与 persistent 版本没有覆盖尚未读取的数据。

报告至少包含：原 Grid、候选 Grid、是否展平、是否 persistent、logical task 数、资源采集来源、active blocks/SM、精度结果、普通/候选耗时和最终保留/回退理由。无法采集的字段写明“未取得”，不能填猜测值。

## 禁止项

| 禁止内容 | 原因 |
|---|---|
| 普通 kernel 使用 `min(logical_grid, sm_count)` | 会丢失任务，除非 kernel 已有正确 grid-stride loop |
| 使用 `sm_count // num_warps` 计算 Grid | 不是 CUDA occupancy 模型 |
| 未编译就猜 active blocks/SM | 忽略 registers/shared memory/threads 联合约束 |
| 无调用上下文时生成默认 wrapper | 无法证明 shape、Grid 和输出契约 |
| 为了展平擅自引入 atomic | 可能改变确定性、精度和性能 |
| 在 RTX 3090 上启用 FP8/TMA/cluster/Hopper 路径 | sm_86 不支持 |

如果不能证明正确性或无法取得 persistent 资源数据，保持原始普通 Grid；正确性优先于形式统一。

所有结论必须可由代码、编译记录或实测结果复核。
