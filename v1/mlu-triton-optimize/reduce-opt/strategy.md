# Triton CUDA reduce类算子综合优化

循环、访存和归约改写保留为通用 Triton 候选；CUDA shared memory、寄存器、occupancy、Libdevice 与设备规则统一读取 `.claude/skills/share/gpu/references/platform-rules.md`。

## 职责概述

本优化套件包含多个归约算子优化策略，按顺序遍历检测triton代码是否通过每个优化策略的优化准入校验，通过则执行对应优化。

| 执行顺序 | 优化策略 | 核心适用场景 | 优化目标 |
|----------|----------|--------------|----------|
| 1 | while循环转for循环候选 | kernel中存在可确定分析为固定步长的计数型 `while` 循环 | 生成语义等价候选；仅实测更快时保留 |
| 2 | 归约轴循环优化 | 归约轴存在显式循环计算，且归约轴大小 ≤ MAX_REDUCE_DIM | 消除归约循环，改用一次性向量化加载提升性能 |
| 3 | 归约三维tile转二维tile候选 | 归约计算作用于三维 tile，且可证明降维等价 | 生成 BLOCK 首维为 1 的候选；仅实测更快时保留 |
| 4 | for循环消除优化 | `@triton.heuristics`中已将循环分块参数设置为循环边界 | 消除只执行一次的分块循环，降低控制流与编译分析开销 |
| 5 | 冗余访存消除优化 | for循环消除后存在多个等价 `tl.load`，且中间无可能改写同一内存的操作 | 复用首次加载结果，减少全局内存读取与重复索引分析 |
| 6 | 全维度Reduce改写Load + Reduce + Atomic模式优化 | 全维度reduce（dim=None或等效全维度规约）当前未采用单kernel + load + reduce + atomic模式 | 消除多kernel启动和中间buffer开销，统一改写为单kernel分块加载、局部规约、原子写标量输出 |
| 7 | layout变换消除 | wrapper函数输入做了layout变换后，再输入给triton | permute融合进triton kernel以减少kernel间的数据搬运开销 |

> 执行逻辑：按**执行顺序**依次对每个策略进行优化准入校验。若通过，则执行该策略的优化流程，优化完成后**继续**检测下一策略；若不通过，则直接检测下一策略。若所有策略均未通过，则返回原始代码。

## 综合优化执行流程
1. **输入准备**：接收包含Triton kernel定义、wrapper函数、测试数据规格及测试逻辑的完整Python脚本。
2. **优化策略遍历检测（按顺序依次处理每个优化策略）**：
   - **优化准入校验**：检测当前 kernel 是否满足该策略的准入条件。
   - **优化执行**：若通过准入校验，则执行该策略的优化流程；若不通过，则跳过。
   - 完成当前策略后，继续检测下一个策略。
3. **统一结果测试**：所有优化策略应用完毕后，运行精度测试与性能测试对最终 kernel 进行验证。

**重要提醒**
- 任一策略若通过准入校验，就必须立即执行其优化流程，不可因已有其他优化而跳过。
- 各优化策略仅负责代码改写，不得自行执行编译或运行时验证，所有测试统一在**统一结果测试**阶段进行。
- 策略 1 与策略 3 来自旧后端经验，在 CUDA 上不是普遍规律。统一结果测试必须分别做等输入 A/B；精度通过但没有稳定性能提升也要回退该候选。
## 策略1：while循环转for循环优化

### 优化原理
CUDA Triton 对 `while` 与 `for` 的生成质量取决于具体版本、边界和循环体，不能静态断言 `for` 一定更快。若 `while` 本质上是固定步长计数循环，可生成语义等价的 `for` 候选；只有在目标 RTX 3090 上实测稳定更快才保留。

### 功能说明
本优化策略针对Triton kernel做分析，自动完成以下操作：
1. 准入校验：识别**可确定分析为固定步长的计数型 while 循环**，即循环变量、边界、步长能被识别，循环变量在每轮迭代中按固定步长单调推进，且循环体内不存在会改变循环次数语义的控制流。若不存在满足上述条件的 `while` 循环，则跳过该优化策略。
2. 核心优化：将满足条件的 `while` 循环改写为语义等价的 `for` 循环，并删除原循环体中的循环变量更新语句，保证计算结果与原代码完全一致。

### 优化准入校验

遍历 kernel 函数体，判断是否存在可确定分析为固定步长的计数型 `while` 循环。

#### 检测方法
1. **识别循环条件**：循环条件必须能够解析为循环变量与边界表达式之间的单调比较关系，并能确定比较方向是正向收敛还是反向收敛。
2. **识别循环变量来源**：循环变量必须在进入循环前具有可确定的起始表达式，且该起始表达式在循环改写过程中可以作为 `for` 循环的起点。
3. **识别循环变量更新**：循环体内必须存在对同一循环变量的更新操作，且该更新操作能够抽象为固定步长的增量或减量。
4. **确认固定净步长**：必须能确定每轮迭代中循环变量的净变化量固定不变。若循环体中存在多处更新，必须能合并为确定的固定净步长。
5. **匹配比较方向与步长方向**：循环条件的收敛方向必须与固定净步长方向一致，确保循环变量每轮迭代都朝终止边界推进；若方向不一致，则跳过该循环。
6. **确认循环变量用途**：若循环体后续语句依赖循环结束后的循环变量最终值，则不优化该循环，避免改变循环后变量值语义。

#### 判断原则
仅当循环变量初值、边界、比较方向和固定净步长均可确定，循环变量每轮按同一方向单调推进，循环结束后的循环变量最终值不被后续语句依赖，并且能够证明该 `while` 循环可以被安全、等价地改写为 `for` 循环时，才允许执行优化。若上述任一条件不满足，或存在会改变循环执行次数的控制流，则跳过该循环。
#### 结果判断
- 如果检测到至少一个可优化的计数型 `while` 循环，记录循环位置、循环变量名、起始表达式、边界表达式、比较操作符、净步长表达式，并进入优化执行流程。
- 如果未检测到可优化循环，则跳过该优化策略。

### 优化执行流程

#### 步骤1：生成等价 for 循环
针对每个通过准入校验的 `while` 循环执行以下改写：
1. **保留循环前变量定义**：保留循环起始表达式、边界表达式和步长表达式相关的定义语句，避免影响循环体外或后续代码对这些变量的使用。
2. **计算 for 循环边界**：根据原 `while` 循环的比较方向、循环变量等于终止边界时是否仍会执行循环体以及固定净步长，推导等价 `for` 循环的起点、终点和步长；若需要为保持边界语义调整终点表达式，必须确保调整后的表达式可安全生成。
3. **替换循环外壳**：将 `while` 控制结构改写为使用同一循环变量、等价起点、等价终点和等价步长的 `for` 控制结构，保证循环变量的取值序列与原 `while` 完全一致。
4. **删除循环变量更新语句**：删除原循环体内用于推进循环变量的固定步长更新语句。若循环体中存在多处可合并的固定更新，删除这些更新语句，并确保删除后循环体其余语句顺序不变。
5. **保持循环体语义**：除删除循环变量更新语句外，循环体内其余语句顺序、缩进、变量名、访存、mask、归约操作均保持不变。
6. **多循环处理**：若 kernel 内存在多个互不嵌套的计数型 `while` 循环，逐个改写；若存在嵌套 `while`，仅改写可以独立证明等价的循环。

**重要提醒**
- 仅修改必要代码。

### 优化示例（while循环转for循环优化）

#### 原始代码
```python
@triton.jit
def kernel(x_ptr, out_ptr, N, stride, BLOCK_N: tl.constexpr):
    output_idx = tl.program_id(0)
    result = 0.0
    off = 0
    while off < N:
        offsets = off + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + output_idx * stride + offsets, mask=mask, other=0.0)
        result += tl.sum(x)
        off += BLOCK_N

    tl.store(out_ptr + output_idx, result)
```
#### 优化后代码
```python
@triton.jit
def kernel(x_ptr, out_ptr, N, stride, BLOCK_N: tl.constexpr):
    output_idx = tl.program_id(0)
    result = 0.0
    off = 0
    for off in range(off, N, BLOCK_N):
        offsets = off + tl.arange(0, BLOCK_N)
        mask = offsets < N
        x = tl.load(x_ptr + output_idx * stride + offsets, mask=mask, other=0.0)
        result += tl.sum(x)

    tl.store(out_ptr + output_idx, result)
```

#### 示例说明：
1. 循环变量 `off` 的初值为 `0`，边界为 `N`，步长为 `BLOCK_N`。
2. 循环条件 `off < N` 与更新 `off += BLOCK_N` 构成正向固定步长计数循环，且循环结束后的 `off` 最终值不被后续语句依赖。
3. 改写为 `for off in range(off, N, BLOCK_N)` 后，每轮 `off` 取值序列与原 `while` 完全一致，循环体内的 load、归约累加的语义保持不变。

## 策略2：归约轴循环优化

### 优化原理
本优化策略针对「存在归约轴循环且归约轴大小 ≤ MAX_REDUCE_DIM」的场景，直接消除该循环，改用一次性向量化加载。这种方式可以降低编译器的分析复杂度，同时当 kernel 内存在多个 reduce loop 时，消除循环后有可能对部分访存进行合并。

### 功能说明
该优化用于分析输入的 Triton kernel 代码，自动检测归约轴上的显式循环计算。当能推导出归约轴大小且该大小不超过 MAX_REDUCE_DIM 时，自动生成并输出优化后的 kernel（消除循环，改用一次性向量化加载），其余代码保持不变；若归约轴过大或无法提取，则返回原始实现。

### 优化准入校验

#### 步骤1：检测归约轴循环
首先分析输入的Triton kernel，判断是否存在归约轴上的显式循环计算，若存在则记录循环位置信息。

##### 检测方法
1. **识别潜在归约循环**：观察kernel中是否存在循环结构，循环边界通常由归约轴大小相关的参数决定。
2. **分析循环体操作**：检查循环体内是否包含与循环变量线性相关的加载操作以及归约操作，且循环结束后中间结果变量被写入输出。
3. **验证循环变量关联性**：确认循环变量用于索引归约轴上的不同位置，加载属于同一个输出位置的归约集合元素（典型形式：`ptr + i * stride_reduce`）。
4. **区分归约与非归约循环**：排除仅用于遍历输出元素的循环，以及完全由向量化指令完成的归约。
5. **处理分块归约**：即使循环以块为单位遍历归约轴（如`for start in range(0, reduce_size, BLOCK_SIZE):`），只要执行归约操作，仍判定为存在归约循环。

##### 判断原则
- **存在归约轴循环**：kernel中包含至少一个循环，其迭代范围覆盖归约轴（或分块覆盖），循环变量用于索引归约轴，并在循环体内执行归约操作。
- **不存在归约轴循环**：kernel中无上述循环；或循环变量不参与归约轴索引；或归约完全通过向量化指令一次性完成。

##### 结果判断
根据步骤1的检测结果：
- 如果检测到**存在归约轴循环**，记录循环位置信息，并继续执行步骤2。
- 如果检测到**不存在归约轴循环**，则终止流程，直接返回原始代码。
#### 步骤2：提取归约轴大小并判断
若步骤1检测到存在归约轴循环，则进入本步骤，尝试**提取归约轴大小**，并决定是否生成优化 kernel。

##### 2.1 提取测试数据规格
从给定的测试数据规格中提取输入 tensor 形状、数据类型等信息，供后续步骤使用。

##### 2.2 定位归约轴并根据测试数据规格提取归约轴大小
- **定位归约轴**：利用步骤1记录的循环位置，确定归约轴对应的参数名或维度索引（如 `dim`）。
- **从测试规格提取大小**：在测试脚本中查找 wrapper 调用处，获取输入张量的固定形状或直接传入的常量值，计算得出归约轴具体数值。

##### 2.3 决策规则
- 预设阈值 MAX_REDUCE_DIM 为 16384（经验值）。
- 若提取的归约轴大小 ≤ MAX_REDUCE_DIM，则进入**优化执行流程**，生成优化后的 kernel。
- 若归约轴大小 > MAX_REDUCE_DIM 或无法从测试规格中提取，则终止优化流程，直接输出原始代码。

### 优化执行流程

#### 步骤1：生成优化kernel

##### 1.1 定位归约轴循环并识别归约轴和分块参数名
- **定位归约轴循环**：直接使用**优化准入校验**时记录的归约轴循环位置信息，定位 kernel 中遍历归约轴的 for 循环。
- **识别归约轴参数名**：确定归约循环依赖的归约轴大小参数名（如 `reduce_size`、`N`、`K`），或其在 wrapper 中对应的维度索引（如 `dim` 参数）。
- **识别分块参数名**：分析归约循环的分块参数名，记录分块步长对应的参数名，例如：
  - `for start in range(0, reduce_size, BLOCK_SIZE):` → 分块参数名为 `BLOCK_SIZE`
  - `for n in range(0, N, BN):` → 分块参数名为 `BN`
  ##### 1.2 创建优化 kernel
- 基于原始 kernel 定义，进行如下修改，生成优化版本（**函数名保持与原始一致**，以直接替换）：
  1. 处理 `@triton.heuristics` 装饰器
    - 检查原始 kernel 是否已有 `@triton.heuristics` 装饰器。
    - **如果没有**：直接添加新的装饰器：`@triton.heuristics({分块参数名: lambda args: triton.next_power_of_2(args[归约轴大小参数名])})`。
    - **如果有**：需要**合并**原有的 heuristic 配置与新配置。
      - 提取原有装饰器中的配置字典（注意支持 `values` 关键字或直接字典两种形式）。
      - 如果原有字典中已经存在 `分块参数名` 键，且其值已经是 `lambda args: triton.next_power_of_2(args[归约轴大小参数名])`，则无需修改；否则，更新该键的值。
      - 将新配置合并到原有字典中（新配置优先，覆盖同名键）。
      - 生成新的装饰器，保持与原代码相同的调用风格（如果原来使用了 `values` 关键字，则继续使用；否则使用直接字典形式）。
      - 如果原有装饰器还使用了其他参数（如 `values` 之外的关键字），应保留。

  2. 处理 `@triton.autotune` 装饰器（若存在）
    - 检查原始 kernel 是否也使用了 `@triton.autotune` 装饰器。
    - 如果存在 `@triton.autotune` 且其配置中包含了**分块参数名**的搜索空间，则需要对该配置进行修改：
      - 提取 autotune 的配置（通常为 `configs` 列表或 `key` 参数指定的可调参数列表）。
      - 遍历配置，移除其中针对分块参数名的条目（即不再对该参数进行自动调优）。
      - 注意保留 autotune 的其他参数（如 `key`、`prune_configs_by`、`warmup`、`rep` 等）不变。
    - 修改后的 `@triton.autotune` 装饰器仍应放置在优化 kernel 上，且通常位于 `@triton.heuristics` 之上（保持与原始代码相同的装饰器顺序）。
    - 如果 autotune 配置中已不存在任何可调参数（所有参数均被移除），则可考虑直接移除 `@triton.autotune` 装饰器。

  3. 修改 kernel 函数体
    - **移除整个归约循环**，将循环体内的向量化加载代码提升到循环外，直接使用 `tl.arange(0, 分块参数名)` 生成偏移。
    - 使用一次性 `tl.load` 加载所有归约元素，然后使用对应的归约函数（如 `tl.sum`、`tl.max`）计算最终结果。
    - 删除原中间结果变量和循环。
    ##### 1.3 修改 wrapper 函数
- 定位 wrapper 函数中调用优化后 kernel 的位置，如果归约轴分块参数使用的是关键字参数形式，则移除已通过 heuristic 绑定的归约轴分块参数。
- 确保仅在归约轴分块参数使用的是关键字参数形式时才移除，避免因位置参数错位导致运行时错误，且其余所有参数必须完整保留，参数顺序、关键字参数形式均不做任何修改。

**重要提醒**
- 必须使用 heuristic 设置分块参数，优化后的 kernel **必须** 通过 `@triton.heuristics` 将分块参数设置为不小于归约轴大小的 2 的幂，并保留边界 mask。
- 若kernel内有多个归约轴循环，对每个归约轴循环都进行修改。
- 若存在 `@triton.autotune`，必须确保其不再包含对分块参数名的调优，避免与 heuristic 的静态设定冲突。
- 仅修改必要代码。
- 严禁在优化执行完成后立即执行编译或运行时验证，所有测试必须严格推迟至**统一结果测试**阶段集中处理。

### 优化示例（归约轴循环优化）

#### 原始代码（存在分块归约循环的场景）
```python
@triton.jit
def sum_kernel_blocked(x_ptr, out_ptr, reduce_size, stride_reduce, stride_out, BK: tl.constexpr):
    output_idx = tl.program_id(0)
    acc = 0.0
    for start in range(0, reduce_size, BK):               # 归约轴循环
        offsets = start + tl.arange(0, BK)
        mask = offsets < reduce_size
        x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, acc)

def wrapper(x, dim):
    reduce_size = x.shape[dim]
    ...                                                   # stride 计算等
    sum_kernel_blocked[grid](x, out, reduce_size, stride_reduce, stride_out, BK=128)
    return out

x = torch.randn(32, 1024, device='cuda', dtype=torch.float32)
out = wrapper(x, dim=1)
```

#### 优化后代码
```python
@triton.heuristics({"BK": lambda args: triton.next_power_of_2(args["reduce_size"])})
@triton.jit
def sum_kernel_blocked(x_ptr, out_ptr, reduce_size, stride_reduce, stride_out, BK: tl.constexpr):
    output_idx = tl.program_id(0)
    offsets = tl.arange(0, BK)                            # 循环消除，直接向量化全轴
    mask = offsets < reduce_size
    x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, result)

def wrapper(x, dim):
    reduce_size = x.shape[dim]
    ...
    # 直接调用优化 kernel，无需传递 BK 参数（由 heuristic 设置）
    sum_kernel_blocked[grid](x, out, reduce_size, stride_reduce, stride_out)
    return out

x = torch.randn(32, 1024, device='cuda', dtype=torch.float32)
out = wrapper(x, dim=1)
```

#### 示例说明：
1. 归约轴识别与分块参数提取：循环 `for start in range(0, reduce_size, BK)` 是典型的分块归约循环，`reduce_size` 为归约轴大小，`BK` 为分块参数名。归约操作为求和，中间结果变量 `acc` 跨块累加。
2. 归约轴大小提取与阈值判断：测试数据中 `x.shape = (32, 1024)`，`dim=1`，因此归约轴大小 `reduce_size = 1024`。由于 `1024 ≤ 16384 = MAX_REDUCE_DIM`，触发优化流程。
3. heuristics 添加：新增 `@triton.heuristics({"BK": lambda args: triton.next_power_of_2(args["reduce_size"])})`，将 `BK` 设置为覆盖归约轴的合法 2 的幂 block 长度，并由 mask 屏蔽尾部。
4. Kernel 函数体优化：消除 `for` 循环，直接使用 `tl.arange(0, BK)` 生成全轴偏移，一次性向量化加载并归约。

归约轴循环优化的更多示例见[references/reduce_dim_loop_opt_example.md](references/reduce_dim_loop_opt_example.md)

## 策略3：归约三维tile转二维tile优化

### 优化原理
CUDA 后端没有“三维 tile 沿中间维归约必然低效”的通用规则。对三维 tile 可把非归约首维 BLOCK 设为 1，生成语义等价的二维候选，以比较访存合并、寄存器压力与 occupancy；只有在目标 RTX 3090 上实测稳定更快才保留，不得把该改写作为强制规则。
### 功能说明
本优化策略针对Triton kernel做分析，自动完成以下操作：
1. 准入校验：识别**归约计算作用于三维tile + 沿中间维度(axis=1)归约**的归约操作，若不满足优化特征，直接返回原始代码。
2. 核心优化：通过`@triton.heuristics`将非归约首维度的分块参数设置为1，移除该维度的并行分块逻辑，改为标量处理，使得归约操作降维为「归约轴维度 + 非归约尾维度」的二维tile计算。
3. 逻辑适配：修改 kernel 内部非归约首维度的索引生成、边界判断、访存偏移、mask 规则等，适配分块大小为 1 的标量计算，保证数学结果与原代码完全等价。

### 优化准入校验
分析输入的Triton kernel，检测是否存在**归约操作作用于三维tile，且归约轴为该三维tile的中间维度（axis=1）**：
1. kernel中存在归约操作，其作用于三维tile。
2. 确认归约操作沿该三维tile的中间维度（axis=1）执行。
- 仅当同时满足上述2项时判断为符合优化条件，进入优化执行流程，否则终止优化，返回原始代码。

**重要提醒**
- 三维tile中的各个维度长度可以是分块大小，也可以是对应维度的全长。

### 优化执行流程

#### 步骤1：关键信息提取
提取优化所需核心信息：
1. 分块信息：三维tile在三个维度上的大小、定义方式、分块参数名。
2. 归约信息：算子类型、原归约轴、输入/输出张量名、keepdim参数值、附加参数。
3. 循环信息：外层遍历循环的变量、范围、步长；程序ID映射逻辑。
4. 访存信息：tl.load/tl.store的基址、偏移公式、mask规则、stride引用。
5. 中间结果变量信息：中间结果变量是归约计算中承载累加 / 聚合结果的核心变量，记录中间结果变量的初始化形状、数据类型。
6. 维度分类：
  - 非归约首维度：三维tile中「归约轴（axis=1）左侧」的维度，对应的分块参数将被设为 1。
  - 归约轴维度：三维tile中 axis=1 对应的维度，是归约操作要消除的维度。
  - 非归约尾维度：三维tile中「归约轴（axis=1）右侧」的维度，是归约后保留的维度。

#### 步骤2：优化kernel生成
**核心约束**：函数名兼容、数学结果完全等价，核心修改如下：
##### 2.1 分块参数的 heuristics 配置
- 检查原始 kernel 是否已有@triton.heuristics装饰器：
  - 无装饰器：新增`@triton.heuristics({非归约首维度分块参数名: lambda args: 1})`。
  - 有装饰器：合并原有配置与新配置（新配置优先覆盖同名键），保持原装饰器调用风格（如保留values关键字）。
- 确保非归约首维度的分块参数被强制设置为 1。
- 定位 wrapper 函数中调用 kernel 的位置，如果非归约首维度分块参数使用的是关键字参数形式，移除已通过 heuristics 设置的非归约首维度分块参数。
- 确保仅在非归约首维度分块参数使用的是关键字参数形式时才移除，避免因位置参数错位导致运行时错误。

##### 2.2 非归约首维度的索引重构（标量化）
1. 非归约首维度：
   - 将原非归约首维度的并行偏移生成逻辑替换为标量偏移计算。
   - 移除该维度的向量广播逻辑，改为标量直接参与地址计算。
2. 归约轴维度、非归约尾维度：
   - 完全保留原有的`tl.arange`并行偏移生成逻辑。
   - 保留原维度的广播规则，仅适配降维后的张量形状。

##### 2.3 中间结果变量重构
1. 中间结果变量识别：定位归约计算中承载累加 / 聚合结果的核心变量。
2. 移除中间结果变量中对应非归约首维度的轴：
  - 原中间结果变量形状（如`[非归约首维度， 非归约尾维度]`）→ 优化后形状（如`[非归约尾维度]`）。
  - 归约操作`keepdim=true`时，保留归约轴的size=1维度，保证张量形状与原代码对齐。
3. 中间结果变量初始化适配新形状，删除与非归约首维度相关的广播逻辑。

##### 2.4 边界判断适配
适配标量索引与降维形状，完成边界判断逻辑的重构，规避越界风险：
1. 非归约首维度的边界判断：完全移除非归约首维度的边界判断。
  - **原因说明**：优化后非归约首维度分块参数被 heuristics 强制设为 1，使得该维度的总任务块数与维度全长相等，索引变量天然被约束在合法区间内，标量偏移无越界风险，无需额外判断。
2. 归约轴、非归约尾维度的边界判断：保留原逻辑，仅适配降维后的张量形状调整广播规则。
3. 边界判断验证：确保所有维度边界覆盖完整，无越界访存风险。

##### 2.5 访存逻辑适配
核心原则：移除非归约首维度的广播冗余，适配降维后的张量形状，保证load/store逻辑与原代码数学等价。
1. **偏移重构**：移除非归约首维度的向量广播偏移，改为标量偏移直接参与地址计算，消除广播冗余。
2. **Mask适配**：移除非归约首维度的广播mask维度；归约轴维度、非归约尾维度的边界判断逻辑完全保留，仅适配降维后的张量形状调整广播规则（如从三维广播调整为二维广播），确保覆盖有效数据区域。
3. **兼容保留**：完整保留原load/store的所有参数，保证访存数据范围、计算结果与原代码完全一致。

##### 2.6 归约轴适配
1. 归约操作的 axis 参数适配：原三维 tile 归约轴为 1 → 降维后二维 tile 归约轴为 0；
2. 保留归约操作的所有语义，仅调整 axis 索引以匹配降维后的张量形状。
**重要提醒**
- 仅修改必要代码。

### 优化示例（归约三维tile转二维tile优化）

#### 原始代码（存在外层任务循环的场景）
```python
@triton.jit
def sum_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_m_blocks = tl.cdiv(M, BLOCK_M)
    total_k_blocks = tl.cdiv(K, BLOCK_K)
    total_tasks = total_m_blocks * total_k_blocks  # Total tasks

    for task_idx in range(pid, total_tasks, num_programs):
        # Convert 1D task_idx back to 2D (m_block_idx, k_block_idx)
        m_block_idx = task_idx // total_k_blocks
        k_block_idx = task_idx % total_k_blocks

        m_offset = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        k_offset = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Boundary masks
        mask_m = m_offset < M
        mask_k = k_offset < K

        result_sum = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
            mask_n = n_offset < N

            m_off = m_offset[:, None, None]
            n_off = n_offset[None, :, None]
            k_off = k_offset[None, None, :]

            offset = m_off * (N * K) + n_off * K + k_off  # [BM, BN, BK]

            mask = mask_m[:, None, None] & mask_n[None, :, None] & mask_k[None, None, :]

            inp_vals = tl.load(inp + offset, mask=mask, other=0)
            sum_val = tl.sum(inp_vals, axis=1)

            # Update global sum
            result_sum += sum_val

        out_offset = m_offset[:, None] * K + k_offset[None, :]
        out_mask = mask_m[:, None] & mask_k[None, :]

        tl.store(out + out_offset, result_sum, mask=out_mask)
        def wrapper(inp, dim):
    ...
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_K"]),)
    sum_kernel[grid](inp, out, M, N, K, BLOCK_M = 8, BLOCK_N = 128, BLOCK_K = 32)
```

#### 优化后代码
```python
@triton.heuristics({"BLOCK_M": lambda args: 1}) # 新增heuristics，将非归约首维度分块参数设为1
@triton.jit
def sum_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,  # 保留BLOCK_M参数，由heuristics设为1
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_m_blocks = tl.cdiv(M, BLOCK_M) # BLOCK_M = 1，等价于total_m_blocks = M
    total_k_blocks = tl.cdiv(K, BLOCK_K)
    total_tasks = total_m_blocks * total_k_blocks

    for task_idx in range(pid, total_tasks, num_programs):
        # Convert 1D task_idx back to 2D (m_block_idx, k_block_idx)
        m_block_idx = task_idx // total_k_blocks
        k_block_idx = task_idx % total_k_blocks

        # 非归约首维度标量化：BLOCK_M=1，且由外层循环约束无越界
        m_offset = m_block_idx
        k_offset = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # 仅保留归约轴与非归约尾维度的边界判断
        mask_k = k_offset < K

        # 中间结果变量形状适配：移除BLOCK_M维度
        result_sum = tl.zeros([BLOCK_K], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
            mask_n = n_offset < N

            # 移除m维度的广播，仅保留二维偏移
            n_off = n_offset[:, None]
            k_off = k_offset[None, :]
            # 地址计算：标量m_offset参与，偏移降为二维
            offset = m_offset * (N * K) + n_off * K + k_off  # [BN, BK]

            # Mask简化为二维（移除mask_m）
            mask = mask_n[:, None] & mask_k[None, :]
            inp_vals = tl.load(inp + offset, mask=mask, other=0)
            # 归约轴从1改为0（适配二维tile）
            sum_val = tl.sum(inp_vals, axis=0)
            result_sum += sum_val

        # 存储偏移标量化：移除m维度广播，且无需mask_m
        out_offset = m_offset * K + k_offset  # [BK]
        out_mask = mask_k
        tl.store(out + out_offset, result_sum, mask=out_mask)

def wrapper(inp, dim):
    ...
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(K, meta["BLOCK_K"]),)
    # 优化后调用：移除BLOCK_M参数（由heuristics自动设为1），保留其他参数
    sum_kernel[grid](inp, out, M, N, K, BLOCK_N = 128, BLOCK_K = 32)
```

#### 示例说明：
1. 此例中，归约操作的输入是一个三维tile（各维度大小为 [BLOCK_M, BLOCK_N, BLOCK_K]），归约轴为该tile的中间维度（axis=1），满足优化准入的2项条件（输入为三维tile + 沿中间维度归约），因此进入优化流程。
2. 维度角色明确：`BLOCK_M` 为非归约首维度、`BLOCK_N` 为归约轴维度、`BLOCK_K` 为非归约尾维度。
3. heuristics 配置：新增`@triton.heuristics({"BLOCK_M": lambda args: 1})`，将非归约首维度的分块参数`BLOCK_M`设为 1；
4. 索引体系重构:
  - 非归约首维度：m_offset从向量（`[BLOCK_M]`）改为标量（`m_block_idx`），因外层循环由`tl.cdiv(M, BLOCK_M)`约束，`m_block_idx`天然在`[0, M)`范围内；
  - 归约轴维度、非归约尾维度：完全保留原 `tl.arange` 并行偏移生成逻辑，仅适配降维后的张量形状调整广播规则。
  5. 边界判断适配:
  - 非归约首维度：移除非归约首维度的边界判断，完全删除`mask_m = m_offset < M`及相关逻辑。因优化后`BLOCK_M=1`，`total_m_blocks = tl.cdiv(M, 1) = M`，`task_idx`映射得到的`m_block_idx`取值范围天然为`[0, M-1]`，`m_offset=m_block_idx`不会越界，无越界风险。
  - 归约轴 / 非归约尾维度：保留原`mask_n/mask_k`逻辑，适配降维形状调整广播规则。
6. 中间结果变量重构：
  - 原始形状：`[BLOCK_M, BLOCK_K]` -> 优化后形状 `[BLOCK_K]`（移除非归约首维度`BLOCK_M`，仅保留归约后剩余的非归约尾维度）
7. 访存逻辑适配：
  - 偏移重构：load/store 均移除非归约首维度的向量广播偏移，改为标量`m_offset`直接参与地址计算，load 偏移从 3 维降为 2 维，store 偏移从 2 维降为 1 维。
  - Mask 适配：移除非归约首维度的mask，load mask 简化为二维分块组合，store mask 简化为一维组合。
8. 归约轴适配：归约操作`tl.sum`的 axis 从 1 改为 0，匹配降维后的二维 tile。
9. Wrapper 函数适配：定位 wrapper 函数中 kernel 调用位置，移除已通过 heuristics 设置的非归约首维度分块参数`BLOCK_M = 8`。

归约三维tile转二维tile优化的更多示例见[references/reduce_3d_to_2d_opt_example.md](references/reduce_3d_to_2d_opt_example.md)

## 策略4：for循环消除优化

### 优化原理
本优化策略针对「kernel 中存在分块 for 循环，且该循环的分块参数已在 `@triton.heuristics` 中被设置为循环边界」的场景。由于此时分块步长等于循环边界，循环实际只会执行一次，但编译器仍会按循环结构进行代码生成和分析。直接使用一次性向量化计算替代循环控制流，可以降低编译器分析复杂度和运行时控制流开销，并有可能与其他访存操作进一步合并，提升性能。

### 功能说明
该优化用于分析输入的 Triton kernel 代码，自动检测 `@triton.heuristics` 中对分块参数的设置，并进一步分析 kernel 中所有分块循环。当发现某个循环满足「循环边界参数 == heuristic 设置后的分块步长参数」时，自动消除该循环并保持数学结果等价。若不存在满足条件的循环，直接返回原始代码。

### 优化准入校验

#### 步骤1：提取 heuristic 分块映射
分析输入的 Triton kernel，识别 `@triton.heuristics` 中的分块参数设置，并提取其设置的分块参数映射关系。
##### 结果判断
- 如果存在至少一个有效的 heuristic 分块参数设置，记录映射关系并继续步骤2。
- 如果不存在 `@triton.heuristics`，或 `@triton.heuristics` 中没有有效分块参数设置，则终止流程。

#### 步骤2：分析 kernel 中的所有分块循环
遍历 kernel 函数体中的所有 `for` 循环，判断是否存在可消除的分块循环。

##### 检测方法
1. **识别分块循环结构**：定位以分块参数作为步长、按区间分批遍历某个维度或数据范围的 `for` 循环。
2. **提取循环边界与步长**：从循环范围中提取下界、上界和步长表达式，重点记录作为步长使用的分块参数及其对应的遍历边界。
3. **匹配 heuristic 映射**：判断循环步长参数是否已在 `@triton.heuristics` 中设置为该循环上界：
   - `@triton.heuristics({"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])})` + `for start_n in range(0, N, BLOCK_N)` → 可消除（`BLOCK_N >= N` 且为合法 block 长度）。
   - `@triton.heuristics({"BLOCK_N": lambda args: args["N"]})` + `for start_k in range(0, K, BLOCK_N)` → 不可消除，边界不匹配。
4. **验证循环变量用途**：确认循环变量只用于生成当前分块偏移、mask 或地址表达式，且循环体没有依赖多次迭代累积才正确的副作用。

##### 可消除判断原则
- **可消除**：循环步长参数已通过 heuristic 设置为循环上界，且循环下界为 `0` 或可安全化简为起始边界；循环实际只执行一次。
- **不可消除**：循环步长参数未被 heuristic 设置；或 heuristic 设置目标与循环上界不一致；或循环体存在必须多次迭代才等价的复杂副作用；或无法确定循环实际只执行一次。

##### 结果判断
- 如果检测到至少一个可消除分块循环，记录所有可消除循环的循环位置、循环变量名、循环上界、分块参数名，并进入优化执行流程。
- 如果所有分块循环均不可消除，则终止流程，返回原始代码。
### 优化执行流程

#### 步骤1：分析并改写可消除的分块循环
针对每个可消除循环执行以下改写：
1. **移除 for 循环外壳**：将循环体整体提升到原循环所在位置，保持循环体内部语句顺序不变。
2. **替换循环变量**：将循环体内对循环变量的引用替换为该循环的起始边界，并同步化简由此产生的偏移表达式。
3. **处理中间结果变量**：
   - 分析原循环外的中间结果变量（如 `acc`、`result_sum`）是否仅用于承载该单次循环的结果。
   - 如果能证明删除该变量后语义等价，则删除变量初始化和累加更新，直接将单次归约结果作为最终值（或直接赋值）。
   - 如果无法安全删除，则保留原中间结果变量与更新形式。
4. **多循环处理**：若 kernel 内存在多个可消除循环，逐个改写；若存在嵌套循环，仅消除满足条件的循环，不影响其他循环。

**重要提醒**
- 仅消除可以证明实际只执行一次的分块循环；如果循环下界、上界、步长之间的关系无法静态确认，则直接跳过该循环。
- kernel 入参不变，wrapper 中调用不变（分块参数仍由 heuristic 注入，wrapper 无需传递）。
- 仅修改必要代码。

### 优化示例（for循环消除优化）

#### 原始代码
```python
@triton.heuristics({"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])})
@triton.jit
def sum_kernel(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offset < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        mask_n = n_offset < N
        vals = tl.load(inp + m_offset[:, None] * N + n_offset[None, :],
                       mask=mask_m[:, None] & mask_n[None, :],
                       other=0.0)
        acc += tl.sum(vals, axis=1)

    tl.store(out + m_offset, acc, mask=mask_m)

```
#### 优化后代码
```python
@triton.heuristics({"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])})
@triton.jit
def sum_kernel(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offset < M

    n_offset = tl.arange(0, BLOCK_N)
    mask_n = n_offset < N
    vals = tl.load(inp + m_offset[:, None] * N + n_offset[None, :],
                   mask=mask_m[:, None] & mask_n[None, :],
                   other=0.0)
    result = tl.sum(vals, axis=1)

    tl.store(out + m_offset, result, mask=mask_m)
```

#### 示例说明：
1. heuristics 映射：`BLOCK_N=triton.next_power_of_2(N)`，保证 `BLOCK_N >= N` 且适配 `tl.arange`。
2. 循环识别：循环 `for start_n in range(0, N, BLOCK_N)` 的步长不小于上界，因此循环只执行一次。
3. 循环改写：移除 `for` 循环外壳，将 `start_n` 替换为 `0`，因此 `start_n + tl.arange(0, BLOCK_N)` 化简为 `tl.arange(0, BLOCK_N)`。
4. 中间结果变量删除：`acc` 仅承载单次循环的归约结果，因此删除初始化和累加更新。

## 策略5：冗余访存消除优化

### 优化原理
for循环消除后，原本位于不同 pass 或单次循环体内的访存语句会被展开到同一顺序代码中。若多个 `tl.load` 读取同一个指针、同一个地址表达式，并使用完全等价的 `mask`、`other`、`cache_modifier`、`eviction_policy` 等参数，且两次加载之间没有任何可能改写该内存区域的 `tl.store` 或原子操作，则后续加载可以直接复用首次加载得到的张量变量。

### 功能说明
该优化用于分析输入 Triton kernel 中的 `tl.load` 序列，自动检测可证明等价的重复加载，并删除后续冗余加载语句，将其使用点替换为首次加载结果变量。若无法证明两次加载等价或无法排除中间写入影响，则跳过该候选，不修改代码。

### 优化准入校验

#### 步骤1：收集候选 `tl.load`
遍历 kernel 函数体，记录所有将 `tl.load` 结果保存到变量中的加载语句，提取以下信息：
1. 接收 `tl.load` 返回值的目标变量名。
2. `tl.load` 访问的基址指针与完整地址表达式。
3. `mask`、`other` 以及所有关键字参数。
4. load 语句所在的控制流路径与顺序位置。
5. load 结果变量在后续语句中的使用范围。

#### 步骤2：判断 load 等价性
两个 `tl.load` 仅在满足以下条件时判定为等价：
1. 基址指针完全相同，地址表达式语义等价。
2. `mask` 表达式等价，`other` 值等价。
3. `tl.load` 的其他参数等价，包括但不限于 `cache_modifier`、`eviction_policy`、`volatile`、`boundary_check`、`padding_option`。
4. 两次 load 位于同一个确定执行路径上；若位于不同 `if` 分支、不同循环迭代，或控制流关系不明确，则不优化。
5. 首次 load 的结果变量在第二次 load 使用点之前没有被重新赋值。

#### 步骤3：检查中间写入与别名风险
在首次 load 与后续重复 load 之间检查是否存在可能影响该内存值的操作：
1. 若存在 `tl.store`、`tl.atomic_add`等写操作，且写入基址与重复 load 的基址相同，或无法排除别名关系，则不消除该 load。
2. 若写操作只写入明确不同的输出指针（如 `output_ptr`），且该指针与被加载指针不同，可继续优化。
3. 若 kernel 参数中输入指针和输出指针可能来自同一 tensor，且无法从 wrapper 或调用约束证明无别名，则保守跳过。
4. 若 load 使用 `volatile=True` 或其他要求每次实际访存的语义，直接跳过。

##### 结果判断
- 如果至少发现一组可安全复用的重复 load，则进入优化执行流程。
- 如果所有候选都无法证明等价或存在写入/别名风险，则终止流程，返回当前代码。

### 优化执行流程

#### 步骤1：选择复用变量
对每组等价 load，选择顺序上最早的 load 结果变量作为复用变量。

#### 步骤2：删除冗余 load
删除后续重复的 `tl.load` 赋值语句，并保持其他计算语句顺序不变。不得删除首次 load，也不得移动 load 跨越可能影响其语义的控制流或写操作。

#### 步骤3：清理冗余索引与 mask
删除重复 load 后，如果紧邻该 load 之前存在仅为该 load 服务的重复索引、mask 或 offset 计算，且这些变量已经在首次 load 前以等价形式定义并且后续无其他用途，则可以删除重复定义并复用已有变量。但必须确保这些变量不会被任何不同语义的语句使用，否则应保留重复计算，以保证正确性。

**重要提醒**
- 不要为了制造等价性而重排计算语句；仅删除可证明冗余的 load 及其专属索引/mask。
### 优化示例（冗余访存消除优化）

#### 原始代码（for循环消除后存在重复 load）
```python
@triton.heuristics({"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])})
@triton.jit
def softmax_kernel(
    x_ptr, output_ptr,
    stride_x0, stride_x1,
    stride_o0, stride_o1,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Pass1
    n_idx = tl.arange(0, BLOCK_N)
    x_index = m_idx[:, None] * stride_x0 + n_idx[None, :] * stride_x1
    mask_x = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    x = tl.load(x_ptr + x_index, mask=mask_x, other=0.0)
    row_max = tl.max(x, axis=1)

    # Pass2
    n_idx = tl.arange(0, BLOCK_N)
    x_index = m_idx[:, None] * stride_x0 + n_idx[None, :] * stride_x1
    mask_x = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    x = tl.load(x_ptr + x_index, mask=mask_x, other=0.0)
    row_sum = tl.sum(tl.exp(x - row_max[:, None]), axis=1)

    # Pass3
    n_idx = tl.arange(0, BLOCK_N)
    x_index = m_idx[:, None] * stride_x0 + n_idx[None, :] * stride_x1
    mask_x = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    x = tl.load(x_ptr + x_index, mask=mask_x, other=0.0)
    y = tl.exp(x - row_max[:, None]) / row_sum[:, None]
    output_index = m_idx[:, None] * stride_o0 + n_idx[None, :] * stride_o1
    tl.store(output_ptr + output_index, y, mask=mask_x)
```

#### 优化后代码
```python
@triton.heuristics({"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])})
@triton.jit
def softmax_kernel(
    x_ptr, output_ptr,
    stride_x0, stride_x1,
    stride_o0, stride_o1,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    n_idx = tl.arange(0, BLOCK_N)
    x_index = m_idx[:, None] * stride_x0 + n_idx[None, :] * stride_x1
    mask_x = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    x = tl.load(x_ptr + x_index, mask=mask_x, other=0.0)
    row_max = tl.max(x, axis=1)

    row_sum = tl.sum(tl.exp(x - row_max[:, None]), axis=1)
    y = tl.exp(x - row_max[:, None]) / row_sum[:, None]
    output_index = m_idx[:, None] * stride_o0 + n_idx[None, :] * stride_o1
    tl.store(output_ptr + output_index, y, mask=mask_x)
```

#### 示例说明：
1. 三次 `tl.load(x_ptr + x_index, mask=mask_x, other=0.0)` 的地址、mask、other 完全一致，均读取同一个 `x_ptr` 输入 tile。
2. 三次 load 之间没有写入 `x_ptr` 指向内存的 store 或原子操作，最终 `tl.store` 只写入不同的 `output_ptr`。
3. 第二次、第三次 load 的结果变量名仍为 `x`，可直接删除这两次重复 load，并在 `row_sum` 和 `y` 的计算中复用首次 load 的 `x`。
4. 第二次、第三次 load 前重复计算的 `n_idx`、`x_index`、`mask_x` 与首次定义等价，删除后后续仍可复用首次变量，因此一并删除。

## 策略6：全维度Reduce改写Load + Reduce + Atomic模式优化

### 优化原理
全维度 reduce（指 dim=None 或显式指定对所有维度进行规约，输出标量）在不同代码中存在多种实现方式。本优化将所有可安全转换的全维度 reduce 统一改写为 单 kernel + 分块 load + 块内局部 reduce + atomic 的模式。每个 block 使用 tl.load 加载一段连续数据，在 block 内完成局部 reduce（如 tl.sum、tl.max 等），最后通过 tl.atomic_add / tl.atomic_max / tl.atomic_min 将局部结果原子 merge 到 DRAM 上的输出标量。
### 功能说明
该优化用于分析输入 Python 脚本中的全维度 reduce 实现，自动检测当前是否已采用 单 kernel， load + Reduce + atomic 模式。如果不是，则生成等价的新 kernel 并改写 wrapper，将所有相关逻辑替换为该统一形式。若原始实现已经符合目标模式，则不做任何修改直接返回。优化过程始终保守——**必须确保被改写的代码块只负责全维度 reduce 逻辑，不存在其他混合运算**；无法确认语义或存在原子操作不支持的情况时，回退到原始代码。

### Atomic 指令白名单与规约映射

策略5只能使用以下 Triton atomic 原语，禁止自行创造未列出的 atomic 指令名：

| Triton atomic 原语 | 可对应的全维度规约语义 |
|-------------------|------------------------|
| `tl.atomic_add` | `sum`、`mean` 的求和阶段、`norm`/平方和的求和阶段、`count` 类加和 |
| `tl.atomic_max` | `max` |
| `tl.atomic_min` | `min` |
| `tl.atomic_and` | 整型/布尔 `bitwise_and` 或等价 all 语义 |
| `tl.atomic_or` | 整型/布尔 `bitwise_or` 或等价 any 语义 |

### 优化准入校验

#### 步骤1：检测全维度 reduce 操作并提取语义
分析完整 Python 脚本，查看精度测试或性能测试中的 baseline 代码是否为全维度规约语义，再回到 wrapper 及其调用的 Triton kernel 中确认待优化实现是否对应该全维度 reduce：

- baseline/reference 中显式指定 dim=None（如 torch.sum(x, dim=None)），或者等价地对所有维度进行规约（如逐元素运算后跟 sum() 且无 dim 参数）。
- wrapper 返回标量输出，或 wrapper 调用的自定义 Triton kernel 明确生成标量输出。
- 规约语义必须属于可结合、可交换且支持原子操作的类型。

若 wrapper 调用自定义 Triton kernel，则分析该 kernel 的输出参数和访存范围。只有当 kernel 的语义明确为全维度 reduce（例如 kernel 内部遍历所有输入元素并写单个标量输出）且不包含其他副作用（如修改输入、额外写中间结果等）时，才判定为全规约候选。若无法从 baseline/reference 和 wrapper 实现之间建立明确对应关系，或怀疑 kernel 中混杂其他逻辑，则不进行改写。

#### 步骤2：判断当前实现是否已是 Load+Reduce+Atomic 形式
对 wrapper 调用的 Triton kernel 实现进行检查，确认是否同时满足以下所有条件：
1. 单 kernel 完成全规约：所有规约工作仅由一个 Triton kernel 完成，不存在中间 buffer 或第二个规约 kernel。
2. 分块 load + 局部 reduce：每个 program（block）通过 tl.load 加载一段数据，并执行与最终规约语义对应的局部 reduce（如 tl.sum、tl.max 等）。
3. 原子写标量输出：局部 reduce 的结果通过原子操作写入输出标量指针，例如 `tl.atomic_add(out, partial)`；输出 `out` 在 wrapper 中必须按上方映射表初始化为对应规约的单位元或初值。
若当前实现已满足上述模式，则准入失败，优化流程终止，原样返回代码。
若不满足（包括两级 kernel、使用 tl.store 写入中间数组、直接调用 PyTorch API 等），则准入成功，继续步骤3。

#### 步骤3：检查改写可行性
在生成新 kernel 前，进一步确认：
- 目标硬件支持所需的原子操作，且所需原子操作必须来自上方 Atomic 指令白名单与规约映射；若规约语义无法映射到白名单中的 `tl.atomic_*` 原语，则直接放弃改写。
- 规约操作不存在无法用原子操作表达的复合运算（如 mean 和 norm 会被分解为 kernel 内的 sum/平方和 与 wrapper 的后处理，因此可行）。
- 被改写代码块的逻辑纯净性：通过静态分析（或保守假设）确保原始 wrapper 中从 reduce 开始到输出标量结束的代码段仅执行全维度 reduce 这一任务，不存在分支、额外写操作、或与规约无关的副作用。若发现任何可能“揉杂”其他逻辑的迹象（如 kernel 内部有条件跳转且依赖非规约数据、同时更新多个输出、或 wrapper 在调用 kernel 前后夹杂不可忽略的计算），则视为风险过高，放弃改写。
- 原始 wrapper 中不存在与输出标量相关的别名风险（如输入和输出为同一内存），若无法排除风险，则保守跳过。
一旦通过以上校验，进入优化执行流程。

### 优化执行流程

#### 步骤1：生成 Load+Reduce+Atomic 单 kernel
基于已提取的规约语义，构造一个新的 Triton kernel。要求该 kernel 内部仅通过分块加载、块内局部规约和单次原子写完成全量归约，不再使用中间 buffer 或多个阶段。生成时需选择合适的块大小与累加精度，并确保新名称不与已有符号冲突。

#### 步骤2：改写 wrapper 函数
保留原始 wrapper 中与规约无关的预/后处理逻辑（如 contiguous、设备检查、dtype 转换等），将其规约调用替换为新 kernel 的调用。对需要标量后处理的归约类型（如 mean 或 L2 norm），在 kernel 调用后补充相应的运算，并保证最终返回值的数据类型与原始一致。

#### 步骤3：清理原始实现残留
移除原 wrapper 中为旧规约实现服务的中间张量分配、已废弃的 kernel 启动及对应辅助函数定义。确保最终代码仅包含新生成的 kernel 和修改后的 wrapper，无冗余、无冲突。

**重要提醒**
- 改写必须保证最终返回值 dtype 与原始 wrapper 完全一致，且规约语义不变。
- 输出标量初始化必须匹配所选 atomic 语义：`tl.atomic_add` 使用加法单位元，`tl.atomic_max` 使用最小初值，`tl.atomic_min` 使用最大初值，按位 atomic 使用对应按位单位元；不得把所有场景都改成 `torch.zeros`。
- 所有 host tensor、同步与设备检查使用 CUDA；不得生成非 CUDA backend 或其它设备字符串。

### 优化示例（full_reduce_load_atomic_opt）

#### 原始代码（两级 kernel 实现，sum 规约）
```python
import math
import triton
import triton.language as tl
import torch
@triton.jit
def sum_kernel_stage1(inp, mid, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(inp + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x)
    tl.store(mid + pid, s)

@triton.jit
def sum_kernel_stage2(mid, out, mid_size, BLOCK_M2: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_M2
    offsets = block_start + tl.arange(0, BLOCK_M2)
    mask = offsets < mid_size
    vals = tl.load(mid + offsets, mask=mask, other=0.0)
    partial = tl.sum(vals)
    tl.atomic_add(out, partial)

def wrapper(inp):
    M = inp.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(math.ceil(math.sqrt(M))), 1024)
    mid_size = triton.cdiv(M, BLOCK_SIZE)
    BLOCK_M2 = min(triton.next_power_of_2(mid_size), 1024)
    mid = torch.empty((mid_size,), dtype=torch.float32, device=inp.device)
    out = torch.zeros([], dtype=torch.float32, device=inp.device)
    sum_kernel_stage1[(mid_size,)](inp, mid, M, BLOCK_SIZE)
    sum_kernel_stage2[(triton.cdiv(mid_size, BLOCK_M2),)](mid, out, mid_size, BLOCK_M2)
    return out.to(inp.dtype)
```

#### 优化后代码
```python
import math
import triton
import triton.language as tl
import torch

@triton.jit
def sum_kernel_atomic(inp, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    acc_dtype = tl.float32 if inp.dtype.element_ty == tl.float16 else inp.dtype.element_ty
    x = tl.load(inp + offsets, mask=mask, other=0.0).to(acc_dtype)
    partial = tl.sum(x)
    tl.atomic_add(out, partial)

def wrapper(inp):
    M = inp.numel()
    BLOCK_SIZE = min(triton.next_power_of_2(math.ceil(math.sqrt(M))), 1024)
    grid = triton.cdiv(M, BLOCK_SIZE)
    out_dtype = torch.float32 if inp.dtype in (torch.float16, torch.bfloat16) else inp.dtype
    out = torch.zeros([], dtype=out_dtype, device=inp.device)
    sum_kernel_atomic[(grid,)](inp, out, M, BLOCK_SIZE)
    return out.to(inp.dtype)
```

#### 示例说明：
1. 原始代码使用两级 kernel：sum_kernel_stage1 将数据分块局部求和存入中间数组 mid，sum_kernel_stage2 再对 mid 进行二次规约并通过原子加写入输出标量。
2. 优化检测到当前实现非 Load+Atomic 模式（存在中间 buffer 和两个 kernel），因此触发改写。
3. 新 kernel sum_kernel_atomic 将分块加载、局部求和、原子加合并到单个 kernel 中，消除了中间数组分配和第二个 kernel 的启动开销。
4. Wrapper 中分配 out 为累加精度标量、调用新 kernel 并将结果转回原始 dtype，语义与原 wrapper 完全一致。

## 策略7：layout 变换消除

### 优化原理
在 Triton 实现的归约算子中，wrapper 可能对输入做 layout 变换（`transpose/permute + .contiguous()`）；在 CUDA 上额外搬运也可能掩盖 kernel 收益，kernel 间的
数据搬运往往比trans指令开销更大。本优化通过消除这类冗余转置，直接在 kernel 内部按照原始维度布局进行归约（即改变 kernel 内的 reduce axis），从而避免形状变换带来的开销，提升性能。

**重点提示**：目前仅支持2维reduce。

### 功能说明
该优化用于分析输入 Python 脚本中的 wrapper 函数，检测是否存在可安全融合到 kernel 内部的二维 `transpose`/`permute` 操作或等价自定义函数。若检测到符合条件的冗余转置，则自动将其移除，同时同步修改 kernel 内部的归约轴及相关索引逻辑，保证功能等价。如果当前实现已经是最优形式（不存在冗余转置），或融合存在风险，则不进行任何修改，返回原始代码。
### 优化准入校验

#### 步骤1：检测冗余 transpose/permute 操作
定位 wrapper 函数中可能存在的冗余转置模式：
- `.permute([1, 0])` 或 `.transpose(0, 1)` 等改变二维张量维度顺序的操作；
- 紧随其后的 `.contiguous()` 调用（如有）；
- 或者功能等价的自定义函数（起到相同维度交换作用）。

#### 步骤2：判断当前实现是否已融合或无需优化
- 检查 wrapper 中上述转置操作的输出是否直接或仅经过一元逐元素运算（单输入单输出）后传递给 Triton kernel 进行归约。
- 若 wrapper 中不存在此类转置，或转置与归约之间存在无法忽略的复杂逻辑（如多个分支、写操作等），则判定当前实现无需优化，准入失败，直接返回原始代码。
- 若归约 kernel 已直接作用于原始维度顺序，而 wrapper 仍显式执行了转置，则属于典型的冗余场景，准入成功。

#### 步骤3：检查改写可行性
在决定改写前，进一步确认：
- 输入张量必须为 **2 维**，归约操作作用在其中一个维度（axis=0 或 axis=1）。
- 转置仅交换两个维度，不涉及更高维或复杂的维度排列。
- 归约操作属于常见的可简单改变轴的类型（如 sum、mean、max、min 等），其语义不受轴变换影响。
- 去除转置后，kernel 内部的 data loading 逻辑可安全调整，且不影响边界处理与其他相关逻辑。
- 不存在依赖原始转置结果的后续运算（即 wrapper 中后续代码使用的是归约输出，而非转置中间量）。
若任一条件不满足，视为风险过高，放弃改写。

### 优化执行流程

#### 步骤1：移除冗余转置
在 wrapper 函数中删除目标 `.permute()`/`.transpose()` 语句以及专门为其服务的 `.contiguous()` 调用，或者删除相同功能的自定义函数。确保不误删其他有用的连续化操作。

#### 步骤2：适配 kernel 内部的索引与加载逻辑
若 kernel 内有基于转置后形状的索引计算（例如假设行长度为 N 进行 load），需同步修改为基于原始形状的索引（例如行长度变为 M）。保证每个 program 加载的数据量与逻辑边界与原实现严格等价。
#### 步骤3：调整归约轴参数
将 wrapper 中传递给 kernel 的 `dim`（或 `axis`）参数，以及 kernel 内部 `tl.reduce` 的 `axis` 由转置前的值修改为映射后的正确轴。例如：若原操作流程为 `input.permute(1,0) -> reduce(dim=0)`，则移除 permute 后应将 dim 改为 `1`。

#### 步骤4：验证语义一致性
优化后，归约输出的形状、数值精度及边界行为必须与原始实现完全一致。最终返回值的数据类型也应保持不变。

### 重要提醒
- **严格限制变换**：只能修改归约轴和相关索引，禁止改变归约语义（如将 sum 改为 max）或输出形状。
- **连续性处理**：若原始 kernel 依赖于连续内存假设，移除 `.contiguous()` 后可能需要根据实际情况在新输入上显式保证连续性（如添加 `.contiguous()` 调用或调整 kernel 的 load 策略）。若无法确保，则保守回退。
- **复杂 layout 回退**：对于涉及三维及以上、或使用非常规维度交换（如 `.permute(2,0,1)`）的情况，本优化不适用，必须原样返回。
- **设备适配**：所有 host tensor、同步与设备检查使用 CUDA，目标为 RTX 3090（sm_86）。
- **必须消除 Host Transpose/Permute 搬运**：本策略不以性能提升作为是否保留改写的条件。若二维 reduce 前存在仅用于调整归约轴的 `.transpose()` /
  `.permute()` / 等价自定义函数及 `.contiguous()`，且可通过调整 kernel 索引保持语义正确，则必须融合到 kernel 中。只要最终精度验证通过，就必须保留
  融合结果；禁止因性能无提升或性能回退而恢复 host 侧 transpose/contiguous。

### 优化示例（reduce_trans_fusing）

#### 原始代码（冗余 transpose + reduce）
```python
import triton
import triton.language as tl
import torch

@triton.jit
def reduce_kernel(inp, out, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # 假设输入形状为 (M, N)，在 axis=0 上 reduce
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row = tl.load(inp + pid * N + offsets, mask=mask, other=0.0)
    result = tl.sum(row, axis=0)
    tl.store(out + pid, result)

def wrapper(inp):
    # inp 原始形状为 (N, M)
    inp_t = inp.permute(1, 0).contiguous()   # 形状变为 (M, N)
    out = reduce_kernel(inp_t, N, M, BLOCK_SIZE=128)
    return out
```

#### 优化后代码
```python
import triton
import triton.language as tl
import torch

@triton.jit
def reduce_kernel(inp, out, N, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # 输入形状为 (N, M)，改为在 axis=1 上 reduce（即对每行的 M 个元素求和）
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    row = tl.load(inp + pid * M + offsets, mask=mask, other=0.0)
    result = tl.sum(row, axis=0)
    tl.store(out + pid, result)

def wrapper(inp):
    # 直接传入原始 (N, M) 输入，去掉 permute，dim 由 0 改为 1
    N, M = inp.shape
    out = reduce_kernel(inp, N, M, BLOCK_SIZE=128)
    return out
```

#### 示例说明：
1. 原始 wrapper 通过 permute(1, 0) 将 reduce 轴从 1 变到 0，再调用 kernel 在轴 0 上进行归约，产生冗余数据搬运。
2. 检测到转置紧邻归约，且输入为 2 维，满足融合条件。
3. 优化移除了 permute 和 .contiguous()，将 wrapper 中的 dim 对应关系和 kernel 内的 axis 从 0 改为 1，同时将每行的加载宽度由 N 修改为 M。
4. 优化后的功能与原实现完全一致，消除了不必要的显式转置开销。

## 统一结果测试

在所有优化策略应用完毕后，运行精度测试与性能测试对最终 kernel 进行验证，若测试过程抛出错误或精度不正确，则根据错误信息进行调试修改，最多尝试 3 次，如果 3 次修正后仍抛出错误或无法满足精度要求，则输出原始代码。

**重要说明**：

- 若编译或运行显示 shared-memory/寄存器超限、spilling 或 occupancy 过低，优先调低非归约轴分块参数，再依据 `share/gpu` 规则评估 `num_stages`、`num_warps`；不要硬编码 RTX 3090 资源数值。
