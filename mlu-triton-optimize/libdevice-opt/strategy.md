# libdevice-opt

## 职责概述

本策略文档用于将 Triton kernel 中低效的计算模式替换为 Cambricon libdevice 库中的高效算子。libdevice 是 Cambricon 加速器提供的高性能数学函数库，包含 `fast_*` 和 `ultra_*` 两类算子：

- **fast_\* 算子**：精度更高，优先使用
- **ultra_\* 算子**：性能更高但精度稍低，作为备选方案

调用方式为 `tl.extra.mlu.libdevice.xxx()`，**无需额外导入**。

## 修改部分

@triton.jit 装饰的Triton kernel (使用libdevice替换了原本的计算模式)。

## 可替换模式

| 原始模式 | 替换为 |
|---------|--------|
| `x * tl.sigmoid(x)` | `tl.extra.mlu.libdevice.fast_silu(x)` |
| `tl.sigmoid(x) * (1.0 + x * (1.0 - tl.sigmoid(x)))` | `tl.extra.mlu.libdevice.fast_silubp(x)` |
| `tl.sigmoid(x)` | `tl.extra.mlu.libdevice.fast_sigmoid(x)` |
| `0.5 * x * (1 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))` | `tl.extra.mlu.libdevice.fast_gelu(x)` |
| `tl.tanh(x)` | `tl.extra.mlu.libdevice.fast_tanh(x)` |
| `tl.log(x)` | `tl.extra.mlu.libdevice.fast_log(x)` |
| `tl.exp(x)` | `tl.extra.mlu.libdevice.fast_expf(x)` |
| `0.5 * x * (1 + tl.erf(x/tl.sqrt(2.0)))` | `tl.extra.mlu.libdevice.fast_gelu(x)` |
| `(tl.exp(2.0 * x) - 1.0) / (tl.exp(2.0 * x) + 1.0)` | `tl.extra.mlu.libdevice.fast_tanh(x)` |
| `tl.erf(x)` | `tl.extra.mlu.libdevice.fast_erf(x)` |
| `tl.sqrt(x)` | `tl.extra.mlu.libdevice.fast_sqrt(x)` |
| `1.0 / x` (仅限分子为常量 1.0) | `tl.extra.mlu.libdevice.fast_rcp(x)` |
| `tl.pow(x, y)` 或 `x ** y` | `tl.extra.mlu.libdevice.fast_powf(x, y)` |
| `tl.maximum(x, y)` | `tl.extra.mlu.libdevice.fast_max(x, y)` |
| `tl.minimum(x, y)` | `tl.extra.mlu.libdevice.fast_min(x, y)` |
| `tl.math.pow(x, n)` (仅限 n 为整数) | `tl.extra.mlu.libdevice.fast_powi(x, n)` |
| `tl.exp2(x)` | `tl.extra.mlu.libdevice.fast_expf(0.6931471805599453 * x)` |
| `tl.log2(x)` | `tl.extra.mlu.libdevice.fast_log2f(x)` |
| `tl.exp10(x)` | `tl.extra.mlu.libdevice.fast_exp10f(x)` |
| `tl.log10(x)` | `tl.extra.mlu.libdevice.fast_log10f(x)` |

### ultra 算子备选（精度稍低但性能更高）(低优先级使用)

| 原始模式 | 替换为 |
|---------|--------|
| `tl.extra.mlu.libdevice.fast_gelu(x)` | `tl.extra.mlu.libdevice.ultra_gelu(x)` |
| `tl.extra.mlu.libdevice.fast_sigmoid(x)` | `tl.extra.mlu.libdevice.ultra_sigmoid(x)` |
| `tl.extra.mlu.libdevice.fast_silu(x)` | `tl.extra.mlu.libdevice.ultra_silu(x)` |
| `tl.extra.mlu.libdevice.fast_tanh(x)` | `tl.extra.mlu.libdevice.ultra_tanh(x)` |
| `tl.extra.mlu.libdevice.fast_powf(x, y)` | `tl.extra.mlu.libdevice.ultra_pow(x, y)` |

## 优化工作流

### 1. 基线采集

运行一次原始代码，完成正确性验证和基线采集：

1. **运行原始代码**：执行输入的测试代码，记录 Triton kernel 的性能指标 `original_triton_ms` 和 `original_triton_bandwidth(gb/s)`
   - 如果运行出错，说明原始代码有问题，**终止优化并报告错误内容**
   - 如果运行成功，此次运行的性能指标即为后续性能对比的基准

**注意**：基线是 Triton kernel 自身的执行耗时，而非 Triton vs PyTorch 的对比数据。

### 2. LLM 检测与替换

分析代码并执行替换，首先把输入文件全部复制到输出文件，**后续操作都在输出文件中完成**：

#### 2.1 扫描代码

识别是否存在上述**可替换模式**中可被替换的计算模式。

**参考案例：**

##### 通用案例：GELU 算子优化

算子功能：GELU 激活函数，标准计算公式为 `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`。

原始代码：
```python
@triton.jit
def gelu(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
    x = 0.5 * x * (1 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    ...
```
优化后代码：
```python
@triton.jit
def gelu(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
    x = tl.extra.mlu.libdevice.fast_gelu(x)  # 使用 libdevice 高效算子
    ...
```

关键修改：

| 原代码 | 修改后 |
|--------|--------|
| `0.5 * x * (1 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))` | `tl.extra.mlu.libdevice.fast_gelu(x)` |

补充说明：对于 `0.5 * x * (1 + tl.erf(x/tl.sqrt(2.0)))` 这种 GELU 公式，同样可以替换为 `fast_gelu`。

**注意：特殊案例仅 LeakyReLU（需检查 `negative_slope` 约束）和 fast_rcp（需检查分子为常量 `1.0`）这两种，其余可替换模式均为通用案例，直接按模式匹配即可执行替换，无需额外约束判断。**

##### 特殊案例1：LeakyReLU 算子优化

算子功能：LeakyReLU 激活函数，计算公式为 `x >= 0 ? x : x * negative_slope`，其中 `negative_slope` 为负斜率（通常为 0.01）。当 `negative_slope < 1` 时，若 `x < 0` 则 `x < x * negative_slope`，若 `x >= 0` 则 `x >= x * negative_slope`，因此可用 `fast_max` 优化。

原始代码：
```python
@triton.jit
def leakyrelu_op(x_ptr, y_ptr, n_elements, negative_slope: tl.constexpr, TILE_SIZE: tl.constexpr):
    ...
    o = tl.where(x >= 0, x, x * negative_slope)  # 低效计算模式
    ...
```

优化后代码：
```python
@triton.jit
def leakyrelu_op(x_ptr, y_ptr, n_elements, negative_slope: tl.constexpr, TILE_SIZE: tl.constexpr):
    ...
    o = tl.extra.mlu.libdevice.fast_max(x, x * negative_slope)  # 使用 libdevice 高效算子
    ...
```

关键修改：

| 原代码 | 修改后 |
|--------|--------|
| `tl.where(x >= 0, x, x * negative_slope)` | `tl.extra.mlu.libdevice.fast_max(x, x * negative_slope)` |

约束条件检查：**对于和输入数据的值有关的变换，必须严格检查约束条件。**

以`tl.where(x >= 0, x, x * negative_slope)`为例：
- 当 `negative_slope < 1` 时：
  - 若 `x < 0`：`x < x * negative_slope`
  - 若 `x >= 0`：`x >= x * negative_slope`
  - 因此等价于 `max(x, x * negative_slope)`
- 当 `negative_slope > 1` 时：
  - 若 `x < 0`：`x > x * negative_slope`
  - 若 `x >= 0`：`x < x * negative_slope`
  - 因此等价于 `min(x, x * negative_slope)`
- 当 `negative_slope = 1` 时：
  - `x * negative_slope = x`，此时无需优化

补充说明：对于 `tl.maximum(x, x * negative_slope)` 或其他手动实现的 LeakyReLU 公式，同样可以替换为 `fast_max`。

##### 特殊案例2：fast_rcp 倒数优化

算子功能：计算倒数 `1.0 / x`，使用 `fast_rcp` 高效实现。**注意：仅当分子为字面常量 `1.0` 时才匹配，其他任意表达式作分子均不匹配。**

原始代码：
```python
@triton.jit
def rcp_op(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
    y = 1.0 / x  # 低效计算模式
    ...
```

优化后代码：
```python
@triton.jit
def rcp_op(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
    y = tl.extra.mlu.libdevice.fast_rcp(x)  # 使用 libdevice 高效算子
    ...
```

关键修改：

| 原代码 | 修改后 |
|--------|--------|
| `1.0 / x` | `tl.extra.mlu.libdevice.fast_rcp(x)` |

匹配规则：仅当分子为字面常量 `1.0` 时才匹配：
```python
# ✅ 匹配：分子是 1.0
y = 1.0 / x                    → y = tl.extra.mlu.libdevice.fast_rcp(x)
y = 1.0 / tl.sqrt(x)           → y = tl.extra.mlu.libdevice.fast_rcp(tl.sqrt(x))
y = 1.0 / (x + 1.0)            → y = tl.extra.mlu.libdevice.fast_rcp(x + 1.0)

# ❌ 不匹配：分子不是 1.0
y = 2.0 / x                    → 不替换，分子是 2.0
y = alpha / x                  → 不替换，分子是变量
y = (x + 1.0) / y              → 不替换，分子是表达式
y = x / y                      → 不替换，分子是变量 x
```

#### 2.2 输入类型检查（必须执行）
**在执行任何替换之前，必须先进行输入类型检查。** 匹配到的计算模式的输入**不能全为标量**，至少有一个输入必须是 tensor。

**标量判断规则：**

以下情况视为标量：
1. **字面常量**：如 `0.0`、`1.0`、`2.0`、`0.5` 等
2. **Python 变量**：函数参数、循环变量、局部变量等（如 `margin`、`i`、`scale` 等）
3. **标量的算术运算结果**：标量之间的加减乘除运算结果仍是标量
4. **归约操作的结果**：`tl.sum()`、`tl.max()`、`tl.min()` 等归约操作的输出是否为标量取决于输入维度和 `axis` 参数：
   - 如果未指定 `axis`（即对所有维度归约），结果一定是标量
   - 如果指定了 `axis` 且输入维度等于 1，归约后结果为标量
   - 如果指定了 `axis` 但输入维度大于 1（归约仅消除其中一个维度），结果仍然是 tensor
5. **标量函数的标量输入结果**：如 `tl.sqrt(scalar)`、`tl.exp(scalar)` 等对标量操作的结果

以下情况视为 tensor：
1. **`tl.load()` 的结果**：从全局内存加载的数据
2. **tensor 之间的运算结果**：tensor 之间的加减乘除等运算
3. **`tl.arange()` 的结果**：生成的范围向量
4. **tensor 的切片或索引结果**：对 tensor 进行操作后的结果

**检查示例：**

```python
# 示例1：所有输入都是标量 - 不可替换
pos_sum = 0.0  # 标量
pos_dist = tl.sqrt(pos_sum)  # 输入是标量，不可替换

# 示例2：所有输入都是标量 - 不可替换
margin = 1.0  # 函数参数，标量
loss = tl.maximum(pos_dist - neg_dist + margin, 0.0)  # 所有输入都是标量，不可替换

# 示例3：输入是 tensor - 可以替换
x = tl.load(ptr + offsets, mask)  # tensor
y = tl.sigmoid(x)  # 输入是 tensor，可以替换为 fast_sigmoid

# 示例4：归约结果是标量 - 不可替换
result = tl.sum(x * x)  # 未指定axis，对所有维度归约，返回标量
out = tl.sqrt(result)  # 输入是标量，不可替换

ptr = input_ptr + (m[:, None] * stride_m + n[None, :] * stride_n)
x = tl.load(ptr, mask)  # tensor，维度为2
row_sum = tl.sum(x, axis=0)  # 指定axis=0，输入维度>1，结果仍是tensor
y = tl.sqrt(row_sum)  # 输入是tensor，可以替换为 fast_sqrt

# 示例6：归约结果为标量（输入维度=1）- 不可替换
ptr = input_ptr + tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr, mask)
total = tl.sum(x, axis=0)  # 指定axis=0，输入维度=1，结果为标量
out = tl.sqrt(total)  # 输入是标量，不可替换
```

**执行规则：**
- 如果匹配到的模式的所有输入都是标量，**跳过该模式的替换**
- 只有当至少有一个输入是 tensor 时，才执行替换

#### 2.3 执行替换

将识别的模式替换为对应的 libdevice 算子：
- 优先使用 `fast_*` 算子（如 `fast_silu`、`fast_gelu`）而非 `ultra_*` 版本
- 使用 `tl.extra.mlu.libdevice.xxx()` 方式调用
- 确保替换后的代码语法正确

#### 2.4 复杂匹配优先

尽可能匹配最复杂的模式进行替换。例如，如果代码中同时存在 `x * tl.sigmoid(x)` 和 `tl.sigmoid(x)`，优先替换前者为 `fast_silu(x)`。

#### 2.5 保持不变

其他不需要替换的代码保持原样。

### 3. 验证替换结果

替换完成后，尝试运行输入的代码，并依次执行以下验证：
**检查项1：是否发生了替换**

- 比较替换后的代码与原始代码
- 如果没有任何变化，说明代码中不存在可替换的模式，优化终止

**检查项2：精度验证**

- 使用精度测试，验证替换后的计算结果精度
- 如果精度测试通过（accuracy=True），进入**性能验证**
- 如果精度测试失败，说明替换无法满足精度要求，**还原代码到替换之前**，退回到 LLM 检测与替换，**尝试匹配更加简单的模式**

**检查项3：性能验证**

对比替换前后的 **Triton kernel 自身执行耗时**。使用步骤2中采集的 `original_triton_ms` 作为基线，与替换后的耗时进行对比：

1. **记录替换后耗时**：替换代码后，使用性能测试，测量 Triton kernel 的执行耗时，记为 `opt_triton_ms`
2. **计算加速比**：`speedup = original_triton_ms / opt_triton_ms`
3. **判定规则**：
   - 如果 `speedup < 1`（替换后更慢），则还原代码到原始代码，优化终止
   - 如果 `speedup >= 1`（替换后性能未下降），则保留替换，优化终止

### 4. 输出最终代码

文件内容必须包含 kernel + wrapper + 测试代码的完整可执行脚本：

- 若优化成功（精度通过且性能未下降）：写入优化后的代码
- 若优化失败（精度失败或性能下降）：将输入代码复制为输出代码

