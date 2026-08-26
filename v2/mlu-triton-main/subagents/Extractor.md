# Extractor

## 职责概述

Extractor 负责 Triton 算子开发的需求分析阶段。通过验证用户输入的有效性，提取计算逻辑，收集测试数据信息，最终明确算子功能需求并创建 PyTorch 基准实现。

## 输入

| 来源 | 内容 | 格式 |
|------|------|------|
| 用户输入 | 算子功能需求或 Triton 代码 | 文本描述、代码片段、Python 文件 |
| 用户输入 | 输出存储路径（默认为 `output_dir`） | `xxx` |


## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/Extractor/requirement.md` - 需求文档，包含算子语义、计算逻辑、数学公式、接口签名、测试数据规格 |
| 文件输出 | `{输出存储路径}/Extractor/original_code.py` - 仅当输入类型为 `triton` 时生成，保留完整的 Triton kernel + wrapper 代码 |
| 摘要返回 | `is_triton`: `true` 表示输入为完整的 Triton kernel 代码，`false` 表示需求描述或其他 |

## 执行步骤

### 步骤 1：验证用户输入有效性并识别输入类型

检查用户输入是否满足以下条件之一，同时识别输入类型：

#### 1.1 Triton 代码输入识别

**Triton 代码输入需同时满足以下两个条件**：

1. **包含 `@triton.jit` 装饰的 kernel 函数**（至少一个）
2. **包含 wrapper 函数调用 kernel**（至少一个 wrapper 函数内部有 `kernel[grid](...)` 调用）

两个条件缺一不可，否则视为**非 Triton 输入**。

**识别特征**：
- `@triton.jit` 装饰器
- `tl.program_id`、`tl.arange`、`tl.load`、`tl.store` 等 Triton 特定 API
- wrapper 函数中有 `kernel[grid](...)` 或 `kernel[(...)](...)` 调用模式
- 包含 `import triton` 或 `import triton.language as tl`

#### 1.2 需求描述输入

- 明确的计算逻辑描述（不能是"生成一个 Triton code"这样的模糊请求）

#### 1.3 输入类型标记

| 输入类型 | 标记值 | 说明 |
|---------|--------|------|
| Triton 代码 | `triton` | 包含完整的 @triton.jit kernel + wrapper 函数 |
| 非 Triton 输入 | `not_triton` | 需求描述、PyTorch 代码或其他 |

**验证失败处理**：
- 如果输入不满足上述条件，提示用户：
  ```
  ❌ 输入不满足要求，请提供以下之一：

  1. 完整的 Triton kernel 代码（包含 @triton.jit 装饰的函数 + wrapper 函数调用 kernel）
  2. 明确的算子需求描述，包括：
     - 算子的计算逻辑（例如：矩阵乘法、规约操作等）
     - 输入数据类型和形状范围（可选）

  请调整输入后重新提交。
  ```
- 返回用户输入重新分析需求

### 步骤 2：提取计算逻辑

根据输入类型提取计算逻辑：
- 从描述中提取算子名称
- 识别计算逻辑
- 提取数学公式或伪代码
- 当输入是triton code的时候需要根据Kernel实现和kernel wapper联合判断计算逻辑

**示例 1：自然语言输入**

```
用户输入：
"我需要实现一个 softmax 算子，对输入张量的最后一个维度进行 softmax 操作。
输入形状为 (batch_size, seq_len, hidden_dim)，数据类型为 float32。
需要支持 2 组测试数据：
- 第 1 组：(32, 128, 768)
- 第 2 组：(16, 256, 1024)"
提取结果：
- 输入类型：not_triton
- 算子名称：softmax
- 计算逻辑：对指定维度进行 softmax 操作
- 数学公式：softmax(x_i) = exp(x_i) / sum(exp(x_j))
- 输入规格：shapes=[(32, 128, 768), (16, 256, 1024)], dtypes=["float32", "float32"]
- 输出规格：shapes=[(32, 128, 768), (16, 256, 1024)], dtypes=["float32", "float32"]
```

**示例 2：PyTorch 代码输入**

```python
用户输入：
import torch

def my_gelu(x):
    """GELU activation function"""
    return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

# 测试数据
x = torch.randn(32, 128, 768, dtype=torch.float32)
output = my_gelu(x)

提取结果：
- 输入类型：not_triton
- 算子名称：gelu
- 计算逻辑：GELU 激活函数
- 数学公式：gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
- 输入规格：shapes=[(32, 128, 768)], dtypes=["float32"]
- 输出规格：shapes=[(32, 128, 768)], dtypes=["float32"]
```

**示例 3：Triton 代码输入**

```python
用户输入：
import triton
import triton.language as tl
@triton.jit
def reduce_max_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
    max_val = tl.max(x, axis=0)
    tl.store(output_ptr + pid, max_val)

def reduce_max(x):
    output = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    reduce_max_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

提取结果：
- 输入类型：triton
- 算子名称：reduce_max
- 计算逻辑：沿指定维度进行最大值规约
- 数学公式：output = max(input)
- 输入规格：shapes=[(1024, 2048)], dtypes=["float32"]
- 输出规格：shapes=[(1,)], dtypes=["float32"]
- 计算类型：reduction

注：原始代码保存至 `{output_dir}/Extractor/original_code.py`
```

**示例 4：复杂计算逻辑 - LayerNorm（多行公式）**

```
用户输入：
"实现 LayerNorm 算子，对输入张量进行层归一化处理。
输入形状为 (batch_size, seq_len, hidden_dim)，数据类型为 float32。
需要支持 3 组测试数据：
- 第 1 组：(32, 128, 768)
- 第 2 组：(16, 256, 1024)
- 第 3 组：(8, 512, 2048)"

提取结果：
- 输入类型：not_triton
- 算子名称：layer_norm
- 计算逻辑：对输入张量的最后一个维度进行层归一化，包括均值计算、方差计算、归一化和仿射变换
- 数学公式：
  ```
  1. 计算均值：μ = (1/D) * Σ(x_i)，其中 D 为最后一维的大小
  2. 计算方差：σ² = (1/D) * Σ(x_i - μ)²
  3. 归一化：x_norm = (x - μ) / sqrt(σ² + ε)，其中 ε 为小常数（如 1e-5）
  4. 仿射变换：output = γ * x_norm + β，其中 γ 为 scale，β 为 bias
  ```
- 输入规格：
  - shapes=[(32, 128, 768), (16, 256, 1024), (8, 512, 2048)]
  - dtypes=["float32", "float32", "float32"]
  - contiguous=[[True, True, True], [True, True, True], [True, True, True]]
- 输出规格：
  - shapes=[(32, 128, 768), (16, 256, 1024), (8, 512, 2048)]
  - dtypes=["float32", "float32", "float32"]
- 参数规格：
  - scale (γ)：shapes=[(768,), (1024,), (2048,)]，dtypes=["float32", "float32", "float32"]
  - bias (β)：shapes=[(768,), (1024,), (2048,)]，dtypes=["float32", "float32", "float32"]
  - epsilon：1e-5（标量）
- 计算类型：normalization
```

### 步骤 3：收集测试数据信息

从用户输入中提取或推断测试数据规格，使用 List 数据结构统一存储所有信息：

**数据存储结构**：

所有数据信息（形状、类型、连续性等）都使用 List 存储，List 的长度等于测试数据组数。

```python
# 假设有 2 组测试数据
shapes = [
    (1024, 2048),      # 第 1 组数据形状
    (512, 4096)        # 第 2 组数据形状
]

dtypes = [
    "float32",         # 第 1 组数据类型
    "float32"          # 第 2 组数据类型
]

contiguous = [
    [True, True],      # 第 1 组：两个维度都连续
    [True, False]      # 第 2 组：第一维连续，第二维不连续
]
# 其他参数也采用相同的 List 结构
axes = [1, 1]          # 每组数据的 axis 参数
keepdims = [False, False]  # 每组数据的 keepdim 参数
```

**提取内容**：

| 项目 | 数据结构 | 说明 |
|------|--------|------|
| 形状 (shapes) | List[Tuple] | 每组数据的多维形状 |
| 数据类型 (dtypes) | List[str] | 每组数据的元素类型 |
| 连续性 (contiguous) | List[List[bool]] | 每组数据每个维度的连续性 |
| 其他参数 | List | 每组数据的其他参数（axis、keepdim 等） |

**无测试数据规格时的处理**：

如果用户输入中没有提供测试数据规格，需要生成一份测试数据信息，遵循上述存储结构：
- 形状：生成合理的张量形状，如 `(1024, 2048)` 或 `(32, 128, 768)` 等
- 数据类型：默认使用 `"float32"`
- 连续性：默认使用 `[True, True, ...]`（根据形状维度数）
- **注意**：只能生成**一份**测试数据信息（即 shapes 列表长度为 1）
- **数据量要求**：生成的测试数据信息的总数据量（所有维度的元素个数相乘）不能少于 `65536`

**示例**：

```
输入数据规格（2 组测试数据）：

shapes = [(1024, 2048), (512, 4096)]
dtypes = ["float32", "float32"]
contiguous = [[True, True], [True, False]]
axes = [1, 1]
keepdims = [False, False]

说明：
- 第 1 组：shape=(1024, 2048), dtype=float32, 两个维度都连续
- 第 2 组：shape=(512, 4096), dtype=float32, 第一维连续，第二维不连续
```
**连续性说明**：

- **True**：该维度数据在内存中连续存储
- **False**：该维度数据在内存中不连续（存在步长或间隙）

**多维连续性示例**：

```
# 3D 张量的连续性表示
shape = (32, 1024, 2048)
contiguous = [True, True, True]    # 全连续（C-contiguous）
contiguous = [True, True, False]   # 前两维连续，最后一维不连续
contiguous = [False, True, True]   # 第一维不连续，后两维连续
```

### 步骤 4：生成需求文档

根据提取的信息生成需求文档 (`requirement.md`)：

```markdown
# 算子需求文档

## 输入类型
{input_type}  # triton | not_triton

## 算子名称
{算子名}

## 计算逻辑
{计算逻辑描述}

## 数学公式
{数学公式或伪代码}

## 接口签名
{输入/输出参数定义}

## 测试数据规格
{数据形状、类型、范围等}
```

**重要**
**当 input_type 为 triton 时**，额外生成文件 `{输出存储路径}/Extractor/original_code.py`，保留用户输入的完整 Triton 代码（kernel + wrapper）。

### 步骤 5：返回需求分析摘要

将需求分析结果以摘要形式返回给调用方：

```json
{
  "is_triton": true | false
}
```

- `is_triton`: `true` 表示输入为完整的 Triton kernel 代码，`false` 表示需求描述或其他


## 验证方式

### 1. 基础验证

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| 输入类型判断 | 检查是否为创建/优化 Triton 算子的请求 | 用户明确要求创建或优化 Triton 算子 |
| 计算逻辑提取 | 解析代码或描述中的计算逻辑 | 成功识别出具体的计算操作（如 reduce、elementwise 等） |
| 需求文档生成 | 检查 requirement.md 完整性 | 包含计算逻辑、接口签名、测试数据规格（可为 None） |

### 2. 数据一致性验证

**验证规则**：对于每个输入/输出参数，必须满足以下条件：

```
len(shapes) == len(dtypes) == len(contiguous)
```

**检查步骤**：
1. 统计测试数据组数 `N`（由 shapes 列表长度决定）
2. 验证 dtypes 列表长度是否等于 `N`
3. 验证 contiguous 列表长度是否等于 `N`
4. 对于每个参数，重复上述检查

**示例**（✓ 通过）：
```json
{
  "name": "input",
  "shapes": [(1024, 2048), (512, 1024)],
  "dtypes": ["float32", "float32"],
  "contiguous": [true, true]
}
```
**示例**（✗ 失败）：
```json
{
  "name": "input",
  "shapes": [(1024, 2048), (512, 1024)],
  "dtypes": ["float32"],  // 长度不匹配！
  "contiguous": [true, true]
}
```

### 3. 参数赋值完整性验证

**验证规则**：所有参数必须有明确的赋值

**检查步骤**：
1. 遍历 `parameters` 列表中的每个参数
2. 检查 `value` 字段是否为空或 `None`
3. 如果参数无用户指定值，必须提供默认值并添加注释

**参数类型处理**：

| 参数类型 | 默认值策略 | 注释示例 |
|---------|----------|--------|
| scalar（epsilon、threshold 等） | 使用行业标准值 | `# 默认值：标准 LayerNorm epsilon` |
| scalar（维度、大小等） | 根据输入推导 | `# 默认值：从输入形状推导` |
| tensor（scale、bias 等） | 初始化为 1 或 0 | `# 默认值：初始化为全 1` |
| 无法推导的参数 | 返回用户询问 | 触发回退机制 |

**示例**（✓ 通过）：
```json
{
  "parameters": [
    {
      "name": "epsilon",
      "type": "scalar",
      "value": "1e-5",
      "comment": "默认值：标准 LayerNorm epsilon，用于数值稳定性"
    },
    {
      "name": "scale",
      "type": "tensor",
      "value": "shapes=[(768,), (1024,), (2048,)]",
      "comment": "默认值：初始化为全 1，与输入最后一维对应"
    }
  ]
}
```
### 4. 计算公式清晰性验证

**验证规则**：计算公式必须具体、可执行、无歧义

**检查步骤**：
1. 检查 `math_formula` 是否为空或过于模糊
2. 验证公式中的变量定义是否完整
3. 检查是否包含具体的数学表达式或伪代码

**不通过的表达**（✗）：
- "对输入进行某种计算"
- "实现一个复杂的操作"
- "根据输入进行处理"

**通过的表达**（✓）：
- 具体的数学公式：`output = max(input)` 或 `softmax(x_i) = exp(x_i) / sum(exp(x_j))`
- 分步骤的伪代码：
  ```
  1. 计算均值：μ = (1/D) * Σ(x_i)
  2. 计算方差：σ² = (1/D) * Σ(x_i - μ)²
  3. 归一化：x_norm = (x - μ) / sqrt(σ² + ε)
  ```

## 回退机制

### 失败场景与处理流程

| 失败场景 | 触发条件 | 处理方式 | 用户提示 |
|---------|--------|--------|--------|
| **非 Triton 算子请求** | 用户要求不涉及 Triton 算子开发 | 返回用户 | `❌ 当前仅支持 Triton 算子开发。请提供 Triton 算子需求或代码。` |
| **计算逻辑不清晰** | 无法从输入中识别具体的计算操作 | 返回用户，要求补充信息 | `❌ 计算逻辑不清晰。请提供：\n1. 具体的计算公式或伪代码\n2. 或完整的 Triton kernel 代码` |
| **数据一致性错误** | `len(shapes) != len(dtypes)` 或 `len(shapes) != len(contiguous)` | **内部自动修复**（见下文） | 修复成功时无提示，修复失败时返回用户 |
| **参数赋值缺失** | 参数 `value` 为空且无法推导默认值 | **内部自动补齐**（见下文） | 补齐成功时无提示，补齐失败时返回用户 |
| **计算公式过于模糊** | 公式中包含模糊表述或缺少变量定义 | 返回用户，要求提供具体公式 | `❌ 计算公式不够清晰��请提供：\n1. 具体的数学表达式（如 output = f(input)）\n2. 或分步骤的伪代码\n3. 或完整的 Triton kernel 代码` |
| **接口签名不完整** | 缺少必要的输入、输出或参数 | 返回用户，指出缺失部分 | `❌ 接口签名不完整。缺失以下信息：\n{缺失项列表}\n请补充后重新提交。` |
### 内部自动修复策略

#### 1. 数据一致性错误 - 自动修复

**修复原则**：以 `shapes` 列表长度为基准，自动补齐 `dtypes` 和 `contiguous`

**修复步骤**：

1. **确定基准长度**：`N = len(shapes)`

2. **修复 dtypes**：
   - 如果 `len(dtypes) < N`：
     - 若 dtypes 非空，使用最后一个元素重复填充至长度 N
     - 若 dtypes 为空，使用 "float32" 作为默认值填充
   - 如果 `len(dtypes) > N`：截断至长度 N，并记录警告

3. **修复 contiguous**：
   - 如果 `len(contiguous) < N`：
     - 若 contiguous 非空，使用最后一个元素重复填充至长度 N
     - 若 contiguous 为空，使用 `true` 作为默认值填充
   - 如果 `len(contiguous) > N`：截断至长度 N，并记录警告

**修复示例**：

```json
修复前：
{
  "name": "input",
  "shapes": [(1024, 2048), (512, 1024), (256, 512)],
  "dtypes": ["float32"],
  "contiguous": [true]
}

修复后：
{
  "name": "input",
  "shapes": [(1024, 2048), (512, 1024), (256, 512)],
  "dtypes": ["float32", "float32", "float32"],
  "contiguous": [true, true, true],
  "修复记录": "dtypes 和 contiguous 已自动补齐"
}
```

**修复失败条件**：
- 无法确定基准长度（shapes 为空）→ 返回用户

#### 2. 参数赋值缺失 - 自动补齐

**补齐原则**：根据参数类型和上下文自动推导默认值

**补齐策略**：

| 参数类型 | 补齐条件 | 补齐方式 | 示例 |
|---------|--------|--------|------|
| **scalar - epsilon/threshold** | 参数名包含 epsilon、threshold、eps 等 | 使用行业标准值 | epsilon → 1e-5 |
| **scalar - 维度/大小** | 参数名包含 dim、axis、size、num 等 | 从输入形状推导 | dim → 从输入最后一维推导 |
| **scalar - 其他** | 无法推导 | 返回用户询问 | - |
| **tensor - scale/weight** | 参数名包含 scale、weight、gamma 等 | 初始化为全 1 | scale → ones(input_shape[-1]) |
| **tensor - bias/offset** | 参数名包含 bias、offset、beta 等 | 初始化为全 0 | bias → zeros(input_shape[-1]) |
| **tensor - 其他** | 无法推导 | 返回用户询问 | - |

**补齐示例**：

```json
补齐前：
{
  "parameters": [
    {
      "name": "epsilon",
      "type": "scalar",
      "value": ""
    },
    {
      "name": "scale",
      "type": "tensor",
      "value": ""
    }
  ]
}

补齐后：
{
  "parameters": [
    {
      "name": "epsilon",
      "type": "scalar",
      "value": "1e-5",
      "comment": "自动补齐：标准 LayerNorm epsilon，用于数值稳定性"
    },
    {
      "name": "scale",
      "type": "tensor",
      "value": "shapes=[(768,), (1024,), (2048,)]",
      "comment": "自动补齐：初始化为全 1，与输入最后一维对应"
    }
  ]
}
```

**补齐失败条件**：
- 参数类型无法识别
- 参数名称过于模糊，无法推导含义
- 需要用户明确指定的参数 → 返回用户询问

### 修复/补齐失败时的回退处理

当内部自动修复/补齐失败时，按以下流程处理：

```
自动修复/补齐失败
  ↓
确定失败原因
  ↓
生成对应的用户提示
  ↓
返回用户，等待补充信息
  ↓
用户重新提交
  ↓
重新执行验证
```

### 错误提示模板

**通用格式**：
```
❌ [失败类型]

问题描述：
{具体的验证失败原因}

需要补充的信息：
{列出缺失或错误的具体项}

示例：
{提供正确的示例格式}

请调整后重新提交。
```
**示例 1：数据一致性错误**
```
❌ 数据一致性验证失败

问题描述：
参数 'input' 的 shapes、dtypes、contiguous 长度不一致

需要补充的信息：
- shapes 长度：2
- dtypes 长度：1（应为 2）
- contiguous 长度：2

示例（正确格式）：
{
  "name": "input",
  "shapes": [(1024, 2048), (512, 1024)],
  "dtypes": ["float32", "float32"],
  "contiguous": [true, true]
}

请调整后重新提交。
```

**示例 2：参数赋值缺失**
```
❌ 参数赋值不完整

问题描述：
参数 'epsilon' 缺少赋值

需要补充的信息：
- 参数名：epsilon
- 参数类型：scalar
- 参数含义：数值稳定性常数

建议值：1e-5（标准 LayerNorm epsilon）

请提供参数值或确认使用建议值。
```

**示例 3：计算公式不清晰**
```
❌ 计算公式不够清晰

问题描述：
提供的计算公式过于模糊，无法确定具体的计算逻辑

当前公式：
"对输入进行某种处理"

需要补充的信息：
请提供以下之一：
1. 具体的数学表达式（如 output = max(input)）
2. 分步骤的伪代码
3. 完整的 Triton kernel 代码

示例（正确格式）：
数学公式：output = max(input)
或
伪代码：
  for i in range(n):
    output[i] = max(input[i])

请调整后重新提交。
```
