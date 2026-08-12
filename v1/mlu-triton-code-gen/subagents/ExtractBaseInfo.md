# ExtractBaseInfo

## 职责概述

ExtractBaseInfo 是 mlu-triton-code-gen 工作流程的第 1 步 subagent。负责从需求文档中提取结构化信息，一次性完成算子的核心计算公式、输入输出形状、轴信息、连续性信息以及拆分轴的分析。

## 输入

| 来源           | 内容                                      |
| -------------- | ----------------------------------------- |
| Extractor 输出 | `{输出存储路径}/Extractor/requirement.md` |

## 输出

| 输出类型 | 说明                                                                                                                             |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 文件输出 | `{输出存储路径}/KernelGen/step1_base_info.json` - 结构化信息分析结果（包含计算公式、io 形状、轴、连续性、reduce_axes、计算说明） |
| 文件输出 | `{输出存储路径}/KernelGen/step1_io_shapes.json` - 单独保存 `io_shapes` 字段，供后续步骤或外部工具直接使用                         |

**输出格式**：

```json
{
  "step": 1,
  "op_name": "{算子名}",
  "compute_type": "reduction|elementwise|matmul|normalization|others",
  "compute_formula": "Y[n,k] = Σ X[n,m,k]",
  "compute_note": {
    "description": "{用自然语言描述算子的计算逻辑}",
    "torch_impl": "{用 torch 参考实现表达计算内容，例如：Y = X.sum(dim=1)}"
  },
  "io_shapes": {
    "X": {
      "type": "input",
      "axis": ["N", "M", "K"],
      "shape": ["{N}", "{M}", "{K}"],
      "contiguity": [true, false, true]
    },
    "Y": {
      "type": "output",
      "axis": ["N", "K"],
      "shape": ["{N}", "{K}"],
      "contiguity": [true, true]
    }
  },
  "reduce_axes": ["M"]
}
```

## 执行步骤
### 步骤 1：读取需求文档

读取 Extractor 生成的需求文档，提取以下信息：

- 算子名称
- 计算逻辑描述
- 数学公式
- 接口签名（输入/输出参数）
- 输入构造方式（`create_inputs` 函数等）

### 步骤 2：分析计算逻辑与输入输出特征

使用 LLM 分析，一次性提取：

**分析内容**：

- 计算类型：elementwise / reduction / matmul / normalization / others
- 计算公式：数学表达式或伪代码
- 计算说明 `compute_note`：
  - `description`：用自然语言描述该算子的计算逻辑
  - `torch_impl`：用 torch 参考代码表达计算内容
- 归约维度 `reduce_axes`（显式或隐式，参见下文"隐式Reduce场景处理"）
- 每个输入/输出的：
  - `type`：input / output
  - `axis`：逻辑轴名称列表（如 `["N","M","K"]`）
  - `shape`：对应各轴的实际形状（从 `create_inputs` 或需求文档中得到）
  - `contiguity`：每个维度在内存中是否连续

**连续性判断方法**：

- 对于维度 i，如果 `stride[i] == shape[i+1] * stride[i+1]`（最后维度 `stride[-1] == 1`），则为连续
- 如果输入通过 slice（如 `raw_x[:, ::2, :]`）构造，则对应维度为非连续

### 步骤 3：保存结果

1. 将完整分析结果保存到 `{输出存储路径}/KernelGen/step1_base_info.json`
2. 将 `io_shapes` 字段单独抽出，保存到 `{输出存储路径}/KernelGen/step1_io_shapes.json`

**`step1_io_shapes.json` 的格式**：直接保存 `io_shapes` 对象本身（不嵌套在外层字典中），例如：

```json
{
  "X": {
    "type": "input",
    "axis": ["N", "M", "K"],
    "shape": [16, 32, 128],
    "contiguity": [true, false, true]
  },
  "Y": {
    "type": "output",
    "axis": ["N", "K"],
    "shape": [16, 128],
    "contiguity": [true, true]
  }
}
```
**一致性要求**：`step1_io_shapes.json` 的内容必须与 `step1_base_info.json` 中的 `io_shapes` 字段**完全一致**（同源写出，禁止单独编辑其中一个）。

## 隐式Reduce场景处理

### 什么是隐式Reduce

某些算子的输入输出形状一致，但计算逻辑内部需要进行归约操作。例如：

- **Softmax**: 输入 (M, N) → 输出 (M, N)，但在 kernel 内部需要沿 N 轴计算 max 和 sum
- **LayerNorm / Softmax Scale** 等

### 识别方法

通过分析 `compute_formula` / `compute_note` 来判断是否存在隐式 reduce：

**关键特征**：

- compute_formula 中包含 `Σ`、`max`、`sum` 等归约操作符
- 归约操作的对象是输入的某个维度，但输出形状中保留了该维度

**处理**：即使输入输出形状一致，也要将被归约的轴添加到 `reduce_axes` 中。

## 参考场景

### 场景1：Reduce Sum with Non-contiguous Dimension

**输入（需求文档/代码片段）：**

```python
N, M, K = 16, 32, 128

def triton_reduce_sum(X):
    N, M, K = X.shape
    Y = torch.empty((N, K), dtype=X.dtype, device=X.device)
    # ...
    return Y

def create_inputs():
    raw_x = torch.rand(N, 2 * M, K, device='cuda', dtype=torch.float32)
    X = raw_x[:, ::2, :]  # stride_x1 = 2*K, M 轴非连续
    return X
```

**输出：**

```json
{
  "step": 1,
  "op_name": "reduce_sum",
  "compute_type": "reduction",
  "compute_formula": "Y[n,k] = Σ X[n,m,k]",
  "compute_note": {
    "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
    "torch_impl": "Y = X.sum(dim=1)"
  },
  "io_shapes": {
    "X": {
      "type": "input",
      "axis": ["N", "M", "K"],
      "shape": [16, 32, 128],
      "contiguity": [true, false, true]
    },
    "Y": {
      "type": "output",
      "axis": ["N", "K"],
      "shape": [16, 128],
      "contiguity": [true, true]
    }
  },
  "reduce_axes": ["M"]
}
```

### 场景2：Transpose + Elementwise Add

**输入：**

```python
M, N = 128, 64

def triton_transpose_add(A, B):
    M, N = A.shape
    C = torch.empty((N, M), dtype=A.dtype, device=A.device)
    # ...
    return C
```

**输出：**

```json
{
  "step": 1,
  "op_name": "transpose_add",
  "compute_type": "elementwise",
  "compute_formula": "C[n,m] = A.T[n,m] + B[n,m]",
  "compute_note": {
    "description": "先对 A 做转置得到形状 (N, M) 的张量，再与同形状的 B 逐元素相加。",
    "torch_impl": "C = A.transpose(0, 1) + B"
  },
  "io_shapes": {
    "A": {
      "type": "input",
      "axis": ["M", "N"],
      "shape": [128, 64],
      "contiguity": [true, true]
    },
    "B": {
      "type": "input",
      "axis": ["N", "M"],
      "shape": [64, 128],
      "contiguity": [true, true]
    },
    "C": {
      "type": "output",
      "axis": ["N", "M"],
      "shape": [64, 128],
      "contiguity": [true, true]
    }
  },
  "reduce_axes": []
}
```

### 场景3：Broadcasting Elementwise with Non-contiguous Input

**输入：**

```python
M, N, K = 32, 64, 128

def create_inputs():
    raw_a = torch.rand(2 * M, N, K, device='cuda', dtype=torch.float32)
    A = raw_a[::2, :, :]  # M 轴非连续
    B = torch.rand(K, device='cuda', dtype=torch.float32)
    C = torch.rand(M, N, device='cuda', dtype=torch.float32)
    return A, B, C
```

**输出：**

```json
{
  "step": 1,
  "op_name": "broadcast_add",
  "compute_type": "elementwise",
  "compute_formula": "Y[m,n,k] = A[m,n,k] + B[k] + C[m,n]",
  "compute_note": {
    "description": "A 为 3D 张量，B 沿 K 维广播，C 沿 K 维广播到 (M, N, K)，三者逐元素相加。",
    "torch_impl": "Y = A + B.view(1, 1, -1) + C.unsqueeze(-1)"
  },
  "io_shapes": {
    "A": {
      "type": "input",
      "axis": ["M", "N", "K"],
      "shape": [32, 64, 128],
      "contiguity": [false, true, true]
    },
    "B": {
      "type": "input",
      "axis": ["K"],
      "shape": [128],
      "contiguity": [true]
    },
    "C": {
      "type": "input",
      "axis": ["M", "N"],
      "shape": [32, 64],
      "contiguity": [true, true]
    },
    "Y": {
      "type": "output",
      "axis": ["M", "N", "K"],
      "shape": [32, 64, 128],
      "contiguity": [true, true, true]
    }
  },
  "reduce_axes": []
}
```

### 场景4：Softmax（隐式 Reduce）

**输入：**

```python
def triton_softmax(X):
    # X: (M, N)
    Y = torch.empty_like(X)
    # ...
    return Y
```

**输出：**

```json
{
  "step": 1,
  "op_name": "softmax",
  "compute_type": "normalization",
  "compute_formula": "Y[m,n] = exp(X[m,n]) / Σ_n exp(X[m,n])",
  "compute_note": {
    "description": "对每一行沿 N 维计算 softmax：先减去该行最大值，再做 exp 并归一化。输入输出形状相同，但内部包含沿 N 维的隐式 reduce。",
    "torch_impl": "Y = torch.softmax(X, dim=-1)"
  },
  "io_shapes": {
    "X": {
      "type": "input",
      "axis": ["M", "N"],
      "shape": ["M", "N"],
      "contiguity": [true, true]
    },
    "Y": {
      "type": "output",
      "axis": ["M", "N"],
      "shape": ["M", "N"],
      "contiguity": [true, true]
    }
  },
  "reduce_axes": ["N"]
}
```

## 注意事项

### 归约维度的定义

归约维度是指在计算过程中被**消除**的维度。常见类型：

| 操作类型           | 归约维度     | 示例                        |
| ------------------ | ------------ | --------------------------- |
| `sum/max/min/mean` | 指定轴被消除 | `X.sum(dim=1)` → M轴被消除  |
| `matmul`           | 公共维度     | `(A,B) @ (B,C)` → B轴被消除 |

### 归约维度的处理原则

⚠️ **归约维度通常不适合作为拆分轴**：

- 每个输出块需要访问全部归约数据
- 拆分归约维度会增加额外的同步/聚合开销
- 归约操作内部通常需要循环处理

但归约维度可以作为内部 BLOCK 调优（如 matmul 的 BLOCK_K）。

### io_shapes 必须存储实际形状

- `shape` 应该基于 **`create_inputs` 函数创建的实际形状**
- 如果需求文档只给出符号形状，则保留符号名（如 `"M"`, `"N"`）

### 候选拆分轴

- 拆分轴从**输出**张量的维度中提取
- 每个输出维度的名字和大小组成一个候选拆分轴
### contiguity 判断方法

- contiguity 反映**原始输入/输出数据在内存中的连续性**
- 对于维度 i，如果 `stride[i] == shape[i+1] * stride[i+1]`（最后维度判断 `stride[-1] == 1`），则为连续

### 显式 Reduce vs 隐式 Reduce

| 特征             | 显式Reduce                 | 隐式Reduce                                 |
| ---------------- | -------------------------- | ------------------------------------------ |
| 输入输出形状     | 不一致（如 (N,M,K)→(N,K)） | 一致（如 (M,N)→(M,N)）                     |
| reduce_axes 来源 | 从形状差异直接推导         | 从 compute_formula / compute_note 分析推断 |
| 示例             | reduce_sum, reduce_max     | softmax, layer_norm                        |

## 验证方式

| 检查项                | 验证方式                                                | 通过条件                                                                |
| --------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------- |
| 需求文档存在          | 检查文件是否存在                                        | 文件存在且可读                                                          |
| 计算公式有效          | 解析 JSON 格式                                          | 包含 compute_formula 字段                                               |
| compute_note 存在     | 解析 JSON 格式                                          | 包含 description 和 torch_impl                                          |
| io_shapes 完整        | 检查所有指针                                            | 每个指针包含 type, axis, shape, contiguity                              |
| reduce_axes 识别      | 检查 reduce_axes                                        | 对于 reduction/normalization 必须有归约维度                             |
| step1_io_shapes.json  | 与 step1_base_info.json 中 `io_shapes` 字段做内容对比   | 两边内容完全相同（键、顺序、值都一致）                                  |


## 回退机制

| 失败场景                             | 处理方式                       |
| ------------------------------------ | ------------------------------ |
| 需求文档不存在或无法读取             | 返回错误，要求先执行 Extractor |
| 输出 JSON 格式无效                   | 内部重试（最多 3 次）          |
| compute_formula 解析失败             | 内部重试（最多 3 次）          |
| io_shapes 缺少 axis/shape/contiguity | 内部重试（最多 3 次）          |
