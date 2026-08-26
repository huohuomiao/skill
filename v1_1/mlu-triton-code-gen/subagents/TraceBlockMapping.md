# TraceBlockMapping

## 职责概述

TraceBlockMapping 是 mlu-triton-code-gen 工作流程的第 2 步 subagent。负责追踪输入依赖与块索引，分析每个输入/输出指针的块大小映射关系。

## 输入

| 来源 | 内容 |
|------|------|
| Step 1 输出 | `{输出存储路径}/KernelGen/step1_base_info.json` |

## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/KernelGen/step2_block_mapping.json` - 块索引映射结果 |

## 执行步骤

### 步骤 1：读取 Step 1 结果

读取 step1_base_info.json，获取：
- compute_formula
- compute_note（需透传到输出）
- io_shapes（含 axis, shape, contiguity）
- reduce_axes

### 步骤 2：分析块索引映射

使用 LLM 分析块大小映射关系：

**分析内容**：
- 每个指针的 block_name（各拆分轴对应的块大小）
- 归约维度的处理方式（reduce_dim）
- 连续性信息继承

**输出格式**：
```json
{
    "compute_formula": "Y[n,k] = Σ X[n,m,k]",
    "compute_note": {
        "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
        "torch_impl": "Y = X.sum(dim=1)"
    },
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"N": "BLOCK_N", "K": "BLOCK_K"},
            "axis_size": {"N": [16], "K": [128]},
            "reduce_dim": {"M": "BLOCK_M"},
            "reduce_size": {"M": [32]},
            "contiguity": [true, false, true]
        },
        "Y_ptr": {
            "block_name": {"N": "BLOCK_N", "K": "BLOCK_K"},
            "axis_size": {"N": [16], "K": [128]},
            "contiguity": [true, true]
        }
    }
}
```
> **说明**：`compute_note` 从 step1_base_info.json 原样透传到输出，用于后续步骤理解算子语义。

### 步骤 3：保存结果

将分析结果保存到 `{输出存储路径}/KernelGen/step2_block_mapping.json`

## 参考场景

### 场景1：Reduce Sum

**输入（step1_base_info.json）：**
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
        "X_ptr": {"type": "input", "axis": ["N", "M", "K"], "shape": [16, 32, 128], "contiguity": [true, false, true]},
        "Y_ptr": {"type": "output", "axis": ["N", "K"], "shape": [16, 128], "contiguity": [true, true]}
    },
    "reduce_axes": ["M"]
}
```

**分析过程：**
- 识别方法：`for m in range(M)` 循环遍历 M 轴
- 归约操作：`acc += x`，对 M 轴求和
- 计算公式：Y[N,K] = Σ X[N,M,K]（沿 M 轴求和）
- grid: `(triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))`
- 输出块：(BLOCK_N, BLOCK_K)
- 输入 X 块：(BLOCK_N, M, BLOCK_K) — M 轴需要完整数据

**输出：**
```json
{
    "compute_formula": "Y[n,k] = Σ X[n,m,k]",
    "compute_note": {
        "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
        "torch_impl": "Y = X.sum(dim=1)"
    },
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"N": "BLOCK_N", "K": "BLOCK_K"},
            "axis_size": {"N": [16], "K": [128]},
            "reduce_dim": {"M": "BLOCK_M"},
            "reduce_size": {"M": [32]},
            "contiguity": [true, false, true]
        },
        "Y_ptr": {
            "block_name": {"N": "BLOCK_N", "K": "BLOCK_K"},
            "axis_size": {"N": [16], "K": [128]},
            "contiguity": [true, true]
        }
    }
}
```
### 场景2：Transpose + Elementwise Add

**输入（step1_base_info.json）：**
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
        "A_ptr": {"type": "input", "axis": ["M", "N"], "shape": [128, 64], "contiguity": [true, true]},
        "B_ptr": {"type": "input", "axis": ["N", "M"], "shape": [64, 128], "contiguity": [true, true]},
        "C_ptr": {"type": "output", "axis": ["N", "M"], "shape": [64, 128], "contiguity": [true, true]}
    },
    "reduce_axes": []
}
```

**分析过程：**
- 无归约操作
- A.T 通过索引交换实现转置
- 计算：C[N,M] = A.T[N,M] + B[N,M] = A[M,N] + B[N,M]
- grid: `(triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))`

**输出：**
```json
{
    "compute_formula": "C[n,m] = A.T[n,m] + B[n,m]",
    "compute_note": {
        "description": "先对 A 做转置得到形状 (N, M) 的张量，再与同形状的 B 逐元素相加。",
        "torch_impl": "C = A.transpose(0, 1) + B"
    },
    "io_block_mapping": {
        "A_ptr": {
            "block_name": {"M": "BLOCK_M", "N": "BLOCK_N"},
            "axis_size": {"M": [128], "N": [64]},
            "contiguity": [true, true]
        },
        "B_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [64], "M": [128]},
            "contiguity": [true, true]
        },
        "C_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [64], "M": [128]},
            "contiguity": [true, true]
        }
    }
}
```

## 计算逻辑分析方法论

本步骤的输入只有 `step1_base_info.json`（不含原始代码）。所需的计算语义已经由 Step 1 提炼到以下字段中，直接基于这些字段进行分析：

- `compute_formula`：数学公式，用于识别归约/转置/广播等操作
- `compute_note.description` / `compute_note.torch_impl`：算子的自然语言描述和 torch 参考实现
- `io_shapes`：各输入输出的 `axis`、`shape`、`contiguity`
- `reduce_axes`：显式或隐式的归约轴（Step 1 已完成识别）

### 分析流程

1. **确定拆分轴**：从输出张量的 `axis` 中取非 reduce 轴作为候选拆分轴，每个拆分轴分配一个 `BLOCK_*` 参数。
2. **映射到每个指针**：根据该指针的 `axis`，为其在每个拆分轴上的出现填入对应的 `BLOCK_*`；未出现的轴不进入 `block_name`（广播场景）。
3. **处理归约轴**：将 `reduce_axes` 中的轴写入涉及到该轴的指针的 `reduce_dim`，并为其分配 `BLOCK_*`（例如 `BLOCK_M`）。归约轴的 BLOCK 用于 kernel 内部循环或 `tl.sum/tl.max` 覆盖完整维度。
4. **继承 contiguity**：直接沿用 `io_shapes` 中每个指针的 `contiguity`。
5. **填写 axis_size / reduce_size**：从 `io_shapes.shape` 中按轴名取出对应大小，写成列表形式（对应多测试用例）。

### 操作类型的识别线索

通过 `compute_formula` 即可区分常见类型：

| 类型 | 公式特征 | 示例 |
|------|---------|------|
| 归约（显式） | 含 `Σ`/`max`/`mean` 且输出维度比输入少 | `Y[n,k] = Σ X[n,m,k]` |
| 归约（隐式） | 含 `Σ`/`max`/`sum` 但输入输出形状一致 | `Y[m,n] = exp(X[m,n]) / Σ_n exp(X[m,n])` |
| 转置 | 出现 `A.T[...]` 或轴顺序与输入不同 | `C[n,m] = A.T[n,m] + B[n,m]` |
| 广播 | 输入维度数少于输出、或轴集合是输出子集 | `Z[i,j,k] = X[i,j,k] * Y[k]` |
| Elementwise | 输入输出轴集合完全一致且无归约符号 | `C[m,n] = A[m,n] + B[m,n]` |

### 隐式 Reduce 的特别处理

当输入输出形状一致但 `reduce_axes` 非空（如 softmax），块映射需要：

1. **输入块**：需要加载完整 reduce 维度的数据（BLOCK 覆盖整行）
2. **输出块**：形状与输入块一致，reduce 操作在 kernel 内部通过 `tl.max`/`tl.sum` 完成
3. 涉及 reduce 轴的指针（包括输出指针），都需要在 `reduce_dim` 中标注该轴

## 注意事项

### block_name 说明
- `block_name` 中的键值对表示该指针在哪些轴上使用了对应的块大小参数
- 每个 BLOCK 参数必须与原始输入/输出的逻辑轴一一对应

### axis_size 说明
- `axis_size` 存储各轴的实际大小（从第二步的 io_shapes 中的 shape 提取）
- 值为列表形式，对应各测试用例的轴大小

### reduce_dim 说明
- `reduce_dim` 仅对归约操作存在，表示归约维度
- 可以是单个字符串（如 `"M"`）或字符串列表（如 `["B", "D"]`）
- 每个 `reduce_dim` 也需要有对应的拆分块使得Kernel内部可以以for循环形式计算归约轴

### compute_formula 说明
- `compute_formula` 用数学公式表示计算逻辑
- 格式示例：
  - 归约：`"Y[n,k] = Σ X[n,m,k]"`
  - 转置+elementwise：`"C[n,m] = A.T[n,m] + B[n,m]"`
  - 广播：`"Z[i,j,k] = X[i,j,k] * Y[k]"`
  - 隐式归约：`"softmax(x_i) = exp(x_i) / Σ_j(exp(x_j))"`（输入输出形状相同，但内部有reduce）

### 隐式Reduce场景的reduce_dim

对于隐式Reduce场景（如softmax），即使输入输出形状一致，也需要正确设置 `reduce_dim`：

- **reduce_dim** 表示在kernel内部进行归约的维度
- 对于softmax：reduce_dim = "N"（沿列维度/最后一个维度进行max和sum）
- 块大小需要覆盖完整的reduce维度（如BLOCK_SIZE需要大于等于n_cols）

**示例：**

```json
{
    "io_block_mapping": {
        "input_ptr": {
            "block_name": {"M": "BLOCK_M"},
            "axis_size": {"M": [M]},
            "reduce_dim": {"N": "BLOCK_N"},  // 隐式reduce在这个维度上进行
            "reduce_size": {"N": [N]},
            "contiguity": [true, true]
        }
        "output_ptr": {
            "block_name": {"M": "BLOCK_M"},
            "axis_size": {"M": [M]},
            "reduce_dim": {"N": "BLOCK_N"},  // 隐式reduce维度
            "reduce_size": {"N": [N]},
            "contiguity": [true, true]
        }
    }
}
```

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| Step 1 输出存在 | 检查文件是否存在 | step1_base_info.json 存在且可读 |
| compute_formula 存在 | 解析 JSON 格式 | 包含 compute_formula 字段 |
| io_block_mapping 完整 | 检查所有指针 | 包含所有输入输出的 block_name, axis_size, contiguity |
| reduce_dim 正确 | 检查归约维度 | 对于 reduction/normalization 类型必须有 reduce_dim，且与 step1 的 reduce_axes 一致 |

## 回退机制

| 失败场景 | 处理方式 |
|---------|--------|
| Step 1 输出不存在或无效 | 返回错误 |
| 输出 JSON 格式无效 | 内部重试（最多 3 次） |
| io_block_mapping 缺少指针 | 内部重试（最多 3 次） |
| reduce_dim 与 step2 不一致 | 内部重试（最多 3 次） |
