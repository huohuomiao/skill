# AxisFusion

## 职责概述

AxisFusion 是 mlu-triton-code-gen 工作流程的第 3 步 subagent。负责轴融合优化方案，分析相邻轴是否可融合以提升内存访问效率。


## 输入

| 来源 | 内容 |
|------|------|
| Step 2 输出 | `{输出存储路径}/KernelGen/step2_block_mapping.json` |

## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/KernelGen/step3_axis_fusion.json` - 轴融合优化结果 |

## 执行步骤

### 步骤 1：读取 Step 2 结果

读取 step2_block_mapping.json，获取：
- compute_formula
- compute_note（需透传到输出）
- io_block_mapping（其中已携带各指针的 axis_size / reduce_size / contiguity，本步分析所需的形状/连续性信息直接来自这里）

### 步骤 2：分析轴融合

使用 LLM 分析轴融合优化方案：

**融合判断规则**：
- **关键条件**：两个相邻维度在**所有输入/输出**上都是连续的才认为可以融合
- 在任意的输入、输出上，某两个轴**跨越**了**reduce/broadcast/trans维度**，都认为不连续。
- **必须遍历所有相邻维度对**：对于每一对相邻维度 (i, i+1)，都需要独立检查
- **融合场景**：
  1. Elementwise 连续维度融合：输入输出形状完全一致，相邻维度都连续
  以下不**跨越有维度变化的轴进行融合**：
  2. Reduce 后连续：融合维度必须在所有输入/输出的物理存储上都是连续的
  3. Transpose 后连续：融合需要顺序且连续，转置的维度对不能融合
  4. Broadcast 后连续：广播维度可以连续

**输出格式**：
> **说明**：`compute_note` 从 step2_block_mapping.json 原样透传到输出。

```json
{
    "compute_formula": "Y[n,k] = Σ X[n,m,k]",
    "compute_note": {
        "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
        "torch_impl": "Y = X.sum(dim=1)"
    },
    "fusion_note": "不融合: M 维度非连续，无法融合 N-M",
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

### 步骤 3：保存结果

将分析结果保存到 `{输出存储路径}/KernelGen/step3_axis_fusion.json`

## 参考场景

### 场景1：Elementwise 完全连续（融合）

**输入（上一步结果）：**
```json
{
    "compute_formula": "Z[n,m] = X[n,m] + Y[n,m]",
    "compute_note": {
        "description": "对 X 和 Y 逐元素相加，得到同形状的 Z。",
        "torch_impl": "Z = X + Y"
    },
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, true]
        },
        "Y_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, true]
        },
        "Z_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, true]
        }
    }
}
```

**分析**：
- 检查所有输入/输出中维度 N 和 M 的连续性
- X_ptr: contiguity = [true, true] → N 连续，M 连续
- Y_ptr: contiguity = [true, true] → N 连续，M 连续
- Z_ptr: contiguity = [true, true] → N 连续，M 连续
- **结论**：在所有输入/输出中，N 和 M 都是连续的，可以融合

**输出（融合后）：**
```json
{
    "compute_formula": "Z[n,m] = X[n,m] + Y[n,m]",
    "compute_note": {
        "description": "对 X 和 Y 逐元素相加，得到同形状的 Z。",
        "torch_impl": "Z = X + Y"
    },
    "fusion_note": "融合: N + M -> NM",
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"NM": "BLOCK_NM"},
            "axis_size": {"NM": [512]},
            "contiguity": [true]
        },
        "Y_ptr": {
            "block_name": {"NM": "BLOCK_NM"},
            "axis_size": {"NM": [512]},
            "contiguity": [true]
        },
        "Z_ptr": {
            "block_name": {"NM": "BLOCK_NM"},
            "axis_size": {"NM": [512]},
            "contiguity": [true]
        }
    }
}
```
### 场景2：Reduce 后连续

**输入（上一步结果）：**
```json
{
    "compute_formula": "Y[n,k] = Σ X[m,n,k]",
    "compute_note": {
        "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
        "torch_impl": "Y = X.sum(dim=0)"
    },
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"N": "BLOCK_N", "K": "BLOCK_K"},
            "axis_size": {"N": [16], "K": [128]},
            "reduce_dim": {"M": "BLOCK_M"},
            "reduce_size": {"M": [32]},
            "contiguity": [true, true, true]
        },
        "Y_ptr": {
            "block_name": {"N": "BLOCK_N", "K": "BLOCK_K"},
            "axis_size": {"N": [16], "K": [128]},
            "contiguity": [true, true]
        }
    }
}
```

**分析**：
- 检查所有输入/输出中维度 N 和 K 的连续性
- X_ptr: contiguity = [true, true, true] → N 连续，M 连续，K 连续 → N 和 K 都连续
- Y_ptr: contiguity = [true, true] → N 连续，K 连续
- **结论**：在所有输入/输出中，N 和 K 都是连续的，可以融合

**输出（融合后）：**
```json
{
    "compute_formula": "Y[n,k] = Σ X[m,n,k]",
    "compute_note": {
        "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
        "torch_impl": "Y = X.sum(dim=0)"
    },
    "fusion_note": "融合: N + K -> NK",
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"NK": "BLOCK_NK"},
            "axis_size": {"NK": [2048]},
            "reduce_dim": {"M": "BLOCK_M"},
            "reduce_size": {"M": [32]},
            "contiguity": [true, true]
        },
        "Y_ptr": {
            "block_name": {"NK": "BLOCK_NK"},
            "axis_size": {"NK": [2048]},
            "contiguity": [true]
        }
    }
}
```

### 场景3：Transpose 后部分维度可融合（NHWC格式）

**输入（上一步结果）：**
```json
{
    "compute_formula": "out[n,h,w,c] = exp(x[n,c,h,w])",
    "compute_note": {
        "description": "对 NCHW 格式的输入逐元素取指数，并排列为 NHWC 格式输出。",
        "torch_impl": "out = torch.exp(x).permute(0, 2, 3, 1).contiguous()"
    },
    "io_block_mapping": {
        "x_ptr": {
            "block_name": {"N": "BLOCK_N", "C": "BLOCK_C", "H": "BLOCK_H", "W": "BLOCK_W"},
            "axis_size": {"N": [3, 7], "C": [5, 9], "H": [17, 33], "W": [35, 67]},
            "contiguity": [true, true, true, true]
        },
        "out_ptr": {
            "block_name": {"N": "BLOCK_N", "H": "BLOCK_H", "W": "BLOCK_W", "C": "BLOCK_C"},
            "axis_size": {"N": [3, 7], "H": [17, 33], "W": [35, 67], "C": [5, 9]},
            "contiguity": [true, true, true, true]
        }
    }
}
```

**分析**：
- 逐对检查相邻维度（按block_name中的顺序）：
  - N 和 C：顺序不一致 → **不能融合**
  - C 和 H：顺序不一致 → **不能融合**
  - H 和 W：顺序一致且连续 → **可以融合**
  - W 和 C：顺序不一致 → **不能融合**
- **结论**：只有H和W在所有输入/输出中顺序一致且连续，可以融合

**输出（融合后）：**
```json
{
    "compute_formula": "out[n,h,w,c] = exp(x[n,c,h,w])",
    "compute_note": {
        "description": "对 NCHW 格式的输入逐元素取指数，并排列为 NHWC 格式输出。",
        "torch_impl": "out = torch.exp(x).permute(0, 2, 3, 1).contiguous()"
    },
    "fusion_note": "融合: H + W -> HW",
    "io_block_mapping": {
        "x_ptr": {
            "block_name": {"N": "BLOCK_N", "C": "BLOCK_C", "HW": "BLOCK_HW"},
            "axis_size": {"N": [3, 7], "C": [5, 9], "HW": [595, 2211]},
            "contiguity": [true, true, true]
        },
        "out_ptr": {
            "block_name": {"N": "BLOCK_N", "HW": "BLOCK_HW", "C": "BLOCK_C"},
            "axis_size": {"N": [3, 7], "HW": [595, 2211], "C": [5, 9]},
            "contiguity": [true, true, true]
        }
    }
}
```

**注意**：融合后 grid 从 4 维变成 3 维：`grid = (N, HW, C)`

### 场景4：不融合（部分指针不连续）

**输入（上一步结果）：**
```json
{
    "compute_formula": "Z[n,m] = X[n,m] + Y[n,m]",
    "compute_note": {
        "description": "对 X 和 Y 逐元素相加，Y 的 M 维度由 slice 构造为非连续。",
        "torch_impl": "Z = X + Y"
    },
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, true]
        },
        "Y_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, false]
        },
        "Z_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, true]
        }
    }
}
```

**分析**：
- 检查所有输入/输出中维度 N 和 M 的连续性
- Y_ptr 中 M 维度不连续，不满足"在所有输入/输出中，这两个维度都是连续的"条件

**输出（不融合）：**
```json
{
    "compute_formula": "Z[n,m] = X[n,m] + Y[n,m]",
    "compute_note": {
        "description": "对 X 和 Y 逐元素相加，Y 的 M 维度由 slice 构造为非连续。",
        "torch_impl": "Z = X + Y"
    },
    "fusion_note": "不融合: Y_ptr 的 M 维度不连续",
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, true]
        },
        "Y_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, false]
        },
        "Z_ptr": {
            "block_name": {"N": "BLOCK_N", "M": "BLOCK_M"},
            "axis_size": {"N": [16], "M": [32]},
            "contiguity": [true, true]
        }
    }
}
```

## 核心融合规则

### 判断两个维度 i 和 i+1 是否可以融合

- **关键条件**：这两个维度在**所有输入/输出**上都是连续的才认为可以融合
- 即：对于每个指针的 contiguity，维度 i 和维度 i+1 都必须为 true 才可融合
- **重要**：逐对检查相邻维度，只需要这一对维度在所有输入/输出中连续即可，不需要所有维度都连续

### 融合场景

1. **场景1: Elementwise 连续维度融合**
   - 条件：输入输出形状完全一致
   - 示例：N, M 两个轴都是连续的，可以融合为 NM

2. **场景2: Reduce 后连续**
   - 条件：需要融合的维度必须在任意输入/输出上都是可以融合的
   - 示例：abc -> ac（b 穿越了 ac），如果需要融合 ac，不可以融合
   3. **场景3: Transpose 后连续**
   - 条件：转置后维度可以连续，如 NCHW -> CNHW 或 NCHW -> NHWC
   - **关键**：融合需要**顺序且连续**——两个维度在所有输入/输出中不仅都要连续，而且顺序要一致

4. **场景4: Broadcast 后连续**
   - 条件：广播维度可以连续，如 HW -> NCHW

### 不融合的情况

- [i+1]维度不连续则i和i+1维度不能融合
- 不是在所有的输入/输出上i+1维度和i维度都连续
- trans的几个维度能融合
- 被reduce的维度穿越的两个轴不融合

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| Step 2 输出存在 | 检查文件是否存在 | step2_block_mapping.json 存在且可读 |
| compute_formula 存在 | 解析 JSON 格式 | 包含 compute_formula 字段 |
| compute_note 透传 | 解析 JSON 格式 | 包含 compute_note 字段（含 description 和 torch_impl） |
| fusion_note 存在 | 检查融合说明 | 包含 fusion_note 字段（描述融合或不融合的原因） |
| io_block_mapping 完整 | 检查所有指针 | 包含所有输入输出的 block_name, axis_size, contiguity |

## 回退机制

| 失败场景 | 处理方式 |
|---------|--------|
| Step 2 输出不存在或无效 | 返回错误 |
| 输出 JSON 格式无效 | 内部重试（最多 3 次） |
| fusion_note 缺失或不合理 | 内部重试（最多 3 次） |
| 轴融合逻辑错误 | 内部重试（最多 3 次） |
