# Code Gen 兼容产物契约

本文档是 `v1_1` 设计代理的字段真源。它保留 `v1` 的 Step 1-4 文件名和下游可见语义。

## Step 1

`step1_base_info.json`：

```json
{
  "step": 1,
  "op_name": "op_name",
  "compute_type": "reduction|elementwise|matmul|normalization|others",
  "compute_formula": "Y[n,k] = sum_m X[n,m,k]",
  "compute_note": {
    "description": "自然语言计算语义",
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

`step1_io_shapes.json` 直接保存上面 `io_shapes` 的值，不加外层键。两者必须结构和值完全
相等。每个 tensor 的 `axis`、`shape`、`contiguity` 数组等长。

## Step 2

`step2_block_mapping.json`：

```json
{
  "compute_formula": "Y[n,k] = sum_m X[n,m,k]",
  "compute_note": {
    "description": "自然语言计算语义",
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

`compute_formula` 与 `compute_note` 从 Step 1 原样透传。非归约指针不强制出现
`reduce_dim` / `reduce_size`。

## Step 3

`step3_axis_fusion.json`：

```json
{
  "compute_formula": "与 Step 2 完全一致",
  "compute_note": {
    "description": "与 Step 2 完全一致",
    "torch_impl": "与 Step 2 完全一致"
  },
  "fusion_note": "融合或不融合的原因",
  "io_block_mapping": {}
}
```

`io_block_mapping` 使用与 Step 2 相同字段；若融合，更新涉及的 block/axis 映射；若不融合，
原样保留。`fusion_note` 必填，必须能追溯到连续性或语义边界。

## Step 4

`step4_code_spec.json`：

```json
{
  "compute_formula": "计算公式",
  "compute_note": {
    "description": "自然语言计算语义",
    "torch_impl": "torch 参考实现"
  },
  "fusion_note": "融合说明",
  "kernel": {
    "block_params": {"BLOCK_N": [16, 32, 64]},
    "aux_params": {"pid_n": "tl.program_id(0)"},
    "loads": {
      "X_ptr": {
        "index_X_ptr": "索引公式",
        "mask_X_ptr": "边界公式"
      }
    },
    "stores": {
      "Y_ptr": {
        "index_Y_ptr": "索引公式",
        "mask_Y_ptr": "边界公式"
      }
    },
    "compute": {
      "formula": "核心计算公式",
      "note": "实现说明"
    }
  },
  "wrapper": {
    "grid": "grid 公式",
    "block_params": {"BLOCK_N": 64}
  }
}
```

`compute_formula`、`compute_note` 必填且原样透传；`fusion_note` 有融合分析时保留。
`kernel.block_params`、`aux_params`、`loads`、`stores`、`compute`、`wrapper.grid`、
`wrapper.block_params` 均必填。

归约时添加 `kernel.reduce_loop`，多遍归约改用 `reduce_loop_pass1`、
`reduce_loop_pass2`……每个对象包含：

| 字段 | 条件 |
|---|---|
| `reduce_dim`、`reduce_var`、`reduce_block`、`accumulator` | 必填 |
| `reduction_strategy` | 必填；`inline_block_reduction` 或 `delayed_block_reduction` |
| `accumulator_shape`、`final_reduction` | delayed 策略必填 |

无归约操作时不得写空的 `reduce_loop`。loads/stores 的 mask 可在已证明无越界时省略，但
必须在对应 index 说明中可验证；否则一律生成 mask。

## 跨文件不变量

1. Step 1 的两个 io_shapes 完全一致。
2. Step 1→2→3→4 的 `compute_formula`、`compute_note` 不发生漂移。
3. Step 2/3 中的每个指针都能追溯到 Step 1 的输入或输出。
4. Step 4 的每个 BLOCK、grid 轴和 load/store 指针都能追溯到 Step 3 映射。
5. JSON 中不写 Markdown 注释、Python 表达式对象或 NaN/Infinity。
