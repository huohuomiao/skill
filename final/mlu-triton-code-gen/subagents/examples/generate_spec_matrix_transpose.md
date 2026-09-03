## 场景：矩阵转置（Transpose）

**输入（上一步结果）：**
```json
{
    "compute_formula": "Y[m,n] = X.T[m,n]",
    "compute_note": {
        "description": "对输入矩阵 X 做转置，得到形状为 (M, N) 的输出 Y。",
        "torch_impl": "Y = X.transpose(0, 1).contiguous()"
    },
    "io_block_mapping": {
        "X_ptr": {
            "block_name": {"M": "BLOCK_M", "N": "BLOCK_N"},
            "axis_size": {"M": [128], "N": [64]},
            "contiguity": [true, true],
            "transpose": true
        },
        "Y_ptr": {
            "block_name": {"M": "BLOCK_M", "N": "BLOCK_N"},
            "axis_size": {"M": [128], "N": [64]},
            "contiguity": [true, true]
        }
    }
}
```

**输出：**
```json
{
    "compute_formula": "Y[m,n] = X.T[m,n]",
    "compute_note": {
        "description": "对输入矩阵 X 做转置，得到形状为 (M, N) 的输出 Y。",
        "torch_impl": "Y = X.transpose(0, 1).contiguous()"
    },
    "kernel": {
        "block_params": {"BLOCK_M": [128], "BLOCK_N": [64]},
        "aux_params": {
            "pid_m": "tl.program_id(0)",
            "pid_n": "tl.program_id(1)",
            "m_offset": "pid_m * BLOCK_M",
            "n_offset": "pid_n * BLOCK_N",
            "m_idx": "pid_m * BLOCK_M + tl.arange(0, BLOCK_M)",
            "n_idx": "pid_n * BLOCK_N + tl.arange(0, BLOCK_N)"
        },
        "loads": {
            "X_ptr": {
                "index_X_ptr": "n_idx[:, None] * stride_x0 + m_idx[None, :] * stride_x1",
                "mask_X_ptr": "(n_idx[:, None] < N) & (m_idx[None, :] < M)"
            }
        },
        "stores": {
            "Y_ptr": {
                "index_Y_ptr": "m_idx[:, None] * stride_y0 + n_idx[None, :] * stride_y1",
                "mask_Y_ptr": "(m_idx[:, None] < M) & (n_idx[None, :] < N)"
            }
        },
        "compute": {
            "formula": "tl.trans(x)",
            "note": "使用 tl.trans 进行转置操作"
        }
    },
    "wrapper": {
        "grid": "(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))",
        "block_params": {"BLOCK_M": 8, "BLOCK_N": 8}
    }
}
```

**说明**：
- 当计算公式包含转置操作（如 `X.T[m,n]`）时，需要在 `compute` 字段中指定 `formula: "tl.trans(x)"`
- 转置场景下，load 时使用转置前的索引顺序（n 行 m 列）
- 转置场景下的 mask 计算需要注意维度的对应关系
