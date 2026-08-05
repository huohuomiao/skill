## 场景：Reduce Sum

**输入（上一步结果）：**
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
            "reduce_dim": "M",
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

**输出：**
```json
{
    "compute_formula": "Y[n,k] = Σ X[n,m,k]",
    "compute_note": {
        "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
        "torch_impl": "Y = X.sum(dim=1)"
    },
    "kernel": {
        "block_params": {"BLOCK_N": [16], "BLOCK_K": [128]},
        "aux_params": {
            "pid_n": "tl.program_id(0)",
            "pid_k": "tl.program_id(1)",
            "n_offset": "pid_n * BLOCK_N",
            "k_offset": "pid_k * BLOCK_K",
            "n_idx": "pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            "k_idx": "pid_k * BLOCK_K + tl.arange(0, BLOCK_K)"
        },
        "loads": {
            "X_ptr": {
                "index_X_ptr": "n_idx[:, None] * stride_x0 + m * stride_x1 + k_idx[None, :] * stride_x2",
                "mask_X_ptr": "(n_idx[:, None] < N) & (k_idx[None, :] < K)"
            }
        },
        "stores": {
            "Y_ptr": {
            "index_Y_ptr": "n_idx[:, None] * stride_y0 + k_idx[None, :] * stride_y1",
                "mask_Y_ptr": "(n_idx[:, None] < N) & (k_idx[None, :] < K)"
            }
        },
        "reduce_loop": {
            "reduce_dim": "M",
            "reduce_var": "m",
            "reduce_block": "BLOCK_M",
            "accumulator": "acc += x",
            "reduction_strategy": "delayed_block_reduction",
            "accumulator_shape": "(BLOCK_N, BLOCK_M, BLOCK_K)",
            "final_reduction": "out = tl.sum(acc, axis=1)"
        },
        "compute": {
            "formula": "acc + x",
            "note": "Reduce sum 操作。循环内只对包含 BLOCK_M 维度的 accumulator 做逐元素累加，循环结束后执行一次块内 tl.sum 得到输出"
        }
    },
    "wrapper": {
        "grid": "(triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))",
        "block_params": {"BLOCK_N": 4, "BLOCK_K": 8}
    }
}
```

**说明**：
- aux_params 中 `n_idx` 和 `k_idx` 是 `tl.arange` 与 offset 相加的结果
- loads 和 stores 的 index 中各自独立进行扩维操作（`n_idx[:, None]` 和 `k_idx[None, :]`）
- 每个指针有独立的 mask 计算逻辑
- **index 计算公式格式**：`idx * stride`，明确表示偏移和索引的组合