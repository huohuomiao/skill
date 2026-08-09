## 场景：Transpose + Elementwise Add（无融合）

**输入（上一步结果）：**
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

**输出：**
```json
{
    "compute_formula": "C[n,m] = A.T[n,m] + B[n,m]",
    "compute_note": {
        "description": "先对 A 做转置得到形状 (N, M) 的张量，再与同形状的 B 逐元素相加。",
        "torch_impl": "C = A.transpose(0, 1) + B"
    },
    "kernel": {
        "block_params": {"BLOCK_M": [128], "BLOCK_N": [64]},
        "aux_params": {
            "pid_n": "tl.program_id(0)",
            "pid_m": "tl.program_id(1)",
            "n_offset": "pid_n * BLOCK_N",
            "m_offset": "pid_m * BLOCK_M",
            "n_idx": "pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            "m_idx": "pid_m * BLOCK_M + tl.arange(0, BLOCK_M)"
        },
        "loads": {
            "A_ptr": {
            "index_A_ptr": "m_idx[:, None] * stride_a0 + n_idx[None, :] * stride_a1",
                "mask_A_ptr": "(m_idx[:, None] < M) & (n_idx[None, :] < N)"
            },
            "B_ptr": {
                "index_B_ptr": "n_idx[:, None] * stride_b0 + m_idx[None, :] * stride_b1",
                "mask_B_ptr": "(n_idx[:, None] < N) & (m_idx[None, :] < M)"
            }
        },
        "stores": {
            "C_ptr": {
                "index_C_ptr": "n_idx[:, None] * stride_c0 + m_idx[None, :] * stride_c1",
                "mask_C_ptr": "(n_idx[:, None] < N) & (m_idx[None, :] < M)"
            }
        },
        "compute": {
            "formula": "tl.trans(a) + b",
            "note": "Elementwise add 操作，A 需要转置后与 B 相加"
        }
    },
    "wrapper": {
        "grid": "(triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))",
        "block_params": {"BLOCK_N": 8, "BLOCK_M": 8}
    }
}
```
