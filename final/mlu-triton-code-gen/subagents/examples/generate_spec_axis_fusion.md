## 场景：轴融合后（H+W -> HW）

**输入（上一步结果）：**
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

**输出（融合后的规范）：**
```json
{
    "compute_formula": "out[n,h,w,c] = exp(x[n,c,h,w])",
    "compute_note": {
        "description": "对 NCHW 格式的输入逐元素取指数，并排列为 NHWC 格式输出。",
        "torch_impl": "out = torch.exp(x).permute(0, 2, 3, 1).contiguous()"
    },
    "fusion_note": "融合: H + W -> HW",
    "kernel": {
        "block_params": {"BLOCK_N": [3, 7], "BLOCK_C": [5, 9], "BLOCK_HW": [595, 2211]},
        "aux_params": {
            "pid_n": "tl.program_id(0)",
            "pid_hw": "tl.program_id(1)",
            "pid_c": "tl.program_id(2)",
            "n_offset": "pid_n * BLOCK_N",
            "hw_offset": "pid_hw * BLOCK_HW",
            "c_offset": "pid_c * BLOCK_C",
            "n_idx": "pid_n * BLOCK_N + tl.arange(0, BLOCK_N)",
            "hw_idx": "pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)",
            "c_idx": "pid_c * BLOCK_C + tl.arange(0, BLOCK_C)"
        },
        "loads": {
        "x_ptr": {
                "index_x_ptr": "n_idx[:, None, None] * stride_x0 + c_idx[None, :, None] * stride_x1 + hw_idx[None, None, :] * stride_x3",
                "mask_x_ptr": "(n_idx[:, None, None] < N) & (c_idx[None, :, None] < C) & (hw_idx[None, None, :] < HW)"
            }
        },
        "stores": {
            "out_ptr": {
                "index_out_ptr": "n_idx[:, None, None] * stride_o0 + hw_idx[None, :, None] * stride_o2 + c_idx[None, None, :] * stride_o3",
                "mask_out_ptr": "(n_idx[:, None, None] < N) & (hw_idx[None, :, None] < HW) & (c_idx[None, None, :] < C)"
            }
        },
        "compute": {
            "formula": "tl.exp(x)",
            "note": "Elementwise 指数操作，对输入 x 计算 e^x"
        }
    },
    "wrapper": {
        "grid": "(triton.cdiv(N, BLOCK_N), triton.cdiv(HW, BLOCK_HW), triton.cdiv(C, BLOCK_C))",
        "block_params": {"BLOCK_N": 4, "BLOCK_C": 4, "BLOCK_HW": 64}
    }
}
```

**说明**：
- 融合轴 HW 对应原始张量的 H 和 W 维度，融合后使用 stride_o2（融合后 HW 维度的 stride, 值为 C）
- x 原始形状 [N, C, H, W]，融合后 [N, C, HW]，融合后stride为 [C*HW, HW, 1]
- out 原始形状 [N, H, W, C]，融合后 [N, HW, C]，融合后stride为 [HW*C, C, 1]
- index 格式统一为 `idx * stride` 形式，每个维度独立处理
