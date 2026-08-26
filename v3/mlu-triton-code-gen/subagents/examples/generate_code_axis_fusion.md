# GenerateCode 示例：轴融合后（H+W -> HW）

**输入（step4_code_spec，有融合）：**
```json
{
    "compute_formula": "out[n,h,w,c] = sigmoid(x[n,c,h,w])",
    "fusion_note": "融合: H + W -> HW",
    "kernel": {
        "block_params": {
            "BLOCK_N": [2, 4],
            "BLOCK_HW": [512, 2048],
            "BLOCK_C": [4, 8]
        },
        "aux_params": {
            "pid_n": "tl.program_id(0)",
            "pid_hw": "tl.program_id(1)",
            "pid_c": "tl.program_id(2)",
            "n_offset": "pid_n * BLOCK_N",
            "hw_offset": "pid_hw * BLOCK_HW",
            "c_offset": "pid_c * BLOCK_C",
            "n_idx": "tl.arange(0, BLOCK_N)[:, None, None]",
            "hw_idx": "tl.arange(0, BLOCK_HW)[None, :, None]",
            "c_idx": "tl.arange(0, BLOCK_C)[None, None, :]"
        },
        "loads": {
            "x_ptr": {
                "index_x_ptr": "n_offset * stride_x0 + c_offset * stride_x1 + hw_offset * stride_x3 + n_idx * stride_x0 + c_idx * stride_x1 + hw_idx * stride_x3",
                "mask_x_ptr": "(n_offset + n_idx < N) & (c_offset + c_idx < C) & (hw_offset + hw_idx < H * W)"
            }
        },
        "stores": {
            "out_ptr": {
                "index_out_ptr": "n_offset * stride_o0 + hw_offset * stride_o2 + c_offset * stride_o3 + n_idx * stride_o0 + hw_idx * stride_o2 + c_idx * stride_o3",
                "mask_out_ptr": "(n_offset + n_idx < N) & (hw_offset + hw_idx < H * W) & (c_offset + c_idx < C)"
            }
        }
    },
    "wrapper": {
        "grid": "(triton.cdiv(N, BLOCK_N), triton.cdiv(HW, BLOCK_HW), triton.cdiv(C, BLOCK_C))",
        "block_params": {
            "BLOCK_N": 4,
            "BLOCK_HW": 64,
            "BLOCK_C": 4
        }
    }
}
```
**输出（关键点说明）：**
- 使用融合后的参数 `BLOCK_HW`，不能拆分为 `BLOCK_H` 和 `BLOCK_W`
- 使用 `pid_hw`, `hw_offset`, `hw_idx`，不能拆分为 `pid_h`, `pid_w`
- **重要**：直接使用 `hw_idx`，不要用取模/除法拆分！
- grid 使用融合后的维度：`triton.cdiv(H * W, BLOCK_HW)`

```python
import torch
import triton
import triton.language as tl

@triton.jit
def transpose_sigmoid_4d_kernel(
    x_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2, x_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
    N, C, H, W,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_c = tl.program_id(2)

    n_offset = pid_n * BLOCK_N
    hw_offset = pid_hw * BLOCK_HW
    c_offset = pid_c * BLOCK_C

    n_idx = tl.arange(0, BLOCK_N)[:, None, None]
    hw_idx = tl.arange(0, BLOCK_HW)[None, :, None]
    c_idx = tl.arange(0, BLOCK_C)[None, None, :]

    x_index = n_offset * x_stride0 + c_offset * x_stride1 + hw_offset * x_stride3 + n_idx * x_stride0 + c_idx * x_stride1 + hw_idx * x_stride3
    x_mask = (n_offset + n_idx < N) & (c_offset + c_idx < C) & (hw_offset + hw_idx < H * W)
    x = tl.load(x_ptr + x_index, mask=x_mask, other=0.0)

    out_val = 1 / (1 + tl.exp(-x))

    out_index = n_offset * out_stride0 + hw_offset * out_stride2 + c_offset * out_stride3 + n_idx * out_stride0 + hw_idx * out_stride2 + c_idx * out_stride3
    out_mask = (n_offset + n_idx < N) & (hw_offset + hw_idx < H * W) & (c_offset + c_idx < C)
    tl.store(out_ptr + out_index, out_val, mask=out_mask)
def triton_transpose_sigmoid(x: torch.Tensor) -> torch.Tensor:
    N, C, H, W = x.shape
    out = torch.empty((N, H, W, C), dtype=x.dtype, device=x.device)

    BLOCK_N = 4
    BLOCK_HW = 64
    BLOCK_C = 4

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(H * W, BLOCK_HW), triton.cdiv(C, BLOCK_C))

    transpose_sigmoid_4d_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        x_stride0=x.stride(0),
        x_stride1=x.stride(1),
        x_stride2=x.stride(2),
        x_stride3=x.stride(3),
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        out_stride2=out.stride(2),
        out_stride3=out.stride(3),
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
    )

    return out
```
