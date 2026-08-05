# GenerateCode 示例：Transpose + Elementwise（无融合）

**输入（step4_code_spec）：**
```json
{
    "compute_formula": "out[h,n,w] = exp(x[n,h,w])",
    "kernel": {
        "block_params": {"BLOCK_H": [16,32,64], "BLOCK_N": [4,8,16], "BLOCK_W": [32,64,16,128]},
        "aux_params": {
            "pid_h": "tl.program_id(0)",
            "pid_n": "tl.program_id(1)",
            "pid_w": "tl.program_id(2)",
            "h_offset": "pid_h * BLOCK_H",
            "n_offset": "pid_n * BLOCK_N",
            "w_offset": "pid_w * BLOCK_W",
            "h_idx": "tl.arange(0, BLOCK_H)[:, None, None]",
            "n_idx": "tl.arange(0, BLOCK_N)[None, :, None]",
            "w_idx": "tl.arange(0, BLOCK_W)[None, None, :]"
        },
        "loads": {
            "x_ptr": {
                "index_x_ptr": "n_offset * stride_x0 + h_offset * stride_x1 + w_offset * stride_x2 + n_idx * stride_x0 + h_idx * stride_x1 + w_idx * stride_x2",
                "mask_x_ptr": "(n_offset + n_idx < N) & (h_offset + h_idx < H) & (w_offset + w_idx < W)"
            }
        },
        "stores": {
            "out_ptr": {
                "index_out_ptr": "h_offset * stride_out0 + n_offset * stride_out1 + w_offset * stride_out2 + h_idx * stride_out0 + n_idx * stride_out1 + w_idx * stride_out2",
                "mask_out_ptr": "(h_offset + h_idx < H) & (n_offset + n_idx < N) & (w_offset + w_idx < W)"
            }
        }
    },
    "wrapper": {
        "grid": "(triton.cdiv(H, BLOCK_H), triton.cdiv(N, BLOCK_N), triton.cdiv(W, BLOCK_W))",
        "block_params": {"BLOCK_H": 32, "BLOCK_N": 8, "BLOCK_W": 64}
    }
}
```

**输出：**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def transpose_exp_3d_kernel(
    x_ptr, out_ptr,
    x_stride0, x_stride1, x_stride2,
    out_stride0, out_stride1, out_stride2,
    N, H, W,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_w = tl.program_id(2)

    h_offset = pid_h * BLOCK_H
    n_offset = pid_n * BLOCK_N
    w_offset = pid_w * BLOCK_W

    h_idx = tl.arange(0, BLOCK_H)[:, None, None]
    n_idx = tl.arange(0, BLOCK_N)[None, :, None]
    w_idx = tl.arange(0, BLOCK_W)[None, None, :]

    x_index = n_offset * x_stride0 + h_offset * x_stride1 + w_offset * x_stride2 + n_idx * x_stride0 + h_idx * x_stride1 + w_idx * x_stride2
    x_mask = (n_offset + n_idx < N) & (h_offset + h_idx < H) & (w_offset + w_idx < W)
    x = tl.load(x_ptr + x_index, mask=x_mask)

    out_val = tl.exp(x)

    out_index = h_offset * out_stride0 + n_offset * out_stride1 + w_offset * out_stride2 + h_idx * out_stride0 + n_idx * out_stride1 + w_idx * out_stride2
    out_mask = (h_offset + h_idx < H) & (n_offset + n_idx < N) & (w_offset + w_idx < W)
    tl.store(out_ptr + out_index, out_val, mask=out_mask)


def triton_transpose_exp(x: torch.Tensor) -> torch.Tensor:
    N, H, W = x.shape
    out = torch.empty((H, N, W), dtype=x.dtype, device=x.device)

    BLOCK_H = 32
    BLOCK_N = 8
    BLOCK_W = 64

    grid = (triton.cdiv(H, BLOCK_H), triton.cdiv(N, BLOCK_N), triton.cdiv(W, BLOCK_W))

    transpose_exp_3d_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        x_stride0=x.stride(0),
        x_stride1=x.stride(1),
        x_stride2=x.stride(2),
        out_stride0=out.stride(0),
        out_stride1=out.stride(1),
        out_stride2=out.stride(2),
        N=N,
        H=H,
        W=W,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_W=BLOCK_W,
    )

    return out
```

