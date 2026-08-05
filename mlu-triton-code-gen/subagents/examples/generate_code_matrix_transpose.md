# GenerateCode 示例：矩阵转置（Transpose）

**输入（step4_code_spec）：**
```json
{
    "compute_formula": "Y[m,n] = X.T[m,n]",
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

**输出：**
```python
import torch
import triton
import triton.language as tl
@triton.jit
def transpose_kernel(
    X_ptr, Y_ptr,
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N

    m_idx = m_offset + tl.arange(0, BLOCK_M)
    n_idx = n_offset + tl.arange(0, BLOCK_N)

    # 加载数据（使用转置前的索引顺序：n行m列）
    x_index = n_idx[:, None] * stride_x0 + m_idx[None, :] * stride_x1
    x_mask = (n_idx[:, None] < N) & (m_idx[None, :] < M)
    x = tl.load(X_ptr + x_index, mask=x_mask, other=0.0)

    # 执行转置操作
    out_val = tl.trans(x)

    # 存储结果
    y_index = m_idx[:, None] * stride_y0 + n_idx[None, :] * stride_y1
    y_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    tl.store(Y_ptr + y_index, out_val, mask=y_mask)


def triton_transpose(X: torch.Tensor) -> torch.Tensor:
    M, N = X.shape
    Y = torch.empty((N, M), dtype=X.dtype, device=X.device)

    BLOCK_M = 8
    BLOCK_N = 8

grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    transpose_kernel[grid](
        X_ptr=X,
        Y_ptr=Y,
        stride_x0=X.stride(0),
        stride_x1=X.stride(1),
        stride_y0=Y.stride(0),
        stride_y1=Y.stride(1),
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return Y
```

**说明**：
- 当计算公式包含转置操作（如 `X.T[m,n]`）时，使用 `tl.trans` 进行转置
- 转置场景下，load 时使用转置前的索引顺序（n 行 m 列）
- store 时使用正常的索引顺序（m 行 n 列）
- mask 计算需要注意维度的对应关系

