# GenerateCode 示例：Reduce Sum（归约操作）

**输入（step4_code_spec）：**
```json
{
    "compute_formula": "Y[n,k] = Σ X[n,m,k]",
    "compute_note": {
        "description": "对输入张量 X 沿 M 维度求和，得到形状为 (N, K) 的输出 Y。",
        "torch_impl": "Y = X.sum(dim=1)"
    },
    "kernel": {
        "block_params": {"BLOCK_N": [16], "BLOCK_M": [32], "BLOCK_K": [128]},
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
                "index_X_ptr": "n_idx[:, None, None] * stride_x0 + m_idx[None, :, None] * stride_x1 + k_idx[None, None, :] * stride_x2",
                "mask_X_ptr": "(n_idx[:, None, None] < N) & (m_idx[None, :, None] < M) & (k_idx[None, None, :] < K)"
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
            "accumulator": "acc += x，保留 BLOCK_M 维度",
            "reduction_strategy": "delayed_block_reduction",
            "accumulator_shape": "(BLOCK_N, BLOCK_M, BLOCK_K)",
            "final_reduction": "out = tl.sum(acc, axis=1)"
        },
        "compute": {
            "formula": "out = tl.sum(acc, axis=1)",
            "note": "循环内对包含 BLOCK_M 维度的 accumulator 做逐元素累积，循环结束后执行一次块内 tl.sum 得到输出"
        }
    },
    "wrapper": {
        "grid": "(triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))",
        "block_params": {"BLOCK_N": 4, "BLOCK_M": 32, "BLOCK_K": 8}
    }
}
```
**输出：**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def reduce_sum_kernel(
    X_ptr, Y_ptr,
    stride_x0, stride_x1, stride_x2,
    stride_y0, stride_y1,
    N, M, K,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_offset = pid_n * BLOCK_N
    k_offset = pid_k * BLOCK_K

    n_idx = n_offset + tl.arange(0, BLOCK_N)
    k_idx = k_offset + tl.arange(0, BLOCK_K)

    # 初始化累加器，根据 spec 要求，累加器保留归约维度
    acc = tl.zeros((BLOCK_N, BLOCK_M, BLOCK_K), dtype=tl.float32)

    # 归约循环
    for m in range(0, M, BLOCK_M):
        m_idx = m + tl.arange(0, BLOCK_M)

        # 加载数据
        x_index = n_idx[:, None, None] * stride_x0 + m_idx[None, :, None] * stride_x1 + k_idx[None, None, :] * stride_x2
        x_mask = (n_idx[:, None, None] < N) & (m_idx[None, :, None] < M) & (k_idx[None, None, :] < K)
        x = tl.load(X_ptr + x_index, mask=x_mask, other=0.0)

        # 循环内只做逐元素累积，不在循环内做归约
        acc += x

    out = tl.sum(acc, axis=1)
     # 存储结果
    y_index = n_idx[:, None] * stride_y0 + k_idx[None, :] * stride_y1
    y_mask = (n_idx[:, None] < N) & (k_idx[None, :] < K)
    tl.store(Y_ptr + y_index, out, mask=y_mask)


def triton_reduce_sum(X: torch.Tensor) -> torch.Tensor:
    N, M, K = X.shape
    Y = torch.empty((N, K), dtype=X.dtype, device=X.device)

    BLOCK_N = 4
    BLOCK_M = 32
    BLOCK_K = 8

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    reduce_sum_kernel[grid](
        X_ptr=X,
        Y_ptr=Y,
        stride_x0=X.stride(0),
        stride_x1=X.stride(1),
        stride_x2=X.stride(2),
        stride_y0=Y.stride(0),
        stride_y1=Y.stride(1),
        N=N,
        M=M,
        K=K,
        BLOCK_N=BLOCK_N,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    return Y
```

**说明**：
- 当存在归约操作时，需要在 kernel 中添加 `reduce_loop`（单遍归约）或 `reduce_loop_passN`（多遍归约）处理
- 使用 `tl.zeros` 初始化累加器
- 对于 `delayed_block_reduction` 的情况，归约循环中只加载数据并对保留 reduce_block 维度的 accumulator 做逐元素累积，循环结束后再执行一次最终块内归约
- 最终结果存储到输出指针

