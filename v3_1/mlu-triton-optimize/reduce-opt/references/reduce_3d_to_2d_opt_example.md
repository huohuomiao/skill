# 归约三维tile转二维tile优化 - 详细示例

## 示例1：无外层任务循环的场景

### 原始代码
```python
@triton.jit
def sum_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # 三维分块偏移生成（BLOCK_M=非归约首维度，BLOCK_N=归约轴axis=1，BLOCK_K=非归约尾维度）
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M] 向量偏移
    k_offset = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K] 向量偏移

    # 边界mask（向量形式）
    mask_m = m_offset < M
    mask_k = k_offset < K

    # 中间结果变量：承载归约结果，形状[BLOCK_M, BLOCK_K]
    result_sum = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # 归约轴（N维度）的遍历循环
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = n_offset < N

        # 三维广播偏移：构造[BLOCK_M, BLOCK_N, BLOCK_K]的索引
        m_off = m_offset[:, None, None]
        n_off = n_offset[None, :, None]
        k_off = k_offset[None, None, :]

        # 三维访存偏移计算
        offset = m_off * (N * K) + n_off * K + k_off  # [BM, BN, BK]
        mask = mask_m[:, None, None] & mask_n[None, :, None] & mask_k[None, None, :]

        # 加载数据+沿axis=1归约
        inp_vals = tl.load(inp + offset, mask=mask, other=0)
        sum_val = tl.sum(inp_vals, axis=1)  # 归约后形状[BLOCK_M, BLOCK_K]

        # 更新累加结果
        result_sum += sum_val

    # 存储最终结果
    out_offset = m_offset[:, None] * K + k_offset[None, :]  # [BM, BK]
    out_mask = mask_m[:, None] & mask_k[None, :]
    tl.store(out + out_offset, result_sum, mask=out_mask)

def wrapper(inp, dim):
    ...
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))
    sum_kernel[grid](inp, out, M, N, K, BLOCK_M = 8, BLOCK_N = 128, BLOCK_K = 32)
```
### 优化后代码
```python
@triton.heuristics({"BLOCK_M": lambda args: 1})  # 新增heuristics，将非归约首维度分块参数设为1
@triton.jit
def sum_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,  # 保留参数，由heuristics强制设为1
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # 非归约首维度标量化：BLOCK_M=1，向量偏移改为标量偏移
    m_offset = pid_m  # 标量（原[BLOCK_M]向量 → 优化后标量）
    k_offset = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K] 保留原向量逻辑

    # 移除非归约首维度的边界mask（标量天然在[0, M)范围内，无越界风险）
    mask_k = k_offset < K

    # 中间结果变量形状适配：移除非归约首维度，从[BLOCK_M, BLOCK_K] → [BLOCK_K]
    result_sum = tl.zeros([BLOCK_K], dtype=tl.float32)

    # 归约轴（N维度）的遍历循环（保留原循环结构，仅适配内部逻辑）
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = n_offset < N

        # 二维广播偏移：移除m维度的广播维度，构造[BLOCK_N, BLOCK_K]的索引
        n_off = n_offset[:, None]
        k_off = k_offset[None, :]

        # 二维访存偏移计算（标量m_offset直接参与，移除三维广播）
        offset = m_offset * (N * K) + n_off * K + k_off  # [BN, BK]（原[BM, BN, BK] → 优化后[BN, BK]）
        mask = mask_n[:, None] & mask_k[None, :]  # 二维mask（移除mask_m相关逻辑）

        # 加载数据+归约轴适配：axis从1改为0（适配二维tile）
        inp_vals = tl.load(inp + offset, mask=mask, other=0)
        sum_val = tl.sum(inp_vals, axis=0)  # 归约后形状[BLOCK_K]（原[BLOCK_M, BLOCK_K] → 优化后[BLOCK_K]）

        # 更新累加结果（适配中间变量形状）
        result_sum += sum_val

    # 存储最终结果：偏移与mask适配标量化后的维度
    out_offset = m_offset * K + k_offset  # [BK]（原[BM, BK] → 优化后[BK]）
    out_mask = mask_k
    tl.store(out + out_offset, result_sum, mask=out_mask)

def wrapper(inp, dim):
    ...
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(K, meta["BLOCK_K"]))
    # 移除已通过heuristics设置的BLOCK_M参数，仅传递剩余分块参数
    sum_kernel[grid](inp, out, M, N, K, BLOCK_N = 128, BLOCK_K = 32)
```

### 示例说明（关键点说明）
1. 索引体系重构:
  - 非归约首维度：m_offset 从向量（`[BLOCK_M]`）改为标量（`pid_m`，因`BLOCK_M=1`，`pid_m * BLOCK_M`等价于`pid_m`），因 2D grid 的第一维范围由`triton.cdiv(M, meta["BLOCK_M"])`约束，`pid_m`天然在`[0, M)`范围内。
  - 归约轴维度、非归约尾维度：完全保留原`tl.arange`并行偏移生成逻辑，仅适配降维后的张量形状调整广播规则。
2. 边界判断适配:
  - 非归约首维度：移除非归约首维度的边界判断，完全删除`mask_m = m_offset < M`及相关逻辑。因优化后`BLOCK_M=1`，2D grid 第一维范围为`triton.cdiv(M, 1)=M`，`pid_m`的取值范围天然为`[0, M-1]`，`m_offset=pid_m`不会越界，无越界风险。
  - 归约轴 / 非归约尾维度：保留原`mask_n/mask_k`逻辑，适配降维形状调整广播规则。
