# 多维 Grid 等价展平示例

本例只展示索引映射。CUDA 原生支持多维 Grid，因此保留原 Grid 也是正确结果；展平候选不得限制到 SM 数。

## 初始代码

```python
@triton.jit
def original_kernel(a_ptr, b_ptr, M, N, K):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    offset = (pid_n * M + pid_m) * K + pid_k
    value = tl.load(a_ptr + offset)
    tl.store(b_ptr + offset, value)

def run():
    M, N, K = 8, 4, 16
    a = torch.randn(N * M * K, device="cuda")
    b = torch.empty_like(a)
    original_kernel[(N, M, K)](a, b, M, N, K, num_warps=4)
    return b
```

## 分析

1. 原 Grid 次序为 `(N, M, K)`，线性任务顺序应与这一维度次序一致。
2. `total_blocks = N * M * K`。
3. 展平后用一个 program 对应一个原逻辑任务，因此 Grid 就是 `(total_blocks,)`，没有 `min` 封顶，也不需要 `tl.num_programs` 循环。

## 普通展平候选

```python
@triton.jit
def original_kernel(a_ptr, b_ptr, M, N, K):
    flat_pid = tl.program_id(0)
    pid_k = flat_pid % K
    pid_m = (flat_pid // K) % M
    pid_n = flat_pid // (M * K)
    offset = (pid_n * M + pid_m) * K + pid_k
    value = tl.load(a_ptr + offset)
    tl.store(b_ptr + offset, value)

def run():
    M, N, K = 8, 4, 16
    total_blocks = N * M * K
    a = torch.randn(total_blocks, device="cuda")
    b = torch.empty_like(a)
    original_kernel[(total_blocks,)](a, b, M, N, K, num_warps=4)
    return b
```

## 校验

- 检查维度顺序与原 Grid 完全一致；不能把 `(N,M,K)` 的恢复公式误写成 `(M,N,K)`。
- 每个 `flat_pid ∈ [0,total_blocks)` 恰好映射到一个 `(pid_n,pid_m,pid_k)`。
- 比较原多维 Grid 与展平 Grid 的精度和真实耗时。CUDA 不要求一维化；无收益则保留多维版本。
- 只有明确的 persistent 候选才记录 `for flat_pid in range(pid,total_blocks,tl.num_programs(0))` 方案；由后续 `gen-autotune-config` 独立调参、按编译后 occupancy 限制 launch，当前轮不写入输出。
