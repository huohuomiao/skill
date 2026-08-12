## 情况 A：无法提取 Grid

**场景**：只有 kernel 定义，没有调用语句，或 launch 表达式无法回溯。

```python
@triton.jit
def elementwise_add(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)

# 未提供 wrapper / launch
```

## 分析步骤

1. 已知每个 program 看似处理一个 `BLOCK_SIZE`，但不知道真实 shape、meta 参数、wrapper 约束和调用次数。
2. 不能据此虚构 `n_elements` 来源、BLOCK_SIZE、Grid 或输出分配。
3. 也不能判断该 kernel 是否被其它包装器以多种 Grid 调用。

**结论**：保持原代码，在报告中写明“缺少可解析 launch，跳过 modify-grid”。请求调用方补充完整 wrapper 后再分析。

以下做法均禁止：

- 默认生成 `grid=min(cdiv(n_elements, BLOCK_SIZE), sm_count)`；普通 CUDA kernel 会因此遗漏尾部逻辑块。
- 凭空添加 `tl.num_programs` 循环；这会改变 kernel 结构且没有实测依据。
- 猜测设备 SM 数或使用固定 RTX 3090 数值。
- 为缺失的 wrapper 猜测输入 shape、dtype、输出初始化或精度阈值。

若完整调用稍后可用，应回到主策略 Step 1，先建立普通 Grid 基线，再决定是否生成展平或 persistent 候选。
