# Div-to-mul

## 功能概述
识别 Triton Kernel 中的除法运算，并将其重构为“乘倒数”形式（即 $x / y \rightarrow x \times (1.0 / y)$）。通过减少高开销的除法指令执行次数，利用乘法单元的高吞吐量提升性能。

---

## Step 1：场景识别 (Pattern Recognition)

**识别准则**：在 Kernel 代码中搜索所有张量除法运算。

只要出现张量除法，即标记为潜在优化点，并按以下位置逻辑执行重构：
1. **优先位置**：若除数被广播（如 `y[:, None]`）或位于循环内。
2. **常规位置**：普通元素级除法。

---

## Step 2：优化执行策略

### ：代码重构重写
1. **插入倒数计算**：在除数维度发生**任何扩张动作之前**，插入 `inv_D = 1.0 / D`。
2. **变换广播对象**：将后续的除法替换为对 `inv_D` 执行广播后的乘法。

**重构示例：**
```python
# [Original]
row_sum = tl.sum(x, axis=1)
result = x / row_sum[:, None]  # 除法发生在扩展后的 [M, N] 空间

# [Refactored]
row_sum = tl.sum(x, axis=1)
inv_row_sum = 1.0 / row_sum    # 核心：在扩展前执行 M 次除法
result = x * inv_row_sum[:, None]  # 转换后执行 M*N 次高速乘法
```
```python
# [Original]

y = 0.5 * x_val * (1.0 + tl.erf(x_val / 1.4142135623730951))

# [Refactored]

inv_constant = 1.0 / 1.4142135623730951
y = 0.5 * x_val * (1.0 + tl.erf(x_val * inv_constant))

```

---

## Step 3：运行与验证逻辑
 
运行生成的代码
- 若精度测试未通过，若运行优化 kernel 抛出错误或精度不正确，则根据错误信息进行调试修改，最多尝试 3 次，如果 3 次修正后仍抛出错误或无法满足精度要求，则输出原始代码
- 若精度测试通过，如果代码性能提升则保留该代码，若下降则回退回初始代码。
运行精度测试与性能测试对优化 kernel 进行验证。