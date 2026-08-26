# Div-to-Mul：BANG C 除法候选

## 功能

只对真实热点且语义允许的除法生成一个“预计算倒数并复用乘法”候选。整数/浮点、常量/runtime 除数、标量/向量路径和快速数学 intrinsic 必须分开处理。

## Step 1：扫描与分类

| 类别 | 默认动作 |
| --- | --- |
| 整数常量除数 | 保持；先看 CNCC/MLISA 是否已强度削减 |
| 整数 runtime 除数 | 保持，除非存在严格等价算法 |
| 浮点常量除数 | 先检查生成代码和精度契约 |
| 浮点 runtime 不变量 | 可尝试每 Task/每 tile 计算一次 reciprocal |
| 被广播/循环复用的除数 | 优先候选，但需证明生命周期不变 |
| 当前 SDK 的 fast reciprocal | 仅在近似预算明确时作为单独候选 |

CNPerf/MLISA 没有显示除法相关热点时，不因“乘法通常更快”自动改写。

## Step 2：语义门禁

替换前确认：

- 除数在复用范围内不变。
- ±0、Inf、NaN、subnormal 和舍入行为符合原测试契约。
- 不改变整数截断、负数除法或余数语义。
- 不把 double/高精度中间值静默降位。
- reciprocal 的 dtype 与 intrinsic 支持来自当前头文件/共享数学文档。
- reciprocal 被复用多次；只用一次通常没有依据。

示例：

```cpp
float inv = 1.0f / denominator;
for (int i = 0; i < count; ++i) {
  output_nram[i] = input_nram[i] * inv;
}
```

若存在经确认的向量 intrinsic，可用它生成另一个独立候选；不要在同轮同时改变 tile 或数学近似。

## Step 3：验证

1. 只改变目标除法。
2. 相同 flags 编译，保存 MLISA/编译器证据（可用时）。
3. correctness 覆盖普通值、极小/极大除数、±0、Inf、NaN 和 subnormal（契约适用时），不放宽容差。
4. MLU590 公平 benchmark 无收益或落在噪声内即回退。
5. 快速候选性能提升但精度失败仍必须回退。

输出 `bangc_optimized.mlu` 和完整 keep/revert 报告。
