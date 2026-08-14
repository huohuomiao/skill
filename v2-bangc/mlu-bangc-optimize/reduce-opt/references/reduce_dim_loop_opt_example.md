# 归约轴分块优化示例

## 原始骨架

```cpp
__mlu_global__ void row_sum(const float *input, float *output,
                            int64_t rows, int64_t cols) {
  __nram__ float input_nram[REDUCE_TILE];
  for (int64_t row = taskId; row < rows; row += taskDim) {
    float acc = 0.0f;
    for (int64_t begin = 0; begin < cols; begin += REDUCE_TILE) {
      int64_t valid = min((int64_t)REDUCE_TILE, cols - begin);
      __memcpy(input_nram, input + row * cols + begin,
               valid * sizeof(float), GDRAM2NRAM);
      // 将 valid 个元素局部求和并合入 acc。
    }
    output[row] = acc;
  }
}
```

局部求和 API 由当前原语表选择。

## 候选 A：整轴一次处理

仅当 `cols*sizeof(float)` 加其他同时存活 buffer 在已确认片上容量内，且局部归约路径支持该长度时，令配置的 `REDUCE_TILE` 覆盖整轴并消除内层循环。不能使用固定元素阈值代替字节/生命周期核算。

## 候选 B：保留分块并调 tile

当整轴不安全时保留循环，在当前值附近生成有限 tile 候选。每个候选：

- 尾块有效字节正确。
- 补齐区按 sum 单位元初始化。
- 累加 dtype 保持精度。
- 编译器无片上资源错误。
- 全部 Shape correctness 通过。
- MLU590 notifier 无回退。

## 不触发示例

- 循环遍历输出行而非归约轴。
- 归约长度或 buffer 生命周期无法确定。
- stride/gather 使一次整轴搬入不成立。
- 当前实现已使用同等 tile 与复用，生成代码无变化。

报告记录实际 bytes expression 和编译证据，不写“长度小于某经验常数所以安全”。
