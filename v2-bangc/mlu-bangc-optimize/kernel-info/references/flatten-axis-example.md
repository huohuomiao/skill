# 多维轴线性 Task 映射例子

## 源码骨架

```cpp
__mlu_global__ void transform(const float *input, float *output,
                              int64_t N, int64_t H, int64_t W,
                              int64_t tiles_h, int64_t tiles_w) {
  int64_t total_tiles = N * tiles_h * tiles_w;
  for (int64_t flat = taskId; flat < total_tiles; flat += taskDim) {
    int64_t n = flat / (tiles_h * tiles_w);
    int64_t rem = flat - n * tiles_h * tiles_w;
    int64_t th = rem / tiles_w;
    int64_t tw = rem - th * tiles_w;
    // 由 n/th/tw 计算本 Task 的 GDRAM offset，搬入片上、计算、写回。
  }
}
```

## 分析

- N/H/W 都映射到同一个线性 Task 维度。
- `flat += taskDim` 使三个轴的 tile 都可能由同一 Task 循环处理，故相关 `has_loop=true`。
- `th/tw` 是 tile 坐标；实际 H/W 元素坐标还需乘对应 tile 大小并处理尾部。
- 除法/取余只是坐标反解，不能据此把任何轴标为归约轴。
- 每个 tensor 必须按自己的 shape/stride 构造 GDRAM 地址；输入输出 layout 不同时不可复用 mapping。

## 输出示例

```json
{
  "axis": ["N", "H", "W"],
  "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
  "tile_size": [null, "TILE_H", "TILE_W"],
  "has_loop": [true, true, true]
}
```

实际 shape/stride 和 tile symbol 必须来自源码；示例只说明结构。

## 风险

- `N*tiles_h*tiles_w` 在乘法前使用足够位宽。
- `tiles_w` 不能为零；空 Shape 在 Host 侧有明确处理。
- 尾部有效字节与片上对齐长度分离。
- 简化 div/mod 后仍保持非负整数语义。
