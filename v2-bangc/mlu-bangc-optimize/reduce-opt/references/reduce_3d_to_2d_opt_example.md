# 三维片上 tile 转二维示例

## 原始逻辑

输入逻辑 `[A,R,B]`，每轮把 `[TILE_A,TILE_R,TILE_B]` 搬入片上并沿 R 归约，局部结果为 `[TILE_A,TILE_B]`。如果当前 intrinsic 对该布局效率低，可测试 `TILE_A=1` 的二维局部表示。

## 候选骨架

```cpp
__mlu_global__ void reduce_arb(const float *input, float *output,
                               int64_t A, int64_t R, int64_t B) {
  int64_t tiles_b = ceil_div(B, (int64_t)TILE_B);
  int64_t total = A * tiles_b;
  for (int64_t tile = taskId; tile < total; tile += taskDim) {
    int64_t a = tile / tiles_b;              // TILE_A 固定为 1 的标量轴
    int64_t tb = tile - a * tiles_b;
    int64_t b0 = tb * TILE_B;
    int64_t valid_b = min((int64_t)TILE_B, B - b0);

    // local accumulator shape: [TILE_B]
    // for r0 in [0,R) by TILE_R:
    //   搬入逻辑 [valid_r, valid_b] 到连续片上二维 buffer
    //   沿 R 做局部归约并合入 accumulator
    // 仅写回 valid_b 个元素到 output[a, b0:]
  }
}
```

`ceil_div` 表示源码中安全的整数表达式/项目 helper，不是某个必然存在的设备 API。

## 必须同步修改

- A 从片上向量轴变为 task/循环标量轴。
- GDRAM 地址仍按原始 `[A,R,B]` stride 计算。
- 局部 buffer 从三维降为二维，accumulator 从 `[TILE_A,TILE_B]` 降为 `[TILE_B]`。
- R/B 尾部与补齐单位元正确。
- Host task dimension 覆盖 `A*tiles_b`，function type 不无证据改变。

## 门禁

先比较片上峰值和生成代码，再运行全精度和 MLU590 notifier。若二维搬运引入低效 stride/gather，或 Task 数/标量开销导致回退，恢复原始实现。
