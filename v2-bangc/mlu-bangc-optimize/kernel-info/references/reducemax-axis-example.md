# ReduceMax 轴分析例子

## 源码骨架

```cpp
__mlu_global__ void row_max(const float *input, float *output,
                            int64_t M, int64_t N) {
  __nram__ float input_nram[TILE_N];
  for (int64_t row = taskId; row < M; row += taskDim) {
    float acc = NEGATIVE_IDENTITY;
    for (int64_t begin = 0; begin < N; begin += TILE_N) {
      int64_t valid = min((int64_t)TILE_N, N - begin);
      __memcpy(input_nram, input + row * N + begin,
               valid * sizeof(float), GDRAM2NRAM);
      // 用当前原语清单支持的局部 max 路径更新 acc。
    }
    output[row] = acc;
  }
}
```

示意代码中的单位元和局部归约 API 必须按 dtype、特殊值语义及当前头文件补全，不能直接复制为成品。

## 分析

- `input` shape 为 `[M,N]`，stride 为 `[N,1]`（只有 wrapper/测试能证明时填写数值）。
- M 轴由 `row=taskId; row+=taskDim` 划分：`PARALLEL`、`has_loop=true`、无独立片上 tile symbol。
- N 轴由 `begin+=TILE_N` 遍历且被合并到一个输出：`REDUCE`、`tile_size=TILE_N`、`has_loop=true`。
- 搬运为连续 `GDRAM2NRAM`，有效字节为 `valid*sizeof(float)`。
- `input_nram` 是 NRAM buffer，live range 为每轮 load 到局部 max 完成。
- `output[row]` 每行仅由一个 Task 写入，无跨 Task 归并。

## 关键检查

- 最后一块的未使用片上元素不能参与 max，或必须填正确单位元。
- 低精度输入的比较/特殊值行为必须与 reference 一致。
- M 小于 Task 数时空闲 Task 不得越界写。
