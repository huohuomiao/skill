# Softmax 轴分析例子

## 源码骨架

```cpp
__mlu_global__ void row_softmax(const float *input, float *output,
                                int64_t rows, int64_t cols) {
  __nram__ float tile_nram[TILE_N];
  for (int64_t row = taskId; row < rows; row += taskDim) {
    // pass 1: 分块读取 cols，得到 row max
    // pass 2: 分块读取 cols，累加 exp(x-max)
    // pass 3: 分块读取 cols，归一化并写回有效元素
  }
}
```

具体 exponential/reduction intrinsic 必须来自当前共享原语/数学文档；骨架不声明某个 API 必然存在。

## 轴信息

```json
{
  "input": {
    "type": "input",
    "shape": [null, null],
    "stride": [null, 1],
    "axis": ["ROW", "COL"],
    "axis_type": ["PARALLEL", "REDUCE"],
    "tile_size": [null, "TILE_N"],
    "has_loop": [true, true],
    "access": ["GDRAM2NRAM"],
    "local_buffers": ["tile_nram"]
  },
  "output": {
    "type": "output",
    "shape": [null, null],
    "stride": [null, 1],
    "axis": ["ROW", "COL"],
    "axis_type": ["PARALLEL", "PARALLEL"],
    "tile_size": [null, "TILE_N"],
    "has_loop": [true, true],
    "access": ["NRAM2GDRAM"],
    "local_buffers": ["tile_nram"]
  }
}
```

COL 对输入统计阶段是归约轴，但输出仍逐元素写回；因此 input/output 对同一逻辑轴的角色可不同。

## 复用分析

若三遍都从 GDRAM 读取同一行，不能仅凭地址相同就删除后两次搬运：整行可能放不进 NRAM，且 pass 间需要 max/sum。只有片上容量、buffer 生命周期和编译/性能证据证明可保留输入时，才生成一次加载候选。

尾 tile 的 exponential、sum 和写回必须排除无效元素；填充值选择影响精度。
