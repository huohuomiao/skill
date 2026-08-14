# BANG C Kernel 获取 Tensor 轴信息

## 返回格式

对每个 GDRAM 输入/输出指针记录：

```json
{
  "input": {
    "type": "input",
    "shape": [null, null],
    "stride": [null, 1],
    "axis": ["M", "N"],
    "axis_type": ["PARALLEL", "REDUCE"],
    "tile_size": ["TILE_M", "TILE_N"],
    "has_loop": [false, true],
    "access": "GDRAM2NRAM",
    "local_buffer": "input_nram"
  }
}
```

`null` 表示源码无法静态确定；禁止用示例常数补空值。

## 分析步骤

### Step 1：绑定 Host 参数

定位 BANG C Kernel 入口与 Host launch，逐项绑定 GDRAM 指针、shape/stride 标量、任务规模和 function type。再从 correctness/performance 的第一个代表性用例恢复真实 shape/stride；无法恢复时写 `null`。

### Step 2：展开任务坐标

把 `taskId/taskDim` 或三维 task ID 展开为逻辑轴的 task 起点。记录：

- 哪个轴由 Task 并行划分。
- 是否通过 `for (tile = taskId; ...; tile += taskDim)` 让每个 Task 处理多个 tile。
- 线性 task ID 如何整除/取余恢复多维坐标。

### Step 3：展开 GDRAM 地址

对每个 `__memcpy` 或直接 GDRAM 访问，将地址规范化为：

```text
base + sum(axis_index * stride)
```

区分元素偏移与字节偏移。记录每个轴的完整 index、stride、tile symbol、循环变量和 tail 计算。只访问标量且无法恢复轴时保守记录，不虚构 tile。

### Step 4：映射片上 buffer

从搬运目的/源识别 NRAM/WRAM/SRAM buffer，记录 dtype、元素/字节表达式、生命周期、是否 ping-pong 和被哪些 intrinsic 使用。一个轴的 `tile_size` 是控制该轴每批处理量的编译期符号或明确表达式。

### Step 5：确定轴角色

- 多个元素被合并为更少输出：`REDUCE`。
- 保持独立输出：`PARALLEL`。
- Task 循环或片上 tile 循环使用该轴时 `has_loop=true`。

归约判断必须追踪完整计算和写回，不能仅因存在循环就认定。

### Step 6：校验

- 所有数组字段长度一致。
- 每个指针的访问方向与 input/output 类型相容。
- 字节表达式与 dtype 一致。
- task 映射覆盖全部逻辑轴且尾部规则明确。
- 结果为标准 JSON。

## 简例

```cpp
__mlu_global__ void row_reduce(const float *input, float *output,
                               int64_t M, int64_t N) {
  __nram__ float input_nram[TILE_N];
  for (int64_t row = taskId; row < M; row += taskDim) {
    for (int64_t begin = 0; begin < N; begin += TILE_N) {
      int64_t valid = min((int64_t)TILE_N, N - begin);
      __memcpy(input_nram, input + row * N + begin,
               valid * sizeof(float), GDRAM2NRAM);
      // local reduction and accumulation
    }
    // write one output for row
  }
}
```

分析结果：M 为并行轴且有 Task 步长循环；N 为归约轴且有片上 tile 循环；连续 stride 为 1；`input_nram` 的有效搬运字节为 `valid*sizeof(float)`。
