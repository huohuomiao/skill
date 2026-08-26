# BANG C Kernel 信息提取

## 职责

从完整 `.mlu` 源码中静态提取目标 Kernel 的指针、逻辑轴、Task 映射、片上 buffer、搬运、循环与归约信息，输出标准 JSON。只做静态分析，不编译、不运行。

## 目标 Kernel

1. 找到 BANG C Kernel 入口限定符及 Host launch。
2. 若存在多个 Kernel，选择 Host 主 wrapper 中第一个实际产生用户输出的目标 Kernel；多阶段算子同时记录其依赖链。
3. 只把 GDRAM 输入/输出指针作为 `tensors` 顶层条目；shape/stride/标量不作为 tensor。

## 输出格式

```json
{
  "kernel": {
    "name": "row_reduce",
    "launch": {
      "dim_expr": "dim",
      "function_type_expr": "ktype",
      "queue_expr": "queue"
    }
  },
  "tensors": {
    "input": {
      "type": "input",
      "shape": [null, null],
      "stride": [null, 1],
      "axis": ["M", "N"],
      "axis_type": ["PARALLEL", "REDUCE"],
      "tile_size": [null, "TILE_N"],
      "has_loop": [true, true],
      "access": ["GDRAM2NRAM"],
      "local_buffers": ["input_nram"]
    }
  },
  "local_buffers": {
    "input_nram": {
      "space": "NRAM",
      "dtype": "float",
      "bytes_expr": "TILE_N * sizeof(float)",
      "live_range": "load_to_reduce",
      "ping_pong": false
    }
  }
}
```

示例中的 `null` 必须在未知时保留，不能用经验值替换。

## 分析步骤

### Step 1：解析 Host/Device 绑定

- 绑定 Kernel 指针参数与 Host device allocation。
- 从代表性 performance case 的第一个用例恢复 shape/stride；无 performance case 时用 correctness 第一个用例。
- 记录 task dimension、function type、Queue 与 launch 参数。

### Step 2：解析任务坐标

记录所有：

- `taskId/taskDim` 与三维 task ID/dimension。
- `clusterId/coreId/coreDim`（若实际使用）。
- 线性 ID 的整除/取余反解。
- `for (... += taskDim)` 一类跨 tile 循环。

将 Task 坐标映射回逻辑轴，证明覆盖和尾部规则。

### Step 3：解析 GDRAM 地址与搬运

对每个 `__memcpy` 或直接访问：

1. 识别 GDRAM base pointer。
2. 将地址规范化为 `base + Σ(axis_index*stride)`。
3. 区分元素偏移与字节偏移。
4. 记录方向、有效字节、片上对端 buffer、对齐/补齐规则。
5. 多维 stride 无法静态确定时写 `null`。

### Step 4：解析 tile 与循环

- `tile_size` 取控制该轴每批处理元素数的宏、模板/常量或表达式名。
- Task 步长循环或片上分块循环遍历该轴时 `has_loop=true`。
- 片上 buffer 记录 storage space、dtype、字节表达式、live range、复用和 ping-pong。

### Step 5：识别归约轴

建立 `reduced_axis_set`：只要某轴的多个元素经本地 intrinsic/标量累加、多阶段 workspace 或经证明的跨 Task 合并成为更少输出，就标为 `REDUCE`。其余为 `PARALLEL`。

不要仅凭循环或搬运判断归约；必须追踪计算与写回。多 Kernel 归并时记录阶段关系。

### Step 6：校验

- JSON 可解析且只含声明字段。
- shape/stride/axis/axis_type/tile_size/has_loop 长度一致。
- axis_type 仅 `PARALLEL|REDUCE`。
- local buffer 字节表达式可追溯到声明。
- 搬运方向与 input/output/inout 角色一致。
- 未知硬件/容量事实保持 `null`/`N/A`。

## 参考例子

| 文件 | 内容 |
| --- | --- |
| `references/reducemax-axis-example.md` | 行归约与 NRAM 分块 |
| `references/softmax-axis-example.md` | 三遍局部 softmax 与 buffer 复用 |
| `references/flatten-axis-example.md` | 线性 Task ID 还原多维 tile |
