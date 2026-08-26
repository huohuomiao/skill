# BANG C Kernel 获取 Tensor 轴信息

本文件与 `config-tuner/references/get-tensor-axis-info.md` 保持同一 schema，供离线配置生成独立加载。

## 输出 schema

```json
{
  "pointer_name": {
    "type": "input|output|inout",
    "shape": [null],
    "stride": [1],
    "axis": ["N"],
    "axis_type": ["PARALLEL"],
    "tile_size": ["TILE_N"],
    "has_loop": [true],
    "access": "GDRAM2NRAM|NRAM2GDRAM|direct",
    "local_buffer": "buffer_name|null"
  }
}
```

未知事实写 `null`。

## 分析流程

1. 定位 Kernel 入口与 Host launch，绑定指针、shape/stride、任务规模和 function type。
2. 从代表性测试恢复 shape/stride；不要运行源码，不要执行动态 import。
3. 展开 `taskId/taskDim` 和三维 task ID，恢复各逻辑轴的并行映射与步长循环。
4. 将每个 GDRAM 地址化为 `base + Σ(index_i*stride_i)`，明确元素/字节单位。
5. 将每次 `__memcpy` 关联到 NRAM/WRAM/SRAM buffer、有效字节、对齐长度和生命周期。
6. 追踪 intrinsic/标量计算与最终写回，判定 `PARALLEL` 或 `REDUCE`。
7. 记录控制每轴每批工作量的 `TILE_*`/常量及其是否参与循环。
8. 校验字段长度、方向、dtype、覆盖和尾块后输出标准 JSON。

## 多轴线性化

若使用线性 `taskId` 处理二维 tile：

```cpp
int64_t tile = taskId;
int64_t tile_m = tile / tiles_n;
int64_t tile_n = tile - tile_m * tiles_n;
```

则 M/N 都由同一 Task 维度并行划分。若外层为 `tile += taskDim`，二者 `has_loop=true`。除法/取模仅说明坐标恢复，不说明哪个轴是归约轴；归约角色必须从计算和输出数量判断。

## 配置生成所需补充

除轴信息外还输出：

- 每个片上 buffer 的 `bytes_expr` 与 live range。
- 是否存在单/双 buffer 或异步搬运。
- 当前 task dimension/function type 表达式。
- 当前 notifier benchmark 的 Shape 列表。

这些信息用于筛除不安全候选，不用于静态声称最佳配置。
