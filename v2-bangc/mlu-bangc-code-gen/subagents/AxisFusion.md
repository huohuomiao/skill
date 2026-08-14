# AxisFusion：BANG C tile 与片上访存规划

## 任务

根据逻辑维度、stride 和初始 task mapping，判断哪些轴可安全线性化，如何形成连续 GDRAM DMA 段，如何规划 NRAM/WRAM/SRAM，以及是否存在有依据的 `__bang_*` 计算或片上重排机会。输出 `step3_axis_fusion.json`。

文件名保留 AxisFusion 以维持原六阶段结构；这里的“融合”是索引域与访存 tile 的融合，不表示自动合并多个 kernel。

## 输入

- `step1_base_info.json`
- `step1_io_shapes.json`
- `step2_block_mapping.json`
- `.claude/skills/share/mlu/references/platform-rules.md`
- `.claude/skills/share/mlu/references/primitives.md`

## 约束

1. 只有 stride 关系被 requirement/Stage 1 证明时才能融合逻辑轴。
2. 融合不能改变 padding、广播、负 stride、切片或 aliasing 语义。
3. 优先让一个 task 搬运长连续 GDRAM 段，并在 NRAM 中复用；不能把非连续访问描述成单段 DMA。
4. `__memcpy` 的方向、起止地址、字节数和 tail 必须明确。
5. `__bang_*` 仅在共享原语表确认签名、dtype、长度与对齐后启用；否则保留标量/分块 baseline。
6. `__nram__`、`__wram__`、`__sram__` 的 byte 公式必须可复算；没有真实容量事实时不能宣称资源合法。
7. 归约轴与输出并行轴不能机械融合；必须保持聚合语义与唯一写入。
8. baseline 不依赖跨 task 同步或未经确认的 cluster/union 行为。
9. 本阶段不运行性能测试，也不声称某个 tile 最优。

## 维度融合判定

对 tensor 的相邻轴 `d` 与 `d+1`，通常只有满足下式才可把二者作为同一连续地址域：

```text
stride[d] == shape[d+1] * stride[d+1]
```

还必须满足：

- 两轴没有独立广播或 reduction 语义。
- 所有参与 tensor 都能从融合 index 恢复正确 offset，或各自有明确转换。
- 轴乘积与 byte offset 不溢出选定类型。
- 片上 buffer 与搬运方案保留原 layout。

连续 `[B,M,N]` 的逐元素算子可融合为 `total=B*M*N`；task 只需处理线性 tile。仍需每轴坐标时使用 64 位安全反解：

```cpp
int64_t t = linear;
const int64_t n_idx = t % n;
t /= n;
const int64_t m_idx = t % m;
const int64_t b_idx = t / m;
```

host 先检查 `B*M*N` 与字节乘积不溢出。

## 不可直接融合的场景

### Broadcast

`y[b,m,n] = x[b,m,n] + bias[n]` 的输出域可线性化，但 bias offset 仍为 `linear % N`。可把 bias 片段搬入 NRAM 复用，但需证明其范围、容量和更新频率。

### Padded/non-contiguous stride

若 `stride_m > N*stride_n`，M/N 间有 padding。task 域可以扁平，GDRAM offset 必须用 runtime stride 重建；每行分别 DMA 或使用保守访问。

### Transpose

`y[n,m]=x[m,n]` 的 input/output 连续方向不同。规划 input NRAM tile 与 output NRAM tile，在片上转置后按另一方向逐行写回；不能宣称两端由同一线性地址直接复制。

### Reduction

`y[m]=sum_k x[m,k]` 中 M 是输出 tile 轴，K 是归约轴。一个 task 应遍历 K 的 NRAM chunks，并唯一写 y tile；不能把 M/K 同时当成独立输出 tile。

## GDRAM ↔ NRAM 规划

对每个 buffer 记录：

- address space：GDRAM/NRAM/WRAM/SRAM。
- element type、shape、alignment evidence。
- declaration 与静态 byte 公式。
- producer/consumer 生命周期。
- copy direction、source offset、destination offset、byte count。
- full tile 与 tail 的不同路径。

连续逐元素 baseline：

```text
x GDRAM --GDRAM2NRAM(valid*sizeof(T))--> x_nram
compute valid elements (or initialize padded lanes then run verified full-width intrinsic)
y_nram --NRAM2GDRAM(valid*sizeof(T))--> y GDRAM
```

不得从 tail GDRAM 多读到 tile 上界。若 intrinsic 必须处理固定/对齐长度，先在 NRAM 初始化无效 lane，再只复制有效输入；写回仍只写有效 bytes。

## `__bang_*` 门禁

启用 intrinsic 前同时确认：

- 名称和参数顺序在目标 `bang.h` 中存在。
- 支持当前 input/output/accumulator dtype。
- 元素数或字节数的单位明确。
- 长度、地址和片上 buffer 对齐满足要求。
- tail 的 padding identity 不改变语义。
- 数值误差符合 requirement。

输出示例：

```json
{
  "enabled": true,
  "primitive": "__bang_add",
  "evidence": "share/mlu/references/primitives.md and target bang.h",
  "dtype": "float32",
  "length_expression": "TILE_ELEMS",
  "tail_strategy": "zero padded NRAM input; copy back valid elements only",
  "scalar_fallback": true
}
```

证据不足时：

```json
{
  "enabled": false,
  "reason": "target signature/alignment is not confirmed",
  "fallback": "scalar loop over valid NRAM elements",
  "defer_to_optimization": true
}
```

不要编造 `__bang_*` 名称。

## WRAM 与 SRAM

- `__wram__` 仅在算子/已确认 intrinsic 需要权重布局时使用，记录布局转换与 bytes。
- `__sram__` 仅在共享规则确认 cluster 参与集合、可见性、同步和容量后使用。
- 第一版 baseline 若不需要二者，显式写 `enabled=false`，不为“看起来更像 MLU 优化”而占用。
- 任何跨 core/cluster 数据交换必须有真实同步原语和参与集合证据，否则回退为 task 独立方案或多 kernel。

## 转置片上重排

记录：

```text
input NRAM tile:  [rows][cols]
output NRAM tile: [cols][rows]
load:  each valid input row with GDRAM2NRAM
reorder: verified intrinsic or scalar nested loop
store: each valid output row with NRAM2GDRAM
```

若使用两个 NRAM tile，总 bytes 为二者之和；若原地重排必须证明不会覆盖未读取元素。tail tile 的每个行/列边界独立计算。

## 输出 Schema

```json
{
  "schema_version": 1,
  "operator_name": "transpose2d",
  "axis_plan": {
    "logical_axes": ["M", "N"],
    "fused_groups": [],
    "preserved_axes": ["M", "N"],
    "linearization": "flat tile id over tiles_m * tiles_n",
    "inverse_mapping": ["tile_m=tile/tiles_n", "tile_n=tile%tiles_n"],
    "continuous_dma_axis": {"input": "N", "output": "M"}
  },
  "task_plan": {
    "mapping": "taskId/taskDim grid-stride over flat tiles",
    "tile_shape": ["TILE_M", "TILE_N"],
    "task_count_source": "EnvConfig/spec",
    "inter_task_sync": "none"
  },
  "access_analysis": [
    {
      "tensor": "x",
      "mode": "read",
      "contiguous_axis": "N",
      "classification": "row_segment_dma",
      "copy_bytes": "valid_n * sizeof(float)",
      "evidence": "input stride along N is 1"
    }
  ],
  "onchip_memory": {
    "nram": [
      {"name": "input_tile", "declaration": "__nram__ float input_tile[TILE_M*TILE_N]", "bytes": "TILE_M*TILE_N*sizeof(float)"},
      {"name": "output_tile", "declaration": "__nram__ float output_tile[TILE_M*TILE_N]", "bytes": "TILE_M*TILE_N*sizeof(float)"}
    ],
    "wram": {"enabled": false, "bytes": 0},
    "sram": {"enabled": false, "bytes": 0},
    "total_nram_bytes": "2*TILE_M*TILE_N*sizeof(float)",
    "capacity_evidence": null
  },
  "intrinsic_plan": {
    "enabled": false,
    "primitive": null,
    "fallback": "scalar NRAM transpose",
    "reason": "example does not assume a transpose intrinsic"
  },
  "tail_plan": "per-row valid lengths; never over-read or over-write GDRAM",
  "index_reuse": ["precompute input/output row bases"],
  "changes_from_step2": ["add two-buffer NRAM transpose plan"],
  "deferred_candidates": [],
  "risks": [],
  "ready_for_spec": true
}
```

示例值必须替换为真实算子设计。

## 参考场景

GenerateSpec/GenerateCode 按 pattern 读取对应文件：

- 连续逐元素/轴融合：[generate_spec_axis_fusion.md](./examples/generate_spec_axis_fusion.md)、[generate_code_axis_fusion.md](./examples/generate_code_axis_fusion.md)
- 矩阵转置：[generate_spec_matrix_transpose.md](./examples/generate_spec_matrix_transpose.md)、[generate_code_matrix_transpose.md](./examples/generate_code_matrix_transpose.md)
- 归约：[generate_spec_reduce_sum.md](./examples/generate_spec_reduce_sum.md)、[generate_code_reduce_sum.md](./examples/generate_code_reduce_sum.md)
- 转置加逐元素：[generate_spec_transpose_elementwise.md](./examples/generate_spec_transpose_elementwise.md)、[generate_code_transpose_elementwise.md](./examples/generate_code_transpose_elementwise.md)

## 验证

- 每个融合组都有逐 tensor stride 证明。
- 每个 DMA 段都有方向、byte offset、byte count 与 tail。
- NRAM/WRAM/SRAM 声明 byte 数可复算；容量结论有 EnvConfig/编译证据。
- intrinsic 选择有目标原语证据与 scalar/tail 路径。
- 线性化与逆映射使用足够宽的整数。
- 不依赖未确认的跨 task/cluster 同步。
- `changes_from_step2` 不破坏 coverage proof；若改变 mapping，要同步提供完整证明。
- `risks=[]` 且 `ready_for_spec=true` 后才能交接。

## 保守回退

- 无法证明连续：保留原轴，逐行/分段搬运。
- 对齐或 intrinsic 不明确：标量 NRAM loop。
- 容量事实缺失：缩小 tile 并交由真实编译验证，不声称已适配上限。
- transpose 无可靠原语：双 NRAM buffer 标量重排。
- SRAM/WRAM 语义不明确：禁用。
- 把潜在优化写入 `deferred_candidates`，留给 optimize 在实机测量。
