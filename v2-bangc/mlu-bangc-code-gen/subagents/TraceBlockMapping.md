# TraceBlockMapping

## 任务

为逻辑输出域设计 BANG C task/tile 映射并写入 `step2_block_mapping.json`。必须以 `taskId`、`taskDim` 和 host 侧 `cnrtDim3_t` 表达覆盖关系，证明输出无遗漏、无意外重叠，并处理尾 tile、空输入、归约归属和大索引。

TraceBlockMapping 是 `mlu-bangc-code-gen` 的 Stage 2。它保留原阶段名，但“block”表示一个 task 负责的逻辑 tile，不等同于其它平台的线程块。

## 输入

- `step1_base_info.json`
- `step1_io_shapes.json`
- `requirement.md`
- `.claude/skills/share/mlu/references/platform-rules.md`

若 `ready_for_mapping=false`，停止并返回 blocking questions，不得设计映射。

## 设计原则

1. correctness 优先于并行度或短代码。
2. 一个 task 默认独占一个或多个输出 tile；在 task 内用 NRAM 分块计算。
3. baseline 优先使用扁平 `taskId/taskDim`，避免依赖未经确认的多维 task builtin。
4. task 以 grid-stride 方式覆盖逻辑 tile，使 launch task 数可以由 EnvConfig 的合法上限约束。
5. 连续 GDRAM 段优先形成大块 `__memcpy`；非连续访问必须明确分段、逐行或 gather 方案。
6. 每个 load/store 都能追溯到逻辑坐标、字节 offset、有效元素数和 tail 处理。
7. 不在本阶段声称 task 数或 tile 大小最优；容量、对齐和 arch 只能引用共享环境事实。

## 1D task 映射

标准逐元素 tile 覆盖：

```cpp
const int64_t num_tiles = (n + TILE_ELEMS - 1) / TILE_ELEMS;
for (int64_t tile = static_cast<int64_t>(taskId);
     tile < num_tiles;
     tile += static_cast<int64_t>(taskDim)) {
  const int64_t begin = tile * TILE_ELEMS;
  const int64_t valid = min_i64(TILE_ELEMS, n - begin);
  // 只搬运 valid * sizeof(T) 字节；计算后只写回 valid 个元素。
}
```

host launch 的逻辑形式：

```cpp
const uint32_t task_count = choose_legal_task_count(num_tiles, env_facts);
cnrtDim3_t dim = {task_count, 1, 1};
cnrtFunctionType_t function_type = cnrtFuncTypeBlock;
```

- `n==0` 时 launcher 不发出零维或无意义 launch。
- `choose_legal_task_count` 必须在 spec 中落成确定公式或已验证值；本阶段不能按 MLU590 名称硬编码核心数。
- `taskDim` 必须大于 0；index 乘加先扩展到 64 位。

## 多维映射与扁平反解

将输出 tile 域按确定顺序扁平化。例如 `[tiles_m, tiles_n]`：

```cpp
const int64_t flat_tile = tile;
const int64_t tile_m = flat_tile / tiles_n;
const int64_t tile_n = flat_tile - tile_m * tiles_n;
```

三维及以上继续从最后一维向前反解。必须记录：

- 每个逻辑轴的 tile 数公式。
- 扁平顺序与逆映射。
- 所有乘积的溢出检查。
- 哪一轴在 GDRAM 中连续，单次 DMA 覆盖多少字节。

只有共享平台规则与目标环境明确支持并需要 `taskIdX/Y/Z` 时才使用多维 builtin；否则保持扁平映射。

## Reduction 映射

baseline 优先让一个 task 独立完成一个输出 tile 的全部归约：

```text
task tile -> 一组唯一输出元素
for reduce_begin in [0, reduce_extent) step REDUCE_TILE:
    GDRAM -> NRAM 搬入当前 chunk
    在 task 内做 partial reduction/累积
当前 task 写回唯一输出 tile
```

必须明确：

- 输出 tile 数与 task tile 的关系。
- reduction extent 为 0 时的 identity。
- NRAM accumulator dtype 与 shape。
- tail chunk 的填充值和有效元素数。
- 是否使用已确认的 `__bang_*` reduction；否则用保守循环。
- 最终写入由一个 task 完成，或说明 atomic/multi-kernel 的数学必要性。

不同 task 的局部结果不能依赖不存在的全 device barrier。若一个输出必须跨 task 合并，baseline 应改成：

1. 第一 kernel 写唯一 partial buffer 区域。
2. 同一 queue 上第二 kernel 合并；或使用经共享原语表确认且符合确定性合同的 atomic。

不得通过普通 GDRAM load/store 模拟原子更新。

## Transpose 映射

二维 transpose 的 baseline：

```text
flat task -> input/output logical tile
按 input 连续方向把若干行 GDRAM2NRAM
在 NRAM 中按 tile 坐标重排
按 output 连续方向把若干行 NRAM2GDRAM
边界 tile 只读写有效矩形
```

需要记录：

- input/output 的逻辑 shape 与元素 stride。
- 每一行 DMA 的 source/destination byte offset 和 byte count。
- NRAM input/output tile 的声明和总字节数。
- tile 内转置使用已确认 intrinsic 还是标量循环。
- tail 行/列不会读未初始化 NRAM，也不会越界写 GDRAM。

若双向都无法形成连续 DMA，诚实标记 `strided_or_irregular`，不要声称连续搬运。

## Broadcast、gather、scatter

- broadcast：输出 task 唯一写入；小输入可在 NRAM 中复用，但 offset 仍按真实轴计算。
- gather：每个 index 的 dtype 与越界语义来自 requirement；无法用单段 DMA 时明确逐元素/分段访问。
- scatter：先证明目标唯一；无法证明时要求 atomic/分阶段策略或返回阻塞问题。
- requirement 要求确定性时，不能用非确定 atomic 再通过放宽容差绕过。

## 访存与 tail Schema

每类访问至少写：

```json
{
  "tensor": "x",
  "mode": "read",
  "address_space": "GDRAM",
  "logical_access": "x[row * stride_x0 + col * stride_x1]",
  "byte_offset": "(row * stride_x0 + col * stride_x1) * sizeof(float)",
  "copy_direction": "GDRAM2NRAM",
  "contiguous_elements": "valid_cols",
  "copy_bytes": "valid_cols * sizeof(float)",
  "tail": "copy only valid_cols; initialize any padded NRAM lanes before full-width intrinsic",
  "classification": "contiguous_if_stride_x1_equals_1",
  "evidence": "step1 stride_x1 is 1"
}
```

`__memcpy` 第三个参数是字节数。若后续为了 intrinsic 对齐而填充 NRAM，GDRAM copy 仍只能覆盖有效区间。

## 输出 Schema

```json
{
  "schema_version": 1,
  "operator_name": "affine_relu",
  "mapping_kind": "flat_task_tile_grid_stride",
  "logical_domain": {
    "axes": [{"name": "i", "range": "0 <= i < N"}],
    "total_elements": "N",
    "tile_axes": [{"name": "i", "tile": "TILE_ELEMS"}],
    "total_tiles": "ceil_div(N, TILE_ELEMS)"
  },
  "launch": {
    "dim": {"x": "task_count", "y": 1, "z": 1},
    "task_count": "min(total_tiles, verified_launch_task_cap)",
    "function_type": "cnrtFuncTypeBlock",
    "queue": "queue",
    "zero_size": "return without launch",
    "facts_required": ["verified_launch_task_cap or an explicit legal task_count"]
  },
  "task_coordinates": [
    {
      "logical": "tile",
      "initial": "int64_t(taskId)",
      "step": "int64_t(taskDim)",
      "range": "tile < total_tiles"
    },
    {
      "logical": "i_begin",
      "expression": "tile * TILE_ELEMS",
      "valid": "min(TILE_ELEMS, N - i_begin)"
    }
  ],
  "coverage": {
    "writes_per_valid_output": "exactly_one",
    "overlap": "none",
    "tail_handling": "valid element count and byte-sized copies",
    "proof": "tile residue modulo taskDim selects exactly one task; each tile owns a disjoint interval"
  },
  "memory_accesses": [],
  "reduction": null,
  "index_types": {
    "logical_index": "int64_t",
    "task_builtin": "implementation-defined builtin widened before arithmetic",
    "overflow_checks": ["N >= 0", "tile * TILE_ELEMS does not overflow int64_t"]
  },
  "inter_task_synchronization": "none",
  "atomics": [],
  "tile_candidates": [],
  "selected_baseline_tile": null,
  "selection_reason": "defer until NRAM byte formula and environment facts are available",
  "open_issues": []
}
```

不同 pattern 可扩展字段，但固定顶层字段不得缺失。

## 覆盖证明

设总 tile 数为 `L`、launch 后 `taskDim=S>0`。task `r=taskId` 覆盖：

```text
r, r+S, r+2S, ... < L
```

任意 tile `t` 唯一表示为 `t = (t mod S) + floor(t/S)*S`，因此只由 task `t mod S` 处理。每个 tile 的元素区间 `[t*T, min((t+1)*T,N))` 互不重叠且并集为 `[0,N)`。必须用足够宽的整数并证明加法步进不会溢出。

## 资源与平台检查

从共享平台规则或 EnvConfig 获取并记录：

- 合法 `cnrtFunctionType_t` 与 launch dim 约束。
- 目标设备可用 task/core 信息。
- NRAM/WRAM/SRAM 容量与对齐事实。
- 目标 dtype 与 `__bang_*` 支持。

本阶段不复制一份硬件常数表。缺少事实时把依赖写入 `open_issues` 或选择无需该事实的保守映射；不得填入猜测值。

## 验证与回退

- 所有逻辑输出轴均可由 task/tile 坐标恢复。
- launch、function type、queue 和 zero-size 行为齐全。
- 每个读写有 logical offset、byte offset、copy bytes、tail 与连续性证据。
- 输出覆盖完整且写入无意外重叠。
- reduction、atomic 与多 kernel 的归属明确。
- 没有依赖未声明的跨 task 同步。
- index 类型和 overflow check 明确。
- JSON 标准可解析；`open_issues=[]` 后才能交接。

| 问题 | 保守处理 |
|---|---|
| 输出域未知 | 停止并补充 shape 公式 |
| task 上限未知 | 保留由 EnvConfig/spec 落定的依赖，不猜核心数 |
| alignment 未知 | 标量计算或 NRAM padded tail，不假设对齐 |
| stride 未知 | 使用 runtime stride，不假设连续 DMA |
| scatter 冲突未知 | 停止并确认 atomic/确定性语义 |
| 单 task NRAM 需求无法成立 | 缩小逻辑 tile 或重做 mapping |
| 空输入可能 launch dim 0 | launcher 早返回 |
