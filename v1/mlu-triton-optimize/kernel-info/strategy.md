# Triton Kernel 轴信息提取

## 职责概述

从包含 Triton kernel、wrapper、测试代码的 Python 脚本中静态提取 kernel 输入输出 tensor 的轴信息，输出json格式的 `kernel info`。**要求**只做静态分析，不运行输入脚本。输出只包含各指针参数的 tensor 轴信息。

## 输出格式

`kernel info` 顶层是一个字典，以 kernel 指针参数名作为 key。只记录作为 `tl.load` 或 `tl.store` 基址使用的指针参数。

```json
{
  "in": {
    "type": "input",
    "shape": [4, 32, 32],
    "stride": [1024, 32, 1],
    "axis": ["M", "N", "K"],
    "has_loop": [false, false, true],
    "axis_type": ["PARALLEL", "PARALLEL", "REDUCE"],
    "block_size": ["BLOCK_MN", "BLOCK_MN", "BLOCK_K"]
  },
  "out": {
    "type": "output",
    "shape": [4, 32],
    "stride": [32, 1],
    "axis": ["M", "N"],
    "has_loop": [false, false],
    "axis_type": ["PARALLEL", "PARALLEL"],
    "block_size": ["BLOCK_MN", "BLOCK_MN"]
  }
}
```

字段规则：

| 字段 | 说明 |
|------|------|
| `type` | `input` 或 `output`。作为 `tl.load` 基址且不作为 `tl.store` 基址的指针为 `input`；作为 `tl.store` 基址的指针为 `output` |
| `shape` | 该指针对应 tensor 在代表性测试规格下的真实 shape，无法静态确定时对应维度写 `null` |
| `stride` | 该指针对应 tensor 在代表性测试规格下的真实 stride，无法静态确定时对应维度写 `null` |
| `axis` | tensor 逻辑轴名，例如 `M`、`N`、`K` |
| `has_loop` | 与 `axis` 一一对应，表示该轴的分块是否被 kernel 内 for loop 遍历 |
| `axis_type` | 与 `axis` 一一对应，只允许 `PARALLEL` 或 `REDUCE` |
| `block_size` | 与 `axis` 一一对应，表示该轴一次处理的块大小参数名；没有明确分块参数时写 `null` |

要求：

- 输出必须是标准 JSON。
- 每个指针条目只能包含上表字段。
- `shape`、`stride`、`axis`、`has_loop`、`axis_type`、`block_size` 的长度必须一致。
- 轴顺序按 tensor 逻辑维度顺序填写；若只能从地址表达式恢复，则按 stride 从大到小排序。
- 分析 `shape`/`stride` 所使用的**代表性测试规格**：优先取 performance test 的第 1 个测例；若不存在 performance test，则取 accuracy test 的第 1 个测例。

## 分析流程

### Step 1：识别目标 kernel 和指针参数

1. 找到输入脚本中的 `@triton.jit` kernel 和 wrapper 中的 `kernel[grid](...)` launch。
2. 若存在多个 kernel 或多个 launch，选择第一个实际 launch 的 Triton kernel 作为目标 kernel。
3. 仅分析目标 kernel，收集作为 `tl.load` 或 `tl.store` 地址基址出现的指针参数。
4. 只输出 tensor 指针参数；shape 参数、stride 参数、编译期常量参数、普通标量参数不作为顶层 key。

### Step 2：从 wrapper 和测试代码确定 shape 与 stride

1. 选择代表性测例：优先使用 performance test 的第 1 个测例；若不存在，则使用 accuracy test 的第 1 个测例。
2. 追踪该测例调用的 wrapper，并找到 Step 1 选定目标 kernel 对应的 `kernel[grid](...)` launch，建立目标 kernel 指针参数到实际 tensor 的绑定关系。
3. 根据该测例中的 tensor 构造和布局变化，确定真实 `shape` 与 `stride`。
4. 若 wrapper 传入 `x.stride(i)`、`x.shape[i]`、`x.numel()` 等表达式，按代表性测例计算实际值。

### Step 3：解析指针地址表达式

对目标 kernel 中每条 `tl.load` 或 `tl.store`，生成一张地址解析表：

1. 将地址写成 `base_ptr + offsets` 形式，识别 `base_ptr` 和总偏移 `offsets`。
2. 将 `offsets` 展开为多维 offset 与 stride 的线性组合：

```text
offsets = sum(offset_i * stride_i)
```

3. 表格字段为 `逻辑轴`、`offset`、`stride`、`loop`：
   - `逻辑轴`：`offset_i` 对应的 tensor 轴名。
   - `offset`：该轴的完整 offset 表达式；解析加法表达式时要合并同一轴的多项，例如 `pid * BLOCK_M + tl.arange(0, BLOCK_M)` 属于同一轴。
   - `stride`：该轴的 stride 表达式；最低维缺省 stride 时可以填 `1`。
   - `loop`：若该轴 `offset` 依赖 for loop 迭代变量，记录相关的所有 loop；否则填“无”。
4. 表格优先按 Step 2 得到的真实 stride 从大到小排列；无法比较时按 tensor 逻辑维度顺序排列。

辅助例子：

```python
rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
for off in range(0, N, BLOCK_N):
    cols = off + tl.arange(0, BLOCK_N)[None, :]
    x = tl.load(input_ptr + rows * stride_m + cols, mask=mask)
```

指针参数 `input_ptr` 的地址计算可解析为下表。表格优先按真实 stride 从大到小排列：

| 逻辑轴 | offset | stride | loop |
|--------|--------|--------|------|
| `M` | `tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)` | `stride_m` | 无 |
| `N` | `off + tl.arange(0, BLOCK_N)` | `1` | `for off in range(0, N, BLOCK_N)` |

### Step 4：分析 block_size 与 has_loop

基于 Step 3 的地址解析表，逐轴补全 `block_size` 和 `has_loop`。

1. 先看该轴的 `offset`。若 `offset` 中包含 `tl.arange(0, BLOCK_*)`，该 `BLOCK_*` 就是该轴一次访问的块大小，填入 `block_size`。
2. 再看该轴的 `loop`。若 `loop` 不是“无”，展开对应 `range(start, end, step)` 的 `start/end/step`：
   - 如果该轴的 `offset` 中没有明确的 `tl.arange(0, BLOCK_*)`，再从 `start/end/step` 中寻找推进该轴的 `BLOCK_*` 作为 `block_size`。
   - 只要该轴的 `offset` 依赖 loop 迭代变量，或该轴的块起点在 loop 中推进，`has_loop` 就填 `true`。
3. 若该轴的 `BLOCK_*` 只出现在 `tl.arange` 中，没有出现在任何 loop 的 `range(start, end, step)` 或 loop 迭代变量相关表达式中，`has_loop` 填 `false`。
4. 若无法从 `offset` 或 `loop` 中找到明确的块大小参数，`block_size` 填 `null`。
5. 对多个逻辑轴的 block 被摊平成一维索引的情况，先通过整除、取余将一维索引还原为各逻辑轴的块索引，再分别填写每个轴的 `block_size`；这些轴继承该一维循环的 `has_loop`。

### Step 5：分析 reduce 轴集合

reduce 轴必须分析整个计算过程。建立全局 `reduced_axis_set`：只要某个逻辑轴在目标 kernel 任意阶段发生过 reduce 操作，就加入集合。

需要同时识别两类 reduce：

1. 单次 API reduce：一次 `tl.sum/tl.max/tl.min/tl.reduce/...` 直接在 tile 的某个维度上完成聚合。
2. 分块 reduce：for loop 分多次 load 同一逻辑轴的不同 block，每轮先做块内 reduce，再用 `acc += local`、`tl.maximum(acc, local)`、value/index 比较等方式合并到跨块 accumulator。

两类情况都必须把被聚合的 tile 维度映射回 Step 3 中的逻辑轴，并加入 `reduced_axis_set`。

#### 常见 reduce 特征

以下特征出现时，需要定位被聚合的 tile 维度，并映射回逻辑轴：

| 类别 | 典型特征 |
|------|----------|
| sum | `tl.sum(x, axis=a)`；循环内 `local_sum = tl.sum(...)`，循环间 `acc += local_sum` |
| mean | 通过 `tl.sum(..., axis=a)` 累加 sum 和 count，循环结束后相除 |
| max | `tl.max(x, axis=a)`；循环内 local max，循环间 `tl.maximum(acc, local_max)` 或等价比较更新 |
| min | `tl.min(x, axis=a)`；循环内 local min，循环间 `tl.minimum(acc, local_min)` 或等价比较更新 |
| argmax | `tl.max(..., axis=a, return_indices=True)`；块间比较 value，并同步更新 value/index |
| argmin | `tl.min(..., axis=a, return_indices=True)`；块间比较 value，并同步更新 value/index |
| generic reduce | `tl.reduce(x, axis=a, combine_fn=...)`；包括 all/any 等逻辑与/或聚合 |
| contraction | `tl.dot(a, b)` 中被 contraction 的逻辑轴，例如 matmul 的 K 轴 |

分块 reduce 判断要点：

- 如果 for loop 遍历某逻辑轴的不同 block，并在每轮对该轴对应 tile 维度执行 reduce，再将结果合入 accumulator，将该逻辑轴加入 `reduced_axis_set`。
- 如果同一逻辑轴在多个阶段发生 reduce，只需在 `reduced_axis_set` 中记录一次。
- 如果多次 API 调用分别 reduce 不同逻辑轴，将这些逻辑轴分别加入 `reduced_axis_set`。
- 不要只根据 `tl.arange`、`BLOCK_*` 或 for loop 判断 reduce；必须确认该轴上的多个元素被聚合进更少的输出元素或中间 accumulator。

### Step 6：确定 axis_type

按以下规则为每个轴填写 `axis_type`：

1. 若逻辑轴属于 `reduced_axis_set`，填 `REDUCE`。
2. 其他逻辑轴填 `PARALLEL`。

### Step 7：组装输出并校验

输出 `kernel info` 并检查：
1. JSON 可被标准 JSON 解析。
2. 顶层 key 全部是指针参数名。
3. 每个指针条目只包含 `type`、`shape`、`stride`、`axis`、`has_loop`、`axis_type`、`block_size`。
4. 每个数组字段长度一致。
5. `axis_type` 只包含 `PARALLEL` 或 `REDUCE`。
6. `type` 只包含 `input` 或 `output`。

## 参考例子

复杂场景的完整分析例子放在 `references/` 目录下，按需加载：

| 文件 | 内容 |
|------|------|
| `references/reducemax-axis-example.md` | 2D `[M,N]` 沿 `N` 轴分块求 max，`M` 轴无分块参数、`N` 轴被 for loop 遍历规约的分析例子 |
| `references/softmax-axis-example.md` | `N/H/W` 布局下沿 `W` 轴 softmax，`N/H` 合用 `BLOCK_NH` 分块的分析例子 |
| `references/flatten-axis-example.md` | 多个逻辑轴的 block 摊平成一维索引后再还原为各轴块索引的分析例子 |
