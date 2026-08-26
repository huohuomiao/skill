# BANG C intrinsic 优化策略

## 职责

在保持算子语义、接口、布局和误差阈值不变的前提下，将 BANG C kernel 中已证明等价的标量/循环计算替换为当前 MLU590 工具链已确认的 `__bang_*` 或数学向量实现。

执行前读取：

- `{BANGC_SKILL_ROOT}/share/mlu/references/platform-rules.md`
- `{BANGC_SKILL_ROOT}/share/mlu/references/primitives.md`
- `{BANGC_SKILL_ROOT}/share/mlu/references/libdevice.md`
- `{output_dir}/EnvConfig/config.md`

本策略不维护一份脱离 SDK 的完整 API 表。symbol、签名、dtype、长度和对齐必须来自目标环境证据。

## 输入与输出

输入：

- 完整 `.mlu` 文件，包含 BANG C kernel、CNRT host launcher/reference/test/benchmark。
- 工作目录。
- 已验证环境配置。

输出：

- `{工作目录}/bangc_optimized.mlu`
- `{工作目录}/bangc_optimized.md`
- `{工作目录}/result.json`

没有安全可用替换或候选退化时，也必须复制输入为 `bangc_optimized.mlu`，并明确 `decision=no_change` 或 `reverted`。

## 候选模式

只把下表用作模式识别入口：

| 原始语义 | 候选族 | 门禁 |
|---|---|---|
| NRAM 向量逐元素加法 | `__bang_add` | 当前原型、dtype、长度/对齐、alias 已确认 |
| NRAM 向量逐元素减法 | `__bang_sub` | 同上 |
| 向量加标量 | `__bang_add_scalar` | scalar dtype 与广播语义已确认 |
| 向量 floor | `__bang_floor` | 舍入与特殊值通过 |
| 固定布局/窗口求和 | `__bang_sumpool` | reduction 维度、窗口、padding、累加精度完全匹配 |
| 片上 buffer 填充值 | `__bang_write_value` | dtype 与长度单位确认 |
| exp/log/sqrt/rsqrt/tanh/sigmoid/erf/pow 等 | 当前 SDK 已确认数学/vector symbol | 不得编造 `fast_*`/`ultra_*` 名称 |

复杂模式优先识别完整语义。例如 GELU/SiLU/softmax/LayerNorm 只有在当前 SDK 存在完整等价实现、常数与数值稳定语义完全一致时才整体替换；否则逐个已确认基本操作优化。

## 工作流

### 1. 基线门禁

运行原始 `.mlu` 的既有测试和 benchmark，记录：

- `compile_pass`
- `accuracy_pass`、`atol`、`rtol`、`max_diff`
- `host_reference_ms`
- `original_bangc_ms`
- 带宽/吞吐量定义
- warmup、重复次数、统计量和计时范围
- CNCC 命令与 `execution_backend`

原始代码未编译或精度失败时停止本策略，`decision=blocked_invalid_baseline`；不得用优化掩盖基线错误。

### 2. 静态扫描

只扫描 device kernel 中的计算表达式与片上 buffer。忽略：

- host reference 计算。
- 测试数据构造。
- 报告/注释中的示例。
- CNRT allocation/copy/queue 逻辑。

对每个候选记录源码位置、原公式、输入来源、dtype、地址空间、元素长度、alias 和边界条件。

### 3. 证明等价

逐项确认：

1. 数学公式与广播/reduction 轴相同。
2. 输入输出 dtype 和累加 dtype 相同或满足 requirement。
3. 原地/非原地与 alias 语义合法。
4. intrinsic 的长度单位、对齐和地址空间满足当前声明。
5. 尾块不会越界，也不会把 padding 写回逻辑输出。
6. NaN/Inf、除零、溢出和舍入语义满足 requirement。

任一项未知时，候选标记 `needs_current_environment_verification`，当前候选不改代码。

### 4. 确认 API

按 `{BANGC_SKILL_ROOT}/share/mlu/references/libdevice.md` 的证据优先级执行：

- 检索当前 `bang.h` 声明。
- 查同一 SDK 的官方 sample。
- 查同项目同版本已运行通过的源码。
- 必要时创建最小临时 probe，使用相同 CNCC/arch/link 参数编译和运行。

记录 exact symbol、原型来源、CNCC 版本与 probe 结果。不得根据其他 SDK 版本或 Triton Libdevice 文档推断。

### 5. 单一替换

每次只应用一种可归因的替换，保留：

- kernel/host ABI。
- task mapping 与 tile。
- CNRT 生命周期。
- reference 与测试 case。
- 编译和 benchmark 配置。

不要同时重排 tile、改搬运和替换数学函数，否则无法归因性能变化。

### 6. 编译和精度验证

在 `EnvConfig/config.md` 指定后端执行相同 CNCC 命令和完整测试：

- 编译失败：保存 stderr，回退。
- 运行/CNRT 失败：保存 stdout/stderr，回退。
- 任一测试 case 超阈值：保存误差，回退。
- 只有全部通过才测性能。

### 7. 性能判定

使用与基线相同的 kernel-only 计时方法：

```text
speedup_opt_vs_original = original_bangc_ms / opt_bangc_ms
speedup_opt_vs_reference = host_reference_ms / opt_bangc_ms
```

只有 `opt_bangc_ms < original_bangc_ms` 且差异超过既定噪声容限时保留候选。否则恢复输入代码并写明：

- `no_improvement`
- `within_measurement_noise`
- `accuracy_regression`
- `compile_or_runtime_failure`

禁止四舍五入制造加速。

## result.json

```json
{
  "strategy": "libdevice-opt",
  "decision": "accepted | no_change | reverted | blocked_invalid_baseline",
  "target_verified": true,
  "execution_backend": "local | worker",
  "symbol": "<exact __bang_* or N/A>",
  "symbol_evidence": "<header/sample/probe path>",
  "compile_pass": true,
  "accuracy_pass": true,
  "atol": null,
  "rtol": null,
  "max_diff": null,
  "host_reference_ms": null,
  "original_bangc_ms": null,
  "opt_bangc_ms": null,
  "speedup_opt_vs_original": null,
  "speedup_opt_vs_reference": null,
  "unavailable_reason": null
}
```

不可用值保留字段并设为 `null`，在 Markdown 中解释原因。

## 报告要求

`bangc_optimized.md` 至少包含：

1. 原始模式与源码位置。
2. 选择或拒绝的 exact symbol。
3. 头文件/sample/probe 证据。
4. 等价性、dtype、地址空间、长度/对齐和尾块分析。
5. 实际 CNCC 命令与编译结果。
6. 每组测试的精度结果。
7. 同口径前后性能和判定。
8. 最终文件路径与是否回退。

## 禁止行为

- 生成 `tl.extra.mlu.libdevice.*`、`tl.math.*` 或 Triton 代码。
- 把 Triton `fast_*`/`ultra_*` 名字假定为 BANG C symbol。
- 未查当前 API 就编造 `__bang_*` 原型。
- 将 host `<cmath>` 调用当作 device vector intrinsic。
- 放宽用户阈值、删除困难 case 或改 reference。
- 只凭静态推测报告性能提升。
