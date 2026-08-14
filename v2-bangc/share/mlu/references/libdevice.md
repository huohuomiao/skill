# BANG C 数学函数与向量 intrinsic 选择

保留 `libdevice.md` 文件名是为了维持 v1 模块引用路径；本文件不再描述 Triton `tl.math` 或 `tl.extra.mlu.libdevice`。BANG C 中应根据当前 CNCC、`bang.h`、官方 sample 和最小编译探测选择数学实现。

## 目录

- 证据优先级
- 实现层级
- 可安全识别的模式
- dtype 与数值门禁
- 最小探测
- 第一版禁区

## 证据优先级

按以下优先级确认函数名与签名：

1. 目标服务器当前 `bang.h`/CNCC resource headers 中的声明。
2. 同一 NeuWare/CNToolkit 安装内的官方 BANG sample。
3. 当前项目中已经在相同 MLU590、相同 CNCC 版本编译运行通过的源码。
4. 本文件中的通用模式说明。

低优先级资料不得覆盖当前头文件和真实编译结果。一个 dtype 的成功不能外推到其他 dtype、长度或地址空间。

## 头文件发现

不要只检查 `$NEUWARE_HOME/include/bang.h`。第一版需要兼容两类布局：

```text
$NEUWARE_HOME/include/bang.h
$NEUWARE_HOME/lib/clang/<version>/include/bang.h
```

CNCC 可能通过其 resource directory 自动解析 `<bang.h>`，即使它不在公共 include。可以读取头文件做审计，但生成的 `.mlu` 仍使用：

```cpp
#include <bang.h>
#include <cnrt.h>
```

## 实现层级

### 1. 清晰正确性基线

使用 BANG C 循环、基本算术和显式片上 buffer 表达公式。该层适合建立语义基线，但仍需 CNCC 编译验证，不保证性能。

### 2. 已确认的 `__bang_*` 向量 intrinsic

当目标头文件/样例确认参数、dtype、长度和地址空间后，用 `__bang_*` 替换片上逐元素或 reduction 逻辑。仓库现有审计确认曾使用或引用以下名称：

- `__bang_add`
- `__bang_sub`
- `__bang_add_scalar`
- `__bang_floor`
- `__bang_sumpool`
- `__bang_write_value`

这只确认名称是候选，不确认所有版本中的完整签名和支持矩阵。任何新 `__bang_*` 必须先探测。

### 3. fast/approximate 或专用数学实现

只有同时满足以下条件才可使用：

- 当前 SDK 明确提供该函数。
- 输入 dtype、地址空间、长度和对齐满足声明。
- 用户允许相应精度，且所有测试 case 通过原阈值。
- 相同 benchmark 下确有性能改善。

若任一条件未知，保留上一份已验证实现。

## 可安全识别的优化模式

以下是“可扫描的语义模式”，不是未经验证即可替换的 API 表：

| 语义模式 | 候选方向 | 必须检查 |
|---|---|---|
| `out[i] = a[i] + b[i]` | 向量 add intrinsic | dtype、片上地址、长度/对齐、alias |
| `out[i] = a[i] - b[i]` | 向量 sub intrinsic | 同上 |
| `out[i] = a[i] + scalar` | add-scalar intrinsic | scalar dtype/广播语义 |
| 向量 floor | floor intrinsic | 特殊值与舍入 |
| 固定窗口/布局求和 | sumpool 类 intrinsic | 维度、窗口、步长、padding、累加 dtype |
| buffer 常量初始化 | write-value 类 intrinsic | 元素数/字节数、dtype |
| exp/log/sqrt/rsqrt/tanh/sigmoid/erf/pow | 当前 SDK 的数学或向量实现 | 函数是否存在、定义域、近似误差、dtype |

不得仅因函数名相似就替换复合公式。例如 GELU、SiLU、softmax 或 LayerNorm 必须验证完整公式、常数、数值稳定化和特殊值行为。

## dtype 与数值门禁

对每个候选记录：

```yaml
symbol: <exact symbol>
header: <resolved bang.h path or official sample>
cncc_version: <version>
input_dtype: <dtype>
output_dtype: <dtype>
accumulation_dtype: <dtype>
length_constraints: <confirmed value or unknown>
alignment_constraints: <confirmed value or unknown>
address_spaces: <e.g. NRAM -> NRAM>
accuracy_class: exact | approximate | unknown
probe_status: compile_pass | run_pass | unavailable
```

规则：

- `float32` 是第一版默认 smoke dtype，不代表所有 intrinsic 支持 float32。
- `half`、bfloat16、整数和混合 dtype 的 spelling/提升规则由当前 SDK 确认。
- 不生成 fp64 支持声明，除非服务器 probe 明确通过。
- approximate 实现必须在最大误差之外同时考虑 `rtol`、NaN/Inf、接近零和极值输入。
- reduction/normalization 优先保证稳定累加，再比较性能。

## 最小编译探测

对不确定 symbol 建立独立临时 `.mlu`，只包含：

1. `<bang.h>`/`<cnrt.h>`。
2. 最小 kernel 与所需片上 buffer。
3. 单次候选 intrinsic 调用。
4. 最小 CNRT host launch。

使用与目标算子相同的 CNCC/arch/link 配置。分别记录：

- symbol 是否声明。
- 模板/重载选择结果。
- 编译器对 dtype、地址空间、长度/对齐的诊断。
- 运行是否成功。
- 与 host reference 的误差。

probe 文件只用于证据，不得把编译失败的 symbol 写进正式候选。

## 性能替换流程

1. 采集原始 BANG C kernel-only 基线。
2. 识别一个语义模式。
3. 查当前头文件/sample 或执行最小 probe。
4. 只替换一个模式并保持接口、tile 和计时范围不变。
5. 重新 CNCC 编译、真机运行、完整精度验证。
6. 重新 benchmark；只有更快且精度通过才保留。
7. 在报告中记录 symbol、证据来源、误差与前后耗时。

具体策略读取 `.claude/skills/share/mlu/optimize/libdevice-opt.md`。

## 第一版禁区

以下均视为 Triton 残留或无证据推断：

- `tl.math.*`
- `tl.extra.mlu.libdevice.*`
- 把 Triton `fast_*`/`ultra_*` 名称直接改前缀后当作 BANG C API
- 声称完整支持某 dtype 表但没有当前 SDK 证据
- 把 scalar C math、device vector intrinsic 和 host `<cmath>` 当成同一调用域
- 为获得性能而放宽 requirement 的误差阈值

## 第二版校准输出

服务器审计应返回当前 MLU590 环境中实际可用 symbol、原型、dtype、长度/对齐限制、官方 sample 位置以及最小 probe 结果。第二版再把已确认项从“条件使用”提升为稳定生成规则。
