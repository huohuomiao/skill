# CUDA libdevice 优化策略（RTX 3090）

目标：只保留有官方 API 依据、数值契约通过且 RTX 3090 实测不回退的替换。libdevice 不是“看到数学表达式就替换”的规则库；多数 `tl.*` 已能被 Triton 编译器有效 lowering。

执行前读取：

- `share/gpu/references/platform-rules.md`
- `share/gpu/references/libdevice.md`
- 目标项目的 correctness test、误差阈值和 benchmark 入口

## 1. 不可违反的边界

1. import 只能使用 `from triton.language.extra import libdevice`，kernel 中调用 `libdevice.xxx(...)`。
2. 禁止 `tl.extra.mlu`、`fast_gelu`、`fast_silu`、`fast_sigmoid`、`fast_rcp`、`fast_sqrt` 和任意 `ultra_*`；它们不是 NVIDIA Triton libdevice 的通用 wrapper。
3. 不把 `tl.exp/log/sqrt/sin/cos` 机械替换为同名 libdevice。先证明目标 semantics 不变，再比较 PTX/SASS、误差与性能。
4. fast wrapper 是 FP32 近似候选，不是精确函数别名。用户未允许近似、没有误差预算或特殊值测试时不得使用。
5. FP16/BF16 输入调用 libdevice 前显式转 FP32；输出是否转回由接口决定。不得静默引入 FP64。
6. RTX 3090 无 FP8 Tensor Core；libdevice 替换不能绕过该硬件门禁。

## 2. 先建立不可变 baseline

在改代码前保存：

- 完整可执行脚本：kernel、wrapper、reference、correctness、benchmark。
- 环境摘要：GPU name/UUID、CC、driver、PyTorch、PyTorch CUDA、Triton、输入 dtype/shape/stride。
- 精度：最大绝对/相对误差，必要时 ULP；NaN/Inf/signed-zero 处理。
- 性能：至少 warmup 后多次测量的 median 与 p20/p80；区分 kernel time 与端到端 time。
- 编译资源：registers/thread、static/dynamic shared memory、spills、`num_warps`、`num_stages`。

同一轮对比固定时钟/功耗策略（若用户环境允许）、输入数据、stream 和同步方式。发现其他进程占用或温度/功耗明显漂移时标记该轮无效，而不是伪造稳定结果。

## 3. 候选 A：用专用函数修复数值不稳定表达式

这些替换有明确数学目的，但仍需实测；它们可能更快、相当或更慢。

| 原模式 | 候选 | 主要收益 | 必测点 |
| --- | --- | --- | --- |
| `tl.log(1.0 + x)` | `libdevice.log1p(x)` | 小 x 避免相消 | x≈0、x=-1、x<-1 |
| `tl.exp(x) - 1.0` | `libdevice.expm1(x)` | 小 x 避免相消 | x≈0、overflow |
| `1.0 - libdevice.erf(x)` | `libdevice.erfc(x)` | 正尾部避免相消 | ±large、NaN/Inf |
| `tl.sin(PI * x)` | `libdevice.sinpi(x)` | argument reduction | integer/half-integer/large x |
| `tl.cos(PI * x)` | `libdevice.cospi(x)` | argument reduction | integer/half-integer/large x |
| `tl.sqrt(x*x + y*y)` | `libdevice.hypot(x, y)` | 降低 overflow/underflow | 极大/极小/0/Inf |
| `x ** (1.0 / 3.0)` | `libdevice.cbrt(x)` | 负数有实根语义 | 负数；只有契约一致才换 |

这类替换优先要求“精度改善且性能可接受”，而不是假设 libdevice 单调用一定快。若原契约明确要求原表达式的 NaN/rounding 行为，不替换。

## 4. 候选 B：缺失的特殊函数

当 Triton core language 没有需要的函数时，可使用当前 wrapper 确认存在的 libdevice：

- inverse trig/hyperbolic：`asin`, `acos`, `atan`, `atan2`, `asinh`, `acosh`, `atanh`
- error/normal：`erfc`, `erfinv`, `erfcinv`, `erfcx`, `normcdf`, `normcdfinv`
- gamma/Bessel：`lgamma`, `tgamma`, `j0`, `j1`, `jn`, `y0`, `y1`, `yn`
- scale/remainder：`ldexp`, `scalbn`, `fmod`, `remainder`

这不是“优化替换”，而是官方设备函数实现选择。必须用参考库验证 domain/特殊值，且确认调用成本没有吞掉 fusion 收益。若 PyTorch reference 的定义与 CUDA libdevice 不同，以用户接口契约为准。

## 5. 候选 C：fast FP32 intrinsic

只有下面三项同时成立才尝试：

1. 需求明确允许近似，且给出或可推导 rtol/atol/ULP 预算；
2. 输入 domain 有界，已包含生产分布与 adversarial 边界；
3. 当前 Triton wrapper 确认该名字/FP32 overload 存在。

官方 wrapper 可能包含：

`fast_sinf`, `fast_cosf`, `fast_tanf`, `fast_expf`, `fast_exp10f`, `fast_logf`, `fast_log2f`, `fast_log10f`, `fast_powf`, `fast_dividef`。

仅允许从对应的非 fast libdevice 调用或明确的复合表达式生成一个实验分支。例如：

```python
# baseline
y = libdevice.log(x)

# candidate：只在 FP32 + 近似预算明确时
y = libdevice.fast_logf(x)
```

不要默认从 `tl.exp` 换成 `libdevice.fast_expf`：Triton core `tl.exp` 已被文档描述为快速近似，替换可能相同、变慢或改变误差。先查看生成 PTX，再决定是否值得跑 benchmark。

## 6. 候选 D：显式 rounding

需要可证明的定向舍入或 nearest-rounding 时，可选 `add_*`, `sub_*`, `mul_*`, `div_*`, `rcp_*`, `sqrt_*`, `fma_*` 的 `rn/rz/rd/ru` wrapper。

这些替换的目标是语义，不是速度。注意：

- `a*b+c` 与 `fma(a,b,c)` 只舍入一次，数值结果可能不同。
- 编译器可能默认做 contraction；若接口要求禁止/要求 FMA，需同时控制编译配置并测试。
- directed rounding 会限制优化空间。性能下降但语义必需时应保留，并在报告中说明。

## 7. 明确禁止的“伪优化”

| 禁止模式 | 原因 |
| --- | --- |
| `x * tl.sigmoid(x)` → `libdevice.fast_silu(x)` | NVIDIA wrapper 没有该 API |
| GELU 公式 → `libdevice.fast_gelu(x)` | NVIDIA wrapper 没有高层 GELU API |
| `1/x` → `libdevice.fast_rcp(x)` | wrapper 名不成立；可评估 `tl.fdiv(1,x)` 或 `libdevice.rcp_rn`，语义不同 |
| `tl.sqrt(x)` → `libdevice.fast_sqrt(x)` | wrapper 名不成立 |
| 任意函数 → `ultra_*` | CUDA Triton 无此系列 |
| `tl.maximum/minimum` → libdevice fast max/min | 无此通用 fast wrapper；还涉及 NaN 语义 |
| FP16/BF16 直接传入 FP32 fast 函数 | overload 不匹配或隐式语义不清 |
| 全标量表达式改成 libdevice tensor 函数 | 应在 host/编译期常量折叠，不制造设备调用 |

## 8. 实施流程

### 8.1 扫描与排序

1. 找出 core math、复合不稳定表达式、已有 libdevice 和 dtype conversion。
2. 先处理单个最有证据的候选；不要一轮改十处后无法归因。
3. 复杂模式先匹配，避免内部子表达式被提前改写。例如先识别 `log(1+x)`，再考虑普通 `log`。
4. 记录候选位置、原语义、目标 wrapper、dtype/domain、预期收益和回退条件。

### 8.2 最小修改

只添加一次 import，并替换必要表达式：

```python
from triton.language.extra import libdevice

@triton.jit
def kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = libdevice.log1p(x)
    tl.store(y_ptr + offsets, y, mask=mask)
```

不要同时改变 tile、grid、`num_warps`、`num_stages` 或算法。先隔离 libdevice 替换的影响，确认后再单独调参。

### 8.3 正确性门禁

至少覆盖：

- 生产 shape/dtype/stride 与非整 tile；
- 0、±0、最小正常数/subnormal（接口若关心）、最大有限值；
- NaN、±Inf；
- 每个函数的 domain 边界：log 的 0/负值，asin 的 ±1，atan2 的象限，pow 的负底数等；
- 随机广域和目标真实分布。

输出 max abs error、max rel error、失败样本和阈值。不能因替换误差更大就临时放宽用户阈值。

### 8.4 性能门禁

1. 各分支预热到 JIT 与 cache 稳定。
2. 交错运行 baseline/candidate，减少温度和 boost 漂移；用相同 stream 同步。
3. 报告 median 和分位数，不用单次最小值。
4. 小 kernel 关注 launch noise；必要时循环多次或 CUDA Graph，但两边方法一致。
5. 使用 NCU 确认指令/throughput 改变，并观察 registers、shared memory、occupancy 是否回退。

默认保留条件：correctness 全部通过，且候选 median 不慢于 baseline。若用户要求显著收益，可设置例如 `candidate_median <= 0.98 * baseline_median`；阈值必须在运行前确定。

### 8.5 回退

以下任一发生即回退该候选：

- wrapper/overload 在安装版本不存在；
- `sm_86` 编译或 launch 失败；
- 精度、特殊值或 domain 语义不合格；
- 性能回退或结果落在测量噪声内但代码复杂度增加；
- register spill、shared memory 或 occupancy 恶化抵消收益。

回退只撤销该候选，保留 baseline、命令和失败证据，继续评估下一个独立候选。

## 9. 结果格式

每个候选输出一行决策：

| 字段 | 内容 |
| --- | --- |
| location | 文件、kernel、表达式 |
| baseline / candidate | 精确代码模式 |
| dtype/domain | 实际输入范围 |
| API evidence | 当前 Triton symbol/source |
| accuracy | abs/rel/ULP 与阈值 |
| performance | median、p20/p80、speedup |
| resources | registers/shared/spills/occupancy |
| decision | keep / revert 与原因 |

最终文件必须仍包含完整 kernel、wrapper、correctness 和 benchmark；若所有候选失败，输出保持原代码，而不是为了产生 diff 留下无收益替换。

## 官方依据

- Triton libdevice 教程：<https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html>
- Triton CUDA libdevice wrapper：<https://github.com/triton-lang/triton/blob/main/third_party/nvidia/language/cuda/libdevice.py>
- NVIDIA libdevice User's Guide：<https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>
- Triton core math API：<https://triton-lang.org/main/python-api/triton.language.html#math-ops>
- NVIDIA Nsight Compute：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
