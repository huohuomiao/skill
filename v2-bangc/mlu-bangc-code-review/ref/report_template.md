# BANG C/CNRT 代码检视报告

## 源码与环境

- 输入：`[absolute .mlu path]`
- 输出：`[absolute bangc_code_fix.mlu path]`
- execution_backend：`local | worker | unavailable`
- 设备：`[model / logical id / identifiers]`
- `cncc`：`[absolute path and version]`
- NeuWare/CNToolkit：`[version/root]`
- 编译命令：`[complete command]`
- 目标架构参数：`[confirmed value | N/A]`

## 原始门禁

| 门禁 | 命令 | 退出码 | 结果 | 证据 |
| --- | --- | ---: | --- | --- |
| 完整性 | `[scan]` | N/A | PASS/FAIL | `[kernel, launch, reference]` |
| 编译/链接 | `[cncc ...]` | | PASS/FAIL/BLOCKED | `[first diagnostic]` |
| CNRT/launch | `[binary]` | | PASS/FAIL/BLOCKED | `[API/queue result]` |
| 精度 | `[binary cases]` | | PASS/FAIL/BLOCKED | `[atol/rtol/max error]` |
| MLU590 | `[probe]` | | PASS/FAIL/BLOCKED | `[device evidence]` |

## 静态契约

| 项目 | 内容 |
| --- | --- |
| Kernel/launch | `[entry, dim, function type, queue]` |
| 输入输出 | `[dtype, shape, stride, ownership, alias]` |
| 任务映射 | `[task IDs -> logical axes]` |
| 片上存储 | `[NRAM/WRAM/SRAM buffers and bytes]` |
| 搬运与 intrinsic | `[directions, byte counts, operations]` |
| reference/容差 | `[independent reference and frozen thresholds]` |

## 静态发现

| 严重度 | 位置 | 类别 | 证据 | 修复或所需确认 |
| --- | --- | --- | --- | --- |
| error/warning/note | `[file:line]` | | | |

## 静态改动

- `[exact edit | no definite static repair]`

## 动态修复迭代

### 迭代 `[N]`

- 失败门禁/规范化签名：
- 根因假设：
- 最小改动：
- 定向重跑：
- 完整门禁：
- 保留/回退：

## 最终结果

- 最后通过门禁：
- 最后失败/阻塞门禁：
- 精度：
- 未验证项：

```text
passed: true|false
blocked: true|false
target_verified: true|false
compile_pass: true|false|unavailable
accuracy_pass: true|false|unavailable
final_code_path: [absolute path]
```
