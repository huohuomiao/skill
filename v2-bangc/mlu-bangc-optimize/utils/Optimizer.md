# Optimizer

## 职责

作为指定 BANG C 优化策略的统一包装器。读取一个策略文档，建立单一可归因候选，使用同一 `cncc`/CNRT/测试/计时契约验证，并输出 best-so-far。不得脱离策略文档盲调。

## 输入输出

| 输入 | 说明 |
| --- | --- |
| 策略名称 | 当前目录对应的策略名 |
| 策略文档路径 | 当前 `strategy.md` 或共享数学策略 |
| 工作目录 | 只在此目录写临时产物 |

固定文件：

```text
input.mlu
candidate.mlu
bangc_optimized.mlu
bangc_optimized.md
result.json
```

无论成功、失败、不适用或未测量，都必须生成后三个结果文件。失败/回退时 `bangc_optimized.mlu` 与 `input.mlu` 逐字一致。

## 执行契约

从工作目录向上读取最近的 `{output_dir}/EnvConfig/config.md`：

- 使用其 `execution_backend`、`cncc`、环境变量、完整 flags 和 Worker 上下文。
- baseline/candidate 只改变策略声明的源码/配置变量。
- 架构 flag、片上容量、Core/Cluster 数、function type 限制与 CNPerf 格式不得猜测。
- Worker 任务前台同步执行；基础设施失败记 `blocked/not_measured`，不修改算法。

## 步骤 1：冻结输入

1. 计算 `input.mlu` 哈希，不覆盖该文件。
2. 确认 Kernel、CNRT launch、reference、correctness、notifier benchmark 完整。
3. 固定 Shape、dtype、stride/layout、seed、容差、Queue、warmup、repeat 和计时范围。
4. 使用 EnvConfig 命令编译运行 baseline；失败则输出原文件和真实错误。
5. 保存 baseline 的编译、精度、notifier、CNPerf/MLISA 可用性证据。

## 步骤 2：建立候选

读取策略文档并先记录：

- 匹配位置与准入条件。
- 预期影响：访存、计算、任务并行、流水、片上资源或数学语义。
- 必须保持的接口与结果。
- 单一变化、风险和回退条件。

不满足准入条件时报告 `not_applicable` 并逐字回退。不得修改 reference、输入、容差或 benchmark 以制造收益。

## 步骤 3：编译和静态资源检查

用相同构建契约编译 candidate，保存完整 stdout/stderr。检查：

- Kernel/launch ABI 与 CNRT 生命周期。
- 编译器明确报告的 NRAM/SRAM/WRAM、spill 或资源错误。
- intrinsic/dtype/address-space 支持。
- 任务规模和 function type 的已确认限制。
- 当前审计兼容项：`bang.h` 定位、头文件 `CNRT_CHECK`、返回码符号与 C++ 链接依赖。

编译失败最多修复三次，且仅修当前候选引入的错误；仍失败即回退。

## 步骤 4：正确性

按原顺序运行全部用例：不改变 seed、Shape、stride、dtype、容差或 pass 条件。记录最大误差、NaN/Inf 和首个失败位置。任务映射、片上搬运、流水、归约、原子/workspace 或索引变化必须覆盖相应边界用例。任何失败立即回退。

## 步骤 5：公平 benchmark

只有真实 MLU590 且正确性通过时执行：

1. baseline/candidate 使用同一 Queue、notifier 范围、warmup/repeat/sample 数。
2. 排除 context/module load 与首次编译。
3. 交错测量并保存样本、median、p20、p80。
4. 收益未超过预先确定的噪声阈值，或任一关键 case 回退，则回退候选。
5. 带宽仅在实际读写字节口径明确时计算。
6. CNPerf replay 时间只作诊断，不替代 acceptance benchmark。

## 步骤 6：输出

`decision=keep` 必须同时满足：

- `compile_pass=true`
- `accuracy_pass=true`
- `target_verified=true`
- `performance_measured=true`
- `same_contract=true`
- `no_regression=true`

否则为 `revert` 或 `not_measured`。

`result.json` 最小契约：

```json
{
  "schema_version": 1,
  "strategy": "name",
  "status": "success|failed|not_applicable|not_measured",
  "decision": "keep|revert|not_measured",
  "execution_backend": "local|worker|unavailable",
  "target_verified": false,
  "compile": {"same_contract": true, "pass": false},
  "correctness": {"pass": false, "atol": null, "rtol": null, "max_abs_error": null, "max_rel_error": null},
  "performance": {"measured": false, "host_reference_ms": null, "original_bangc_ms": null, "opt_bangc_ms": null, "p20_ms": null, "p80_ms": null, "speedup": null, "no_regression": null},
  "resources": {"nram_bytes": null, "sram_bytes": null, "wram_bytes": null},
  "evidence": {"cnperf_raw": null, "mlisa": null},
  "output_code": "bangc_optimized.mlu",
  "reason": ""
}
```

未知值使用 `null`，报告显示 `N/A`。不得用 0 表示未测量。

## 文件权限

只读：当前 `input.mlu`、当前策略文档及其 `references/`、最近的 EnvConfig、共享 MLU 平台/原语/数学文档。只写当前工作目录。禁止读取其他策略候选来挑选有利代码；跨轮 best-so-far 由主流程传入。

## 禁止事项

- 禁止用 Host reference、高层库或框架算子替代 BANG C Kernel。
- 禁止凭静态估算、其他设备或单次最小值声明 MLU590 提速。
- 禁止同时改变多个参数族并归因给单一策略。
- 禁止隐藏编译失败、精度失败、性能回退或工具不可用。
- 禁止把失败候选传给下一策略。
