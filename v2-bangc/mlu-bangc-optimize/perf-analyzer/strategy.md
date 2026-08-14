# BANG C Kernel 性能分析

## 职责

对可独立编译运行的 `.mlu` 算子采集三类证据：

1. correctness harness 内的 CNRT notifier 稳态时间（acceptance benchmark）。
2. CNPerf 原始报告（瓶颈诊断）。
3. CNCC 可用的中间产物/MLISA（静态佐证）。

本策略只分析并生成建议，不修改 Kernel。CNPerf 或 MLISA 不可用时降级并明确写 `N/A`；不得猜命令参数、输出格式或硬件指标。

## 执行契约

读取最近的 `{output_dir}/EnvConfig/config.md` 并沿用：目标 MLU590、`cncc`、环境变量、完整编译命令和 `execution_backend`。

- local：直接前台执行。
- worker：通过 `mlu-bangc-main/subagents/scripts/submit_task_to_worker.py` 前台同步执行。
- EnvConfig 缺失、目标未确认或 infrastructure 失败：输出 blocked 报告，不改代码。

禁止新建 Job、修改系统环境、安装/升级工具或用其他设备数据做 keep 决策。

## 输入输出

- `input_file`：完整 `.mlu` 文件。
- `output_dir`：本轮目录。
- 输出：`report.md`、notifier 原始样本、CNPerf 原始文件、解析 JSON、CNCC/MLISA 产物清单（能取得时）。

## Step 1：建立可归因 binary

1. 用 EnvConfig 的相同 release 编译契约构建 input，不改变源码或 flags。
2. 确认 binary 的 correctness 先通过。
3. 识别目标 Kernel 名；多 Kernel 逻辑调用要记录完整 stage 范围。
4. notifier benchmark 保留原 Shape/dtype/stride、Queue、warmup/repeat。

编译/精度失败先返回 code-review；不能在分析阶段修 Kernel。

## Step 2：notifier 基线

运行原 harness，保存每个 case 的原始样本、median、p20、p80 和样本数。计时范围必须只包含一致的完整设备逻辑调用；若含输出初始化或第二阶段归并，两侧都必须计入。

统一输出字段：`host_reference_ms`、`original_bangc_ms`、`opt_bangc_ms`。本策略分析 input 时 `opt_bangc_ms=N/A`。

## Step 3：CNPerf 采集

共享脚本契约为：

```bash
bash .claude/skills/share/mlu/perf-analyzer/analyzer.sh \
  <output_dir> <binary_or_command> [artifact_dir]
```

本策略先自行使用 EnvConfig 编译 `.mlu`，通常把第二参数传为可执行 binary；只有确需包装完整逻辑调用时才传可复现命令。可选 `artifact_dir` 指向隔离的 CNCC 中间产物目录。若 `cnperf-cli` 不存在或子命令/参数与脚本不兼容：

- 保存版本/错误原文。
- 标记 `cnperf_status=unavailable|partial|failed`。
- 继续使用 notifier 和源码/CNCC 证据，不伪造计数器。

运行成功时必须保留 CNPerf 原始报告。`scripts/analyzer_rep.py` 仅做 best-effort 宽松解析；解析失败不覆盖原始证据。CNPerf replay duration 不作为最终性能验收时间。

## Step 4：CNCC/MLISA 产物

只有当前 `cncc --help` 明确支持相应选项时，才在隔离目录生成汇编/MLISA/IR/对象临时产物。记录完整命令和产物类型；不硬编码某版本必有的 flag。产物清单由共享 `analyzer_cncc_artifacts.py` 生成；识别不到的文件保持 `unknown`。

可用时分析：

- 重复标量 div/mod 或地址计算。
- 搬运、向量 intrinsic 与同步序列。
- 编译器明确报告的 NRAM/SRAM/WRAM 或 spill。
- 候选前后是否真的改变生成代码。

无法关联目标 Kernel 时写 `N/A`，不得拿其他 entry 的数据代替。

## Step 5：证据驱动建议

| 证据 | 可建议策略 |
| --- | --- |
| GDRAM 访问/搬运主导、复用低 | `retiling` 或 `index-computation-simplify` |
| 片上利用低且搬运批次多 | `config-tuner`（增大安全 tile） |
| 编译器明确片上资源压力 | `config-tuner`（减 tile/缩短 live range） |
| Task 负载不均或尾批 | `modify-grid`/`config-tuner` |
| 归约/跨 Task 合并热点 | `reduce-opt` |
| 数学 intrinsic/除法热点 | `libdevice-opt` 或 `div-to-mul` |
| 流水空隙且 SDK 支持异步搬运 | `retiling`/`config-tuner` |

只选择证据最直接的一个建议。数据不足时写“无可执行建议”，停止深度迭代。

## Step 6：报告

按 `references/report_template.md` 写 `report.md`，每个数字链接到原始文件/命令。解析器无法识别的 CNPerf 字段保留原文并写 `N/A`，不按旧版列位置猜测。
