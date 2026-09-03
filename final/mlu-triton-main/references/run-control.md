# 最终集成版运行控制契约

`run_control.py` 只管理外层阶段状态、内容指纹、缓存副本和恢复顺序，不替代任何 Skill 的业务逻辑。

## 阶段 DAG

```text
env_config -> extractor -> kernel_gen -> optimizer -> finalize
                                 \--------------------^  correctness 模式
```

- `correctness` 把 `optimizer` 标为 `skipped`；其他模式必须执行或恢复它。
- `env_config` 每次真实运行，不缓存，防止复用过期设备可用性。
- `extractor`、`kernel_gen`、`optimizer` 可缓存。
- `kernel_gen`、`optimizer` 的指纹包含 `hardware_key` / `toolchain_key` 上下文；上下文不同不得复用运行结果。
- `finalize` 不缓存，因为总结必须反映本次 cache hit、恢复和失效原因。

## 新运行

```bash
python .claude/skills/mlu-triton-main/scripts/run_control.py init \
  --output-dir <output_dir> \
  --input <输入文件路径或需求文本> \
  --mode <correctness|balanced|max-performance> \
  [--budget-file <budget.json>] \
  [--cache-dir <共享缓存目录>]
```

默认清单位于 `{output_dir}/run_manifest.json`，默认共享缓存位于 `{output_dir}` 的同级 `.mlu-triton-cache/`。已有清单时 `init` 必须失败，禁止覆盖。

## 阶段事务

每个阶段严格执行：

1. `next --manifest <manifest>`：只能服从返回的 `run`、`restore`、`blocked` 或 `done`。
2. `run` 时先 `start --stage <stage>`，然后调用对应 Skill；成功后执行 `complete`，并显式传入已通过的验证等级。
3. `restore` 时执行 `restore --stage <stage>`；控制器先验缓存哈希，再原子替换产物。
4. 业务执行失败时执行 `fail --reason <真实原因>`；不得把缺失产物标成完成。

```bash
python .claude/skills/mlu-triton-main/scripts/run_control.py start --manifest <manifest> --stage <stage>
python .claude/skills/mlu-triton-main/scripts/run_control.py complete --manifest <manifest> --stage <stage> \
  [--validation-level l1] [--validation-level l2] [--validation-level l3]
python .claude/skills/mlu-triton-main/scripts/run_control.py fail --manifest <manifest> --stage <stage> --reason <reason>
```

`complete` 会校验 `stage-sources.json` 声明的必需产物非空，自动收集本阶段已存在的可选审计产物并记录 SHA-256。只有验证等级满足阶段要求时才写入不可变缓存：Extractor 要求 L1+L2，Kernel Gen 和 Optimizer 要求 L1+L2+L3。验证不足仍可完成本次阶段，但 `cache_key=null`，后续运行不得恢复该结果。额外产物可重复传 `--artifact <output-relative-path>`。

Code Gen 的长阶段可在外层 `running` 期间保存内部检查点：

```bash
python .claude/skills/mlu-triton-main/scripts/run_control.py checkpoint-save \
  --manifest <manifest> --stage kernel_gen --name design \
  --artifact KernelGen/step1_base_info.json \
  --artifact KernelGen/step1_io_shapes.json \
  --artifact KernelGen/step2_block_mapping.json \
  --artifact KernelGen/step3_axis_fusion.json \
  --artifact KernelGen/step4_code_spec.json \
  --validation-level l1 --validation-level l2
python .claude/skills/mlu-triton-main/scripts/run_control.py checkpoint-status \
  --manifest <manifest> --stage kernel_gen --name design
```

最终流程使用 `design`、`build`、`review` 三个分组检查点；为兼容旧流程，控制器仍识别 Step 1-7 名称。Design/Build 要求 L1+L2，Review 要求 L1+L2+L3。只有阶段指纹、验证等级与各产物哈希都一致时才返回 `reusable=true`；输入或 Skill 源变化引发的阶段失效会清空其检查点。

## 绑定真实运行上下文

EnvConfig 完成后必须绑定：

```bash
python .claude/skills/mlu-triton-main/scripts/run_control.py bind-context \
  --manifest <manifest> \
  --context-file <output_dir>/EnvConfig/run_context.json
```

上下文 JSON 必须含非空 `execution_backend`、`hardware_key`、`toolchain_key`。未绑定时控制器会阻断 `kernel_gen`；上下文变化时 `kernel_gen` 及其下游自动失效。

## 中断恢复

```bash
python .claude/skills/mlu-triton-main/scripts/run_control.py resume --manifest <manifest>
```

恢复规则：

- `complete` / `cached` 阶段只有在当前源指纹与产物哈希都一致时才跳过。
- `running` / `failed` 阶段重新打开为 `pending`，但不删除已有目录和内部状态。
- 指纹变化、产物缺失或哈希错误会使该阶段和全部下游失效。
- 中断在 Optimizer 内部时，外层只重开 `optimizer`；内层必须用 `optimization_control.py plan --resume` 继续同一个 `optimization_state.json`，不得重置预算。
- 缓存损坏会阻断恢复并报告原因，不会静默执行或覆盖不可变缓存。

需要人工强制重做某阶段时：

```bash
python .claude/skills/mlu-triton-main/scripts/run_control.py invalidate \
  --manifest <manifest> --stage <stage> --reason <reason>
```

## 指纹边界

阶段指纹由以下内容的规范化 JSON 计算 SHA-256：

- 输入文件内容或需求文本哈希；
- 对模式/预算敏感的阶段，优化模式和预算文件内容哈希；Extractor 与 Code Gen 不因只改变优化模式而失去可复用缓存；
- 当前阶段相关 Skill、共享规则和脚本的文件哈希；
- 上游阶段指纹；
- 对运行态阶段，真实硬件与工具链上下文哈希。

因此只修改无关 Skill 不会清空所有缓存；修改某阶段契约只会使该阶段及下游失效。

## 安全边界

- 产物必须位于 `output_dir` 内，缓存恢复路径禁止绝对路径和 `..`。
- 缓存以 `<stage>/<fingerprint>` 寻址，命中后逐文件验证 SHA-256。
- 控制器不自动删除缓存；清理策略由仓库或 CI 显式管理，避免误删共享结果。
- 同一个 `run_manifest.json` 只允许一个主流程写入；不要并发执行阶段控制命令。
