# final 验证说明

所有命令从 `final` 根目录执行。离线工具使用 Python 3.9+ 标准库，不要求 MLU 或第三方验证包。

## 1. L1 与 L2 离线门禁

```bash
python -B validation/validate.py l1
python -B validation/validate.py l2
# 或一次执行
python -B validation/validate.py all
```

L1 检查四个 Skill 入口、合并角色、共享资源、Python/JSON/Markdown 结构和全部 Schema。L2 检查路由、分组检查点、验证等级、缓存/恢复、失效传播、路径边界、优化预算续跑、调度度量以及回归正负例。

通过条件是同时出现 `[PASS] L1` 和 `[PASS] L2`。

## 2. 阶段完成与缓存晋级

`--validation-level` 可重复传入。Extractor 缓存需要 L1+L2：

```bash
python -B mlu-triton-main/scripts/run_control.py complete \
  --manifest <manifest> --stage extractor \
  --validation-level l1 --validation-level l2
```

KernelGen 与 Optimizer 缓存需要 L1+L2+L3：

```bash
python -B mlu-triton-main/scripts/run_control.py complete \
  --manifest <manifest> --stage kernel_gen \
  --validation-level l1 --validation-level l2 --validation-level l3
```

缺少要求的等级时，阶段可以完成本次运行，但不会发布共享缓存。KernelGen 的 L3 完成还必须交付非空的 `review_result.json` 和 `dispatch_metrics.json`；Main 只应在 Review 的真实精度证据通过后传入 `l3`。

## 3. Code Gen 分组检查点

```bash
python -B mlu-triton-main/scripts/run_control.py checkpoint-save \
  --manifest <manifest> --stage kernel_gen --name design \
  --artifact KernelGen/step1_base_info.json \
  --artifact KernelGen/step1_io_shapes.json \
  --artifact KernelGen/step2_block_mapping.json \
  --artifact KernelGen/step3_axis_fusion.json \
  --artifact KernelGen/step4_code_spec.json \
  --validation-level l1 --validation-level l2

python -B mlu-triton-main/scripts/run_control.py checkpoint-save \
  --manifest <manifest> --stage kernel_gen --name review \
  --artifact KernelGen/step6_test_code_fix.py \
  --artifact KernelGen/step6_test_code_fix.md \
  --artifact KernelGen/review_result.json \
  --validation-level l1 --validation-level l2 --validation-level l3
```

恢复前使用 `checkpoint-status`；只有 `reusable=true` 才能跳过该组。输入、相关源文件或外层阶段指纹变化后，旧检查点会被拒绝。

## 4. 缓存与中断恢复 smoke test

```bash
python -B mlu-triton-main/scripts/run_control.py init \
  --output-dir <temp-output> --input "reduce sum" --mode balanced
python -B mlu-triton-main/scripts/run_control.py next \
  --manifest <temp-output>/run_manifest.json
python -B mlu-triton-main/scripts/run_control.py resume \
  --manifest <temp-output>/run_manifest.json
```

验证点：

- EnvConfig 每次运行，且 KernelGen 前必须绑定 `run_context.json`。
- `running` 中断后回到 `pending`，attempts 和同指纹有效检查点保留。
- 第二个相同输入、上下文、模式/预算和 cache directory 的运行返回 `restore`。
- 修改已完成产物或缓存内容后，哈希校验拒绝复用并向下游传播失效。
- cache metadata 包含阶段配置、源快照、上下文、验证等级和产物哈希。

## 5. Optimizer 预算恢复

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> --output-dir <temp-output>/Optimizer --mode balanced
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> --output-dir <temp-output>/Optimizer --mode balanced --resume
```

`--resume` 必须保留 usage、best、history 和 patience 状态。已有状态时不带 `--resume` 的普通 plan 必须拒绝覆盖；输入、模式或预算不一致时恢复必须失败。

## 6. 真实产物与 L3

增量和完整检查：

```bash
python -B validation/validate.py artifacts --output-dir <output_dir>
python -B validation/validate.py artifacts --output-dir <output_dir> --require-complete
```

真实 MLU 环境检查与单算子运行：

```bash
python -B validation/run_mlu_integration.py
python -B validation/run_mlu_integration.py --operator <triton_final.py>
```

L3 必须来自与 `run_context.json` 一致的真实 MLU/工具链。CPU fixture 只能证明 L1/L2 控制逻辑，不能写成精度或性能通过。Nightly 至少覆盖 Elementwise 和 Reduce；Release 覆盖支持的硬件/工具链矩阵。

## 7. 回归比较

```bash
python -B validation/regression.py compare \
  --baseline validation/fixtures/valid/Regression/baseline.json \
  --current validation/fixtures/valid/Regression/current_pass.json \
  --policy validation/regression_policy.json \
  --report-json <report.json> --report-md <report.md>
```

正例退出 0；精度/性能回退退出 1；非法输入退出 2。硬件或工具链不同必须标记 `not_comparable`，不得作为性能通过。

## 8. 最终验收

- L1、L2 全部通过。
- 验证等级不足不能发布缓存；损坏缓存不能恢复。
- Design/Build 可在 L1+L2 复用，Review 只有 L1+L2+L3 才可复用。
- 相同有效输入的第二次运行产生 cache hit。
- Optimizer 恢复不重置全局预算。
- 真实 MLU Elementwise、Reduce 的 L3 与回归门禁通过。

最后一项必须在可用 MLU 环境执行；离线通过不能替代动态精度和性能结论。
