# Artifact Layout Contract

Use `{output_dir}` as the runtime output root. If the user does not provide it, use `output_triton_kernel_workflow` under the current working directory.

## Stage ownership and handoffs

| Stage | Owner | Required input | Required output |
| --- | --- | --- | --- |
| Environment | Environment Checker | Current runtime and optional `JOB_ID` | `EnvConfig/config.json`, `EnvConfig/config.md`, `EnvConfig/runtime_info.txt` |
| Requirement extraction | Requirement Analyzer | User requirement or existing Triton code | `Extractor/requirement.md`; `Extractor/original_code.py` for the fast path |
| Code generation | Code Generation workflow | `Extractor/requirement.md` | `KernelGen/triton_code_fix.py`, `KernelGen/triton_report.md` |
| Performance tuning | Performance Tuning workflow | `KernelGen/triton_code_fix.py`; skipped in `correctness` | `Optimizer/best_so_far.py`, `Optimizer/best_so_far.json`, `Optimizer/triton_optimized.py`, `Optimizer/triton_optimized.md` |
| Finalization | Finalization workflow | Validated code; selected tuning output outside `correctness` | `triton_final.py`, `summary.md` |
| State summary | Outer workflow and stage owner | Verified stage result | `run_manifest.json` |

The owner of a stage writes its files. The outer workflow only verifies and reads downstream handoff files; it must not fabricate or rewrite another stage's outputs.

## Runtime output tree

```text
{output_dir}/
├── run_manifest.json              # 下游读取的精简状态摘要
├── RunState/
│   ├── resume_plan.json          # 本次 reuse/rerun/skip 计划
│   ├── fingerprints/             # 当前期望的阶段指纹
│   ├── cache/                    # 内容寻址的不可变阶段快照
│   └── quarantine/               # 可恢复的损坏缓存隔离记录
├── EnvConfig/
│   ├── config.json                 # 唯一机器事实源
│   ├── config.md                   # 从 JSON 生成的人类报告
│   └── runtime_info.txt
├── Extractor/
│   ├── requirement.md
│   └── original_code.py             # Existing-code fast path only
├── KernelGen/
│   ├── step1_base_info.json
│   ├── step1_io_shapes.json
│   ├── step2_block_mapping.json
│   ├── step3_axis_fusion.json
│   ├── step4_code_spec.json
│   ├── step5_kernel_code.py
│   ├── step6_test_code.py
│   ├── step6_test_code_fix.py
│   ├── triton_code_fix.py
│   └── triton_report.md
├── Optimizer/
│   ├── strategy_plan.json          # 确定性静态路由（非 correctness）
│   ├── tuning_state.json           # 全局预算状态（非 correctness）
│   ├── baseline/
│   │   ├── input.py
│   │   ├── baseline.md
│   │   └── candidate.json
│   ├── {n}_{strategy}/
│   │   └── candidate.json           # 仅有真实测量时生成
│   ├── triton_oob_optimized.md
│   ├── best_so_far.py               # 精度通过且同条件下实测最快
│   ├── best_so_far.json             # 确定性选择记录
│   ├── triton_optimized.py
│   └── triton_optimized.md
├── triton_final.py
└── summary.md
```

## Required handoff checks

1. Read stage status and artifact paths from `run_manifest.json`, then verify the referenced files before handoff.
2. Do not enter code generation until requirement extraction is `completed` and `Extractor/requirement.md` is readable.
3. Do not enter performance tuning until code generation and validation are `completed` and `KernelGen/triton_code_fix.py` is readable.
4. In `correctness`, enter finalization only when performance tuning is `skipped` and the validated code-generation artifact is readable. In `balanced` and `max-performance`, require performance tuning to be `completed` and `Optimizer/best_so_far.json`, `Optimizer/triton_optimized.py`, and `Optimizer/triton_optimized.md` to be readable.
5. Do not finish until finalization is `completed` and `triton_final.py` and `summary.md` are non-empty.
6. A resumed handoff additionally requires a verified cache record and restored artifact hashes. `run_manifest.json` status without cache evidence is insufficient.

`run_manifest.json` is a state index, not evidence. A consumer may open the one artifact needed for its task, but must not reload every upstream report merely to reconstruct status.
