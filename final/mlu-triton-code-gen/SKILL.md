---
name: mlu-triton-code-gen
description: "Generate, test, validate, resume, and checkpoint MLU Triton kernels from requirement files or existing Triton code. Use for MLU Triton kernel design/build workflows that must preserve Step 1-7 artifacts while reducing agent dispatch and supporting L1/L2/L3 validation."
---

# MLU Triton Code Gen

## Operating Principle

Generate the same Step 1-7 artifacts as the legacy workflow with two build roles instead of six:

```text
normal:      DesignKernel -> BuildKernel -> Code Review
triton_fast:                BuildKernel -> Code Review
```

Pass only paths and short route values to agents. Keep requirements, JSON, source code, platform rules, and logs in files.

## Inputs And Outputs

Read `{output_dir}/Extractor/requirement.md`. For a Triton input, also read
`{output_dir}/Extractor/original_code.py`.

Always preserve these public outputs:

- `{output_dir}/KernelGen/triton_code_fix.py`
- `{output_dir}/KernelGen/triton_report.md`
- `{output_dir}/KernelGen/dispatch_metrics.json`

The legacy Step 1-7 filenames and field meanings remain compatible. Read
`references/artifact-contracts.md` before producing Step 1-4 JSON.

## Route Selection

Read `输入类型` or `Input Type` from `requirement.md`:

| Route | Condition | Agent dispatch |
|---|---|---|
| `normal` | A non-Triton requirement | `DesignKernel`, then `BuildKernel` |
| `triton_fast` | Existing Triton code | Skip design; run `BuildKernel` only |

Do not fall back to the legacy six-agent chain after a failure.

## Main-Controlled Resume

When called by `mlu-triton-main`, use `{output_dir}/run_manifest.json`. The outer
`kernel_gen` stage must already be `running`. This Skill owns only the following inner
checkpoints; it must not mark the outer stage complete:

| Checkpoint | Artifacts | Required validation |
|---|---|---|
| `design` | Step 1-4 JSON files | L1 + L2 |
| `build` | `step5_kernel_code.py`, `step6_test_code.py` | L1 + L2 |
| `review` | `step6_test_code_fix.py`, `step6_test_code_fix.md`, `review_result.json` | L1 + L2 + L3 |

Before each group, run `checkpoint-status`. Skip the group only when it returns
`reusable=true`. After validating a group, save it with the matching validation levels:

```bash
python .claude/skills/mlu-triton-main/scripts/run_control.py checkpoint-save \
  --manifest <run_manifest> --stage kernel_gen --name <design|build|review> \
  --artifact <output-relative-path> \
  --validation-level l1 --validation-level l2 [--validation-level l3]
```

For a standalone call without a Main manifest, execute normally and do not create a fake
manifest or cache entry.

## Design Group: Step 1-4

Run only on the `normal` route. Dispatch one blocking `DesignKernel` agent:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
Read {skill_root}/mlu-triton-code-gen/subagents/DesignKernel.md and execute it.

route: normal
requirement_path: {requirement_path}
kernelgen_dir: {output_dir}/KernelGen
artifact_contract: {skill_root}/mlu-triton-code-gen/references/artifact-contracts.md
primitives_path: {skill_root}/share/mlu/references/primitives.md
platform_rules_path: {skill_root}/share/mlu/references/platform-rules.md

Write all results to kernelgen_dir. Return only status and paths.
"""
)
```

Require all five compatible artifacts:

1. `step1_base_info.json`
2. `step1_io_shapes.json`
3. `step2_block_mapping.json`
4. `step3_axis_fusion.json`
5. `step4_code_spec.json`

Validate JSON parsing, the schemas under `share/contracts/`, and all cross-file invariants
from `artifact-contracts.md`. Allow one self-correction inside the same agent context. If it
still fails, stop without starting Build.

## Build Group: Step 5-6

Dispatch one blocking `BuildKernel` agent:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
Read {skill_root}/mlu-triton-code-gen/subagents/BuildKernel.md and execute it.

route: {route}
requirement_path: {requirement_path}
original_code_path: {output_dir}/Extractor/original_code.py
kernelgen_dir: {output_dir}/KernelGen
primitives_path: {skill_root}/share/mlu/references/primitives.md
platform_rules_path: {skill_root}/share/mlu/references/platform-rules.md

For normal, read only step1_io_shapes.json and step4_code_spec.json.
For triton_fast, read original_code_path and skip Step 1-4.
Write step5_kernel_code.py and step6_test_code.py. Do not execute the test.
Return only status and paths.
"""
)
```

Before saving the Build checkpoint, require:

- both files exist, are non-empty, and parse as Python;
- Step 6 preserves the Step 5 kernel/wrapper;
- Step 6 includes input construction, a PyTorch reference, `torch.allclose`, and
  `triton.testing.do_bench`;
- no accuracy or performance result is fabricated.

Allow one self-correction in the same context. Otherwise stop.

## Review Group: Step 7

Call Code Review with exactly one path:

```python
Skill(
    skill="mlu-triton-code-review",
    args="{output_dir}/KernelGen/step6_test_code.py"
)
```

Code Review owns all real execution. It must read the fixed EnvConfig backend, run
synchronously, and write:

- `step6_test_code_fix.py`
- `step6_test_code_fix.md`
- `review_result.json`

Require `review_result.json.status` to be `passed` or `repaired`,
`validation_level=l3`, and `accuracy.pass=true`. Infrastructure errors never trigger code
repair and never create a reusable Review checkpoint.

Copy the fixed code to `triton_code_fix.py`; copy or normalize the report to
`triton_report.md`. Preserve measured accuracy and baseline performance fields. Do not claim
success when the report is missing, unparseable, or not converged.

## Dispatch Metrics

Record the actual route and Review outcome:

```bash
python .claude/skills/mlu-triton-code-gen/scripts/dispatch_metrics.py analyze \
  --route <normal|triton-fast> --outcome <direct-pass|repair> \
  --output <output_dir>/KernelGen/dispatch_metrics.json
```

This is a stable static-context proxy, not exact tokenizer billing. The normal route must
retain at least 50% reductions in dispatches and stable scheduled context versus the legacy
chain.

## Invariants

- Keep all agent and Worker calls blocking and serial.
- The normal route uses exactly two Code Gen agents; the fast route uses one.
- Do not let Build read Step 2/3 intermediate JSON.
- Do not duplicate shared MLU rules inside this Skill.
- Do not repair with CPU, pure PyTorch, or scalar-loop kernel substitutions.
- Return only route, group status, final paths, and dispatch metrics path.
