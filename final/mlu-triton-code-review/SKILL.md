---
name: mlu-triton-code-review
description: "Execute, validate, and conservatively repair complete MLU Triton test files. Use when a Triton kernel must pass real MLU accuracy checks; direct passes use zero agents, while business failures use one ReviewAndFix agent and emit L3 evidence."
---

# MLU Triton Code Review

## Contract

Accept exactly one absolute `.py` path containing the Triton kernel, wrapper, input
construction, reference implementation, accuracy assertion, and benchmark. Do not accept a
code snippet or a separate output directory.

For `xxx.py`, write in the same directory:

| Artifact | Meaning |
|---|---|
| `xxx_fix.py` | Final candidate; an unchanged copy when the original passes |
| `xxx_fix.md` | Execution, repair, and final evidence report |
| `review_result.json` | Machine-readable L3 result conforming to `share/contracts/review_result.schema.json` |

The JSON artifact is additive; the legacy `xxx.py -> xxx_fix.py + xxx_fix.md` contract remains
valid for existing callers.

## 1. Bind The Execution Backend

Walk upward from the input directory to find `{output_dir}/EnvConfig/config.md` and
`run_context.json`.

- `execution_backend=local`: synchronously run `python <input_code_path>` locally.
- `execution_backend=worker`: synchronously call
  `mlu-triton-main/subagents/scripts/submit_task_to_worker.py` with
  `--timeout-sec 1800 --task-type accuracy`.
- Missing, unknown, or mismatched context: stop as `infrastructure_error`; do not modify the
  kernel.

Never submit concurrent/background Worker tasks, create another Job, or bypass the submit
script.

Classify exit code `0` as execution and accuracy pass, `1` as a repairable business/accuracy
failure, and `2` or a device/Worker/path failure as infrastructure failure.

## 2. Execute First

Run the original file once before dispatching an agent:

- Pass: copy it unchanged to `xxx_fix.py`, write `xxx_fix.md`, and write
  `review_result.json` with `status=passed`, `validation_level=l3`, and
  `accuracy.pass=true`. Do not dispatch an agent.
- Business failure: record the first log and continue to one `ReviewAndFix` agent.
- Infrastructure failure: write the Markdown and JSON failure evidence, do not produce a
  false passing candidate, and stop.

## 3. One-Agent Review And Repair

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
Read {skill_root}/mlu-triton-code-review/ReviewAndFix.md and execute it in one context.

input_code_path: {input_code_path}
initial_log_path: {initial_log_path}
env_config_path: {output_dir}/EnvConfig/config.md
run_context_path: {output_dir}/EnvConfig/run_context.json
primitives_path: {skill_root}/share/mlu/references/primitives.md
platform_rules_path: {skill_root}/share/mlu/references/platform-rules.md
libdevice_path: {skill_root}/share/mlu/references/libdevice.md
common_error_path: {skill_root}/mlu-triton-code-review/ref/common_error.md
troubleshooting_path: {skill_root}/mlu-triton-code-review/ref/troubleshooting.md
report_template_path: {skill_root}/mlu-triton-code-review/ref/report_template.md
review_schema_path: {skill_root}/share/contracts/review_result.schema.json
worker_submit_path: {skill_root}/mlu-triton-main/subagents/scripts/submit_task_to_worker.py

Write xxx_fix.py, xxx_fix.md, and review_result.json beside the input.
Return only status and paths.
"""
)
```

After the agent returns, check that the fixed code parses as Python, the Markdown has an
explicit conclusion, and the JSON conforms to the schema. Do not run the code again in the
caller. Accept only `passed` or `repaired` with `accuracy.pass=true`.

## Repair Limits

- Make the smallest change supported by real stderr, traceback, or accuracy values.
- Stop on pass, two consecutive identical failures, five repair rounds, or infrastructure
  failure.
- Never replace Triton with CPU, pure PyTorch computation, or a scalar element loop.
- Direct pass uses zero agents; a repair path uses at most one agent.
