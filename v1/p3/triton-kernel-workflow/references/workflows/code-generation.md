# Code Generation Workflow

Generate a runnable Triton kernel while preserving the existing Step 1–6 artifact contract. The normal path uses two outer delegated roles: one design task and one build task. Code validation remains a separate workflow.

## Inputs and shared state

- `requirement`: `{output_dir}/Extractor/requirement.md`
- Optional fast-path source: `{output_dir}/Extractor/original_code.py`
- `output_dir`
- `skill_root`
- `optimization_intent`: `handoff-to-tuning` when the caller will run performance tuning, otherwise `standalone`
- State summary: `{output_dir}/run_manifest.json`

Read execution state only from `{output_dir}/EnvConfig/config.json`. Read `{skill_root}/references/contracts/execution-backend.md` only when entering real compilation or execution. Read only the needed sections of `{skill_root}/references/backend/platform-rules.md`; never copy platform or execution rules into this workflow.

## Stable outputs

Normal-path intermediate artifacts remain compatible:

```text
KernelGen/
├── step1_base_info.json
├── step1_io_shapes.json
├── step2_block_mapping.json
├── step3_axis_fusion.json
├── step4_code_spec.json
├── step5_kernel_code.py
├── step6_test_code.py
├── step6_test_code_fix.py
├── triton_code_fix.py
└── triton_report.md
```

Downstream stages read artifact paths and stage status from `run_manifest.json`; they do not reload every intermediate report.

## Step 0: classify the input

Verify `requirement.md` exists and is non-empty.

- **Existing-code fast path**: `Extractor/original_code.py` exists and contains a Triton kernel plus wrapper. Skip the design task and pass the source to Kernel Builder.
- **Normal path**: use the requirement and run Kernel Designer.

Do not derive `optimization_intent` from the operator name. The full pipeline passes `handoff-to-tuning`; a partial Code Generation request defaults to `standalone`. Existing-code fast-path source is preserved under either intent.

Do not infer fast-path eligibility from conversational text when `original_code.py` is absent.

Mark code generation as running:

```bash
python {skill_root}/scripts/state/update-run-manifest.py \
  --manifest {output_dir}/run_manifest.json \
  --stage code-generation \
  --status running \
  --metadata optimization_intent={optimization_intent}
```

## Step 1: one design dispatch

Skip this step only on the verified existing-code fast path.

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
    Read {skill_root}/references/roles/kernel-designer.md and perform the complete
    design task.

    requirement: {output_dir}/Extractor/requirement.md
    output_dir: {output_dir}
    skill_root: {skill_root}
    optimization_intent: {optimization_intent}

    Write all Step 1–4 compatibility artifacts. Load only a matching example and
    only the platform-rule sections needed by this operator.
    """
)
```

After the role finishes, require non-empty:

- `step1_base_info.json`
- `step1_io_shapes.json`
- `step2_block_mapping.json`
- `step3_axis_fusion.json`
- `step4_code_spec.json`

Parse all JSON files and verify `step1_io_shapes.json` is identical to the `io_shapes` field in `step1_base_info.json`. Retry Kernel Designer only when this design bundle is invalid, at most three total attempts.

When `step1_base_info.json` has a non-empty `reduce_axes`, validate the retained optimization surface before building:

```bash
python {skill_root}/scripts/validation/validate-optimization-surface.py \
  --base-info {output_dir}/KernelGen/step1_base_info.json \
  --spec {output_dir}/KernelGen/step4_code_spec.json \
  --intent {optimization_intent}
```

For `handoff-to-tuning`, a failed surface check returns only to Kernel Designer. It is a design-contract failure, not permission for the outer workflow to rewrite the spec.

## Step 2: one build dispatch

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
    Read {skill_root}/references/roles/kernel-builder.md and perform the complete
    build task.

    requirement: {output_dir}/Extractor/requirement.md
    original_code: {output_dir}/Extractor/original_code.py (only when present)
    output_dir: {output_dir}
    skill_root: {skill_root}
    optimization_intent: {optimization_intent}

    Write step5_kernel_code.py and step6_test_code.py. Preserve the Step 5
    kernel/wrapper exactly when adding the test harness.
    """
)
```

Require both outputs to be non-empty and Python-parseable. Verify Step 6 contains a Triton kernel, wrapper, accuracy test, performance test, and executable entrypoint. Retry Kernel Builder only when its own outputs are invalid, at most three total attempts.

The normal path therefore performs two outer Code Gen role dispatches instead of six. A fast path performs one.

## Step 3: validate the built program

Read and execute `code-validation.md` with:

- `input_code_path={output_dir}/KernelGen/step6_test_code.py`
- `output_dir={output_dir}`
- `skill_root={skill_root}`

Real compilation and accuracy execution follow `execution-backend.md`. Require:

- `{output_dir}/KernelGen/step6_test_code_fix.py`
- `{output_dir}/KernelGen/step6_test_code_fix.md`

An infrastructure failure stops this workflow without rewriting the kernel. A test-harness defect returns only to Kernel Builder. Return to Kernel Designer only when validation evidence identifies an invalid design contract such as a wrong shape, dtype, formula, or mapping.

## Step 4: publish the handoff

Copy the validated program to:

- `{output_dir}/KernelGen/triton_code_fix.py`

Write `{output_dir}/KernelGen/triton_report.md` with:

- input path and fast/normal route
- optimization intent and reduction-surface validation result when applicable
- design/build attempt counts
- validation backend and real result status
- accuracy evidence path
- generated artifact index
- explicit statement that performance is unverified unless real benchmark evidence exists

Update the manifest after all required files exist:

```bash
python {skill_root}/scripts/state/update-run-manifest.py \
  --manifest {output_dir}/run_manifest.json \
  --stage code-generation \
  --status completed \
  --artifact code={output_dir}/KernelGen/triton_code_fix.py \
  --artifact report={output_dir}/KernelGen/triton_report.md \
  --artifact io_shapes={output_dir}/KernelGen/step1_io_shapes.json
```

On the fast path, omit `io_shapes` only when no compatible design artifact exists; the manifest must record the route in stage metadata.

## Rollback scope

| Failure | Smallest retry target |
| --- | --- |
| Invalid Step 1–4 JSON bundle | Kernel Designer |
| Invalid Step 5 kernel | Kernel Builder, then regenerate Step 6 |
| Invalid test harness only | Kernel Builder test phase; preserve Step 5 |
| Runtime kernel defect | Code Validation runtime repair |
| Shape/dtype/formula contract defect | Kernel Designer, then downstream artifacts |
| Backend or Worker infrastructure failure | Environment/backend configuration; do not modify code |

Each retry target has at most three total attempts. Preserve the last verified error and never fabricate compilation, accuracy, or performance success.
