# Full Pipeline

Use this workflow for a new Triton operator, a natural-language operator requirement, an existing Triton implementation that must complete the whole lifecycle, or an end-to-end development request.

Read these contracts before execution:

- `../contracts/artifact-layout.md`
- `../contracts/execution-backend.md`
- `../contracts/rollback-policy.md`
- `../contracts/tuning-policy.md`
- `../contracts/cache-resume.md`

Use `{output_dir}/run_manifest.json` as the compact state index. Update it only through `{skill_root}/scripts/state/update-run-manifest.py`; stage owners still write their own artifacts.

Select one immutable `optimization_mode` before execution: `correctness`, `balanced`, or `max-performance`. Use `balanced` when the caller does not specify a mode. Mode behavior and the fixed global tuning budget are owned by `tuning-policy.md`.

## Step 0: Cache and resume preflight

Before delegating a stage:

1. Obtain current dependency, hardware, toolchain, and backend identity. If current identity cannot be verified, force `environment`; do not trust an old `EnvConfig/config.json` by itself.
2. Generate available desired fingerprints in stage order under `{output_dir}/RunState/fingerprints/` using `fingerprint-stage.py`. Use current inputs or verified upstream cache snapshots. Once a stage is known to rerun, downstream fingerprints may be deferred.
3. Generate `{output_dir}/RunState/resume_plan.json` with `plan-resume.py`, then apply it with `apply-resume-plan.py`.
4. Do not dispatch stages marked `reuse` or `skip`. If `resume_from` is `null`, verify the restored final outputs and finish without dynamic execution.
5. For every `rerun` stage, compute its final desired fingerprint immediately before execution. After the stage completes and its artifacts pass their existing handoff checks, record them with `stage-cache.py record` and add `cache_status=recorded`, `cache_fingerprint`, and `cache_record` to its manifest metadata.

Only completed, verified files inside `{output_dir}` enter the cache. A cache or resume-script failure disables reuse and causes a normal stage rerun; it never authorizes a performance claim.

## Delegation rule

Delegate only the roles explicitly assigned below. Run one delegated role at a time, wait for it to finish, and validate its required files before continuing. The outer workflow must not take over a delegated role while it is running.

## Step 1: Environment preparation

Delegate the Environment Checker using `../roles/environment-checker.md`.

Pass:

- `{output_dir}`
- `{skill_root}`
- `mode=full`
- `optimization_mode={optimization_mode}`
- The current `JOB_ID` when available

Require:

- `{output_dir}/EnvConfig/config.json`
- `{output_dir}/EnvConfig/runtime_info.txt`

Stop if neither local nor worker execution passes the environment contract.

Record the verified environment artifact paths and `execution_backend` in `run_manifest.json` before continuing.

## Step 2: Requirement extraction

Delegate the Requirement Analyzer using `../roles/requirement-analyzer.md`.

Pass the original user requirement or existing Triton code, `{output_dir}`, and `{skill_root}`. Require `{output_dir}/Extractor/requirement.md`. Existing Triton input may also produce `{output_dir}/Extractor/original_code.py` for the fast path.

After verification, mark requirement extraction completed and record only the requirement and optional original-code paths in the manifest.

## Step 3: Code generation and validation

Read and execute `code-generation.md` with:

- `requirement={output_dir}/Extractor/requirement.md`
- `output_dir={output_dir}`
- `skill_root={skill_root}`
- `optimization_intent=standalone` for `correctness`, otherwise `handoff-to-tuning`

That workflow includes the code-validation stage. Require:

- `{output_dir}/KernelGen/triton_code_fix.py`
- `{output_dir}/KernelGen/triton_report.md`

The code-generation report is a stage result, not the end of the full pipeline.

Use the manifest handoff written by `code-generation.md`; do not reread all Step 1–6 artifacts in the outer workflow.

## Step 4: Mode-controlled performance tuning

For `correctness`, do not enter the performance-tuning workflow. Mark the stage skipped and record why:

```bash
python {skill_root}/scripts/state/update-run-manifest.py \
  --manifest {output_dir}/run_manifest.json \
  --stage performance-tuning \
  --status skipped \
  --metadata reason=correctness-mode
```

For `balanced` or `max-performance`, read and execute `performance-tuning.md` with:

- `triton_code={output_dir}/KernelGen/triton_code_fix.py`
- `output_dir={output_dir}`
- `skill_root={skill_root}`
- `optimization_mode={optimization_mode}`

Require:

- `{output_dir}/Optimizer/best_so_far.json`
- `{output_dir}/Optimizer/triton_optimized.py`
- `{output_dir}/Optimizer/triton_optimized.md`

Only the performance-tuning workflow and its delegated optimizer role may create or modify those outputs.

Require the performance-tuning manifest stage to identify `best_so_far.json`, final code, report, and the selected checkpoint.

## Step 5: Finalization

Read and execute `finalization.md`. In `correctness`, it finalizes the accuracy-validated code without claiming performance; in the other modes, it reads the selected checkpoint. Require non-empty `{output_dir}/triton_final.py` and `{output_dir}/summary.md` and a completed finalization manifest stage before reporting completion.
