# Finalization Workflow

Run only after the manifest handoff checks in `../contracts/artifact-layout.md` pass.

Render `summary.md` using `../templates/final-summary.md`. This rendering step is presentation-only and must not compile, execute, profile, or benchmark a kernel. A template-only fingerprint change may therefore rerun finalization while reusing all upstream computation.

## Inputs

Read:

- `{output_dir}/run_manifest.json`
- `{output_dir}/EnvConfig/config.json`

For `correctness`, read the validated code path from the manifest's `code-generation` artifacts (normally `{output_dir}/KernelGen/triton_code_fix.py`). For `balanced` and `max-performance`, also read:

- `{output_dir}/Optimizer/triton_optimized.py`
- `{output_dir}/Optimizer/best_so_far.json`

Use stage status and artifact paths from `run_manifest.json`. Do not load the full requirement, generation report, optimization report, or every strategy report merely to reconstruct status. Open one detailed artifact only when a user-requested summary needs information absent from the manifest and selected checkpoint.

## Outputs

1. Copy the accuracy-validated code unchanged to `{output_dir}/triton_final.py`: use the code-generation artifact in `correctness`, otherwise use `{output_dir}/Optimizer/triton_optimized.py`.
2. Write `{output_dir}/summary.md`.

## Summary contract

Include:

1. Route, execution backend, and stage statuses from the manifest.
2. Final code path and verified code-generation/validation status.
3. For `balanced` and `max-performance`, selected candidate, measured latency, hardware, backend, and benchmark signature from `best_so_far.json`.
4. For `correctness`, state that performance tuning was skipped by mode and that latency and speedup are `N/A`; do not infer a selected checkpoint.
5. An artifact index using manifest paths. Link detailed reports without rereading them.
6. `N/A（原因：...）` for genuinely absent values; never infer a number or success state.

Do not claim performance improvement without real measurements. Do not delete intermediate artifacts.

After both outputs are non-empty, mark finalization completed:

```bash
python {skill_root}/scripts/state/update-run-manifest.py \
  --manifest {output_dir}/run_manifest.json \
  --stage finalization \
  --status completed \
  --artifact final_code={output_dir}/triton_final.py \
  --artifact summary={output_dir}/summary.md
```

Before reporting the entire Job complete, use the active-task mechanism documented by `execution-backend.md` for the current `JOB_ID` and confirm no queued, leased, or running task remains.
