# Performance Tuning Workflow

Use this workflow only for `balanced` or `max-performance`. A `correctness` full run marks performance tuning skipped and proceeds directly to finalization.

Enter only when the p4 resume plan marks `performance-tuning` as `rerun`. A verified cache hit is restored by the outer cache workflow and must not initialize a new tuning budget or execute hardware commands. A rerun always initializes a fresh p3 budget with the unchanged fixed limits.

## Inputs and contracts

Inputs:

- `triton_code`: complete runnable Triton code as a file path or code text
- `output_dir`
- `skill_root`
- `optimization_mode`: `balanced` or `max-performance`; default `balanced`

Read before execution:

- `../contracts/artifact-layout.md`
- `../contracts/execution-backend.md`
- `../contracts/tuning-policy.md`
- `../contracts/rollback-policy.md`

Read execution state only from `{output_dir}/EnvConfig/config.json`. All compilation, accuracy, profiling, and benchmark commands follow `execution-backend.md` and run serially. p3 keeps the p2.1 candidate schema and `latency_ms` selector unchanged; do not add measurement-noise fields or a different scoring rule.

## Step 0: Initialize the bounded run

1. Create `{output_dir}/Optimizer/` and normalize the input into `{output_dir}/Optimizer/baseline/input.py`. Reject an incomplete input that lacks a `@triton.jit` kernel, wrapper, accuracy test, or performance test.
2. Initialize the immutable fixed budget before the baseline measurement:

   ```bash
   python {skill_root}/scripts/state/manage-tuning-budget.py init \
     --state {output_dir}/Optimizer/tuning_state.json \
     --mode {optimization_mode}
   ```

3. Generate the deterministic static plan:

   ```bash
   python {skill_root}/scripts/state/plan-strategies.py \
     --input {output_dir}/Optimizer/baseline/input.py \
     --output {output_dir}/Optimizer/strategy_plan.json \
     --mode {optimization_mode}
   ```

4. Validate the two JSON files against `tuning-state.schema.json` and `strategy-plan.schema.json`. Mark the manifest stage `running` and record their paths. For a focused tuning request whose manifest does not yet exist, also pass `--mode performance-tuning --optimization-mode {optimization_mode}`. For a full run, omit those two creation arguments and verify that the existing manifest already has the same immutable optimization mode.

   ```bash
   python {skill_root}/scripts/state/update-run-manifest.py \
     --manifest {output_dir}/run_manifest.json \
     --stage performance-tuning \
     --status running \
     --artifact strategy_plan={output_dir}/Optimizer/strategy_plan.json \
     --artifact tuning_state={output_dir}/Optimizer/tuning_state.json \
     --metadata optimization_mode={optimization_mode}
   ```

Never dispatch a strategy whose plan decision is `skip`. Treat the plan as immutable for this run.

## Step 1: Establish the measured baseline

Run a budget `check`, then execute baseline accuracy and performance using the selected backend. Wrap local commands with `run-budgeted-local.py`. Each Worker command must include:

```bash
--budget-state {output_dir}/Optimizer/tuning_state.json \
--budget-label baseline-<accuracy-or-performance>
```

Write real output and measurement conditions to `{output_dir}/Optimizer/baseline/baseline.md`. Only when accuracy passes and a finite positive hardware latency exists, write `baseline/candidate.json` conforming to `optimization-candidate.schema.json`.

The candidate `benchmark_signature` covers all comparison-affecting conditions, including shape, dtype, warmup, repeat, and benchmark entrypoint. All candidates in one selection must use the same backend, hardware model, and signature.

If no valid baseline candidate can be produced, stop. Budget exhaustion before a baseline candidate produces no performance conclusion. Workload failure and infrastructure failure retain their original p2.1 classification.

## Step 2: Execute only admitted OOB strategies

Read the OOB rows of `strategy_plan.json` in `order` order:

| Strategy | Name | Strategy document |
| --- | --- | --- |
| Tiling | `retiling` | `{skill_root}/references/strategies/tiling.md` |
| Reduction | `reduce-opt` | `{skill_root}/references/strategies/reduction.md` |
| Grid | `modify-grid` | `{skill_root}/references/strategies/grid-layout.md` |
| Index simplification | `index-computation-simplify` | `{skill_root}/references/strategies/index-simplification.md` |
| Autotune configuration | `gen-autotune-config` | `{skill_root}/references/strategies/autotune-config.md` |

For a `skip` row, add its router reason to the progress report without creating a strategy workdir or dispatching an optimizer.

For each `apply` row, serially:

1. Run `manage-tuning-budget.py check`. Exit `3` means stop scheduling and continue to Step 4 using completed candidates.
2. Create `{output_dir}/Optimizer/{order}_{name}/` and copy the previous OOB output to `input.py`; the first admitted strategy uses `baseline/input.py`.
3. Delegate exactly one Performance Optimizer using `{skill_root}/references/roles/performance-optimizer.md`. Pass the strategy name, absolute strategy document, absolute workdir, `skill_root`, config path, and tuning-state path.
4. Wait for completion. Require non-empty `triton_optimized.py` and `triton_optimized.md`. Validate an optional `candidate.json` against `optimization-candidate.schema.json` and admit it only when it is a real, accuracy-passing, comparable measurement.
5. If outputs are missing, inspect the budget state before retrying. Do not retry `budget-stopped`; otherwise retain the existing maximum of two retries. After final failure, carry `input.py` forward unchanged and write the failure reason.

The delegated optimizer passes `--budget-state` to every Worker submission. Agent dispatches do not consume Worker-call budget; real remote submissions do.

Write `{output_dir}/Optimizer/triton_oob_optimized.md` with every apply/skip/failure decision and copy the last carried OOB code to `{output_dir}/Optimizer/triton_oob_optimized.py`. This carried code is not automatically the final winner.

## Step 3: Bounded deep optimization

For `balanced`, skip this step and record `disabled-by-mode` in the final report.

For `max-performance`, first inspect the advanced rows in the static plan. If none has decision `apply`, record `no-applicable-advanced-strategy` and do not start a round.

Otherwise create `{output_dir}/Optimizer/Advanced_Optimization/` and execute at most three serial rounds:

1. Before round `i`, reserve it atomically:

   ```bash
   python {skill_root}/scripts/state/manage-tuning-budget.py start-round \
     --state {output_dir}/Optimizer/tuning_state.json \
     --label iter_<i>
   ```

   Exit `3` is a normal bounded stop and exits the loop. Never start a fourth round.
2. Create `Advanced_Optimization/iter_{i}/input.py` from the previous round, or from `triton_oob_optimized.py` for round 1.
3. Delegate the performance analyzer using `{skill_root}/references/strategies/performance-analysis.md`. It checks the same global state and passes `--budget-state` to profiling Worker calls.
4. Select at most one recommendation whose advanced plan row is `apply`. Prefer a not-yet-used admitted strategy. If no admitted recommendation remains, stop deep optimization.
5. Delegate the Performance Optimizer for that one strategy, serially. Apply the same output, candidate, retry, budget-stop, and rollback rules as OOB.
6. Continue only while the existing p2.1 termination rules permit it: stop after three rounds, after three consecutive no-improvement rounds, or when analysis has no admitted suggestion. The hard budget may stop earlier.

Write `triton_advanced_optimized.md` and the last carried code as `triton_advanced_optimized.py`. Static candidate inspection may be parallel, but performance analysis, compilation, accuracy, and benchmark execution must never overlap.

## Step 4: Deterministic final selection

Preserve every completed candidate when the budget stops. Discover all `candidate.json` files under `Optimizer/` and run the existing selector unchanged:

```bash
python {skill_root}/scripts/state/select-best-candidate.py \
  --candidate-root {output_dir}/Optimizer \
  --output-dir {output_dir}/Optimizer
```

The selector accepts only accuracy-passing, real, comparable measurements; it selects the minimum `latency_ms` and breaks exact ties by `candidate_id`. It must produce:

- `{output_dir}/Optimizer/best_so_far.py`
- `{output_dir}/Optimizer/best_so_far.json`
- `{output_dir}/Optimizer/triton_optimized.py`

Do not select by stage order, modification time, the last attempted strategy, confidence intervals, variance, or inferred performance. If the selector fails, stop without a fabricated winner.

Combine the OOB summary, optional advanced summary, budget state, plan decisions, and selected checkpoint into `{output_dir}/Optimizer/triton_optimized.md`. Include the budget usage and stop reason. Use `N/A` for missing data.

If the budget remains active after all planned work, mark it complete. If a non-budget termination rule stopped deep optimization, record that reason:

```bash
python {skill_root}/scripts/state/manage-tuning-budget.py complete \
  --state {output_dir}/Optimizer/tuning_state.json \
  --reason <planned-work-complete-or-termination-reason>
```

Finally update the manifest from `best_so_far.json`:

```bash
python {skill_root}/scripts/state/update-run-manifest.py \
  --manifest {output_dir}/run_manifest.json \
  --stage performance-tuning \
  --status completed \
  --artifact strategy_plan={output_dir}/Optimizer/strategy_plan.json \
  --artifact tuning_state={output_dir}/Optimizer/tuning_state.json \
  --artifact best={output_dir}/Optimizer/best_so_far.json \
  --artifact code={output_dir}/Optimizer/triton_optimized.py \
  --artifact report={output_dir}/Optimizer/triton_optimized.md \
  --metadata optimization_mode={optimization_mode} \
  --metadata budget_stop_reason=<value-from-tuning-state-or-none> \
  --selected-candidate-id <candidate_id> \
  --selected-code-path <code_path> \
  --selected-latency-ms <latency_ms>
```

Downstream consumers read only the manifest and selected checkpoint, not every strategy report.
