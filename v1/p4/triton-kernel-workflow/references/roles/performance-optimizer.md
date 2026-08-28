# Performance Optimizer

Apply one named optimization strategy without extending its scope. Backend execution rules come only from `{skill_root}/references/contracts/execution-backend.md`; platform constraints come only from the relevant sections of `{skill_root}/references/backend/platform-rules.md`.

## Inputs

- Strategy name and strategy document path
- `{workdir}/input.py`
- `{output_dir}/EnvConfig/config.json`
- `{output_dir}/Optimizer/tuning_state.json`
- `skill_root`

Read only the input code, the named strategy, a reference explicitly selected by that strategy, `config.json`, `tuning_state.json`, `execution-backend.md` when executing, and the relevant platform-rule section. Do not read another strategy workdir or its reports.

## Outputs

- `{workdir}/triton_optimized.py`
- `{workdir}/triton_optimized.md`
- Optional `{workdir}/candidate.json` only after real accuracy and comparable performance measurements

Always produce code and a report. If the strategy is inapplicable or fails, carry `input.py` forward unchanged and record the reason. Do not fabricate a candidate.

## Procedure

1. Read the strategy's admission conditions, transformation steps, and invariants.
2. Load at most the one example/reference matching the detected code pattern.
3. For a target-platform constraint, load only the needed section:
   - Device literal or synchronization → Runtime and device adaptation
   - Grid transformation → Grid and persistent kernels
   - Resource/config tuning → NRAM and tuning
   - Primitive, dtype, or Libdevice → Dtype, primitives, and device math
4. If admission fails, copy the input and write a skipped report.
5. If admitted, make only the documented transformation and write `triton_optimized.py`.
6. Run syntax checks locally. Run compilation, accuracy, and benchmark commands only through `execution-backend.md` using the backend recorded in `config.json`.
   - Before any dynamic command, run `manage-tuning-budget.py check`. For local execution, use `run-budgeted-local.py` so the command cannot outlive the remaining wall-time.
   - For Worker execution, pass `--budget-state {output_dir}/Optimizer/tuning_state.json` and a unique `--budget-label`; the submission script performs the reservation.
   - Run compilation, accuracy, and benchmark serially. Never overlap hardware measurements.
   - Treat Worker exit `4` as a normal budget stop: carry `input.py` forward unchanged, report `budget-stopped`, and do not create a candidate or retry.
7. Keep the change only when accuracy passes. Record real stdout/stderr and measurement conditions.
8. Write `candidate.json` conforming to `optimization-candidate.schema.json` only when latency is a real finite measurement and its hardware/backend/benchmark signature is known.

## Report contract

`triton_optimized.md` records:

- Strategy name and applied/skipped/failed status
- Admission evidence and exact transformation
- Accuracy result and tolerances
- Original and optimized latency/bandwidth when measured
- Execution backend, hardware, and benchmark signature
- Output and evidence paths

Use `N/A` with a reason for missing measurements; never fill them with inferred values.

## Invariants

- No undocumented optimization is introduced.
- Infrastructure failure stops the strategy without modifying the kernel in response.
- Accuracy-passing input remains the rollback checkpoint.
- Benchmark comparisons use identical input, timing method, hardware, and backend.
- File reads stay inside the declared whitelist and all paths are absolute.
- Budget exhaustion never triggers a kernel modification or retry.
