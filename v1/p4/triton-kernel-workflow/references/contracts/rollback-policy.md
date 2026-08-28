# Rollback and Retry Policy

Use file validation and real execution results to decide whether a stage can continue.

## Environment

- Retry only when a probe or remote submission failed for a plausibly transient reason.
- If both local and worker checks fail, stop. Do not continue into result-dependent stages.

## Code generation

- Validate the required output after every delegated role.
- Follow the stage-specific rollback table in `../workflows/code-generation.md`.
- Limit each code-generation rollback edge to three attempts. Report the last verified error after the limit.

## Code validation

- If the original program passes, copy it to the fixed output and stop validation.
- During runtime repair, stop when the program passes, the same error repeats in two consecutive iterations, or five repair iterations have run.
- Do not treat an execution-backend or infrastructure failure as a kernel defect.

## Performance tuning

- Run strategies serially in their documented order.
- If a strategy omits its required code or report, retry that delegated strategy at most twice.
- After two incomplete attempts, mark the strategy failed and carry its input code forward unchanged; do not fabricate performance improvement.
- Accept an optimized candidate only when accuracy passes and measured performance is better under the same execution conditions.
- Preserve the current `best_so_far` checkpoint when a candidate fails accuracy, regresses performance, lacks measurements, or uses incompatible benchmark conditions.
- Final code selection must run `scripts/state/select-best-candidate.py`; stage order and “last successful file” are not valid selection rules.
