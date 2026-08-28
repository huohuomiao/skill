---
name: triton-kernel-workflow
description: "Plan, generate, validate, debug, and optimize Triton kernels for supported accelerator backends. Use for complete operator-development workflows or focused code generation, correctness validation, runtime repair, and performance tuning."
---

# Triton Kernel Workflow

Use one entrypoint for the complete Triton kernel lifecycle. Treat the directory containing this file as `{skill_root}` and resolve every bundled resource relative to it.

## Select the mode

- **Full workflow**: For a new operator, a requirement description, or an end-to-end development request, read [references/workflows/full-pipeline.md](references/workflows/full-pipeline.md).
- **Code generation**: For requirement-to-kernel generation without running the complete outer workflow, read [references/workflows/code-generation.md](references/workflows/code-generation.md).
- **Code validation**: For correctness checking, static review, execution-driven debugging, or repair of an existing kernel, read [references/workflows/code-validation.md](references/workflows/code-validation.md).
- **Performance tuning**: For optimization of a complete runnable kernel and benchmark, read [references/workflows/performance-tuning.md](references/workflows/performance-tuning.md).

If the request does not explicitly select a partial mode, use the full workflow.

For a full or performance-tuning request, select an optimization mode from `correctness`, `balanced`, and `max-performance`; default to `balanced`. Read [references/contracts/tuning-policy.md](references/contracts/tuning-policy.md) before deciding which tuning stages may run.

## Shared contracts

Before starting a mode, read the contracts it depends on:

- Read [references/contracts/artifact-layout.md](references/contracts/artifact-layout.md) for stage inputs, outputs, ownership, and handoff checks.
- Read [references/contracts/execution-backend.md](references/contracts/execution-backend.md) before compilation, device execution, accuracy testing, or benchmarking.
- Read only the relevant sections of [references/backend/platform-rules.md](references/backend/platform-rules.md) for device, Grid, on-chip memory, dtype, primitive, or device-math constraints.
- Read [references/contracts/rollback-policy.md](references/contracts/rollback-policy.md) when a stage fails, produces incomplete artifacts, or needs another attempt.
- Read [references/contracts/tuning-policy.md](references/contracts/tuning-policy.md) for optimization modes, deterministic strategy routing, and the fixed p3 global budget.
- Read [references/contracts/cache-resume.md](references/contracts/cache-resume.md) before reusing outputs, resuming an interrupted run, or selecting change-affected validation.

## Maintenance validation

When this Skill or one of its bundled resources changes, read [references/contracts/validation-gates.md](references/contracts/validation-gates.md). Run L1 for every change, L1+L2 before committing, and L3 only for scheduled or release regression after the lower gates pass.

## Invariants

- Preserve the full-workflow order: environment check → requirement extraction → code generation → code validation → mode-controlled performance tuning → finalization.
- Run delegated roles serially. Wait for one role to finish and validate its files before starting the next role.
- Exchange stage results through the documented files instead of relying on conversational summaries.
- Do not claim compilation, accuracy, or performance results without actual output from the selected execution backend.
- Read execution state only from `EnvConfig/config.json`; treat `config.md` as a derived human report.
- Read cross-stage status and artifact paths from `run_manifest.json`; do not reconstruct state by reloading all upstream reports.
- Treat `execution-backend.md` and `platform-rules.md` as the only fact sources for their respective domains; downstream documents may point to them but must not copy their rules.
- Select the final optimized kernel only through `scripts/state/select-best-candidate.py`; never choose it from stage order or the last attempted strategy.
- In `correctness`, stop after compilation and accuracy validation and mark performance tuning skipped. In `balanced`, run applicable OOB strategies only. In `max-performance`, additionally allow applicable deep strategies.
- Generate `Optimizer/strategy_plan.json` before scheduling strategies. Never dispatch a strategy whose deterministic plan decision is `skip`.
- Enforce the p3 tuning limits through `Optimizer/tuning_state.json`: at most 3 deep rounds, 16 Worker calls, and 1800 seconds. Real hardware executions remain serial.
- Preserve p2.1 candidate eligibility and deterministic `latency_ms` selection; p3 does not add noise-aware scoring.
- Build stage fingerprints from declared inputs, stage-owned Skill resources, dependency versions, hardware, toolchain, configuration, and upstream fingerprints. Reuse only a verified content-addressed cache entry.
- On the first cache miss, invalid entry, or forced stage, rerun that stage and every active downstream stage while preserving valid upstream cache hits.
- Restore cache snapshots before marking a stage completed. Clear stale artifacts and selected checkpoints for rerun stages; never treat manifest status alone as a cache hit.
- Keep the established runtime artifact directories (`EnvConfig`, `Extractor`, `KernelGen`, and `Optimizer`) unchanged.
- When delegating a role, resolve its reference to an absolute path under `{skill_root}` and include that path in the task message.
- In normal code generation, use one Kernel Designer dispatch and one Kernel Builder dispatch while preserving Step 1–6 files.
- When a full-workflow reduction will enter performance tuning, preserve its validated chunked `optimization_surface`; do not collapse the reduction during Code Gen or preselect an autotune winner from a static estimate.
