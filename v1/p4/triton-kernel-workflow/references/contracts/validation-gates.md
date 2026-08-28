# Validation Gates Contract

Use these gates when changing or releasing this Skill. They validate the workflow implementation; they do not replace per-operator accuracy or performance tests.

For a changed-file set, first run `scripts/validation/plan-change-impact.py`. It may narrow L3 operator and Worker coverage according to `cache-resume.md`, but L1 always runs and L2 always runs before any selected L3 check. An impact plan is routing evidence, not a passing test result.

## Gate order

1. **L1 static** runs without a device and must finish within 30 seconds.
2. **L2 offline behavior** runs fixed scenarios and mock artifacts without a device. L1 and L2 together must finish within 300 seconds.
3. **L3 hardware integration** may run only when fresh L1 and L2 reports pass for the current Skill fingerprint.

Run the gate driver:

```bash
python {skill_root}/scripts/validation/run-validation.py \
  --level {l1|l2|l3} \
  --report-dir <absolute-report-directory> \
  [--integration-suite <absolute-suite-json>] \
  [--impact-plan <absolute-change-impact-json>]
```

Selecting `l2` runs L1 before L2. Selecting `l3` runs L1 and L2 first, then runs the hardware suite only if both reports pass and satisfy their time budgets. A fresh impact plan whose Skill fingerprint matches the lower-gate reports narrows L3 to its selected operator classes and Worker checks; omission runs the complete suite.

## L1 static checks

L1 must not compile or execute a kernel. It checks:

- `SKILL.md` frontmatter and unfinished scaffold markers.
- Local Markdown links, `{skill_root}` references, and JSON Schema references.
- Markdown headings and fenced blocks.
- Empty files, placeholder-only files, Python syntax, JSON syntax, and Shell syntax when Bash is available.
- Repeated directive rules and Markdown size thresholds.
- Artifact naming and handoff contracts.
- The invariant that final optimized code comes only from `select-best-candidate.py`.
- p4 stage-dependency registry coverage, fingerprint fields, cache snapshot paths, forward invalidation, and resume-plan contracts.

Warnings are recorded but do not fail the gate. Syntax errors, missing paths, broken contracts, empty files, and hard size-limit violations fail it.

## L2 offline behavior

L2 reads `../evals/offline-scenarios.json` and evaluates every scenario deterministically. It also validates mock intermediate artifacts against the schemas under `../schemas/` and reruns the P0 contract tests.

The fixed suite covers:

- Full, code-generation, validation, and tuning input routing.
- Environment failure stopping downstream dynamic stages.
- Observed file reads staying within the declared role whitelist.
- Valid and invalid intermediate JSON handling.
- Code-generation, infrastructure, and performance rollback targets.
- Suppression of performance claims without real measurements and passing accuracy.
- Deterministic cache hits, corrupted-cache rejection, downstream invalidation, artifact restoration, and change-impact routing.

## L3 hardware integration

L3 requires a project-provided suite conforming to `../schemas/integration-suite.schema.json`. It must include representative `elementwise`, `reduction`, and `layout` cases. Every case provides serial commands for compilation, accuracy, and performance. The suite also provides Worker submission and failure-recovery commands.

L3 executes command arrays directly without a shell, records real exit codes and bounded stdout/stderr, and never manufactures missing metrics. Real benchmark commands remain serial so they do not interfere with one another.

Do not describe L3 as passed when the suite is absent, the hardware identity is unknown, a command fails, or L1/L2 reports are stale or failed.
