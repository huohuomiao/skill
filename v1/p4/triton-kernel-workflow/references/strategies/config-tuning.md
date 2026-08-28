# Configuration Tuning

Tune existing block sizes, `num_warps`, and `num_stages` without changing the algorithm. Read Grid and NRAM constraints only from `{skill_root}/references/backend/platform-rules.md`.

## Admission conditions

- At least one configuration parameter is safely adjustable.
- A real benchmark and accuracy test are runnable.
- The baseline is accuracy-passing and measured.

Otherwise skip without generating a performance candidate.

## Steps

1. Extract and rank tensor axes. Load `{skill_root}/references/strategies/config-tuning-tensor-axis.md` only when the mapping is nontrivial.
2. Classify the kernel bottleneck and detect persistent loops or widening/transpose operations.
3. Form a small candidate set from the current config and platform constraints.
4. Change one coherent configuration dimension at a time so regressions are attributable.
5. Run candidates serially through `execution-backend.md`; reject any accuracy failure immediately.
6. Keep the lowest-latency accuracy-passing configuration under the same benchmark signature.

## Invariants

- Kernel computation, indexing, shapes, and tolerance remain unchanged.
- Heuristic-fixed block parameters are not tuned.
- Resource failure is not a kernel correctness failure.
- Missing or incompatible measurements cannot become a candidate.
- Output always preserves the best measured accuracy-passing checkpoint.
