# Autotune Configuration Generation

Use only when the kernel has tunable block parameters or an incomplete autotune decorator. Read NRAM/Grid constraints from `{skill_root}/references/backend/platform-rules.md` rather than copying device constants.

## Admission conditions

- No `@triton.autotune` exists and at least one safe tunable parameter exists; or
- The decorator lacks `num_warps`/`num_stages`; or
- Existing keys/config fields do not match the wrapper/kernel contract.

Skip when no parameter can vary safely or the existing single configuration is complete and valid.

## Steps

1. Extract tensor axes, strides, reduction/parallel type, block parameter, and loop presence. Load `{skill_root}/references/strategies/autotune-tensor-axis.md` only if this mapping cannot be derived directly.
2. Rank axes: reduction before parallel; within a type, smaller stride receives higher priority.
3. Generate candidates that respect shape, dtype, Grid, NRAM, and kernel parameter constraints from `platform-rules.md`.
4. Keep decorator keys aligned with runtime shape/dtype fields that truly affect the best configuration.
5. Benchmark candidates serially through `execution-backend.md` using identical inputs and timing.
6. Emit one measured best accuracy-passing configuration. If measurements are unavailable, do not label any configuration best.

## Candidate generation versus elimination

Generate a useful search space before eliminating candidates. A static NRAM estimate is a feasibility screen, not a substitute for compilation or measurement:

- Model live tensor lifetimes and reuse. Do not sum every intermediate tensor when their lifetimes do not overlap.
- Do not multiply the complete tile estimate by `num_stages` unless the generated code and backend prove those buffers are simultaneously resident.
- Reject a candidate statically only when it is provably outside a hard shape, dtype, Grid, or NRAM bound. When the estimate is uncertain but plausible, compile it serially and let a real resource error reject it.
- Record statically rejected, compile-rejected, accuracy-rejected, and benchmarked candidates separately.

For an I/O-bound row reduction or softmax-style kernel with a tunable non-reduction block, retain a small parallel candidate. Include `4` when its indexing and output ownership are valid, then include larger candidates such as `8` and `16` when they fit the shape. When the backend supports software-pipeline stages, include both `num_stages=1` and `num_stages=3`; do not declare either best before measurement.

When Code Gen provides `optimization_surface`, seed the search from its candidates and then apply the rules above. The surface is not permission to skip the real NRAM compile check.

## Invariants

- Every config key names a real kernel `tl.constexpr` parameter.
- Heuristic-fixed parameters are not simultaneously tuned.
- Input/test shapes and tolerances do not change across candidates.
- The selected config fits platform constraints and has real evidence.
- A conservative estimate does not silently remove every small-block or pipelined candidate from an otherwise valid search space.
- The output decorator and launch call remain syntactically and semantically consistent.
