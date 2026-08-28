# Reduction Optimization

Use this strategy only when static analysis finds a reduction. Read the Grid, NRAM, and backend-optimization sections of `{skill_root}/references/backend/platform-rules.md`; do not reload unrelated platform sections.

Apply the following candidates in order. A candidate that does not meet every admission condition is skipped. Keep each accepted rewrite only after the shared accuracy/performance validation phase.

## Candidate routing

| Order | Candidate | Admission condition |
| --- | --- | --- |
| 1 | Counting `while` to `for` | Start, bound, monotonic comparison, and fixed step are provable; the final loop-variable value is unused |
| 2 | Remove reduction-axis loop | The loop covers one reduction axis, the extent is known and within the supported vectorized reduction size |
| 3 | Lower 3D tile to 2D | A three-dimensional tile reduces its middle dimension and setting the first non-reduction block to one preserves indexing |
| 4 | Remove single-iteration loop | A heuristic fixes the loop block to its entire bound, proving one iteration |
| 5 | Eliminate redundant loads | Equivalent loads have identical pointer/mask/dtype and no intervening write or aliasing hazard |
| 6 | Full reduction to load/reduce/atomic | The result is a true scalar full reduction and the atomic dtype/identity are supported |
| 7 | Absorb layout transform | Wrapper layout movement can be represented exactly by kernel index/stride changes |

If no candidate is admitted, return the input unchanged and record every skip reason.

## Execution steps

### 1. Analyze before editing

Identify:

- Reduction axes, extents, identities, accumulation dtype, and output dtype
- Loop variables, bounds, steps, and whether their final values escape
- Load/store pointer expressions, masks, and aliasing relationships
- Wrapper layout operations and the exact logical output layout
- Grid and NRAM consequences of each possible rewrite

Do not use runtime guesses to fill missing shape or stride information.

### 2. Apply admitted rewrites

#### Counting loop conversion

Generate the same loop-variable value sequence, remove only the original induction update, and preserve the remaining body order. Reject control flow with data-dependent termination, `break`, incompatible step direction, or an externally used final induction value.

#### Reduction-loop removal

Replace a block loop with one vectorized offset range only when the supported size and NRAM bounds are satisfied. Preserve masks, identity values, accumulation dtype, and the exact reduced element set.

When the input carries an `optimization_surface` from Code Gen, treat the loop and pass structure as the pre-strategy baseline. Evaluate loop removal and load reuse here rather than assuming Code Gen already performed them. The surface records optimization opportunity only; keep a rewrite solely through the normal accuracy and comparable-performance gate.

#### 3D-to-2D lowering

Set the first non-reduction tile block to one and rebuild indices so the reduction becomes the highest active tile dimension. Reject if broadcast, transpose, output ownership, or pointer reconstruction changes.

#### Single-iteration and redundant-load removal

Inline the only loop iteration before considering load reuse. Reuse a load only with identical pointer, mask, `other`, dtype, and cache semantics and no possible intervening mutation.

#### Full reduction rewrite

Use one kernel with block load, local reduction, and atomic scalar update only when the identity and atomic operation are valid for the accumulation/output dtype. Initialize the destination exactly once outside the kernel launch and remove obsolete intermediate buffers and launches.

#### Layout absorption

Replace wrapper layout movement with equivalent kernel index/stride formulas. Remove the wrapper operation only after output shape, stride interpretation, and reference comparison remain identical.

### 3. Validate once after the candidate sequence

Run the shared accuracy and performance entrypoints through `execution-backend.md`. Compare against the pre-strategy input using identical input, tolerance, hardware, warmup, repeat count, and timing method.

If accuracy fails, restore the last accuracy-passing checkpoint. If performance regresses, retain the faster accuracy-passing checkpoint. Missing hardware evidence produces no performance claim or candidate manifest.

## Invariants

- Mathematical reduction set, identity, and output shape do not change.
- Tail masks remain correct for non-multiple extents.
- Accumulation and output dtype follow the original accuracy contract.
- Grid compression covers every logical output.
- No intermediate allocation or launch is removed while still referenced.
- A rewrite never introduces CPU/framework fallback.
- Only measured, accuracy-passing code may become a candidate.

## Conditional references

Load only the reference matching the admitted candidate:

| Candidate | Reference |
| --- | --- |
| Reduction-axis loop removal | `{skill_root}/references/examples/optimization/reduction-dimension-loop.md` |
| 3D-to-2D lowering | `{skill_root}/references/examples/optimization/reduction-3d-to-2d.md` |
| Reduction axis analysis | `{skill_root}/references/examples/optimization/kernel-axis-reducemax.md` |
| Softmax-style reduction mapping | `{skill_root}/references/examples/optimization/kernel-axis-softmax.md` |

Do not load all references by default.
