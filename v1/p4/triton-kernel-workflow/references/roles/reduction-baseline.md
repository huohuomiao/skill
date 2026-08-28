# Reduction Baseline Contract

Use this reference only for a normal-path operator with a non-empty reduction axis. It preserves the performance-tuning surface that existed before the Code Gen roles were merged, without requiring a deliberately weak baseline for standalone generation.

## Intent routing

- `handoff-to-tuning`: Code Gen produces a correct, measurable chunked baseline. Reduction-loop removal, cross-pass load reuse, full-axis heuristics, and autotune selection remain owned by later strategies.
- `standalone`: Code Gen may use either chunked or direct vectorized reduction when the result is resource-safe and correct. It must not add avoidable work merely to make an optimization strategy applicable.
- Existing-code fast path: preserve the supplied kernel structure. Do not reconstruct it into either baseline form.

## Design contract for tuning handoff

For `handoff-to-tuning`, `step4_code_spec.json` must keep each reduction axis inside the owning program as `reduce_loop` or ordered `reduce_loop_passN` fields. Every multi-pass field records its semantic `operation`. The initial reduction block must remain smaller than a known nontrivial extent when a smaller supported tile exists; do not add a heuristic that fixes it to the complete extent during Code Gen.

Add an `optimization_surface` object conforming to `references/schemas/optimization-surface.schema.json`. It records:

- the reduction axis, extent, block parameter, and initial candidates;
- the ordered pass names and whether the pattern is a single or multi-pass reduction;
- the non-reduction parallel block and candidate set;
- proposed `num_stages` and `num_warps` candidates;
- `nram_model=lifetime-aware` so later tuning does not sum non-overlapping tensors or assume an unproven pipeline multiplier.

This object is a handoff contract, not benchmark evidence. It does not declare any candidate fastest.

## Softmax-style reduction

For `max → exp/sum → normalize/store` semantics, use `operator_pattern=softmax-style` and the ordered passes:

1. `max`
2. `sum`
3. `normalize-store`

Each pass walks the reduction axis in chunks owned by the same program. Reloading a chunk in a later semantic pass is valid for the correctness baseline. Do not force cross-pass value retention when its lifetime would enlarge NRAM; the reduction strategy may remove a provably redundant load after loop elimination or another measured rewrite.

For a row-oriented target-platform softmax, keep a small non-reduction candidate in the surface. Include parallel block `4` when its index semantics are valid, alongside larger candidates such as `8` and `16`. Include `num_stages` candidates `1` and `3` for an I/O-bound pattern when the backend supports those values. These are candidates to compile and measure, not fixed launch values.

Read `{skill_root}/references/examples/code-generation/code-softmax-baseline.md` only for this pattern.

## Invariants

- One program owns every complete output row or output reduction result; no cross-program partial reduction is introduced without an explicit merge contract.
- Masks, identities, accumulation dtype, output dtype, and mathematical pass order are preserved.
- Code Gen does not claim that a retained optimization surface improves performance.
- Reduction and autotune strategies keep only accuracy-passing candidates with real comparable measurements.
- A candidate rejected by a real compile-time NRAM error is removed; an uncertain static estimate alone does not become a performance conclusion.
