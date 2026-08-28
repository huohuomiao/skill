# Kernel Designer

Convert one operator requirement into the four compatible design artifacts used by code generation. Perform the former base extraction, block mapping, axis planning, and code specification inside one delegated task so the same semantic context is not reloaded four times.

## Inputs

- `{output_dir}/Extractor/requirement.md`
- Optional `{output_dir}/Extractor/original_code.py` for the existing-code fast path
- `optimization_intent`: `handoff-to-tuning` or `standalone`
- `{skill_root}/references/backend/supported-primitives.md`
- Only the relevant sections of `{skill_root}/references/backend/platform-rules.md` when targeting that platform

Do not read environment logs or optimization reports. This is a static design task and does not execute the kernel.

## Outputs

Write all normal-path artifacts under `{output_dir}/KernelGen/`:

1. `step1_base_info.json`
2. `step1_io_shapes.json`
3. `step2_block_mapping.json`
4. `step3_axis_fusion.json`
5. `step4_code_spec.json`

Intermediate filenames are stable compatibility contracts. Write each file as soon as its internal phase is complete so failures remain diagnosable.

## Internal design phases

### 1. Extract computation semantics

Create `step1_base_info.json` with at least:

- `op_name`
- `compute_type`
- `compute_formula`
- `compute_note`
- `io_shapes`, including tensor axis names, shapes, dtype, and contiguity when known
- `reduce_axes`

Write the exact `io_shapes` object separately to `step1_io_shapes.json`. The two copies must be structurally identical; never edit them independently.

Reject ambiguous shape or dtype assumptions that would change the request. Record unknown information explicitly instead of inventing a convenient test size.

### 2. Trace block mapping

Create `step2_block_mapping.json`. Preserve `compute_formula` and `compute_note`, then describe how every input/output logical axis maps to program IDs, block offsets, masks, and reduction axes.

Invariants:

- Every output element has one defined ownership path unless an intentional atomic reduction is specified.
- Broadcast, transpose, and reduction axes retain their mathematical meaning.
- Tail elements have an explicit mask strategy.

### 3. Plan axis fusion

Create `step3_axis_fusion.json`. Preserve prior semantic fields and add `fusion_note` plus the resulting `io_block_mapping`.

Fuse axes only when shape, stride, index reconstruction, and output layout remain equivalent. If fusion is unsafe or useless, record `fusion_note` as skipped and carry the mapping forward unchanged.

### 4. Build the code specification

Create `step4_code_spec.json` containing a kernel specification and wrapper specification. Include:

- Kernel parameters and `tl.constexpr` block parameters
- Grid formula and logical-task coverage
- Auxiliary index expressions
- Load/store pointer formulas and masks
- Reduction or atomic behavior
- Wrapper allocation, dtype, shape, launch arguments, and return values
- Required accuracy and performance test entrypoints

For a target platform, load only the applicable sections of `platform-rules.md`: device adaptation, Grid, NRAM, or dtype/primitives. Do not copy those platform rules into the JSON; record the chosen design consequence.

When `reduce_axes` is non-empty, read `{skill_root}/references/roles/reduction-baseline.md`. Under `handoff-to-tuning`, encode its required `optimization_surface` in `step4_code_spec.json` and keep the reduction-axis chunking visible to the later reduction and autotune strategies. Under `standalone`, choose the simplest resource-safe correct form and do not manufacture redundant work merely to trigger an optimizer.

## Conditional examples

Load at most the matching example rather than the entire example directory:

| Match | Design example |
| --- | --- |
| Axis fusion | `{skill_root}/references/examples/code-generation/spec-axis-fusion.md` |
| Matrix transpose | `{skill_root}/references/examples/code-generation/spec-matrix-transpose.md` |
| Reduction | `{skill_root}/references/examples/code-generation/spec-reduce-sum.md` |
| Softmax-style multi-pass reduction | `{skill_root}/references/examples/code-generation/code-softmax-baseline.md` |
| Transpose plus elementwise | `{skill_root}/references/examples/code-generation/spec-transpose-elementwise.md` |

If no row matches, do not load an example.

## Validation and failure scope

Before returning:

1. Parse all five JSON files.
2. Verify `step1_io_shapes.json == step1_base_info.json["io_shapes"]`.
3. Verify formula and shape semantics are preserved through Steps 2–4.
4. Verify every required code-spec field is present.

On failure, repair only the earliest invalid internal phase and regenerate its downstream design files. Do not restart requirement extraction or modify `Extractor/` artifacts.
