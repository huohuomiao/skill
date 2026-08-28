# Grid Layout Optimization

Use when logical Grid size exceeds or poorly matches target hardware. Read only the Grid and device-properties sections of `{skill_root}/references/backend/platform-rules.md`.

## Admission conditions

Apply only when at least one is true:

- A Grid dimension can exceed the target limit.
- Logical task count greatly exceeds usable physical parallelism.
- A multidimensional Grid can be flattened with exact index reconstruction.
- Grid reduction by block enlargement conflicts with on-chip memory limits and a persistent loop can preserve coverage.

Skip when the existing Grid is within constraints and changing it has no measurable scheduling benefit.

## Steps

1. Derive logical task counts from wrapper Grid expressions and kernel `program_id` use.
2. Read physical capacity and Grid constraints from `platform-rules.md`; do not copy or hard-code them here.
3. Choose one transformation:
   - cap a one-dimensional physical Grid and stride over logical tasks;
   - flatten multiple logical dimensions and reconstruct indices;
   - add missing persistent coverage after a capped Grid;
   - preserve a reduction-output ownership mapping while compressing only parallel axes.
4. Update wrapper Grid and kernel traversal together.
5. Prove that every logical task is visited exactly once, except intentional atomic aggregation.
6. Run shared accuracy/performance validation and keep only an accuracy-passing non-regression.

## Invariants

- No logical block is omitted or processed twice unintentionally.
- Flatten/unflatten formulas are inverse over the valid range.
- Masks still protect tail elements.
- Grid changes do not alter output ownership or reduction semantics.
- Hardware properties come from `platform-rules.md` or the installed driver, never a copied constant.

## Conditional references

Load one matching example only:

| Pattern | Reference |
| --- | --- |
| One-dimensional cap/persistent loop | `{skill_root}/references/examples/optimization/grid-layout-1d.md` |
| Three-dimensional flattening | `{skill_root}/references/examples/optimization/grid-layout-3d.md` |
| Compile-time Grid parameters | `{skill_root}/references/examples/optimization/grid-layout-constexpr.md` |
| Missing persistent coverage | `{skill_root}/references/examples/optimization/grid-layout-missing.md` |
| Reduction output Grid | `{skill_root}/references/examples/optimization/grid-layout-reduction.md` |

Do not load the whole example directory.
