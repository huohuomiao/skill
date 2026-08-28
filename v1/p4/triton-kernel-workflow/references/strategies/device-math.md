# Device Math Optimization

Replace a matching mathematical expression with a supported target device-math primitive. Read API and dtype support from `{skill_root}/references/backend/math-functions.md` and the dtype/device-math section of `{skill_root}/references/backend/platform-rules.md`.

## Admission conditions

- The expression matches a documented primitive pattern exactly.
- At least one operand is tensor-valued in kernel execution; do not replace host scalar math.
- Input/output dtype and accuracy requirements are supported.
- The replacement preserves exceptional-value behavior required by the user.

Prefer ordinary documented primitives. Consider lower-accuracy fast variants only when the accuracy contract explicitly permits them and real validation passes.

## Steps

1. Scan kernel expressions and identify an exact pattern from `math-functions.md`.
2. Confirm tensor provenance, dtype, broadcasting, and expression domain.
3. Replace only the matched expression; preserve surrounding indexing and masks.
4. Parse and compile through the shared execution contract.
5. Run the original accuracy suite, including boundary values relevant to the function.
6. Benchmark under the same signature and keep the replacement only if it is an accuracy-passing improvement.

## Invariants

- No undocumented Libdevice symbol is invented.
- Scalar host expressions remain host expressions.
- Fast approximations never silently weaken tolerance.
- A failed or unmeasured replacement is rolled back and produces no candidate.
