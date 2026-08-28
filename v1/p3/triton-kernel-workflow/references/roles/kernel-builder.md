# Kernel Builder

Generate the Triton implementation and its executable accuracy/performance tests inside one delegated task. The kernel and tests share one context, but remain separate compatibility artifacts.

## Inputs

Normal path:

- `{output_dir}/Extractor/requirement.md`
- `{output_dir}/KernelGen/step1_io_shapes.json`
- `{output_dir}/KernelGen/step4_code_spec.json`

Existing-code fast path:

- `{output_dir}/Extractor/original_code.py`

Shared rules, loaded only when needed:

- `{skill_root}/references/backend/supported-primitives.md`
- Relevant sections of `{skill_root}/references/backend/platform-rules.md`
- `optimization_intent` supplied by the Code Generation workflow

Do not read Step 2/3 reports, optimization reports, or unrelated strategy directories.

## Outputs

- `{output_dir}/KernelGen/step5_kernel_code.py`
- `{output_dir}/KernelGen/step6_test_code.py`

`step5_kernel_code.py` contains the kernel and wrapper. `step6_test_code.py` contains the exact Step 5 kernel/wrapper plus input creation, reference computation, accuracy validation, performance measurement, and an executable entrypoint.

## Build phases

### 1. Generate or preserve the kernel

For the normal path, implement `step4_code_spec.json` without changing its computation or shape contract. For the fast path, copy the existing kernel/wrapper unless a missing test harness requires additions.

Required invariants:

- Wrapper arguments and kernel parameters match exactly.
- Grid covers every logical output and respects the selected platform rule.
- Loads and stores use correct pointer formulas and tail masks.
- Reduction identities, accumulation dtype, and output dtype preserve the requirement.
- CPU or framework fallback is not a substitute for the Triton implementation.

For a reduction normal path, read `{skill_root}/references/roles/reduction-baseline.md` and implement the validated `optimization_surface` from Step 4. With `handoff-to-tuning`, do not silently replace required chunked reduction passes with a direct full-axis vectorized load, add heuristics that fix the reduction block to the full extent, or collapse softmax-style max/sum/normalize passes during Code Gen. Those transformations belong to the measured reduction strategy. With `standalone`, direct vectorization remains allowed when platform bounds are satisfied.

Write the result to `step5_kernel_code.py` and run Python syntax parsing. Do not claim compilation at this stage.

### 2. Add the executable test harness

Copy the Step 5 kernel and wrapper byte-for-byte into `step6_test_code.py`, then add:

- `create_inputs()` using the requested shapes and dtypes
- A framework reference implementation
- `accuracy_test()` using the user tolerance, or a conservative dtype-aware tolerance when none was supplied
- `performance_test()` using the same inputs and byte-counting convention for reference and Triton paths
- A main entrypoint that exposes real failure through a nonzero process exit

For multi-input/output, quantized, in-place, scale, index, or mask operators, count every actual tensor read/write separately. Do not assume output dtype equals input dtype.

After writing Step 6, verify that the Step 5 kernel/wrapper region did not change and parse the complete Python file.

## Conditional examples

Load at most one matching implementation example:

| Match | Code example |
| --- | --- |
| Axis fusion | `{skill_root}/references/examples/code-generation/code-axis-fusion.md` |
| Matrix transpose | `{skill_root}/references/examples/code-generation/code-matrix-transpose.md` |
| Reduction | `{skill_root}/references/examples/code-generation/code-reduce-sum.md` |
| Softmax-style multi-pass reduction | `{skill_root}/references/examples/code-generation/code-softmax-baseline.md` |
| Transpose plus elementwise | `{skill_root}/references/examples/code-generation/code-transpose-elementwise.md` |

If no row matches, do not load an example.

## Failure scope

- Invalid Step 4 or I/O metadata: stop and return the exact design artifact error; do not guess.
- Kernel generation or syntax failure: repair Step 5, then regenerate Step 6.
- Test-only failure: keep Step 5 unchanged and repair only the harness portion of Step 6.
- Infrastructure or execution-backend failure belongs to later validation and must not trigger a kernel rewrite here.
