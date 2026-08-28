# Tuning Mode, Routing, and Budget Contract

Read this contract when entering a full workflow or focused performance-tuning workflow. It owns optimization-mode behavior, static strategy admission, and the global tuning budget. Backend invocation details remain owned by `execution-backend.md`; platform constraints remain owned by `platform-rules.md`.

## Optimization modes

| Mode | Behavior |
| --- | --- |
| `correctness` | Complete generation, compilation, and accuracy validation. Skip performance tuning and make no performance claim. |
| `balanced` | Measure a baseline, execute only statically applicable OOB strategies, select the best measured accuracy-passing candidate, and skip deep optimization. This is the default. |
| `max-performance` | Run the balanced path, then execute bounded deep-optimization rounds before final selection. |

The caller selects one mode before environment preparation. Record it as immutable `optimization_mode` in `run_manifest.json`. Do not infer `max-performance` merely because the user asks for a fast kernel.

## Static routing

Before any performance strategy is delegated, generate `{output_dir}/Optimizer/strategy_plan.json` with `scripts/state/plan-strategies.py`. The router performs static analysis only and never compiles or executes the kernel.

Required routing decisions:

- No reduction operation: skip `reduce-opt`.
- Grid already has bounded persistent coverage: skip `modify-grid`.
- No floor-division or modulo indexing in the kernel: skip `index-computation-simplify`.
- No tunable `tl.constexpr` parameter: skip `gen-autotune-config` and `config-tuner`.
- No supported device-math call pattern: skip `libdevice-opt`.
- No tensor division expression in the Triton kernel: skip `div-to-mul`.
- No block/tile surface: skip `retiling`.

`balanced` disables all advanced strategies even when a pattern exists. `correctness` disables every performance strategy. `max-performance` may admit both OOB and advanced strategies.

Static analysis of independent candidates may run in parallel, but real compilation, accuracy execution, profiling, and benchmark commands remain serial.

## Global budget

Initialize `{output_dir}/Optimizer/tuning_state.json` through `scripts/state/manage-tuning-budget.py`. The limits are fixed for p3:

- Deep-optimization rounds: at most `3`.
- Worker submissions during performance tuning: at most `16`.
- Total performance-tuning wall time: at most `1800` seconds (`30` minutes).

The budget begins before baseline measurement. Check it before each strategy delegation and before each deep round. Every performance-tuning Worker submission must pass `--budget-state {output_dir}/Optimizer/tuning_state.json` to `submit-remote-task.py`; that script reserves the Worker call before submission.

When any limit is reached:

1. Stop scheduling new strategies and Worker tasks.
2. Preserve completed reports and candidate manifests.
3. Run the existing deterministic candidate selector over the candidates already measured.
4. Record the stop reason in `tuning_state.json`, the tuning report, and the performance-tuning manifest metadata.

Budget exhaustion is a normal bounded stop, not an accuracy success, performance claim, or infrastructure failure.

## Unchanged selection behavior

p3 does not change measurement statistics or selection scoring. Candidate eligibility and final selection continue to use `optimization-candidate.schema.json` and `select-best-candidate.py` exactly as in p2.1.

