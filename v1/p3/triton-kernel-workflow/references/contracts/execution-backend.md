# Execution Backend Contract

Apply this contract before any real compilation, device execution, accuracy test, or performance benchmark. Static text analysis and Python syntax checks may run locally without a device.

`{output_dir}/EnvConfig/config.json` is the only machine-readable source of execution state. `config.md` is derived for humans and must never be parsed by downstream automation.

## Environment selection

Run the local probes serially:

```bash
python {skill_root}/scripts/environment/inspect-device.py
python {skill_root}/scripts/environment/verify-runtime.py
```

- If both commands return exit code `0`, generate `{output_dir}/EnvConfig/config.json` with `execution_backend=local` through `{skill_root}/scripts/state/write-env-config.py`, then run later dynamic commands locally.
- If either command fails, use the current `JOB_ID` to run the same probes through the remote execution script. Do not create another Job.
- Record `execution_backend=worker` only when both remote probes succeed.
- If neither backend passes the probes, stop the workflow and report the real stdout, stderr, and exit codes. Do not infer device results.

Validate the generated object against `{skill_root}/references/schemas/env-config.schema.json` before entering any downstream dynamic stage.

Remote environment check:

```bash
python {skill_root}/scripts/execution/submit-remote-task.py \
    --task-type custom \
    --workdir <repository-root-absolute-path> \
    --timeout-sec 600 \
    --command "python {skill_root}/scripts/environment/inspect-device.py && python {skill_root}/scripts/environment/verify-runtime.py"
```

## Dynamic execution

When `execution_backend=local`, normally run the required command directly in its working directory. During p3 performance tuning, run it through the wall-time guard:

```bash
python {skill_root}/scripts/execution/run-budgeted-local.py \
    --budget-state {output_dir}/Optimizer/tuning_state.json \
    --workdir <absolute-path> \
    --label <baseline-or-strategy-or-profile-label> \
    -- <executable> <arguments...>
```

The guard does not consume Worker-call budget. It stops a local command when the 1800-second tuning wall-time expires and returns `4`.

When `execution_backend=worker`, run:

```bash
python {skill_root}/scripts/execution/submit-remote-task.py \
    --task-type {accuracy|performance|custom} \
    --workdir <absolute-path> \
    --timeout-sec <seconds> \
    --command "<command>"
```

During p3 performance tuning, append:

```bash
    --budget-state {output_dir}/Optimizer/tuning_state.json \
    --budget-label <baseline-or-strategy-or-profile-label>
```

The submission script reserves one Worker call and clamps its timeout to the remaining tuning wall-time before posting. Do not reserve Worker calls separately. Both local and Worker hardware operations remain serial.

Every remote task must run synchronously in the foreground. Wait for the script to exit before starting another stage or task. Do not use background execution or submit tasks concurrently.

Interpret the remote script result as follows:

- `0`: task succeeded.
- `1`: the executed workload failed; use its logs for debugging.
- `2`: infrastructure or submission failure; do not modify kernel code in response to an infrastructure failure.
- `3`: Worker task canceled.
- `4`: tuning budget exhausted before submission; stop scheduling tuning work and finalize from completed candidates.

Use the printed `task_output_dir` and its `stdout.log`, `stderr.log`, and `result.json` as the evidence source. Do not bypass the submission script with handwritten HTTP requests, and do not use a worker task to install or modify shared dependencies or device toolchains.
