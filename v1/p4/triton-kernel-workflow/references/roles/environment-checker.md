# Environment Checker

Confirm one usable execution backend and create the canonical environment artifacts. All backend selection, Worker invocation, timeout, and exit-code behavior comes exclusively from `{skill_root}/references/contracts/execution-backend.md`.

## Inputs

- `output_dir`
- `skill_root`
- `mode` supplied by the caller
- `optimization_mode` supplied by the caller; default `balanced`
- Current `JOB_ID` when available

## Outputs

- `{output_dir}/EnvConfig/runtime_info.txt`
- `{output_dir}/EnvConfig/config.json` — the only machine-readable environment source
- `{output_dir}/EnvConfig/config.md` — a derived human report
- Updated `{output_dir}/run_manifest.json`

## Procedure

1. Read `execution-backend.md` and run its environment probes in the documented order. Do not reproduce or reinterpret its backend decision table here.
2. Stop when the contract reports that neither execution backend is usable. Preserve real stdout, stderr, exit codes, and Worker result paths.
3. Combine the successful backend's device/runtime probe output into non-empty `EnvConfig/runtime_info.txt`.
4. Extract device model and installed Triton, Torch, and toolchain versions from that real output.
5. Generate both configuration files through the deterministic writer:

   ```bash
   python {skill_root}/scripts/state/write-env-config.py \
     --output-dir {output_dir} \
     --backend <backend selected by execution-backend.md> \
     --env-check-task-id <local or real Worker task id> \
     --runtime-info-path {output_dir}/EnvConfig/runtime_info.txt \
     --device-model <measured model> \
     --triton-version <measured version> \
     --torch-version <measured version> \
     --toolchain-version <measured version>
   ```

   Supply `--worker-submit-url` only when the selected backend requires it.

6. Validate `config.json` against `{skill_root}/references/schemas/env-config.schema.json`. Downstream automation must never parse `config.md`.
7. Record the verified artifacts and backend in the manifest:

   ```bash
   python {skill_root}/scripts/state/update-run-manifest.py \
     --manifest {output_dir}/run_manifest.json \
     --mode <mode supplied by the caller> \
     --optimization-mode <optimization mode supplied by the caller> \
     --stage environment \
     --status completed \
     --execution-backend <local-or-worker> \
     --artifact config={output_dir}/EnvConfig/config.json \
     --artifact runtime_info={output_dir}/EnvConfig/runtime_info.txt \
     --artifact report={output_dir}/EnvConfig/config.md
   ```

## Invariants

- No device, compilation, accuracy, or performance claim is inferred from missing probe output.
- Environment/infrastructure failure never triggers a kernel change.
- `config.json` and `config.md` are written together by the shared writer.
- The returned summary contains `execution_backend`, `env_check_task_id`, `runtime_info_path`, `config_path`, and `report_path`.
