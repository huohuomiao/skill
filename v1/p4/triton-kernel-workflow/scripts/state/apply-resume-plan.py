#!/usr/bin/env python3
"""Restore p4 cache hits and reset rerun stages in run_manifest.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(VALIDATION_DIR))

from json_schema import SchemaValidationError, validate  # noqa: E402
from validation_common import load_json  # noqa: E402


CLEAR_CHECKPOINT_STAGES = {
    "environment",
    "requirement-extraction",
    "code-generation",
    "code-validation",
    "performance-tuning",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def _run(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def _selected_checkpoint(stage: dict, restored: dict) -> list[str]:
    if stage["stage"] != "performance-tuning":
        return []
    artifacts = restored["artifacts"]
    if "best" not in artifacts or "code" not in artifacts:
        raise ValueError("restored performance cache requires best and code artifacts")
    best = load_json(Path(artifacts["best"]))
    selected = best["selected_candidate"]
    return [
        "--selected-candidate-id",
        str(selected["candidate_id"]),
        "--selected-code-path",
        artifacts["code"],
        "--selected-latency-ms",
        str(selected["latency_ms"]),
    ]


def main() -> int:
    args = _arguments()
    try:
        plan_path = args.plan.resolve()
        plan = load_json(plan_path)
        skill_root = Path(__file__).resolve().parents[2]
        plan_schema = skill_root / "references/schemas/resume-plan.schema.json"
        validate(plan, load_json(plan_schema), plan_schema)
        output_dir = Path(plan["output_dir"]).resolve()
        cache_root = Path(plan["cache_root"]).resolve()
        expected_cache_root = output_dir / "RunState" / "cache"
        if cache_root != expected_cache_root:
            raise ValueError("resume plan cache root is outside the canonical output location")
        cache_root.mkdir(parents=True, exist_ok=True)
        manifest = args.manifest.resolve() if args.manifest else output_dir / "run_manifest.json"
        cache_script = skill_root / "scripts/state/stage-cache.py"
        updater = skill_root / "scripts/state/update-run-manifest.py"
        results: list[dict] = []

        for stage in plan["stages"]:
            common = [
                sys.executable,
                str(updater),
                "--manifest",
                str(manifest),
                "--mode",
                plan["mode"],
                "--optimization-mode",
                plan["optimization_mode"],
                "--stage",
                stage["stage"],
                "--resume-plan",
                str(plan_path),
                "--cache-root",
                str(cache_root),
            ]
            if stage["action"] == "reuse":
                restored_result = _run(
                    [
                        sys.executable,
                        str(cache_script),
                        "restore",
                        "--output-dir",
                        str(output_dir),
                        "--record",
                        stage["cache_record"],
                    ],
                    f"restore {stage['stage']}",
                )
                restored = json.loads(restored_result.stdout)
                if (
                    restored.get("stage") != stage["stage"]
                    or restored.get("fingerprint") != stage["fingerprint"]
                ):
                    raise ValueError(
                        f"restored cache identity does not match plan for {stage['stage']}"
                    )
                update = [
                    *common,
                    "--status",
                    "completed",
                    "--metadata",
                    "cache_status=restored",
                    "--metadata",
                    f"cache_fingerprint={stage['fingerprint']}",
                    "--metadata",
                    f"cache_record={stage['cache_record']}",
                ]
                for key, path in sorted(restored["artifacts"].items()):
                    update.extend(("--artifact", f"{key}={path}"))
                if stage["stage"] == "environment" and "config" in restored["artifacts"]:
                    environment = load_json(Path(restored["artifacts"]["config"]))
                    update.extend(("--execution-backend", environment["execution_backend"]))
                update.extend(_selected_checkpoint(stage, restored))
                _run(update, f"manifest reuse {stage['stage']}")
            elif stage["action"] == "skip":
                update = [
                    *common,
                    "--status",
                    "skipped",
                    "--metadata",
                    f"reason={stage['reason']}",
                ]
                if stage["stage"] == "performance-tuning":
                    update.append("--clear-selected-checkpoint")
                _run(update, f"manifest skip {stage['stage']}")
            else:
                update = [*common, "--status", "pending", "--clear-stage"]
                if stage["stage"] in CLEAR_CHECKPOINT_STAGES:
                    update.append("--clear-selected-checkpoint")
                _run(update, f"manifest reset {stage['stage']}")
            results.append(
                {"stage": stage["stage"], "action": stage["action"], "reason": stage["reason"]}
            )

        print(
            json.dumps(
                {
                    "status": "applied",
                    "manifest": str(manifest),
                    "resume_from": plan["resume_from"],
                    "stages": results,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        SchemaValidationError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"apply-resume-plan: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
