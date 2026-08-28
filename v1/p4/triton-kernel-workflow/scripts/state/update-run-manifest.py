#!/usr/bin/env python3
"""Create or atomically update the compact workflow run manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(VALIDATION_DIR))

from json_schema import SchemaValidationError, validate  # noqa: E402
from validation_common import load_json, utc_now, write_json  # noqa: E402


STAGES = (
    "environment",
    "requirement-extraction",
    "code-generation",
    "code-validation",
    "performance-tuning",
    "finalization",
)

MODE_TERMINAL_STAGES = {
    "full": ("environment", "requirement-extraction", "code-generation", "code-validation", "performance-tuning", "finalization"),
    "code-generation": ("code-generation", "code-validation"),
    "code-validation": ("code-validation",),
    "performance-tuning": ("performance-tuning",),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "code-generation", "code-validation", "performance-tuning"))
    parser.add_argument(
        "--optimization-mode",
        choices=("correctness", "balanced", "max-performance"),
    )
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--status", choices=("pending", "running", "completed", "failed", "skipped"), required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--metadata", action="append", default=[])
    parser.add_argument("--error")
    parser.add_argument("--execution-backend", choices=("local", "worker"))
    parser.add_argument("--resume-plan", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--clear-stage", action="store_true")
    parser.add_argument("--clear-selected-checkpoint", action="store_true")
    parser.add_argument("--selected-candidate-id")
    parser.add_argument("--selected-code-path", type=Path)
    parser.add_argument("--selected-latency-ms", type=float)
    return parser.parse_args()


def _pairs(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires key=value: {value}")
        key, item = value.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"invalid or duplicate {option}: {value}")
        result[key] = item
    return result


def _new_manifest(mode: str, optimization_mode: str) -> dict:
    return {
        "schema_version": "1.0",
        "mode": mode,
        "optimization_mode": optimization_mode,
        "workflow_status": "running",
        "current_stage": None,
        "execution_backend": None,
        "resume": {"plan_path": None, "cache_root": None, "last_applied_at": None},
        "updated_at": utc_now(),
        "stages": {
            stage: {"status": "pending", "artifacts": {}, "error": None, "metadata": {}}
            for stage in STAGES
        },
        "selected_checkpoint": None,
    }


def _artifact_paths(values: dict[str, str], require_existing: bool) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for key, value in values.items():
        path = Path(value).resolve()
        if require_existing and (not path.is_file() or path.stat().st_size == 0):
            raise ValueError(f"completed artifact is missing or empty: {path}")
        artifacts[key] = str(path)
    return artifacts


def _selected_checkpoint(args: argparse.Namespace) -> dict | None:
    values = (args.selected_candidate_id, args.selected_code_path, args.selected_latency_ms)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError("selected checkpoint requires id, code path, and latency")
    code_path = args.selected_code_path.resolve()
    if not code_path.is_file() or code_path.stat().st_size == 0:
        raise ValueError(f"selected code is missing or empty: {code_path}")
    latency = float(args.selected_latency_ms)
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError("selected latency must be finite and positive")
    return {
        "candidate_id": args.selected_candidate_id,
        "code_path": str(code_path),
        "latency_ms": latency,
    }


def main() -> int:
    args = _arguments()
    manifest_path = args.manifest.resolve()
    try:
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            manifest.setdefault(
                "resume", {"plan_path": None, "cache_root": None, "last_applied_at": None}
            )
            if args.mode is not None and args.mode != manifest["mode"]:
                raise ValueError("cannot change manifest mode")
            if (
                args.optimization_mode is not None
                and args.optimization_mode != manifest["optimization_mode"]
            ):
                raise ValueError("cannot change manifest optimization mode")
        else:
            manifest = _new_manifest(
                args.mode or "full", args.optimization_mode or "balanced"
            )

        artifacts = _artifact_paths(
            _pairs(args.artifact, "--artifact"), args.status == "completed"
        )
        metadata = _pairs(args.metadata, "--metadata")
        if args.clear_stage:
            if args.status != "pending" or artifacts or metadata or args.error:
                raise ValueError("--clear-stage requires an otherwise empty pending update")
            manifest["stages"][args.stage] = {
                "status": "pending",
                "artifacts": {},
                "error": None,
                "metadata": {},
            }
        if args.status == "completed" and not (artifacts or manifest["stages"][args.stage]["artifacts"]):
            raise ValueError("a completed stage requires at least one verified artifact")
        if args.status == "failed" and not args.error:
            raise ValueError("a failed stage requires --error")

        stage = manifest["stages"][args.stage]
        stage["status"] = args.status
        stage["artifacts"].update(artifacts)
        stage["metadata"].update(metadata)
        stage["error"] = args.error if args.status == "failed" else None
        manifest["current_stage"] = args.stage
        if args.execution_backend is not None:
            manifest["execution_backend"] = args.execution_backend
        if args.resume_plan is not None:
            resume_plan = args.resume_plan.resolve()
            if not resume_plan.is_file() or resume_plan.stat().st_size == 0:
                raise ValueError(f"resume plan is missing or empty: {resume_plan}")
            manifest["resume"]["plan_path"] = str(resume_plan)
            manifest["resume"]["last_applied_at"] = utc_now()
        if args.cache_root is not None:
            cache_root = args.cache_root.resolve()
            if not cache_root.is_dir():
                raise ValueError(f"cache root does not exist: {cache_root}")
            manifest["resume"]["cache_root"] = str(cache_root)
        if args.clear_selected_checkpoint:
            manifest["selected_checkpoint"] = None
        selected = _selected_checkpoint(args)
        if selected is not None:
            manifest["selected_checkpoint"] = selected
        if any(item["status"] == "failed" for item in manifest["stages"].values()):
            manifest["workflow_status"] = "failed"
        elif all(
            manifest["stages"][stage_name]["status"] in {"completed", "skipped"}
            for stage_name in MODE_TERMINAL_STAGES[manifest["mode"]]
        ):
            manifest["workflow_status"] = "completed"
        else:
            manifest["workflow_status"] = "running"
        manifest["updated_at"] = utc_now()

        schema_path = (
            Path(__file__).resolve().parents[2]
            / "references"
            / "schemas"
            / "run-manifest.schema.json"
        )
        schema = load_json(schema_path)
        validate(manifest, schema, schema_path)
        write_json(manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    except (KeyError, OSError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"update-run-manifest: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
