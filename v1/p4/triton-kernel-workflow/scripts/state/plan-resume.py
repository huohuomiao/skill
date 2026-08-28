#!/usr/bin/env python3
"""Plan safe p4 cache reuse and forward-only stage invalidation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(VALIDATION_DIR))

from json_schema import SchemaValidationError, validate  # noqa: E402
from validation_common import load_json, utc_now, write_json  # noqa: E402


FULL_STAGES = (
    "environment",
    "requirement-extraction",
    "code-generation",
    "code-validation",
    "performance-tuning",
    "finalization",
)
MODE_STAGES = {
    "full": FULL_STAGES,
    "code-generation": ("code-generation", "code-validation"),
    "code-validation": ("code-validation",),
    "performance-tuning": ("performance-tuning",),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE_STAGES), default="full")
    parser.add_argument("--optimization-mode", choices=("correctness", "balanced", "max-performance"), default="balanced")
    parser.add_argument("--fingerprint", action="append", default=[], help="stage=fingerprint-json")
    parser.add_argument("--force-stage", action="append", default=[], choices=FULL_STAGES)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _fingerprints(values: list[str], schema_path: Path) -> dict[str, tuple[Path, dict]]:
    result: dict[str, tuple[Path, dict]] = {}
    schema = load_json(schema_path)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--fingerprint requires stage=path: {value}")
        stage, raw_path = value.split("=", 1)
        if stage in result:
            raise ValueError(f"duplicate stage fingerprint: {stage}")
        path = Path(raw_path).resolve()
        payload = load_json(path)
        validate(payload, schema, schema_path)
        if payload["stage"] != stage:
            raise ValueError(f"fingerprint stage mismatch: {stage} != {payload['stage']}")
        recomputed = hashlib.sha256(
            json.dumps(
                {"stage": payload["stage"], "factors": payload["factors"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if recomputed != payload["fingerprint"]:
            raise ValueError(f"fingerprint digest mismatch for {stage}")
        if len(payload["fingerprint"]) != 64:
            raise ValueError(f"invalid SHA-256 fingerprint for {stage}")
        result[stage] = (path, payload)
    return result


def _cache_valid(
    skill_root: Path,
    output_dir: Path,
    record_path: Path,
    expected_stage: str,
    expected_fingerprint: str,
) -> bool:
    cache_script = skill_root / "scripts/state/stage-cache.py"
    result = subprocess.run(
        [
            sys.executable,
            str(cache_script),
            "verify",
            "--output-dir",
            str(output_dir),
            "--record",
            str(record_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return (
        payload.get("stage") == expected_stage
        and payload.get("fingerprint") == expected_fingerprint
    )


def main() -> int:
    args = _arguments()
    try:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        skill_root = Path(__file__).resolve().parents[2]
        fingerprint_schema = skill_root / "references/schemas/stage-fingerprint.schema.json"
        fingerprints = _fingerprints(args.fingerprint, fingerprint_schema)
        active = MODE_STAGES[args.mode]
        skipped = (
            {"performance-tuning"}
            if args.mode == "full" and args.optimization_mode == "correctness"
            else set()
        )
        forced = set(args.force_stage)
        invalid_forces = sorted((forced - set(active)) | (forced & skipped))
        if invalid_forces:
            raise ValueError(f"forced stages are not executable in this mode: {invalid_forces}")
        unexpected = sorted(set(fingerprints) - set(active))
        if unexpected:
            raise ValueError(f"fingerprints are outside selected mode: {unexpected}")

        cache_root = output_dir / "RunState" / "cache"
        stages: list[dict] = []
        downstream_invalid = False
        resume_from = None
        for stage in active:
            if stage in skipped:
                stages.append(
                    {
                        "stage": stage,
                        "action": "skip",
                        "reason": "disabled-by-correctness-mode",
                        "fingerprint": None,
                        "fingerprint_path": None,
                        "cache_record": None,
                    }
                )
                continue

            fingerprint_entry = fingerprints.get(stage)
            fingerprint_path = fingerprint_entry[0] if fingerprint_entry else None
            fingerprint = fingerprint_entry[1]["fingerprint"] if fingerprint_entry else None
            record_path = None
            if downstream_invalid:
                action, reason = "rerun", "upstream-invalidated"
            elif stage in forced:
                action, reason = "rerun", "explicitly-forced"
            else:
                if fingerprint_entry is None:
                    raise ValueError(
                        f"missing desired fingerprint before first invalidated stage: {stage}"
                    )
                record_path = cache_root / stage / fingerprint / "record.json"
                if not record_path.is_file():
                    older_records = (
                        list((cache_root / stage).glob("*/record.json"))
                        if (cache_root / stage).is_dir()
                        else []
                    )
                    action = "rerun"
                    reason = "fingerprint-changed" if older_records else "cache-record-missing"
                elif not _cache_valid(
                    skill_root, output_dir, record_path, stage, fingerprint
                ):
                    action, reason = "rerun", "cache-record-invalid"
                else:
                    action, reason = "reuse", "verified-cache-hit"

            if action == "rerun":
                downstream_invalid = True
                if resume_from is None:
                    resume_from = stage
            stages.append(
                {
                    "stage": stage,
                    "action": action,
                    "reason": reason,
                    "fingerprint": fingerprint,
                    "fingerprint_path": str(fingerprint_path) if fingerprint_path else None,
                    "cache_record": str(record_path) if action == "reuse" else None,
                }
            )

        plan = {
            "schema_version": "1.0",
            "mode": args.mode,
            "optimization_mode": args.optimization_mode,
            "generated_at": utc_now(),
            "output_dir": str(output_dir),
            "cache_root": str(cache_root),
            "resume_from": resume_from,
            "stages": stages,
        }
        schema_path = skill_root / "references/schemas/resume-plan.schema.json"
        validate(plan, load_json(schema_path), schema_path)
        write_json(args.output.resolve(), plan)
        print(json.dumps(plan, ensure_ascii=False))
        return 0
    except (KeyError, OSError, TypeError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"plan-resume: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
