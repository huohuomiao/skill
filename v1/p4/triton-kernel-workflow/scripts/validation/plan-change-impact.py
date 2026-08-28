#!/usr/bin/env python3
"""Map changed Skill paths to deterministic p4 validation scope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from json_schema import SchemaValidationError, validate  # noqa: E402
from validation_common import load_json, skill_fingerprint, write_json  # noqa: E402


STAGE_ORDER = (
    "environment",
    "requirement-extraction",
    "code-generation",
    "code-validation",
    "performance-tuning",
    "finalization",
)
CASE_ORDER = ("elementwise", "reduction", "layout")
WORKER_ORDER = ("submission", "failure-recovery")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--changed", action="append", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _normalize(root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            path = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"changed path escapes skill root: {resolved}") from exc
    if any(part == ".." for part in path.parts):
        raise ValueError(f"invalid changed path: {value}")
    text = path.as_posix()
    if text.startswith("./"):
        text = text[2:]
    if not text:
        raise ValueError(f"invalid changed path: {value}")
    return text


def _impact(path: str) -> tuple[set[str], set[str], set[str], str]:
    if path == "references/templates/final-summary.md" or path == "references/workflows/finalization.md":
        return {"finalization"}, set(), set(), "presentation-only change"
    if path.startswith("references/templates/"):
        return {"performance-tuning", "finalization"}, set(), set(), "performance-report presentation change"
    if path == "references/strategies/reduction.md":
        return {"performance-tuning", "finalization"}, {"reduction"}, set(), "reduction optimization change"
    if path == "scripts/execution/submit-remote-task.py" or path == "references/contracts/execution-backend.md":
        return {"environment"}, set(), {"submission", "failure-recovery"}, "Worker/backend integration change"
    if path.startswith("scripts/environment/") or path == "references/roles/environment-checker.md":
        return {"environment"}, set(), {"submission"}, "environment probe change"
    if path == "references/backend/platform-rules.md":
        return set(STAGE_ORDER), set(CASE_ORDER), set(WORKER_ORDER), "platform rule change"
    if path.startswith("scripts/validation/") or path.startswith("references/evals/"):
        return set(), set(), set(), "offline validation implementation change"
    if path.startswith("scripts/state/") and any(
        name in path for name in ("fingerprint-stage.py", "stage-cache.py", "plan-resume.py", "apply-resume-plan.py")
    ):
        return set(), set(), set(), "cache/resume implementation change"
    return set(STAGE_ORDER), set(CASE_ORDER), set(WORKER_ORDER), "unclassified execution-affecting change"


def main() -> int:
    args = _arguments()
    try:
        root = args.skill_root.resolve()
        changed = sorted({_normalize(root, value) for value in args.changed})
        stages: set[str] = set()
        cases: set[str] = set()
        workers: set[str] = set()
        reasons: list[str] = []
        for path in changed:
            path_stages, path_cases, path_workers, reason = _impact(path)
            stages.update(path_stages)
            cases.update(path_cases)
            workers.update(path_workers)
            reasons.append(f"{path}: {reason}")
        hardware = bool(cases or workers)
        plan = {
            "schema_version": "1.0",
            "skill_fingerprint": skill_fingerprint(root),
            "changed_paths": changed,
            "required_levels": ["L1", "L2", *( ["L3"] if hardware else [] )],
            "affected_stages": [stage for stage in STAGE_ORDER if stage in stages],
            "l3_cases": [case for case in CASE_ORDER if case in cases],
            "worker_checks": [check for check in WORKER_ORDER if check in workers],
            "hardware_required": hardware,
            "reasons": reasons,
        }
        schema_path = root / "references/schemas/change-impact-plan.schema.json"
        validate(plan, load_json(schema_path), schema_path)
        if args.output:
            write_json(args.output.resolve(), plan)
        print(json.dumps(plan, ensure_ascii=False))
        return 0
    except (OSError, TypeError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"plan-change-impact: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
