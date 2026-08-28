#!/usr/bin/env python3
"""Run fixed offline behavior scenarios and mock-artifact validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from json_schema import SchemaValidationError, validate, validate_file
from validation_common import (
    finish,
    load_json,
    make_check,
    make_report,
    skill_fingerprint,
    utc_now,
)


ROUTES = {
    "auto": "full-pipeline",
    "full": "full-pipeline",
    "code-generation": "code-generation",
    "code-validation": "code-validation",
    "performance-tuning": "performance-tuning",
}
VALID_INPUTS = {
    "auto": {"requirement", "existing-code", "runnable-kernel"},
    "full": {"requirement", "existing-code", "runnable-kernel"},
    "code-generation": {"requirement", "existing-code"},
    "code-validation": {"existing-code", "runnable-kernel"},
    "performance-tuning": {"runnable-kernel"},
}
MODE_STAGES = {
    "code-generation": ["code-generation", "code-validation"],
    "code-validation": ["code-validation"],
    "performance-tuning": ["performance-tuning"],
}
ROLE_READ_WHITELISTS = {
    "performance-optimizer": {"current-code", "strategy", "strategy-reference", "env-config", "tuning-state"}
}
FULL_STAGES = [
    "environment",
    "requirement-extraction",
    "code-generation",
    "code-validation",
    "performance-tuning",
    "finalization",
]
ROLLBACK_TARGETS = {
    "codegen-step2": "codegen-step2",
    "codegen-step3": "codegen-step3",
    "codegen-step4": "codegen-step4",
    "codegen-step5": "codegen-step5",
    "codegen-step6": "codegen-step6",
    "codegen-final-normal": "codegen-step1",
    "codegen-final-fast": "codegen-step0",
    "infrastructure": "environment",
    "performance-regression": "best-so-far",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _evaluate(category: str, inputs: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if category == "routing":
        mode = inputs.get("mode")
        if mode not in ROUTES:
            raise ValueError(f"unsupported mode: {mode}")
        return {
            "workflow": ROUTES[mode],
            "accepted": inputs.get("input_kind") in VALID_INPUTS[mode],
        }
    if category == "stage-gate":
        mode = inputs.get("mode")
        if mode != "full":
            stages = MODE_STAGES.get(mode)
            if stages is None:
                raise ValueError(f"unsupported mode: {mode}")
            return {"execute": stages, "skip": [item for item in FULL_STAGES if item not in stages]}
        if inputs.get("backend_ready") is True:
            return {"execute": FULL_STAGES, "skip": []}
        return {"execute": ["environment"], "skip": FULL_STAGES[1:]}
    if category == "file-access":
        role = inputs.get("role")
        if role not in ROLE_READ_WHITELISTS:
            raise ValueError(f"unsupported role: {role}")
        observed = set(inputs.get("observed", []))
        allowed = ROLE_READ_WHITELISTS[role]
        return {"allowed": observed <= allowed}
    if category == "schema":
        artifact = (base_dir / inputs["artifact"]).resolve()
        schema = (base_dir / inputs["schema"]).resolve()
        try:
            validate_file(artifact, schema)
        except (SchemaValidationError, OSError, json.JSONDecodeError):
            return {"valid": False}
        return {"valid": True}
    if category == "rollback":
        failure = inputs.get("failure")
        if failure not in ROLLBACK_TARGETS:
            raise ValueError(f"unsupported rollback failure: {failure}")
        return {"target": ROLLBACK_TARGETS[failure]}
    if category == "evidence":
        may_claim = (
            inputs.get("backend_ready") is True
            and inputs.get("measured") is True
            and inputs.get("accuracy_pass") is True
        )
        return {"may_claim_performance": may_claim}
    if category == "dispatch":
        route = inputs.get("route")
        if route == "normal":
            return {"outer_dispatches": 2, "roles": ["kernel-designer", "kernel-builder"]}
        if route == "fast":
            return {"outer_dispatches": 1, "roles": ["kernel-builder"]}
        raise ValueError(f"unsupported Code Gen route: {route}")
    if category == "generation-policy":
        intent = inputs.get("optimization_intent")
        pattern = inputs.get("operator_pattern")
        if intent == "handoff-to-tuning" and pattern == "softmax-style":
            return {
                "baseline_form": "chunked-reduction-loop",
                "passes": ["max", "sum", "normalize-store"],
                "required_parallel_candidates": [4],
                "required_stage_candidates": [1, 3],
            }
        if intent == "standalone":
            return {
                "baseline_form": "resource-safe-choice",
                "passes": [],
                "required_parallel_candidates": [],
                "required_stage_candidates": [],
            }
        raise ValueError(f"unsupported generation policy: {intent}/{pattern}")
    if category == "optimization-mode":
        mode = inputs.get("optimization_mode")
        behavior = {
            "correctness": {
                "performance_stage": "skipped",
                "allows_oob": False,
                "allows_deep": False,
                "allows_performance_claim": False,
            },
            "balanced": {
                "performance_stage": "completed",
                "allows_oob": True,
                "allows_deep": False,
                "allows_performance_claim": True,
            },
            "max-performance": {
                "performance_stage": "completed",
                "allows_oob": True,
                "allows_deep": True,
                "allows_performance_claim": True,
            },
        }
        if mode not in behavior:
            raise ValueError(f"unsupported optimization mode: {mode}")
        return behavior[mode]
    if category == "resume-policy":
        stages = inputs.get("stages", [])
        reusable = set(inputs.get("reusable", []))
        skipped = set(inputs.get("skipped", []))
        actions: dict[str, str] = {}
        downstream_invalid = False
        for stage in stages:
            if stage in skipped:
                actions[stage] = "skip"
            elif downstream_invalid:
                actions[stage] = "rerun"
            elif stage in reusable:
                actions[stage] = "reuse"
            else:
                actions[stage] = "rerun"
                downstream_invalid = True
        return {"actions": actions}
    raise ValueError(f"unsupported scenario category: {category}")


def _scenario_checks(root: Path, scenario_path: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    scenario_schema = root / "references/schemas/offline-scenarios.schema.json"
    try:
        validate_file(scenario_path, scenario_schema)
    except (SchemaValidationError, OSError, json.JSONDecodeError) as exc:
        return [make_check("offline-scenario-schema", "fail", error=str(exc))]
    checks.append(make_check("offline-scenario-schema", "pass", path=str(scenario_path)))
    suite = load_json(scenario_path)
    identifiers: list[str] = []
    for scenario in suite["scenarios"]:
        identifier = scenario["id"]
        identifiers.append(identifier)
        try:
            actual = _evaluate(scenario["category"], scenario["input"], scenario_path.parent)
            passed = actual == scenario["expected"]
            checks.append(
                make_check(
                    f"scenario:{identifier}",
                    "pass" if passed else "fail",
                    category=scenario["category"],
                    expected=scenario["expected"],
                    actual=actual,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            checks.append(
                make_check(
                    f"scenario:{identifier}", "fail", category=scenario["category"], error=str(exc)
                )
            )
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    checks.append(
        make_check(
            "scenario-identifiers",
            "fail" if duplicates else "pass",
            count=len(identifiers),
            duplicates=duplicates,
        )
    )
    return checks


def _p0_regression(root: Path) -> dict[str, Any]:
    test_path = root / "scripts/validation/test-p0-contracts.py"
    result = subprocess.run(
        [sys.executable, str(test_path)], check=False, capture_output=True, text=True, timeout=60
    )
    return make_check(
        "p0-contract-regression",
        "pass" if result.returncode == 0 else "fail",
        exit_code=result.returncode,
        stdout=result.stdout[-4000:],
        stderr=result.stderr[-4000:],
    )


def _stage2_regression(root: Path) -> dict[str, Any]:
    test_path = root / "scripts/validation/test-stage2-contracts.py"
    result = subprocess.run(
        [sys.executable, str(test_path)], check=False, capture_output=True, text=True, timeout=60
    )
    return make_check(
        "stage2-contract-regression",
        "pass" if result.returncode == 0 else "fail",
        exit_code=result.returncode,
        stdout=result.stdout[-4000:],
        stderr=result.stderr[-4000:],
    )


def _p21_regression(root: Path) -> dict[str, Any]:
    test_path = root / "scripts/validation/test-p21-contracts.py"
    result = subprocess.run(
        [sys.executable, str(test_path)], check=False, capture_output=True, text=True, timeout=60
    )
    return make_check(
        "p2.1-reduction-regression",
        "pass" if result.returncode == 0 else "fail",
        exit_code=result.returncode,
        stdout=result.stdout[-4000:],
        stderr=result.stderr[-4000:],
    )


def _p3_regression(root: Path) -> dict[str, Any]:
    test_path = root / "scripts/validation/test-p3-contracts.py"
    result = subprocess.run(
        [sys.executable, str(test_path)], check=False, capture_output=True, text=True, timeout=60
    )
    return make_check(
        "p3-mode-routing-budget-regression",
        "pass" if result.returncode == 0 else "fail",
        exit_code=result.returncode,
        stdout=result.stdout[-4000:],
        stderr=result.stderr[-4000:],
    )


def _p4_regression(root: Path) -> dict[str, Any]:
    test_path = root / "scripts/validation/test-p4-contracts.py"
    result = subprocess.run(
        [sys.executable, str(test_path)], check=False, capture_output=True, text=True, timeout=90
    )
    return make_check(
        "p4-cache-resume-regression",
        "pass" if result.returncode == 0 else "fail",
        exit_code=result.returncode,
        stdout=result.stdout[-4000:],
        stderr=result.stderr[-4000:],
    )


def _behavior_contract_markers(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    optimizer = (root / "references/roles/performance-optimizer.md").read_text(encoding="utf-8")
    rollback = (root / "references/contracts/rollback-policy.md").read_text(encoding="utf-8")
    for token in ("full-pipeline.md", "code-generation.md", "code-validation.md", "performance-tuning.md"):
        if token not in skill:
            errors.append(f"SKILL.md missing route marker {token}")
    for token in ("input.py", "named strategy", "EnvConfig/config.json", "another strategy workdir"):
        if token not in optimizer:
            errors.append(f"performance-optimizer.md missing access marker {token}")
    for token in ("infrastructure failure", "best_so_far", "select-best-candidate.py"):
        if token not in rollback:
            errors.append(f"rollback-policy.md missing rollback marker {token}")
    return make_check("behavior-contract-markers", "fail" if errors else "pass", errors=errors)


def _report_schema_self_check(root: Path) -> dict[str, Any]:
    schema_path = root / "references/schemas/validation-report.schema.json"
    schema = load_json(schema_path)
    probe = {
        "schema_version": "1.0",
        "level": "L2",
        "status": "pass",
        "started_at": "2026-08-28T00:00:00+00:00",
        "finished_at": "2026-08-28T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "skill_fingerprint": "0" * 64,
        "hardware_evidence": False,
        "checks": [{"name": "probe", "status": "pass", "details": {}}],
    }
    try:
        validate(probe, schema, schema_path)
    except SchemaValidationError as exc:
        return make_check("validation-report-schema", "fail", error=str(exc))
    return make_check("validation-report-schema", "pass", schema=str(schema_path))


def main() -> int:
    args = _arguments()
    root = args.skill_root.resolve()
    scenario_path = (
        args.scenarios.resolve()
        if args.scenarios
        else root / "references/evals/offline-scenarios.json"
    )
    started_at = utc_now()
    started_counter = perf_counter()
    checks = _scenario_checks(root, scenario_path)
    checks.append(_behavior_contract_markers(root))
    checks.append(_p0_regression(root))
    checks.append(_stage2_regression(root))
    checks.append(_p21_regression(root))
    checks.append(_p3_regression(root))
    checks.append(_p4_regression(root))
    checks.append(_report_schema_self_check(root))
    report = make_report(
        level="L2",
        started_at=started_at,
        started_counter=started_counter,
        fingerprint=skill_fingerprint(root),
        checks=checks,
    )
    return finish(report, args.report.resolve() if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
