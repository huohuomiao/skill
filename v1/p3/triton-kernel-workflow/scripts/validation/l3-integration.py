#!/usr/bin/env python3
"""Run gated, serial hardware integration regression commands."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any

from json_schema import SchemaValidationError, validate_file
from validation_common import (
    finish,
    load_json,
    make_check,
    make_report,
    skill_fingerprint,
    utc_now,
)


REQUIRED_OPERATOR_CLASSES = {"elementwise", "reduction", "layout"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--l1-report", type=Path, required=True)
    parser.add_argument("--l2-report", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def _gate_check(root: Path, l1_path: Path, l2_path: Path) -> dict[str, Any]:
    report_schema = root / "references/schemas/validation-report.schema.json"
    try:
        validate_file(l1_path, report_schema)
        validate_file(l2_path, report_schema)
        l1 = load_json(l1_path)
        l2 = load_json(l2_path)
    except (SchemaValidationError, OSError, json.JSONDecodeError) as exc:
        return make_check("lower-level-gates", "fail", error=str(exc))
    fingerprint = skill_fingerprint(root)
    errors: list[str] = []
    if l1.get("level") != "L1" or l1.get("status") != "pass":
        errors.append("L1 report is not passing")
    if l2.get("level") != "L2" or l2.get("status") != "pass":
        errors.append("L2 report is not passing")
    if l1.get("skill_fingerprint") != fingerprint or l2.get("skill_fingerprint") != fingerprint:
        errors.append("L1/L2 report fingerprint is stale")
    l1_elapsed = l1.get("elapsed_seconds")
    l2_elapsed = l2.get("elapsed_seconds")
    if not isinstance(l1_elapsed, (int, float)) or l1_elapsed > 30:
        errors.append("L1 exceeds 30-second budget")
    if (
        not isinstance(l1_elapsed, (int, float))
        or not isinstance(l2_elapsed, (int, float))
        or l1_elapsed + l2_elapsed > 300
    ):
        errors.append("L1+L2 exceeds 300-second budget")
    return make_check(
        "lower-level-gates",
        "fail" if errors else "pass",
        errors=errors,
        l1_report=str(l1_path),
        l2_report=str(l2_path),
    )


def _load_suite(root: Path, suite_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    schema_path = root / "references/schemas/integration-suite.schema.json"
    try:
        validate_file(suite_path, schema_path)
        suite = load_json(suite_path)
    except (SchemaValidationError, OSError, json.JSONDecodeError) as exc:
        return None, make_check("integration-suite", "fail", error=str(exc))
    classes = {case["operator_class"] for case in suite["cases"]}
    missing = sorted(REQUIRED_OPERATOR_CLASSES - classes)
    hardware_model = suite["hardware_model"].strip().lower()
    errors: list[str] = []
    if missing:
        errors.append(f"missing operator classes: {missing}")
    if not hardware_model or hardware_model in {"unknown", "fixture", "mock", "none"}:
        errors.append("hardware_model must identify real hardware")
    identifiers = [case["id"] for case in suite["cases"]]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate case ids: {duplicates}")
    return suite, make_check(
        "integration-suite",
        "fail" if errors else "pass",
        errors=errors,
        suite_id=suite["suite_id"],
        hardware_model=suite["hardware_model"],
    )


def _run_command(name: str, command: list[str], workdir: Path, timeout: float) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return make_check(
            name,
            "pass" if result.returncode == 0 else "fail",
            command=command,
            workdir=str(workdir),
            exit_code=result.returncode,
            elapsed_seconds=round(perf_counter() - started, 6),
            stdout=result.stdout[-4000:],
            stderr=result.stderr[-4000:],
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return make_check(
            name,
            "fail",
            command=command,
            workdir=str(workdir),
            elapsed_seconds=round(perf_counter() - started, 6),
            error=str(exc),
        )


def main() -> int:
    args = _arguments()
    root = args.skill_root.resolve()
    started_at = utc_now()
    started_counter = perf_counter()
    checks = [_gate_check(root, args.l1_report.resolve(), args.l2_report.resolve())]
    suite, suite_check = _load_suite(root, args.suite.resolve())
    checks.append(suite_check)
    if any(item["status"] == "fail" for item in checks) or suite is None:
        report = make_report(
            level="L3",
            started_at=started_at,
            started_counter=started_counter,
            fingerprint=skill_fingerprint(root),
            checks=checks,
            hardware_evidence=False,
        )
        return finish(report, args.report.resolve() if args.report else None)

    timeout = float(suite["timeout_seconds"])
    suite_base = args.suite.resolve().parent
    for case in suite["cases"]:
        workdir = Path(case["workdir"])
        if not workdir.is_absolute():
            workdir = (suite_base / workdir).resolve()
        if not workdir.is_dir():
            checks.append(
                make_check(
                    f"case:{case['id']}:workdir", "fail", error=f"missing directory: {workdir}"
                )
            )
            continue
        for stage in ("compile", "accuracy", "performance"):
            checks.append(
                _run_command(
                    f"case:{case['id']}:{stage}", case["commands"][stage], workdir, timeout
                )
            )
    for name in ("submission", "failure_recovery"):
        checks.append(
            _run_command(
                f"worker:{name}", suite["worker_checks"][name], suite_base, timeout
            )
        )
    report = make_report(
        level="L3",
        started_at=started_at,
        started_counter=started_counter,
        fingerprint=skill_fingerprint(root),
        checks=checks,
        hardware_evidence=True,
    )
    return finish(report, args.report.resolve() if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
