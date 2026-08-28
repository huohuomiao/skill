#!/usr/bin/env python3
"""Run L1/L2/L3 validation in order and enforce time and dependency gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from json_schema import SchemaValidationError, validate_file
from validation_common import load_json, write_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("l1", "l2", "l3"), required=True)
    parser.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--integration-suite", type=Path)
    return parser.parse_args()


def _run(command: list[str]) -> int:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    return result.returncode


def main() -> int:
    args = _arguments()
    root = args.skill_root.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    requested = {"l1": 1, "l2": 2, "l3": 3}[args.level]
    started = perf_counter()
    reports: dict[str, str] = {}
    status = "pass"
    gate_errors: list[str] = []
    report_schema = root / "references/schemas/validation-report.schema.json"

    l1_path = report_dir / "l1-report.json"
    l1_command = [
        sys.executable,
        str(script_dir / "l1-static.py"),
        "--skill-root",
        str(root),
        "--report",
        str(l1_path),
    ]
    if _run(l1_command) != 0:
        status = "fail"
    reports["L1"] = str(l1_path)
    if l1_path.is_file():
        try:
            validate_file(l1_path, report_schema)
        except (SchemaValidationError, OSError, json.JSONDecodeError) as exc:
            status = "fail"
            gate_errors.append(f"invalid L1 report: {exc}")

    l2_path = report_dir / "l2-report.json"
    if requested >= 2 and status == "pass":
        l2_command = [
            sys.executable,
            str(script_dir / "l2-offline.py"),
            "--skill-root",
            str(root),
            "--report",
            str(l2_path),
        ]
        if _run(l2_command) != 0:
            status = "fail"
        reports["L2"] = str(l2_path)
        if l2_path.is_file():
            try:
                validate_file(l2_path, report_schema)
            except (SchemaValidationError, OSError, json.JSONDecodeError) as exc:
                status = "fail"
                gate_errors.append(f"invalid L2 report: {exc}")

    elapsed_lower = perf_counter() - started
    budget_errors: list[str] = []
    if l1_path.is_file() and load_json(l1_path).get("elapsed_seconds", 31) > 30:
        budget_errors.append("L1 exceeded 30 seconds")
    if requested >= 2 and elapsed_lower > 300:
        budget_errors.append("L1+L2 exceeded 300 seconds")
    if budget_errors:
        status = "fail"
    gate_errors.extend(budget_errors)

    if requested >= 3 and status == "pass":
        if args.integration_suite is None:
            status = "fail"
            gate_errors.append("--integration-suite is required for L3")
        else:
            l3_path = report_dir / "l3-report.json"
            l3_command = [
                sys.executable,
                str(script_dir / "l3-integration.py"),
                "--skill-root",
                str(root),
                "--l1-report",
                str(l1_path),
                "--l2-report",
                str(l2_path),
                "--suite",
                str(args.integration_suite.resolve()),
                "--report",
                str(l3_path),
            ]
            if _run(l3_command) != 0:
                status = "fail"
            reports["L3"] = str(l3_path)

    summary = {
        "schema_version": "1.0",
        "requested_level": args.level.upper(),
        "status": status,
        "elapsed_seconds": round(perf_counter() - started, 6),
        "budget_errors": budget_errors,
        "gate_errors": gate_errors,
        "reports": reports,
    }
    summary_path = report_dir / "gate-summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
