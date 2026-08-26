#!/usr/bin/env python3
"""Measure the v1 -> v1_1 dispatch contract without third-party packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "mlu-triton-code-gen" / "references" / "dispatch-contract.json"


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reduction_pct(baseline: int, optimized: int) -> float:
    if baseline <= 0:
        return 0.0
    return round((baseline - optimized) * 100.0 / baseline, 2)


def static_context_bytes(
    contract: dict[str, Any], variant: str, route: str, outcome: str
) -> tuple[int, list[dict[str, Any]]]:
    calls = contract["static_context_calls"][variant][route][outcome]
    measured: list[dict[str, Any]] = []
    total = 0
    for call in calls:
        files = []
        call_bytes = 0
        for relative in call["files"]:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"contract path does not exist: {relative}")
            size = path.stat().st_size
            call_bytes += size
            files.append({"path": relative, "bytes": size})
        total += call_bytes
        measured.append(
            {"stage": call["stage"], "static_context_bytes": call_bytes, "files": files}
        )
    return total, measured


def analyze(route: str, outcome: str) -> dict[str, Any]:
    contract = load_contract()
    dispatch = contract["routes"][route][outcome]
    baseline_dispatches = dispatch["baseline_dispatches"]["total"]
    optimized_dispatches = dispatch["optimized_dispatches"]["total"]
    baseline_bytes, baseline_calls = static_context_bytes(
        contract, "baseline", route, outcome
    )
    optimized_bytes, optimized_calls = static_context_bytes(
        contract, "optimized", route, outcome
    )

    report = {
        "schema_version": contract["schema_version"],
        "version": contract["version"],
        "phase": contract["phase"],
        "route": route,
        "outcome": outcome,
        "dispatches": {
            "baseline": dispatch["baseline_dispatches"],
            "optimized": dispatch["optimized_dispatches"],
            "reduction_pct": reduction_pct(baseline_dispatches, optimized_dispatches),
        },
        "static_context": {
            "baseline_bytes": baseline_bytes,
            "optimized_bytes": optimized_bytes,
            "reduction_pct": reduction_pct(baseline_bytes, optimized_bytes),
            "baseline_calls": baseline_calls,
            "optimized_calls": optimized_calls,
        },
        "measurement_note": contract["measurement"],
    }

    if route == "normal":
        targets = contract["targets"]
        if report["dispatches"]["reduction_pct"] < targets["normal_dispatch_reduction_pct"]:
            raise RuntimeError("normal dispatch reduction target not met")
        if (
            report["static_context"]["reduction_pct"]
            < targets["normal_static_context_reduction_pct"]
        ):
            raise RuntimeError("normal static-context reduction target not met")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="analyze one route/outcome")
    analyze_parser.add_argument(
        "--route", choices=("normal", "triton-fast"), required=True
    )
    analyze_parser.add_argument(
        "--outcome", choices=("direct-pass", "repair"), required=True
    )
    analyze_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze(args.route, args.outcome)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
