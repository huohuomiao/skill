#!/usr/bin/env python3
"""Compare normalized MLU Triton run results against a pinned baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "validation" / "regression_policy.json"


class RegressionError(ValueError):
    """A malformed or incomparable regression input."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RegressionError(f"JSON object required: {path}")
    return value


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def require_result(value: dict[str, Any], label: str) -> None:
    required_strings = ("run_id", "hardware_key", "toolchain_key")
    if value.get("schema_version") != 1:
        raise RegressionError(f"{label}.schema_version must be 1")
    for key in required_strings:
        if not isinstance(value.get(key), str) or not value[key]:
            raise RegressionError(f"{label}.{key} must be a non-empty string")
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RegressionError(f"{label}.cases must be a non-empty list")
    ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise RegressionError(f"{label}.cases[{index}] must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RegressionError(f"{label}.cases[{index}].case_id must be non-empty")
        ids.append(case_id)
        if case.get("status") not in {"completed", "failed", "skipped"}:
            raise RegressionError(f"{label}.{case_id}.status is invalid")
        accuracy = case.get("accuracy")
        performance = case.get("performance")
        if not isinstance(accuracy, dict) or not isinstance(accuracy.get("pass"), bool):
            raise RegressionError(f"{label}.{case_id}.accuracy.pass must be boolean")
        for metric in ("atol", "rtol"):
            if finite_number(accuracy.get(metric)) is None:
                raise RegressionError(f"{label}.{case_id}.accuracy.{metric} must be finite")
        if accuracy.get("max_diff") is not None and finite_number(accuracy.get("max_diff")) is None:
            raise RegressionError(f"{label}.{case_id}.accuracy.max_diff must be finite or null")
        if not isinstance(performance, dict):
            raise RegressionError(f"{label}.{case_id}.performance must be an object")
    if len(ids) != len(set(ids)):
        raise RegressionError(f"{label}.case_id values must be unique")


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        converted = float(value)
        if converted == converted and abs(converted) != float("inf"):
            return converted
    return None


def percent_change(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None or baseline <= 0:
        return None
    return (current - baseline) / baseline * 100.0


def positive_number(value: Any) -> float | None:
    converted = finite_number(value)
    return converted if converted is not None and converted > 0 else None


def indexed_cases(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in result["cases"]}


def compare_results(
    baseline: dict[str, Any], current: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    require_result(baseline, "baseline")
    require_result(current, "current")
    if policy.get("schema_version") != 1:
        raise RegressionError("policy.schema_version must be 1")
    for key in (
        "max_latency_regression_pct",
        "max_wall_time_regression_pct",
        "max_token_regression_pct",
        "max_subagent_call_increase",
        "max_worker_call_increase",
    ):
        threshold = finite_number(policy.get(key))
        if threshold is None or threshold < 0:
            raise RegressionError(f"policy.{key} must be a non-negative number")

    same_context = (
        baseline["hardware_key"] == current["hardware_key"]
        and baseline["toolchain_key"] == current["toolchain_key"]
    )
    global_violations: list[str] = []
    if policy.get("require_same_context", True) and not same_context:
        global_violations.append("hardware_or_toolchain_context_mismatch")

    baseline_cases = indexed_cases(baseline)
    current_cases = indexed_cases(current)
    rows: list[dict[str, Any]] = []
    for case_id, base in baseline_cases.items():
        now = current_cases.get(case_id)
        violations: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        if now is None:
            violations.append("case_missing_from_current")
            rows.append(
                {"case_id": case_id, "outcome": "fail", "violations": violations, "warnings": warnings, "metrics": metrics}
            )
            continue
        if now["status"] != "completed":
            violations.append(f"current_status_{now['status']}")
        if policy.get("require_accuracy_pass", True) and not now["accuracy"]["pass"]:
            violations.append("accuracy_failed")
        elif base["accuracy"]["pass"] and not now["accuracy"]["pass"]:
            violations.append("accuracy_regressed")

        base_latency = positive_number(base["performance"].get("latency_ms"))
        current_latency = positive_number(now["performance"].get("latency_ms"))
        latency_pct = percent_change(base_latency, current_latency) if same_context else None
        metrics["baseline_latency_ms"] = base_latency
        metrics["current_latency_ms"] = current_latency
        metrics["latency_change_pct"] = latency_pct
        if not same_context:
            warnings.append("latency_not_comparable_context_mismatch")
        elif latency_pct is None:
            if policy.get("require_performance_metric", True):
                violations.append("latency_metric_missing_or_invalid")
        elif latency_pct > float(policy["max_latency_regression_pct"]):
            violations.append("latency_regression_exceeded")

        base_resources = base.get("resources") if isinstance(base.get("resources"), dict) else {}
        now_resources = now.get("resources") if isinstance(now.get("resources"), dict) else {}
        for metric, policy_key in (
            ("wall_time_sec", "max_wall_time_regression_pct"),
            ("token_count", "max_token_regression_pct"),
        ):
            base_value = finite_number(base_resources.get(metric))
            current_value = finite_number(now_resources.get(metric))
            change = percent_change(base_value, current_value)
            metrics[f"baseline_{metric}"] = base_value
            metrics[f"current_{metric}"] = current_value
            metrics[f"{metric}_change_pct"] = change
            if change is None and policy.get("require_resource_metrics", True):
                violations.append(f"{metric}_missing_or_invalid")
            elif change is not None and change > float(policy[policy_key]):
                violations.append(f"{metric}_regression_exceeded")

        for metric, policy_key in (
            ("subagent_calls", "max_subagent_call_increase"),
            ("worker_calls", "max_worker_call_increase"),
        ):
            base_calls = finite_number(base_resources.get(metric))
            current_calls = finite_number(now_resources.get(metric))
            call_increase = None if base_calls is None or current_calls is None else current_calls - base_calls
            metrics[f"{metric}_increase"] = call_increase
            if call_increase is None and policy.get("require_resource_metrics", True):
                violations.append(f"{metric}_missing_or_invalid")
            elif call_increase is not None and call_increase > float(policy[policy_key]):
                violations.append(f"{metric}_increase_exceeded")

        outcome = "fail" if violations else ("not_comparable" if not same_context else "pass")
        rows.append(
            {"case_id": case_id, "outcome": outcome, "violations": violations, "warnings": warnings, "metrics": metrics}
        )

    for case_id in sorted(set(current_cases) - set(baseline_cases)):
        rows.append(
            {
                "case_id": case_id,
                "outcome": "not_comparable",
                "violations": [],
                "warnings": ["new_case_without_baseline"],
                "metrics": {},
            }
        )

    failed = sum(row["outcome"] == "fail" for row in rows)
    passed = sum(row["outcome"] == "pass" for row in rows)
    not_comparable = sum(row["outcome"] == "not_comparable" for row in rows)
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "baseline_run_id": baseline["run_id"],
        "current_run_id": current["run_id"],
        "context_compatible": same_context,
        "passed": not global_violations and failed == 0,
        "global_violations": global_violations,
        "summary": {"passed": passed, "failed": failed, "not_comparable": not_comparable},
        "cases": rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    verdict = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# 持续回归对比报告",
        "",
        f"- 总结：**{verdict}**",
        f"- 基线运行：`{report['baseline_run_id']}`",
        f"- 当前运行：`{report['current_run_id']}`",
        f"- 硬件/工具链可比：`{str(report['context_compatible']).lower()}`",
        f"- 通过 / 失败 / 不可比：{report['summary']['passed']} / {report['summary']['failed']} / {report['summary']['not_comparable']}",
        "",
    ]
    if report["global_violations"]:
        lines.extend(["## 全局阻断", ""])
        lines.extend(f"- `{item}`" for item in report["global_violations"])
        lines.append("")
    lines.extend(
        [
            "## 用例结果",
            "",
            "| 用例 | 判定 | 延迟变化 | 问题 |",
            "|---|---|---:|---|",
        ]
    )
    for row in report["cases"]:
        latency = row["metrics"].get("latency_change_pct")
        latency_text = "N/A" if latency is None else f"{latency:+.2f}%"
        issues = row["violations"] + row["warnings"]
        lines.append(f"| `{row['case_id']}` | {row['outcome']} | {latency_text} | {'; '.join(issues) or '-'} |")
    lines.extend(
        [
            "",
            "## 判定说明",
            "",
            "性能只在 `hardware_key` 与 `toolchain_key` 都一致时比较；上下文不一致不会产生伪性能结论。",
            "精度失败、用例缺失或超过策略阈值会使报告失败。",
            "",
        ]
    )
    return "\n".join(lines)


def command_compare(args: argparse.Namespace) -> int:
    baseline = read_json(args.baseline.resolve())
    current = read_json(args.current.resolve())
    policy = read_json(args.policy.resolve())
    report = compare_results(baseline, current, policy)
    write_json_atomic(args.report_json.resolve(), report)
    write_text_atomic(args.report_md.resolve(), markdown_report(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare", help="Compare current results with a baseline")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--current", required=True, type=Path)
    compare.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    compare.add_argument("--report-json", required=True, type=Path)
    compare.add_argument("--report-md", required=True, type=Path)
    compare.set_defaults(func=command_compare)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (OSError, RegressionError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
