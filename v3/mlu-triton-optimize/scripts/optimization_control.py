#!/usr/bin/env python3
"""Deterministic optimization routing and budget control for MLU Triton."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import time
import tokenize
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES = ("correctness", "balanced", "max-performance")
OOB_STRATEGIES = (
    (1, "retiling"),
    (2, "reduce-opt"),
    (3, "modify-grid"),
    (4, "index-computation-simplify"),
    (5, "gen-autotune-config"),
)
ADVANCED_STRATEGIES = ("libdevice-opt", "config-tuner", "div-to-mul")
DEFAULT_LIMITS: dict[str, dict[str, int | float]] = {
    "correctness": {
        "max_wall_time_sec": 0,
        "max_subagent_calls": 0,
        "max_worker_calls": 0,
        "max_strategy_attempts": 0,
        "max_advanced_iterations": 0,
        "advanced_patience": 0,
        "min_improvement_pct": 2.0,
    },
    "balanced": {
        "max_wall_time_sec": 1800,
        "max_subagent_calls": 8,
        "max_worker_calls": 8,
        "max_strategy_attempts": 8,
        "max_advanced_iterations": 0,
        "advanced_patience": 0,
        "min_improvement_pct": 2.0,
    },
    "max-performance": {
        "max_wall_time_sec": 7200,
        "max_subagent_calls": 16,
        "max_worker_calls": 16,
        "max_strategy_attempts": 12,
        "max_advanced_iterations": 3,
        "advanced_patience": 2,
        "min_improvement_pct": 2.0,
    },
}
LIMIT_KEYS = frozenset(next(iter(DEFAULT_LIMITS.values())))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def strip_comments_and_strings(source: str) -> str:
    """Remove comments and string literals while retaining Python operators."""
    try:
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in {tokenize.COMMENT, tokenize.STRING}:
                token = token._replace(string=" ")
            tokens.append(token)
        return tokenize.untokenize(tokens)
    except (IndentationError, tokenize.TokenError):
        return source


def detect_max_grid_dimensions(source: str) -> int:
    """Return the largest statically visible grid tuple dimension."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    maximum = 0
    for node in ast.walk(tree):
        value: ast.AST | None = None
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "grid" for target in targets):
                value = node.value
        if isinstance(value, ast.Lambda):
            value = value.body
        if isinstance(value, ast.Tuple):
            maximum = max(maximum, len(value.elts))
    return maximum


def extract_triton_kernels(source: str) -> str:
    """Extract @triton.jit function bodies so test/reference code does not affect routing."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    segments: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [ast.get_source_segment(source, item) or "" for item in node.decorator_list]
        if not any("triton.jit" in decorator for decorator in decorators):
            continue
        start_lines = [node.lineno, *(item.lineno for item in node.decorator_list)]
        start = max(1, min(start_lines)) - 1
        end = node.end_lineno or node.lineno
        segments.append("".join(lines[start:end]))
    return "\n".join(segments) if segments else source


def detect_signals(source: str) -> dict[str, Any]:
    code = strip_comments_and_strings(source)
    kernel_code = strip_comments_and_strings(extract_triton_kernels(source))
    block_params = sorted(
        set(re.findall(r"\b([A-Z][A-Z0-9_]*)\s*:\s*tl\.constexpr\b", kernel_code))
    )
    program_axes = sorted(
        {
            int(axis)
            for axis in re.findall(
                r"tl\.program_id\s*\(\s*(?:axis\s*=\s*)?([0-9]+)\s*\)", kernel_code
            )
        }
    )
    reduce_pattern = re.compile(
        r"\b(?:tl|torch)\.(?:sum|max|min|argmax|argmin|cumsum|cumprod)\s*\("
    )
    math_pattern = re.compile(
        r"\btl\.(?:math\.)?(?:exp|exp2|log|log2|sqrt|rsqrt|sin|cos|sigmoid|erf)\s*\("
        r"|\btl\.extra\.mlu\.libdevice\."
    )
    division_pattern = re.compile(r"(?<!/)/(?!/)")
    integer_index_pattern = re.compile(r"//|%")
    grid_present = bool(re.search(r"\bgrid\s*=|\[[^\]\n]+\]\s*\(", code))
    max_grid_dimensions = detect_max_grid_dimensions(source)
    multi_dim_grid = any(axis > 0 for axis in program_axes) or max_grid_dimensions > 1
    core_cap = bool(
        re.search(
            r"\b(?:TOTAL_CORE_NUM|total_core_num|core_num|device_core|cluster_count|num_cores)\b",
            code,
            flags=re.IGNORECASE,
        )
    )
    return {
        "has_triton_kernel": bool(re.search(r"@\s*triton\.jit\b", code)),
        "has_kernel_launch": bool(re.search(r"\[[^\]\n]+\]\s*\(", code)),
        "has_grid": grid_present,
        "program_id_axes": program_axes,
        "has_multidimensional_grid": multi_dim_grid,
        "max_static_grid_dimensions": max_grid_dimensions,
        "has_core_cap": core_cap,
        "has_reduction": bool(reduce_pattern.search(kernel_code)),
        "has_index_div_or_mod": bool(integer_index_pattern.search(kernel_code)),
        "has_tensor_division": bool(division_pattern.search(kernel_code)),
        "has_libdevice_candidate": bool(math_pattern.search(kernel_code)),
        "has_autotune": bool(re.search(r"@\s*triton\.autotune\b", code)),
        "block_params": block_params,
    }


def route_oob(mode: str, signals: dict[str, Any]) -> list[dict[str, Any]]:
    if mode == "correctness":
        return [
            {"order": order, "name": name, "selected": False, "reason": "correctness mode skips optimization"}
            for order, name in OOB_STRATEGIES
        ]

    block_param_count = len(signals["block_params"])
    decisions = {
        "retiling": (
            signals["has_reduction"] or block_param_count >= 2,
            "reduction or multiple block parameters require tiling review",
            "no reduction and fewer than two block parameters",
        ),
        "reduce-opt": (
            signals["has_reduction"],
            "reduction primitive detected",
            "no reduction primitive detected",
        ),
        "modify-grid": (
            signals["has_grid"]
            and (signals["has_multidimensional_grid"] or not signals["has_core_cap"]),
            "grid needs flattening or an explicit physical-core cap",
            "no grid found, or grid is already single-dimensional and core-capped",
        ),
        "index-computation-simplify": (
            signals["has_index_div_or_mod"],
            "integer division or modulo detected in code",
            "no integer division or modulo detected",
        ),
        "gen-autotune-config": (
            block_param_count > 0 and not signals["has_autotune"],
            "tunable constexpr block parameters found without autotune",
            "no tunable block parameters, or autotune already exists",
        ),
    }
    routed: list[dict[str, Any]] = []
    for order, name in OOB_STRATEGIES:
        selected, selected_reason, skipped_reason = decisions[name]
        routed.append(
            {
                "order": order,
                "name": name,
                "selected": bool(selected),
                "reason": selected_reason if selected else skipped_reason,
            }
        )
    return routed


def route_advanced(mode: str, signals: dict[str, Any]) -> dict[str, Any]:
    if mode != "max-performance":
        return {
            "enabled": False,
            "reason": f"{mode} mode does not run advanced optimization",
            "candidates": [],
        }
    candidates: list[str] = []
    if signals["has_libdevice_candidate"]:
        candidates.append("libdevice-opt")
    if signals["block_params"]:
        candidates.append("config-tuner")
    if signals["has_tensor_division"]:
        candidates.append("div-to-mul")
    return {
        "enabled": True,
        "reason": "max-performance mode enables perf-guided iterations",
        "candidates": candidates,
    }


def load_limits(mode: str, budget_file: Path | None) -> dict[str, int | float]:
    limits = dict(DEFAULT_LIMITS[mode])
    if budget_file is not None:
        override = read_json(budget_file)
        unknown = set(override) - LIMIT_KEYS
        if unknown:
            raise ValueError(f"Unknown budget key(s): {sorted(unknown)}")
        for key, value in override.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Budget {key} must be a non-negative number")
            limits[key] = value
    integer_keys = LIMIT_KEYS - {"min_improvement_pct"}
    for key in integer_keys:
        if int(limits[key]) != limits[key]:
            raise ValueError(f"Budget {key} must be an integer")
        limits[key] = int(limits[key])
    limits["min_improvement_pct"] = float(limits["min_improvement_pct"])
    return limits


def build_plan(
    source: str,
    input_path: str,
    mode: str,
    limits: dict[str, int | float] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    signals = detect_signals(source)
    effective_limits = dict(limits or DEFAULT_LIMITS[mode])
    oob = route_oob(mode, signals)
    advanced = route_advanced(mode, signals)
    manual_review = not signals["has_triton_kernel"] or not signals["has_kernel_launch"]
    return {
        "version": 1,
        "created_at": utc_now(),
        "mode": mode,
        "input_path": input_path,
        "input_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "manual_review_required": manual_review,
        "manual_review_reason": (
            "Triton kernel or launch expression was not detected" if manual_review else None
        ),
        "limits": effective_limits,
        "signals": signals,
        "oob_strategies": oob,
        "selected_oob_strategies": [item["name"] for item in oob if item["selected"]],
        "advanced": advanced,
    }


def build_state(plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "mode": plan["mode"],
        "plan_path": str(plan_path.resolve()),
        "started_at": utc_now(),
        "started_at_epoch": time.time(),
        "limits": plan["limits"],
        "usage": {
            "subagent_calls": 0,
            "worker_calls": 0,
            "strategy_attempts": 0,
            "advanced_iterations": 0,
        },
        "advanced_no_improvement": 0,
        "best": {
            "performance_ms": None,
            "artifact_path": None,
            "strategy": None,
        },
        "stop_reason": None,
        "history": [],
    }


def budget_decision(state: dict[str, Any], phase: str) -> tuple[bool, str | None]:
    if state.get("stop_reason"):
        return False, str(state["stop_reason"])
    limits = state["limits"]
    usage = state["usage"]
    elapsed = max(0.0, time.time() - float(state["started_at_epoch"]))
    checks = (
        (elapsed >= limits["max_wall_time_sec"], "max_wall_time_reached"),
        (usage["subagent_calls"] >= limits["max_subagent_calls"], "max_subagent_calls_reached"),
        (usage["worker_calls"] >= limits["max_worker_calls"], "max_worker_calls_reached"),
        (usage["strategy_attempts"] >= limits["max_strategy_attempts"], "max_strategy_attempts_reached"),
    )
    for exhausted, reason in checks:
        if exhausted:
            return False, reason
    if phase == "advanced":
        if usage["advanced_iterations"] >= limits["max_advanced_iterations"]:
            return False, "max_advanced_iterations_reached"
        if state["advanced_no_improvement"] >= limits["advanced_patience"]:
            return False, "advanced_patience_exhausted"
    return True, None


def parse_accuracy(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def append_history(state: dict[str, Any], entry: dict[str, Any]) -> None:
    state["history"].append(entry)
    if len(state["history"]) > 200:
        state["history"] = state["history"][-200:]


def record_event(
    state: dict[str, Any],
    event: str,
    phase: str,
    strategy: str | None,
    status: str,
    accuracy_pass: bool | None,
    baseline_ms: float | None,
    candidate_ms: float | None,
    artifact_path: str | None,
) -> None:
    usage = state["usage"]
    if event == "subagent":
        usage["subagent_calls"] += 1
    elif event == "worker":
        usage["worker_calls"] += 1
    elif event == "strategy":
        usage["strategy_attempts"] += 1
    elif event == "advanced_iteration":
        usage["advanced_iterations"] += 1

    improvement_pct: float | None = None
    if baseline_ms is not None and candidate_ms is not None and baseline_ms > 0:
        improvement_pct = (baseline_ms - candidate_ms) / baseline_ms * 100.0

    advanced_threshold_met = (
        phase != "advanced"
        or (
            improvement_pct is not None
            and improvement_pct >= float(state["limits"]["min_improvement_pct"])
        )
    )
    if (
        event == "strategy"
        and accuracy_pass is True
        and candidate_ms is not None
        and artifact_path is not None
        and advanced_threshold_met
    ):
        best_ms = state["best"]["performance_ms"]
        if best_ms is None or candidate_ms < best_ms:
            state["best"] = {
                "performance_ms": candidate_ms,
                "artifact_path": artifact_path,
                "strategy": strategy,
            }

    if event == "advanced_iteration":
        threshold = float(state["limits"]["min_improvement_pct"])
        improved = accuracy_pass is True and improvement_pct is not None and improvement_pct >= threshold
        state["advanced_no_improvement"] = 0 if improved else state["advanced_no_improvement"] + 1

    append_history(
        state,
        {
            "timestamp": utc_now(),
            "event": event,
            "phase": phase,
            "strategy": strategy,
            "status": status,
            "accuracy_pass": accuracy_pass,
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
            "improvement_pct": improvement_pct,
            "artifact_path": artifact_path,
        },
    )


def command_plan(args: argparse.Namespace) -> int:
    input_path = args.input.resolve()
    if not input_path.is_file():
        print(f"input file does not exist: {input_path}", file=sys.stderr)
        return 2
    source = input_path.read_text(encoding="utf-8-sig")
    try:
        limits = load_limits(args.mode, args.budget_file)
        plan = build_plan(source, str(input_path), args.mode, limits)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output_dir = args.output_dir.resolve()
    plan_path = output_dir / "optimization_plan.json"
    state_path = output_dir / "optimization_state.json"
    write_json_atomic(plan_path, plan)
    write_json_atomic(state_path, build_state(plan, plan_path))
    print(json.dumps({"plan_path": str(plan_path), "state_path": str(state_path)}, ensure_ascii=False))
    return 0


def command_check(args: argparse.Namespace) -> int:
    state = read_json(args.state)
    allowed, reason = budget_decision(state, args.phase)
    if not allowed and not state.get("stop_reason"):
        state["stop_reason"] = reason
        write_json_atomic(args.state, state)
    print(
        json.dumps(
            {
                "allowed": allowed,
                "reason": reason,
                "phase": args.phase,
                "usage": state["usage"],
                "limits": state["limits"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if allowed else 3


def command_record(args: argparse.Namespace) -> int:
    state = read_json(args.state)
    record_event(
        state=state,
        event=args.event,
        phase=args.phase,
        strategy=args.strategy,
        status=args.status,
        accuracy_pass=parse_accuracy(args.accuracy_pass),
        baseline_ms=args.baseline_ms,
        candidate_ms=args.candidate_ms,
        artifact_path=args.artifact_path,
    )
    write_json_atomic(args.state, state)
    print(
        json.dumps(
            {
                "usage": state["usage"],
                "advanced_no_improvement": state["advanced_no_improvement"],
                "best": state["best"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create a routed plan and initial budget state")
    plan.add_argument("--input", required=True, type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)
    plan.add_argument("--mode", choices=MODES, default="balanced")
    plan.add_argument("--budget-file", type=Path)
    plan.set_defaults(func=command_plan)

    check = subparsers.add_parser("check", help="Check whether another optimization action is allowed")
    check.add_argument("--state", required=True, type=Path)
    check.add_argument("--phase", choices=("oob", "advanced"), required=True)
    check.set_defaults(func=command_check)

    record = subparsers.add_parser("record", help="Record budget usage or a strategy result")
    record.add_argument("--state", required=True, type=Path)
    record.add_argument("--event", choices=("subagent", "worker", "strategy", "advanced_iteration"), required=True)
    record.add_argument("--phase", choices=("oob", "advanced"), required=True)
    record.add_argument("--strategy")
    record.add_argument("--status", default="completed")
    record.add_argument("--accuracy-pass", choices=("true", "false", "unknown"), default="unknown")
    record.add_argument("--baseline-ms", type=float)
    record.add_argument("--candidate-ms", type=float)
    record.add_argument("--artifact-path")
    record.set_defaults(func=command_record)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
