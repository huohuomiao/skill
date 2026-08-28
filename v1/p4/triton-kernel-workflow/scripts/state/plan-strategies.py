#!/usr/bin/env python3
"""Build a deterministic static optimization strategy plan from one Triton program."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(VALIDATION_DIR))

from json_schema import SchemaValidationError, validate  # noqa: E402
from validation_common import load_json, utc_now, write_json  # noqa: E402


REDUCTION_CALLS = {"tl.sum", "tl.max", "tl.min", "tl.reduce", "tl.argmax", "tl.argmin"}
DEVICE_MATH_CALLS = {
    "tl.exp",
    "tl.exp2",
    "tl.log",
    "tl.log2",
    "tl.sqrt",
    "tl.rsqrt",
    "tl.erf",
    "tl.sin",
    "tl.cos",
    "tl.tanh",
    "tl.sigmoid",
}
STRATEGIES = (
    (1, "retiling", "oob", "references/strategies/tiling.md", "has_tiling_surface"),
    (2, "reduce-opt", "oob", "references/strategies/reduction.md", "has_reduction"),
    (3, "modify-grid", "oob", "references/strategies/grid-layout.md", "grid_needs_change"),
    (4, "index-computation-simplify", "oob", "references/strategies/index-simplification.md", "has_complex_index"),
    (5, "gen-autotune-config", "oob", "references/strategies/autotune-config.md", "has_tunable_parameters"),
    (6, "libdevice-opt", "advanced", "references/strategies/device-math.md", "has_device_math_pattern"),
    (7, "config-tuner", "advanced", "references/strategies/config-tuning.md", "has_tunable_parameters"),
    (8, "div-to-mul", "advanced", "references/strategies/division-to-multiplication.md", "has_division"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("correctness", "balanced", "max-performance"),
        default="balanced",
    )
    return parser.parse_args()


def _qualified_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_jit(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(_qualified_name(decorator.func if isinstance(decorator, ast.Call) else decorator) in {"triton.jit", "jit"} for decorator in function.decorator_list)


def _function_arguments(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    return [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]


def _is_tunable_parameter(name: str) -> bool:
    upper = name.upper()
    compact_block = (
        2 <= len(upper) <= 4
        and upper.startswith("B")
        and all(character in "MNKSHWCDXYZ0123456789_" for character in upper[1:])
    )
    return upper.startswith(("BLOCK", "TILE", "CHUNK", "GROUP", "SPLIT", "NUM_")) or compact_block


def _analyze(source: str, path: Path) -> dict:
    tree = ast.parse(source, filename=str(path))
    kernels = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_jit(node)
    ]
    if not kernels:
        raise ValueError("no @triton.jit kernel found")

    call_names: set[str] = set()
    tunable_parameters: set[str] = set()
    has_complex_index = False
    has_division = False
    has_for_loop = False
    for kernel in kernels:
        for argument in _function_arguments(kernel):
            if (
                _qualified_name(argument.annotation)
                in {"tl.constexpr", "triton.language.constexpr"}
                and _is_tunable_parameter(argument.arg)
            ):
                tunable_parameters.add(argument.arg)
        for node in ast.walk(kernel):
            if isinstance(node, ast.Call):
                call_names.add(_qualified_name(node.func))
            elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                has_complex_index = True
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                has_division = True
            elif isinstance(node, ast.For):
                has_for_loop = True

    has_arange = "tl.arange" in call_names
    has_num_programs = "tl.num_programs" in call_names
    return {
        "jit_kernels": [kernel.name for kernel in kernels],
        "has_reduction": bool(call_names & REDUCTION_CALLS),
        "grid_bounded": has_num_programs and has_for_loop,
        "has_complex_index": has_complex_index,
        "tunable_parameters": sorted(tunable_parameters),
        "has_device_math_pattern": bool(call_names & DEVICE_MATH_CALLS) or any(
            name.startswith("tl.libdevice.") for name in call_names
        ),
        "has_division": has_division,
        "has_tiling_surface": has_arange or bool(tunable_parameters),
    }


def _decision(mode: str, phase: str, signal: str, detected: dict) -> tuple[str, str]:
    if mode == "correctness":
        return "skip", "performance strategies are disabled by correctness mode"
    if phase == "advanced" and mode != "max-performance":
        return "skip", "advanced strategy is disabled by balanced mode"

    signals = {
        "has_tiling_surface": (detected["has_tiling_surface"], "block/tile surface detected", "no block/tile surface detected"),
        "has_reduction": (detected["has_reduction"], "reduction operation detected", "no reduction operation detected"),
        "grid_needs_change": (not detected["grid_bounded"], "bounded persistent Grid not proven", "Grid already has bounded persistent coverage"),
        "has_complex_index": (detected["has_complex_index"], "floor-division or modulo indexing detected", "no floor-division or modulo indexing detected"),
        "has_tunable_parameters": (bool(detected["tunable_parameters"]), "tunable tl.constexpr parameters detected", "no tunable tl.constexpr parameter detected"),
        "has_device_math_pattern": (detected["has_device_math_pattern"], "supported device-math call pattern detected", "no supported device-math call pattern detected"),
        "has_division": (detected["has_division"], "division expression detected in Triton kernel", "no division expression detected in Triton kernel"),
    }
    admitted, positive, negative = signals[signal]
    return ("apply", positive) if admitted else ("skip", negative)


def main() -> int:
    args = _arguments()
    try:
        input_path = args.input.resolve()
        if not input_path.is_file() or input_path.stat().st_size == 0:
            raise ValueError(f"input is missing or empty: {input_path}")
        source = input_path.read_text(encoding="utf-8")
        detected = _analyze(source, input_path)
        skill_root = Path(__file__).resolve().parents[2]
        strategies = []
        for order, name, phase, relative_doc, signal in STRATEGIES:
            decision, reason = _decision(args.mode, phase, signal, detected)
            strategies.append(
                {
                    "order": order,
                    "name": name,
                    "phase": phase,
                    "decision": decision,
                    "reason": reason,
                    "strategy_doc": str((skill_root / relative_doc).resolve()),
                }
            )
        plan = {
            "schema_version": "1.0",
            "optimization_mode": args.mode,
            "input_path": str(input_path),
            "generated_at": utc_now(),
            "detected": detected,
            "strategies": strategies,
            "dynamic_execution": "serial",
        }
        schema_path = skill_root / "references/schemas/strategy-plan.schema.json"
        validate(plan, load_json(schema_path), schema_path)
        write_json(args.output.resolve(), plan)
        print(json.dumps(plan, ensure_ascii=False))
        return 0
    except (OSError, SyntaxError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"plan-strategies: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
