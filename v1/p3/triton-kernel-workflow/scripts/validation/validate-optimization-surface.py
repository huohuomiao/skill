#!/usr/bin/env python3
"""Validate a reduction design's optimization handoff before kernel generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from json_schema import SchemaValidationError, validate
from validation_common import load_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-info", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--intent", choices=("handoff-to-tuning", "standalone"), required=True
    )
    return parser.parse_args()


def _has_reduction(base_info: dict) -> bool:
    reduce_axes = base_info.get("reduce_axes")
    return reduce_axes not in (None, "", [], {})


def _loop_contracts(kernel: dict) -> list[dict]:
    return [
        value
        for key, value in kernel.items()
        if (key == "reduce_loop" or key.startswith("reduce_loop_pass"))
        and isinstance(value, dict)
    ]


def _validate_handoff(spec: dict, surface: dict) -> None:
    if surface["baseline_form"] != "chunked-reduction-loop":
        raise ValueError("tuning handoff requires a chunked reduction baseline")

    kernel = spec.get("kernel")
    if not isinstance(kernel, dict):
        raise ValueError("step4_code_spec.kernel must be an object")
    loops = _loop_contracts(kernel)
    if not loops:
        raise ValueError("tuning handoff reduction requires reduce_loop or reduce_loop_passN")

    reduction = surface["reduction_axis"]
    block_parameter = reduction["block_parameter"]
    block_params = kernel.get("block_params", {})
    if not isinstance(block_params, dict) or block_parameter not in block_params:
        raise ValueError(f"reduction block parameter is absent from kernel.block_params: {block_parameter}")
    if not any(loop.get("reduce_block") == block_parameter for loop in loops):
        raise ValueError("no reduction loop uses optimization_surface.reduction_axis.block_parameter")

    extent = reduction["extent"]
    candidates = reduction["block_candidates"]
    if extent is not None and extent > 1 and not any(candidate < extent for candidate in candidates):
        raise ValueError("tuning handoff must retain at least one reduction tile below the full extent")

    parallel_parameter = surface["parallel_block_parameter"]
    if parallel_parameter is not None and parallel_parameter not in block_params:
        raise ValueError(
            f"parallel block parameter is absent from kernel.block_params: {parallel_parameter}"
        )

    if surface["operator_pattern"] == "softmax-style":
        if surface["passes"] != ["max", "sum", "normalize-store"]:
            raise ValueError("softmax-style handoff requires ordered max/sum/normalize-store passes")
        if len(loops) < 3:
            raise ValueError("softmax-style handoff requires three explicit loop-pass contracts")
        operations = [loop.get("operation") for loop in loops[:3]]
        if operations != surface["passes"]:
            raise ValueError("softmax loop-pass operations do not match the declared pass order")
        if 4 not in surface["parallel_block_candidates"]:
            raise ValueError("softmax-style handoff must retain parallel block candidate 4")
        stages = surface["autotune"]["num_stages_candidates"]
        if 1 not in stages or 3 not in stages:
            raise ValueError("softmax-style handoff must allow num_stages candidates 1 and 3")


def main() -> int:
    args = _arguments()
    try:
        base_info = load_json(args.base_info.resolve())
        spec = load_json(args.spec.resolve())
        surface = spec.get("optimization_surface")
        has_reduction = _has_reduction(base_info)
        if not has_reduction:
            if surface is None:
                print(json.dumps({"status": "pass", "reason": "non-reduction"}))
                return 0
        if has_reduction and surface is None and args.intent == "standalone":
            print(json.dumps({"status": "pass", "reason": "standalone-reduction"}))
            return 0
        if not isinstance(surface, dict):
            raise ValueError("reduction step4_code_spec requires optimization_surface")

        schema_path = (
            Path(__file__).resolve().parents[2]
            / "references"
            / "schemas"
            / "optimization-surface.schema.json"
        )
        validate(surface, load_json(schema_path), schema_path)
        if surface["intent"] != args.intent:
            raise ValueError(
                f"optimization_surface intent {surface['intent']} does not match {args.intent}"
            )
        if args.intent == "handoff-to-tuning":
            _validate_handoff(spec, surface)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "intent": args.intent,
                    "operator_pattern": surface["operator_pattern"],
                }
            )
        )
        return 0
    except (KeyError, OSError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"validate-optimization-surface: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
