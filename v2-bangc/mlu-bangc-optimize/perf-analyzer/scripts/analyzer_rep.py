#!/usr/bin/env python3
"""Best-effort parser for CNPerf text while preserving format uncertainty.

The raw report remains authoritative. This parser intentionally accepts loose
``key: value unit`` and ``key: unit value`` layouts and compound durations.
Unknown rows are counted, not silently reinterpreted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
COMPOUND_TIME_RE = re.compile(
    rf"(?:(?P<s>{NUMBER})\s*s\s*)?"
    rf"(?:(?P<ms>{NUMBER})\s*ms\s*)?"
    rf"(?:(?P<us>{NUMBER})\s*(?:us|µs)\s*)?"
    rf"(?:(?P<ns>{NUMBER})\s*ns\s*)?$",
    re.IGNORECASE,
)
SIMPLE_TIME_RE = re.compile(
    rf"^(?P<value>{NUMBER})\s*(?P<unit>s|ms|us|µs|ns)$", re.IGNORECASE
)
KEY_VALUE_RE = re.compile(r"^\s*([^:#][^:]*)\s*:\s*(.*?)\s*$")
VALUE_UNIT_RE = re.compile(rf"^({NUMBER})\s*([A-Za-zµ/%][\wµ/%.-]*)?$")
UNIT_VALUE_RE = re.compile(rf"^([A-Za-zµ/%][\wµ/%.-]*)\s+({NUMBER})$")


def duration_ns(text: str) -> float | None:
    value = text.strip()
    simple = SIMPLE_TIME_RE.fullmatch(value)
    if simple:
        amount = float(simple.group("value"))
        unit = simple.group("unit").lower().replace("µ", "u")
        return amount * {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}[unit]

    compound = COMPOUND_TIME_RE.fullmatch(value)
    if not compound or not any(compound.groupdict().values()):
        return None
    factors = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
    total = 0.0
    for name, raw in compound.groupdict().items():
        if raw is not None:
            total += float(raw) * factors[name]
    return total


def numeric_field(text: str) -> dict[str, Any] | None:
    value = text.strip().replace(",", "")
    match = VALUE_UNIT_RE.fullmatch(value)
    if match:
        return {"value": float(match.group(1)), "unit": match.group(2)}
    match = UNIT_VALUE_RE.fullmatch(value)
    if match:
        return {"value": float(match.group(2)), "unit": match.group(1)}
    return None


def parse_text(text: str) -> dict[str, Any]:
    durations: list[dict[str, Any]] = []
    counters: list[dict[str, Any]] = []
    unparsed_key_values: list[dict[str, str]] = []

    for line_number, line in enumerate(text.splitlines(), 1):
        match = KEY_VALUE_RE.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        raw_value = match.group(2).strip()

        ns = duration_ns(raw_value)
        if ns is not None and any(token in key.lower() for token in ("duration", "time", "latency")):
            durations.append(
                {"line": line_number, "key": key, "raw": raw_value, "nanoseconds": ns}
            )
            continue

        field = numeric_field(raw_value)
        if field is not None:
            counters.append(
                {
                    "line": line_number,
                    "key": key,
                    "raw": raw_value,
                    **field,
                }
            )
        else:
            unparsed_key_values.append(
                {"line": line_number, "key": key, "raw": raw_value}
            )

    return {
        "schema_version": 1,
        "parser_mode": "best_effort",
        "durations": durations,
        "numeric_counters": counters,
        "unparsed_key_values": unparsed_key_values,
        "counts": {
            "durations": len(durations),
            "numeric_counters": len(counters),
            "unparsed_key_values": len(unparsed_key_values),
        },
        "warning": "Raw CNPerf output is authoritative; missing fields are not zero.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Raw CNPerf text report")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    result = parse_text(args.input.read_text(encoding="utf-8", errors="replace"))
    result["source"] = str(args.input.resolve())
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
