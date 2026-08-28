#!/usr/bin/env python3
"""Create and update the fixed p3 performance-tuning budget state."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(VALIDATION_DIR))

from json_schema import SchemaValidationError, validate  # noqa: E402
from validation_common import load_json, utc_now, write_json  # noqa: E402


LIMITS = {
    "max_deep_rounds": 3,
    "max_worker_calls": 16,
    "max_elapsed_seconds": 1800,
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("init", "check", "reserve-worker", "start-round", "stop", "complete", "status"),
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--mode", choices=("balanced", "max-performance"))
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--reason")
    return parser.parse_args()


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _elapsed_seconds(state: dict) -> int:
    return max(0, int((datetime.now(timezone.utc) - _parse_time(state["started_at"])).total_seconds()))


def _event(state: dict, event: str, label: str) -> None:
    state["events"].append({"at": utc_now(), "event": event, "label": label})
    state["updated_at"] = utc_now()


def _stop(state: dict, reason: str) -> None:
    if state["status"] == "active":
        state["status"] = "stopped"
        state["stop_reason"] = reason
        _event(state, "stopped", reason)


def _check_active(state: dict) -> tuple[bool, str | None]:
    if state["status"] != "active":
        return False, state["stop_reason"] or state["status"]
    if _elapsed_seconds(state) >= state["limits"]["max_elapsed_seconds"]:
        _stop(state, "elapsed-limit")
        return False, "elapsed-limit"
    if state["usage"]["worker_calls"] >= state["limits"]["max_worker_calls"]:
        _stop(state, "worker-call-limit")
        return False, "worker-call-limit"
    return True, None


def _validate_state(state: dict, schema_path: Path) -> None:
    validate(state, load_json(schema_path), schema_path)
    if state["usage"]["deep_rounds_started"] > LIMITS["max_deep_rounds"]:
        raise ValueError("deep-round usage exceeds fixed limit")
    if state["usage"]["worker_calls"] > LIMITS["max_worker_calls"]:
        raise ValueError("worker-call usage exceeds fixed limit")


def _public(state: dict) -> dict:
    return {
        **state,
        "elapsed_seconds": _elapsed_seconds(state),
        "remaining_worker_calls": max(0, state["limits"]["max_worker_calls"] - state["usage"]["worker_calls"]),
        "remaining_deep_rounds": max(0, state["limits"]["max_deep_rounds"] - state["usage"]["deep_rounds_started"]),
        "remaining_seconds": max(0, state["limits"]["max_elapsed_seconds"] - _elapsed_seconds(state)),
    }


def main() -> int:
    args = _arguments()
    state_path = args.state.resolve()
    schema_path = Path(__file__).resolve().parents[2] / "references/schemas/tuning-state.schema.json"
    try:
        if args.action == "init":
            if args.mode is None:
                raise ValueError("init requires --mode")
            if state_path.exists():
                raise ValueError(f"budget state already exists: {state_path}")
            now = utc_now()
            state = {
                "schema_version": "1.0",
                "optimization_mode": args.mode,
                "status": "active",
                "started_at": now,
                "updated_at": now,
                "limits": dict(LIMITS),
                "usage": {"deep_rounds_started": 0, "worker_calls": 0},
                "stop_reason": None,
                "events": [],
            }
            _event(state, "initialized", args.mode)
            exit_code = 0
        else:
            state = load_json(state_path)
            _validate_state(state, schema_path)
            exit_code = 0
            if args.action in {"check", "status", "reserve-worker", "start-round"}:
                active, _ = _check_active(state)
                if not active:
                    exit_code = 3
            if exit_code == 0 and args.action == "reserve-worker":
                state["usage"]["worker_calls"] += 1
                _event(state, "worker-call", args.label)
            elif exit_code == 0 and args.action == "start-round":
                if state["optimization_mode"] != "max-performance":
                    raise ValueError("deep rounds require max-performance mode")
                if state["usage"]["deep_rounds_started"] >= state["limits"]["max_deep_rounds"]:
                    _stop(state, "deep-round-limit")
                    exit_code = 3
                else:
                    state["usage"]["deep_rounds_started"] += 1
                    _event(state, "deep-round", args.label)
            elif args.action == "stop":
                if not args.reason:
                    raise ValueError("stop requires --reason")
                _stop(state, args.reason)
            elif args.action == "complete" and state["status"] == "active":
                state["status"] = "completed"
                state["stop_reason"] = args.reason
                _event(state, "completed", args.reason or args.label)

        _validate_state(state, schema_path)
        write_json(state_path, state)
        print(json.dumps(_public(state), ensure_ascii=False))
        return exit_code
    except (KeyError, OSError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"manage-tuning-budget: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

