#!/usr/bin/env python3
"""Run one local tuning command within the remaining p3 wall-time budget."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-state", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--label", default="local-command")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def _manager(*arguments: str) -> subprocess.CompletedProcess[str]:
    manager = Path(__file__).resolve().parents[1] / "state" / "manage-tuning-budget.py"
    return subprocess.run(
        [sys.executable, str(manager), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = _arguments()
    try:
        state_path = args.budget_state.resolve()
        workdir = args.workdir.resolve()
        if not state_path.is_file():
            raise ValueError(f"budget state does not exist: {state_path}")
        if not workdir.is_dir():
            raise ValueError(f"workdir does not exist: {workdir}")
        if not args.command:
            raise ValueError("command is required after --")

        checked = _manager("check", "--state", str(state_path))
        if checked.returncode == 3:
            print(f"run-budgeted-local: budget exhausted: {checked.stdout.strip()}", file=sys.stderr)
            return 4
        if checked.returncode != 0:
            raise ValueError(checked.stderr.strip() or "cannot check tuning budget")
        status = json.loads(checked.stdout)
        remaining = int(status["remaining_seconds"])
        if remaining <= 0:
            return 4

        try:
            completed = subprocess.run(
                args.command,
                cwd=workdir,
                check=False,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            _manager("stop", "--state", str(state_path), "--reason", "elapsed-limit")
            print(
                f"run-budgeted-local: wall-time budget expired during {args.label}",
                file=sys.stderr,
            )
            return 4
        return completed.returncode
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"run-budgeted-local: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
