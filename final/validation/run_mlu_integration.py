#!/usr/bin/env python3
"""Run the L3 local-MLU preflight and an optional operator smoke test."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def run_python(path: Path, timeout_sec: int) -> int:
    print(f"[RUN] {path}", flush=True)
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"[FAIL] timeout after {timeout_sec}s: {path}", file=sys.stderr)
        return 124
    if completed.returncode != 0:
        print(f"[FAIL] exit={completed.returncode}: {path}", file=sys.stderr)
    else:
        print(f"[PASS] {path}")
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator",
        type=Path,
        help="Optional generated Triton .py file to execute after the environment checks pass",
    )
    parser.add_argument("--env-timeout-sec", type=int, default=600)
    parser.add_argument("--operator-timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    checks = (
        ROOT / "share" / "mlu" / "runtime" / "get_device_info.py",
        ROOT / "share" / "mlu" / "runtime" / "test_env_code.py",
    )
    for check in checks:
        if not check.is_file():
            print(f"[FAIL] missing environment check: {check}", file=sys.stderr)
            return 2
        code = run_python(check, args.env_timeout_sec)
        if code != 0:
            return code

    if args.operator is not None:
        operator = args.operator
        if not operator.is_absolute():
            operator = (Path.cwd() / operator).resolve()
        if not operator.is_file() or operator.suffix != ".py":
            print(f"[FAIL] operator must be an existing .py file: {operator}", file=sys.stderr)
            return 2
        return run_python(operator, args.operator_timeout_sec)

    print("[PASS] L3 local MLU environment preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
