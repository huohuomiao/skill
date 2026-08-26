#!/usr/bin/env python3
"""Collect MLU device facts with cnmon and require an MLU590 by default."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEVICE_RE = re.compile(
    r"\|\s*(?P<card>\d+)\s*/\s*(?P<name>MLU[0-9A-Za-z_-]+)"
    r"(?:\s+(?P<firmware>v[0-9.]+))?\s*\|\s*(?P<bus>[0-9A-Fa-f:.]+)?"
)
MEMORY_RE = re.compile(r"(?P<used>\d+)\s*MiB\s*/\s*(?P<total>\d+)\s*MiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-other-mlu",
        action="store_true",
        help="accept any parsed MLU device instead of requiring an MLU590",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args()


def find_executable(name: str, extra: tuple[str, ...] = ()) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in extra:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def run_command(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def parse_devices(output: str) -> list[dict[str, object]]:
    lines = output.splitlines()
    devices: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        match = DEVICE_RE.search(line)
        if not match:
            continue
        memory_used = None
        memory_total = None
        for nearby in lines[index : min(index + 3, len(lines))]:
            memory = MEMORY_RE.search(nearby)
            if memory:
                memory_used = int(memory.group("used"))
                memory_total = int(memory.group("total"))
                break
        devices.append(
            {
                "card_id": int(match.group("card")),
                "device_name": match.group("name"),
                "firmware": match.group("firmware") or None,
                "bus_id": match.group("bus") or None,
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
            }
        )
    return devices


def main() -> int:
    args = parse_args()
    cnmon = find_executable("cnmon", ("/usr/bin/cnmon", "/usr/local/bin/cnmon"))
    if not cnmon:
        print("ERROR: cnmon not found in PATH or standard executable locations", file=sys.stderr)
        return 1

    try:
        result = run_command([cnmon], args.timeout)
    except subprocess.TimeoutExpired:
        print(f"ERROR: cnmon timed out after {args.timeout:g}s", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: failed to execute cnmon: {exc}", file=sys.stderr)
        return 1

    print("=== CNMON RAW OUTPUT ===")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print("=== CNMON STDERR ===", file=sys.stderr)
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        print(f"ERROR: cnmon exited with code {result.returncode}", file=sys.stderr)
        return 1

    devices = parse_devices(result.stdout)
    if not devices:
        print("ERROR: cnmon returned no device row that this script could parse", file=sys.stderr)
        return 1

    target_devices = [d for d in devices if "MLU590" in str(d["device_name"]).upper()]
    accepted = devices if args.allow_other_mlu else target_devices
    payload = {
        "cnmon_path": cnmon,
        "required_device": "any MLU" if args.allow_other_mlu else "MLU590",
        "devices": devices,
        "accepted_device_count": len(accepted),
        "mlu_visible_devices": os.environ.get("MLU_VISIBLE_DEVICES"),
    }
    print("=== PARSED DEVICE INFO ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("DEVICE_INFO_JSON=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    for device in devices:
        used = device.get("memory_used_mib")
        if isinstance(used, int) and used > 0:
            print(
                f"WARNING: card {device['card_id']} already uses {used} MiB; "
                "performance measurements may be noisy",
                file=sys.stderr,
            )

    if not accepted:
        names = ", ".join(str(d["device_name"]) for d in devices)
        print(f"ERROR: target MLU590 not found; detected: {names}", file=sys.stderr)
        return 1
    print(f"MLU_DEVICE_CHECK_PASS: selected {accepted[0]['device_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
