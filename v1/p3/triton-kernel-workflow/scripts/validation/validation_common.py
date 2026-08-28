#!/usr/bin/env python3
"""Shared report and fingerprint helpers for layered validation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


REPORT_SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def skill_fingerprint(skill_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(skill_root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(skill_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def make_check(name: str, status: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "details": details}


def make_report(
    *,
    level: str,
    started_at: str,
    started_counter: float,
    fingerprint: str,
    checks: list[dict[str, Any]],
    hardware_evidence: bool = False,
) -> dict[str, Any]:
    status = "fail" if any(item["status"] == "fail" for item in checks) else "pass"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "level": level,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(perf_counter() - started_counter, 6),
        "skill_fingerprint": fingerprint,
        "hardware_evidence": hardware_evidence,
        "checks": checks,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def strict_json_loads(content: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    return json.loads(content, parse_constant=reject_constant)


def load_json(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def finish(report: dict[str, Any], report_path: Path | None) -> int:
    if report_path is not None:
        write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1
