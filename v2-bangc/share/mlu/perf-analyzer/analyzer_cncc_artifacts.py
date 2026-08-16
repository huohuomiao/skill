#!/usr/bin/env python3
"""Inventory CNCC assembly, MLISA, compiler IR, object, and binary artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


ASSEMBLY_SUFFIXES = {".s", ".asm", ".mlisa"}
IR_SUFFIXES = {".ll", ".bc", ".mlui"}
BINARY_SUFFIXES = {".o", ".cnbin", ".fatbin", ".cnfatbin"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="artifact files or directories")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def iter_files(paths: list[Path]):
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                yield candidate.resolve()


def read_text_prefix(path: Path, limit: int = 131072) -> str:
    try:
        data = path.read_bytes()[:limit]
    except OSError:
        return ""
    if b"\x00" in data[:4096]:
        return ""
    return data.decode("utf-8", errors="replace")


def classify(path: Path) -> tuple[str | None, list[str]]:
    suffix = path.suffix.lower()
    markers: list[str] = []
    if suffix in ASSEMBLY_SUFFIXES:
        text = read_text_prefix(path)
        lowered = text.lower()
        if ".mlisa" in lowered:
            markers.append(".mlisa")
        if ".arch" in lowered:
            markers.append(".arch")
        if "cambricon" in lowered or "bang" in lowered:
            markers.append("cambricon_or_bang")
        return "assembly_or_mlisa", markers
    if suffix in IR_SUFFIXES:
        return "compiler_ir", markers
    if suffix in BINARY_SUFFIXES:
        return "binary_or_object", markers
    return None, markers


def assembly_facts(path: Path) -> dict[str, str]:
    text = read_text_prefix(path)
    facts: dict[str, str] = {}
    mlisa = re.search(r"(?m)^\s*\.mlisa\s+([^\s#]+)", text)
    arch = re.search(r"(?m)^\s*\.arch\s+([^\s#]+)", text)
    if mlisa:
        facts["mlisa_version"] = mlisa.group(1)
    if arch:
        facts["arch"] = arch.group(1)
    if "CNCC MLISA Back-End" in text:
        facts["producer"] = "CNCC MLISA Back-End"
    return facts


def main() -> int:
    args = parse_args()
    artifacts: list[dict[str, object]] = []
    for path in iter_files(args.paths):
        kind, markers = classify(path)
        if not kind:
            continue
        try:
            stat = path.stat()
            size = stat.st_size
            modified_ns = stat.st_mtime_ns
        except OSError:
            size = None
            modified_ns = None
        artifacts.append(
            {
                "path": str(path),
                "kind": kind,
                "suffix": path.suffix,
                "size_bytes": size,
                "modified_ns": modified_ns,
                "markers": markers,
                "facts": assembly_facts(path) if kind == "assembly_or_mlisa" else {},
            }
        )

    artifacts.sort(key=lambda item: (str(item["kind"]), str(item["path"])))
    counts: dict[str, int] = {}
    for artifact in artifacts:
        kind = str(artifact["kind"])
        counts[kind] = counts.get(kind, 0) + 1

    payload = {
        "status": "AVAILABLE" if artifacts else "UNAVAILABLE",
        "searched_paths": [str(path.expanduser()) for path in args.paths],
        "counts": counts,
        "artifacts": artifacts,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    print("CNCC_ARTIFACTS_JSON=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + os.linesep, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
