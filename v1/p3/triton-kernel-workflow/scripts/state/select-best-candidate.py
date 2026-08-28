#!/usr/bin/env python3
"""Select the fastest accuracy-passing candidate under comparable conditions."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FIELDS = {
    "schema_version",
    "candidate_id",
    "source_stage",
    "code_path",
    "report_path",
    "accuracy_pass",
    "latency_ms",
    "execution_backend",
    "hardware_model",
    "benchmark_signature",
}


class CandidateError(ValueError):
    """Raised when candidate metadata cannot support a safe selection."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", action="append", default=[])
    parser.add_argument("--candidate-manifest", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _manifest_paths(args: argparse.Namespace) -> list[Path]:
    paths = {Path(value).resolve() for value in args.candidate_manifest}
    for root_value in args.candidate_root:
        root = Path(root_value).resolve()
        if not root.is_dir():
            raise CandidateError(f"candidate root does not exist: {root}")
        paths.update(root.rglob("candidate.json"))
    if not paths:
        raise CandidateError("no candidate manifests were provided or discovered")
    return sorted(paths)


def _resolve_file(manifest_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CandidateError(f"{manifest_path}: {field} must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise CandidateError(f"{manifest_path}: {field} does not exist: {path}")
    return path


def _load_candidate(manifest_path: Path) -> dict:
    try:
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateError(f"cannot read {manifest_path}: {exc}") from exc

    if not isinstance(candidate, dict):
        raise CandidateError(f"{manifest_path}: candidate must be a JSON object")
    missing = REQUIRED_FIELDS - candidate.keys()
    if missing:
        raise CandidateError(f"{manifest_path}: missing fields: {sorted(missing)}")
    if candidate["schema_version"] != "1.0":
        raise CandidateError(f"{manifest_path}: unsupported schema_version")
    if candidate["source_stage"] not in {"baseline", "oob", "advanced"}:
        raise CandidateError(f"{manifest_path}: invalid source_stage")
    if candidate["execution_backend"] not in {"local", "worker"}:
        raise CandidateError(f"{manifest_path}: invalid execution_backend")
    if not isinstance(candidate["accuracy_pass"], bool):
        raise CandidateError(f"{manifest_path}: accuracy_pass must be boolean")
    latency = candidate["latency_ms"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency <= 0
    ):
        raise CandidateError(f"{manifest_path}: latency_ms must be finite and positive")
    for field in ("candidate_id", "hardware_model", "benchmark_signature"):
        if not isinstance(candidate[field], str) or not candidate[field]:
            raise CandidateError(f"{manifest_path}: {field} must be non-empty")

    candidate["code_path"] = str(_resolve_file(manifest_path, candidate["code_path"], "code_path"))
    candidate["report_path"] = str(
        _resolve_file(manifest_path, candidate["report_path"], "report_path")
    )
    candidate["_manifest_path"] = str(manifest_path)
    return candidate


def _select(candidates: list[dict]) -> tuple[dict, list[dict]]:
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    duplicate_ids = sorted(
        candidate_id
        for candidate_id in set(candidate_ids)
        if candidate_ids.count(candidate_id) > 1
    )
    if duplicate_ids:
        raise CandidateError(f"duplicate candidate_id values: {duplicate_ids}")

    eligible = [candidate for candidate in candidates if candidate["accuracy_pass"]]
    if not eligible:
        raise CandidateError("no accuracy-passing optimization candidate")

    comparison_keys = {
        (
            candidate["execution_backend"],
            candidate["hardware_model"],
            candidate["benchmark_signature"],
        )
        for candidate in eligible
    }
    if len(comparison_keys) != 1:
        raise CandidateError(
            "accuracy-passing candidates use incompatible execution or benchmark conditions"
        )

    eligible.sort(key=lambda item: (float(item["latency_ms"]), item["candidate_id"]))
    return eligible[0], eligible


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    try:
        args = _parse_args()
        candidates = [_load_candidate(path) for path in _manifest_paths(args)]
        selected, eligible = _select(candidates)
        output_dir = Path(args.output_dir).resolve()
        best_code_path = output_dir / "best_so_far.py"
        final_code_path = output_dir / "triton_optimized.py"
        source_code_path = Path(selected["code_path"])
        _atomic_copy(source_code_path, best_code_path)
        _atomic_copy(source_code_path, final_code_path)

        selected_public = {
            key: value for key, value in selected.items() if key != "_manifest_path"
        }
        metadata = {
            "schema_version": "1.0",
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "selected_candidate": selected_public,
            "compared_candidates": [item["candidate_id"] for item in eligible],
            "best_code_path": str(best_code_path),
            "final_code_path": str(final_code_path),
        }
        _atomic_write_json(output_dir / "best_so_far.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False))
        return 0
    except CandidateError as exc:
        print(f"select-best-candidate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
