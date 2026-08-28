#!/usr/bin/env python3
"""Record, verify, or restore immutable p4 stage cache snapshots."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"
sys.path.insert(0, str(VALIDATION_DIR))

from json_schema import SchemaValidationError, validate  # noqa: E402
from validation_common import load_json, utc_now, write_json  # noqa: E402


STAGES = (
    "environment",
    "requirement-extraction",
    "code-generation",
    "code-validation",
    "performance-tuning",
    "finalization",
)
REQUIRED_ARTIFACT_KEYS = {
    "environment": {"config", "runtime_info", "report"},
    "requirement-extraction": {"requirement"},
    "code-generation": {"code", "report"},
    "code-validation": {"code", "report"},
    "performance-tuning": {"strategy_plan", "tuning_state", "best", "code", "report"},
    "finalization": {"final_code", "summary"},
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("record", "verify", "restore"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fingerprint-file", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--fingerprint")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--artifact", action="append", default=[], help="key=path; record only")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes output_dir: {resolved}") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _pairs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--artifact requires key=path: {value}")
        key, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key) or key in result:
            raise ValueError(f"invalid or duplicate artifact key: {key}")
        result[key] = Path(raw_path)
    if not result:
        raise ValueError("record requires at least one --artifact")
    return result


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _validate_stage_artifacts(
    root: Path,
    skill_root: Path,
    stage: str,
    artifacts: dict[str, Path],
    check_live_links: bool,
) -> None:
    code_key = "final_code" if stage == "finalization" else "code"
    if code_key in artifacts:
        try:
            ast.parse(artifacts[code_key].read_text(encoding="utf-8"), filename=str(artifacts[code_key]))
        except (SyntaxError, UnicodeError) as exc:
            raise ValueError(f"cached code is not valid Python: {artifacts[code_key]}") from exc

    schema_by_key = {}
    if stage == "environment":
        schema_by_key["config"] = skill_root / "references/schemas/env-config.schema.json"
    elif stage == "performance-tuning":
        schema_by_key = {
            "strategy_plan": skill_root / "references/schemas/strategy-plan.schema.json",
            "tuning_state": skill_root / "references/schemas/tuning-state.schema.json",
            "best": skill_root / "references/schemas/best-candidate.schema.json",
        }
    loaded: dict[str, dict] = {}
    for key, schema_path in schema_by_key.items():
        payload = load_json(artifacts[key])
        validate(payload, load_json(schema_path), schema_path)
        loaded[key] = payload

    if stage == "environment" and check_live_links:
        runtime_info = _inside(
            root, Path(loaded["config"]["runtime_info_path"]), "environment runtime info"
        )
        if runtime_info != artifacts["runtime_info"].resolve():
            raise ValueError("environment config runtime_info_path does not match cached artifact")

    if stage == "performance-tuning":
        if loaded["tuning_state"]["status"] == "active":
            raise ValueError("completed performance cache cannot contain an active tuning budget")
        if loaded["strategy_plan"]["optimization_mode"] != loaded["tuning_state"]["optimization_mode"]:
            raise ValueError("strategy plan and tuning state modes do not match")
        if loaded["best"]["selected_candidate"]["accuracy_pass"] is not True:
            raise ValueError("performance cache selected candidate did not pass accuracy")
        if check_live_links:
            final_code = _inside(root, Path(loaded["best"]["final_code_path"]), "best final code")
            if not final_code.is_file() or _sha256(final_code) != _sha256(artifacts["code"]):
                raise ValueError("best checkpoint final code does not match cached code artifact")


def _record_path(args: argparse.Namespace, root: Path) -> Path:
    if args.record:
        return _inside(root, args.record, "record")
    if not args.stage or not args.fingerprint:
        raise ValueError("verify/restore requires --record or both --stage and --fingerprint")
    if len(args.fingerprint) != 64:
        raise ValueError("--fingerprint must be a SHA-256 string")
    return root / "RunState" / "cache" / args.stage / args.fingerprint / "record.json"


def _verify(
    root: Path,
    skill_root: Path,
    record_path: Path,
    schema_path: Path,
    fingerprint_schema_path: Path,
) -> tuple[dict, dict[str, Path]]:
    if not record_path.is_file():
        raise ValueError(f"cache record is missing: {record_path}")
    record = load_json(record_path)
    validate(record, load_json(schema_path), schema_path)
    missing_keys = REQUIRED_ARTIFACT_KEYS[record["stage"]] - set(record["artifacts"])
    if missing_keys:
        raise ValueError(f"cache record is missing required artifact keys: {sorted(missing_keys)}")
    if len(record["fingerprint"]) != 64:
        raise ValueError("cache record fingerprint is not SHA-256")
    fingerprint_path = _inside(root, root / record["fingerprint_path"], "fingerprint snapshot")
    fingerprint_payload = load_json(fingerprint_path)
    validate(
        fingerprint_payload,
        load_json(fingerprint_schema_path),
        fingerprint_schema_path,
    )
    recomputed = _canonical_hash(
        {"stage": fingerprint_payload["stage"], "factors": fingerprint_payload["factors"]}
    )
    if recomputed != fingerprint_payload["fingerprint"]:
        raise ValueError("fingerprint snapshot digest is invalid")
    if fingerprint_payload.get("fingerprint") != record["fingerprint"]:
        raise ValueError("fingerprint snapshot does not match cache record")
    if fingerprint_payload.get("stage") != record["stage"]:
        raise ValueError("fingerprint snapshot stage does not match cache record")
    snapshots: dict[str, Path] = {}
    for key, artifact in record["artifacts"].items():
        snapshot = _inside(root, root / artifact["cache_path"], f"cache artifact {key}")
        _inside(root, root / artifact["target_path"], f"target artifact {key}")
        if not snapshot.is_file() or snapshot.stat().st_size == 0:
            raise ValueError(f"cache artifact is missing or empty: {snapshot}")
        if snapshot.stat().st_size != artifact["size_bytes"] or _sha256(snapshot) != artifact["sha256"]:
            raise ValueError(f"cache artifact hash mismatch: {snapshot}")
        snapshots[key] = snapshot
    if not snapshots:
        raise ValueError("cache record has no artifacts")
    _validate_stage_artifacts(root, skill_root, record["stage"], snapshots, False)
    if record["stage"] == "environment":
        config = load_json(snapshots["config"])
        runtime_target = _inside(
            root,
            root / record["artifacts"]["runtime_info"]["target_path"],
            "environment runtime target",
        )
        if Path(config["runtime_info_path"]).resolve() != runtime_target:
            raise ValueError("cached environment config does not match the current output root")
    elif record["stage"] == "performance-tuning":
        best = load_json(snapshots["best"])
        code_target = _inside(
            root,
            root / record["artifacts"]["code"]["target_path"],
            "performance code target",
        )
        if Path(best["final_code_path"]).resolve() != code_target:
            raise ValueError("cached best checkpoint does not match the current output root")
    return record, snapshots


def _record(
    args: argparse.Namespace,
    root: Path,
    schema_path: Path,
    fingerprint_schema_path: Path,
) -> tuple[dict, Path | None]:
    if args.fingerprint_file is None:
        raise ValueError("record requires --fingerprint-file")
    fingerprint_source = args.fingerprint_file.resolve()
    fingerprint_payload = load_json(fingerprint_source)
    validate(
        fingerprint_payload,
        load_json(fingerprint_schema_path),
        fingerprint_schema_path,
    )
    stage = fingerprint_payload.get("stage")
    fingerprint = fingerprint_payload.get("fingerprint")
    if stage not in STAGES or not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("invalid stage fingerprint file")
    recomputed = _canonical_hash(
        {"stage": fingerprint_payload["stage"], "factors": fingerprint_payload["factors"]}
    )
    if recomputed != fingerprint:
        raise ValueError("stage fingerprint digest is invalid")
    sources = _pairs(args.artifact)
    missing_keys = REQUIRED_ARTIFACT_KEYS[stage] - set(sources)
    if missing_keys:
        raise ValueError(f"record is missing required artifact keys: {sorted(missing_keys)}")
    resolved_sources = {
        key: _inside(root, raw_source, f"artifact {key}")
        for key, raw_source in sources.items()
    }
    for key, source in resolved_sources.items():
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"artifact is missing or empty: {source}")
    _validate_stage_artifacts(
        root, Path(__file__).resolve().parents[2], stage, resolved_sources, True
    )
    entry = root / "RunState" / "cache" / stage / fingerprint
    record_path = entry / "record.json"
    quarantined = None
    if record_path.exists():
        try:
            existing, _ = _verify(
                root,
                Path(__file__).resolve().parents[2],
                record_path,
                schema_path,
                fingerprint_schema_path,
            )
        except (KeyError, OSError, TypeError, ValueError, SchemaValidationError, json.JSONDecodeError):
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            quarantined = (
                root
                / "RunState"
                / "quarantine"
                / stage
                / f"{fingerprint}-{suffix}-{uuid.uuid4().hex[:8]}"
            )
            quarantined.parent.mkdir(parents=True, exist_ok=True)
            entry.replace(quarantined)
        else:
            if set(sources) != set(existing["artifacts"]):
                raise ValueError("immutable cache entry already exists with different artifact keys")
            for key, raw_source in sources.items():
                source = _inside(root, raw_source, f"artifact {key}")
                stored = existing["artifacts"][key]
                if (
                    not source.is_file()
                    or source.stat().st_size != stored["size_bytes"]
                    or _sha256(source) != stored["sha256"]
                    or _relative(root, source) != stored["target_path"]
                ):
                    raise ValueError(
                        "immutable cache entry already exists with different artifact content"
                    )
            return existing, None

    artifacts: dict[str, dict] = {}
    for key, raw_source in sorted(sources.items()):
        source = resolved_sources[key]
        snapshot = entry / "artifacts" / key
        _atomic_copy(source, snapshot)
        artifacts[key] = {
            "target_path": _relative(root, source),
            "cache_path": _relative(root, snapshot),
            "sha256": _sha256(snapshot),
            "size_bytes": snapshot.stat().st_size,
        }

    fingerprint_snapshot = entry / "fingerprint.json"
    _atomic_copy(fingerprint_source, fingerprint_snapshot)
    record = {
        "schema_version": "1.0",
        "stage": stage,
        "fingerprint": fingerprint,
        "created_at": utc_now(),
        "fingerprint_path": _relative(root, fingerprint_snapshot),
        "artifacts": artifacts,
    }
    validate(record, load_json(schema_path), schema_path)
    write_json(record_path, record)
    _verify(
        root,
        Path(__file__).resolve().parents[2],
        record_path,
        schema_path,
        fingerprint_schema_path,
    )
    return record, quarantined


def main() -> int:
    args = _arguments()
    try:
        root = args.output_dir.resolve()
        if args.action == "record":
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            raise ValueError(f"output_dir does not exist: {root}")
        skill_root = Path(__file__).resolve().parents[2]
        schema_path = skill_root / "references/schemas/stage-cache-record.schema.json"
        fingerprint_schema_path = skill_root / "references/schemas/stage-fingerprint.schema.json"
        if args.action == "record":
            record, quarantined = _record(args, root, schema_path, fingerprint_schema_path)
            record_path = (
                root / "RunState" / "cache" / record["stage"] / record["fingerprint"] / "record.json"
            )
            result = {
                "status": "recorded",
                "record_path": str(record_path),
                "quarantined_path": str(quarantined) if quarantined else None,
                "record": record,
            }
        else:
            record_path = _record_path(args, root)
            record, snapshots = _verify(
                root, skill_root, record_path, schema_path, fingerprint_schema_path
            )
            if args.action == "restore":
                for key, snapshot in snapshots.items():
                    target = _inside(root, root / record["artifacts"][key]["target_path"], f"target artifact {key}")
                    _atomic_copy(snapshot, target)
                status = "restored"
            else:
                status = "valid"
            result = {
                "status": status,
                "record_path": str(record_path),
                "stage": record["stage"],
                "fingerprint": record["fingerprint"],
                "artifacts": {
                    key: str(_inside(root, root / item["target_path"], f"target artifact {key}"))
                    for key, item in record["artifacts"].items()
                },
            }
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (KeyError, OSError, TypeError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"stage-cache: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
