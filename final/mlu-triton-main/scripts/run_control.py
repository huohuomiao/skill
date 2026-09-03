#!/usr/bin/env python3
"""Content-addressed cache and resumable stage controller for MLU Triton runs.

The controller is intentionally standard-library-only. It owns the outer workflow
manifest; individual Skills still own their stage artifacts and business logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE_CONFIG_PATH = ROOT / "mlu-triton-main" / "references" / "stage-sources.json"
MANIFEST_NAME = "run_manifest.json"
MODES = ("correctness", "balanced", "max-performance")
TERMINAL_SUCCESS = {"complete", "cached", "skipped"}
VALIDATION_LEVELS = ("l1", "l2", "l3")


class ControlError(ValueError):
    """An expected, user-actionable controller error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ControlError(f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def ensure_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ControlError(f"{label} must stay inside {root}: {resolved}") from exc
    return resolved


def safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ControlError(f"unsafe relative artifact path: {value!r}")
    return path


def load_stage_config() -> dict[str, Any]:
    config = read_json(STAGE_CONFIG_PATH)
    stages = config.get("stages")
    if config.get("version") != 2 or not isinstance(stages, list) or not stages:
        raise ControlError(f"unsupported stage config: {STAGE_CONFIG_PATH}")
    names = [item.get("name") for item in stages if isinstance(item, dict)]
    if len(names) != len(stages) or len(names) != len(set(names)):
        raise ControlError("stage names must be non-empty and unique")
    seen: set[str] = set()
    for stage in stages:
        dependencies = stage.get("dependencies", [])
        if not isinstance(dependencies, list) or any(dep not in seen for dep in dependencies):
            raise ControlError(f"stage dependencies must point backward: {stage.get('name')}")
        required = stage.get("cache_validation_required", [])
        if not isinstance(required, list) or any(level not in VALIDATION_LEVELS for level in required):
            raise ControlError(f"invalid cache validation requirements: {stage.get('name')}")
        checkpoints = stage.get("checkpoint_validation_required", {})
        if not isinstance(checkpoints, dict):
            raise ControlError(f"invalid checkpoint validation requirements: {stage.get('name')}")
        for checkpoint_name, levels in checkpoints.items():
            if not checkpoint_name or not isinstance(levels, list) or any(
                level not in VALIDATION_LEVELS for level in levels
            ):
                raise ControlError(
                    f"invalid checkpoint validation requirements: {stage.get('name')}/{checkpoint_name}"
                )
        seen.add(stage["name"])
    return config


def normalize_validation_levels(values: list[str] | None) -> list[str]:
    supplied = set(values or [])
    unknown = supplied.difference(VALIDATION_LEVELS)
    if unknown:
        raise ControlError(f"unknown validation levels: {sorted(unknown)}")
    return [level for level in VALIDATION_LEVELS if level in supplied]


def validation_satisfies(actual: list[str], required: list[str]) -> tuple[bool, list[str]]:
    missing = [level for level in required if level not in actual]
    return not missing, missing


def validate_kernel_review_evidence(manifest: dict[str, Any]) -> None:
    """Reject a claimed KernelGen handoff unless its real L3 evidence is usable."""
    output_dir = Path(manifest["output_dir"])
    review_path = ensure_within(
        output_dir / "KernelGen" / "review_result.json", output_dir, "review evidence"
    )
    if not review_path.is_file() or review_path.stat().st_size == 0:
        raise ControlError(f"passing review evidence is required: {review_path}")
    review = read_json(review_path)
    if (
        review.get("schema_version") != 1
        or review.get("validation_level") != "l3"
        or review.get("status") not in {"passed", "repaired"}
        or not isinstance(review.get("attempts"), int)
        or review.get("attempts", 0) < 1
        or not isinstance(review.get("accuracy"), dict)
        or review["accuracy"].get("pass") is not True
    ):
        raise ControlError("review_result.json does not contain passing L3 evidence")
    expected_backend = (manifest.get("run_context") or {}).get("execution_backend")
    if expected_backend and review.get("execution_backend") != expected_backend:
        raise ControlError("review_result.json execution_backend does not match run_context")
    artifacts = review.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ControlError("review_result.json artifacts object is required")
    for key in ("fixed_code", "report"):
        relative = safe_relative(str(artifacts.get(key, "")))
        path = ensure_within(output_dir / relative, output_dir, f"review {key}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ControlError(f"review evidence artifact is missing or empty: {path}")


def validate_dispatch_evidence(manifest: dict[str, Any]) -> None:
    output_dir = Path(manifest["output_dir"])
    metrics_path = ensure_within(
        output_dir / "KernelGen" / "dispatch_metrics.json", output_dir, "dispatch evidence"
    )
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        raise ControlError(f"dispatch evidence is required: {metrics_path}")
    metrics = read_json(metrics_path)
    if (
        metrics.get("schema_version") != 1
        or metrics.get("version") != "final"
        or metrics.get("route") not in {"normal", "triton-fast"}
        or metrics.get("outcome") not in {"direct-pass", "repair"}
        or not isinstance(metrics.get("dispatches"), dict)
        or not isinstance(metrics.get("static_context"), dict)
    ):
        raise ControlError("dispatch_metrics.json does not match the final dispatch contract")


def source_snapshot(patterns: list[str]) -> list[dict[str, str]]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file() and "__pycache__" not in path.parts)
    rows = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.as_posix())
    ]
    if not rows:
        raise ControlError(f"source pattern matched no files: {patterns}")
    return rows


def digest_input(value: str) -> dict[str, str]:
    try:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            resolved = candidate.resolve()
            return {"kind": "file", "reference": str(resolved), "sha256": sha256_file(resolved)}
    except OSError:
        pass
    return {"kind": "literal", "reference": "<inline-request>", "sha256": sha256_bytes(value.encode("utf-8"))}


def digest_optional_file(path: Path | None, label: str) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ControlError(f"{label} does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def append_event(manifest: dict[str, Any], event: str, stage: str | None, detail: str) -> None:
    manifest.setdefault("events", []).append(
        {"timestamp": utc_now(), "event": event, "stage": stage, "detail": detail}
    )


def stage_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in config["stages"] if isinstance(item, dict)]


def config_by_name(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in stage_configs(config)}


def enabled(stage: dict[str, Any], mode: str) -> bool:
    return mode in stage.get("enabled_modes", [])


def compute_fingerprints(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for stage in stage_configs(config):
        name = stage["name"]
        dependency_fingerprints = {dep: fingerprints[dep] for dep in stage.get("dependencies", [])}
        payload: dict[str, Any] = {
            "contract": "mlu-triton-stage-fingerprint-v1",
            "stage": name,
            "stage_contract": stage,
            "stage_config_version": config["version"],
            "input": {
                "kind": manifest["request"]["input"]["kind"],
                "sha256": manifest["request"]["input"]["sha256"],
            },
            "dependencies": dependency_fingerprints,
            "sources": source_snapshot(stage.get("sources", [])),
        }
        if stage.get("mode_sensitive"):
            payload["mode"] = manifest["mode"]
        if stage.get("budget_sensitive"):
            budget = manifest["request"].get("budget_file")
            payload["budget_sha256"] = None if budget is None else budget.get("sha256")
        if stage.get("context_sensitive"):
            context = manifest.get("run_context")
            payload["run_context"] = (
                None
                if not context
                else {
                    "sha256": context.get("sha256"),
                    "execution_backend": context.get("execution_backend"),
                    "hardware_key": context.get("hardware_key"),
                    "toolchain_key": context.get("toolchain_key"),
                }
            )
        fingerprints[name] = sha256_bytes(canonical_json(payload))
    return fingerprints


def downstream_names(config: dict[str, Any], start: str) -> list[str]:
    rows = stage_configs(config)
    index = next((i for i, item in enumerate(rows) if item["name"] == start), None)
    if index is None:
        raise ControlError(f"unknown stage: {start}")
    return [item["name"] for item in rows[index:]]


def reset_stage(row: dict[str, Any], status: str, reason: str) -> None:
    row["status"] = status
    row["reason"] = reason
    row["artifacts"] = []
    row["cache_key"] = None
    row["started_at"] = None
    row["completed_at"] = None
    row["checkpoints"] = {}
    row["validation_levels"] = []


def reopen_stage(row: dict[str, Any], reason: str) -> None:
    """Reopen interrupted work while retaining hash-checked inner checkpoints."""
    row["status"] = "pending"
    row["reason"] = reason
    row["artifacts"] = []
    row["cache_key"] = None
    row["started_at"] = None
    row["completed_at"] = None
    row["validation_levels"] = []


def invalidate_from(manifest: dict[str, Any], config: dict[str, Any], stage: str, reason: str) -> None:
    config_map = config_by_name(config)
    for name in downstream_names(config, stage):
        if enabled(config_map[name], manifest["mode"]):
            reset_stage(manifest["stages"][name], "pending", reason)
        else:
            reset_stage(manifest["stages"][name], "skipped", "disabled_by_mode")
    append_event(manifest, "invalidate", stage, reason)


def artifact_records_valid(manifest: dict[str, Any], stage_name: str) -> tuple[bool, str | None]:
    output_dir = Path(manifest["output_dir"])
    records = manifest["stages"][stage_name].get("artifacts", [])
    if not records:
        return False, "artifact_records_missing"
    for record in records:
        try:
            relative = safe_relative(str(record["path"]))
            path = ensure_within(output_dir / relative, output_dir, "artifact")
        except (ControlError, KeyError, TypeError) as exc:
            return False, f"artifact_record_invalid:{exc}"
        if not path.is_file():
            return False, f"artifact_missing:{relative.as_posix()}"
        if sha256_file(path) != record.get("sha256"):
            return False, f"artifact_hash_mismatch:{relative.as_posix()}"
    return True, None


def refresh_manifest(
    manifest: dict[str, Any], config: dict[str, Any], recover_running: bool = False
) -> dict[str, str]:
    new_fingerprints = compute_fingerprints(manifest, config)
    for stage in stage_configs(config):
        name = stage["name"]
        old = manifest["stages"][name].get("fingerprint")
        if old and old != new_fingerprints[name]:
            invalidate_from(manifest, config, name, "fingerprint_changed")
            break
    for name, fingerprint in new_fingerprints.items():
        manifest["stages"][name]["fingerprint"] = fingerprint

    for stage in stage_configs(config):
        name = stage["name"]
        row = manifest["stages"][name]
        if not enabled(stage, manifest["mode"]):
            if row["status"] != "skipped":
                reset_stage(row, "skipped", "disabled_by_mode")
            continue
        if recover_running and row["status"] in {"running", "failed"}:
            reopen_stage(row, "interrupted_or_failed_stage_reopened")
            append_event(manifest, "recover", name, "stage reopened without resetting inner budget")
        if row["status"] in {"complete", "cached"}:
            valid, reason = artifact_records_valid(manifest, name)
            if not valid:
                invalidate_from(manifest, config, name, reason or "artifact_invalid")
                break
    manifest["updated_at"] = utc_now()
    return new_fingerprints


def cache_entry(manifest: dict[str, Any], stage_name: str) -> Path:
    fingerprint = manifest["stages"][stage_name]["fingerprint"]
    return Path(manifest["cache_dir"]) / stage_name / fingerprint


def cache_metadata_valid(
    entry: Path, stage: dict[str, Any], fingerprint: str
) -> tuple[bool, str | None]:
    stage_name = stage["name"]
    metadata_path = entry / "metadata.json"
    if not metadata_path.is_file():
        return False, "cache_metadata_missing"
    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError, ControlError) as exc:
        return False, f"cache_metadata_invalid:{exc}"
    if metadata.get("schema_version") != 2:
        return False, "cache_metadata_schema_mismatch"
    if metadata.get("stage_config_version") != 2:
        return False, "cache_stage_config_version_mismatch"
    if metadata.get("stage") != stage_name or metadata.get("fingerprint") != fingerprint:
        return False, "cache_identity_mismatch"
    actual_levels = metadata.get("validation_levels")
    if not isinstance(actual_levels, list) or any(
        level not in VALIDATION_LEVELS for level in actual_levels
    ):
        return False, "cache_validation_levels_invalid"
    satisfied, missing = validation_satisfies(
        normalize_validation_levels(actual_levels),
        stage.get("cache_validation_required", []),
    )
    if not satisfied:
        return False, f"cache_validation_missing:{','.join(missing)}"
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False, "cache_artifacts_missing"
    for record in artifacts:
        try:
            relative = safe_relative(str(record["path"]))
            cached = ensure_within(entry / "artifacts" / relative, entry, "cached artifact")
        except (ControlError, KeyError, TypeError) as exc:
            return False, f"cache_artifact_invalid:{exc}"
        if not cached.is_file():
            return False, f"cache_artifact_missing:{relative.as_posix()}"
        if sha256_file(cached) != record.get("sha256"):
            return False, f"cache_hash_mismatch:{relative.as_posix()}"
    return True, None


def dependency_ready(manifest: dict[str, Any], config: dict[str, Any], stage: dict[str, Any]) -> tuple[bool, str | None]:
    for dependency in stage.get("dependencies", []):
        status = manifest["stages"][dependency]["status"]
        if status not in TERMINAL_SUCCESS:
            return False, f"dependency_not_ready:{dependency}:{status}"
    if stage.get("context_sensitive") and not manifest.get("run_context"):
        return False, "run_context_not_bound"
    return True, None


def next_action(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    for stage in stage_configs(config):
        name = stage["name"]
        row = manifest["stages"][name]
        status = row["status"]
        if status in TERMINAL_SUCCESS:
            continue
        if status == "running":
            return {"action": "blocked", "stage": name, "reason": "stage_already_running"}
        if status == "failed":
            return {"action": "blocked", "stage": name, "reason": "run_resume_before_retry"}
        ready, reason = dependency_ready(manifest, config, stage)
        if not ready:
            return {"action": "blocked", "stage": name, "reason": reason}
        if stage.get("cacheable"):
            entry = cache_entry(manifest, name)
            valid, cache_reason = cache_metadata_valid(entry, stage, row["fingerprint"])
            if valid:
                return {"action": "restore", "stage": name, "cache_key": f"{name}/{row['fingerprint']}"}
            if entry.exists() and cache_reason != "cache_metadata_missing":
                return {"action": "blocked", "stage": name, "reason": cache_reason}
        return {"action": "run", "stage": name, "reason": row.get("reason")}
    return {"action": "done", "stage": None, "reason": None}


def manifest_summary(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"],
        "mode": manifest["mode"],
        "output_dir": manifest["output_dir"],
        "cache_dir": manifest["cache_dir"],
        "run_context": manifest.get("run_context"),
        "stages": [
            {
                "name": stage["name"],
                "status": manifest["stages"][stage["name"]]["status"],
                "attempts": manifest["stages"][stage["name"]]["attempts"],
                "reason": manifest["stages"][stage["name"]].get("reason"),
                "fingerprint": manifest["stages"][stage["name"]]["fingerprint"],
                "validation_levels": manifest["stages"][stage["name"]].get(
                    "validation_levels", []
                ),
            }
            for stage in stage_configs(config)
        ],
        "next": next_action(manifest, config),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ControlError(f"manifest does not exist: {resolved}")
    manifest = read_json(resolved)
    if manifest.get("schema_version") != 1:
        raise ControlError("unsupported manifest schema_version")
    if resolved != (Path(manifest["output_dir"]) / MANIFEST_NAME).resolve():
        raise ControlError("manifest path does not match output_dir")
    return manifest


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    write_json_atomic(path.resolve(), manifest)


def command_init(args: argparse.Namespace) -> int:
    config = load_stage_config()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise ControlError(f"manifest already exists; use status/resume: {manifest_path}")
    cache_dir = (args.cache_dir or (output_dir.parent / ".mlu-triton-cache")).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage_config_version": config["version"],
        "run_id": str(uuid.uuid4()),
        "mode": args.mode,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "request": {
            "input": digest_input(args.input),
            "budget_file": digest_optional_file(args.budget_file, "budget file"),
        },
        "run_context": None,
        "created_at": now,
        "updated_at": now,
        "stages": {},
        "events": [],
    }
    for stage in stage_configs(config):
        status = "pending" if enabled(stage, args.mode) else "skipped"
        manifest["stages"][stage["name"]] = {
            "status": status,
            "fingerprint": "",
            "attempts": 0,
            "reason": None if status == "pending" else "disabled_by_mode",
            "artifacts": [],
            "cache_key": None,
            "started_at": None,
            "completed_at": None,
            "checkpoints": {},
            "validation_levels": [],
        }
    refresh_manifest(manifest, config)
    append_event(manifest, "init", None, "run manifest initialized")
    save_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), **manifest_summary(manifest, config)}, ensure_ascii=False, indent=2))
    return 0


def command_bind_context(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    context_path = args.context_file.expanduser().resolve()
    ensure_within(context_path, Path(manifest["output_dir"]), "run context")
    context = read_json(context_path)
    required = ("execution_backend", "hardware_key", "toolchain_key")
    missing = [key for key in required if not isinstance(context.get(key), str) or not context[key]]
    if missing:
        raise ControlError(f"run context missing non-empty fields: {missing}")
    if context.get("schema_version") != 1:
        raise ControlError("run context schema_version must be 1")
    if context["execution_backend"] not in {"local", "worker"}:
        raise ControlError("run context execution_backend must be local or worker")
    digest = sha256_bytes(canonical_json(context))
    old_digest = (manifest.get("run_context") or {}).get("sha256")
    manifest["run_context"] = {
        "path": str(context_path),
        "sha256": digest,
        "execution_backend": context["execution_backend"],
        "hardware_key": context["hardware_key"],
        "toolchain_key": context["toolchain_key"],
    }
    refresh_manifest(manifest, config)
    append_event(manifest, "bind_context", "env_config", f"context {old_digest or 'none'} -> {digest}")
    save_manifest(args.manifest, manifest)
    print(json.dumps(manifest_summary(manifest, config), ensure_ascii=False, indent=2))
    return 0


def command_status(args: argparse.Namespace, recover: bool = False) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    refresh_manifest(manifest, config, recover_running=recover)
    save_manifest(args.manifest, manifest)
    print(json.dumps(manifest_summary(manifest, config), ensure_ascii=False, indent=2))
    return 0


def require_stage(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stage = config_by_name(config).get(stage_name)
    if stage is None:
        raise ControlError(f"unknown stage: {stage_name}")
    return stage


def command_start(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    refresh_manifest(manifest, config)
    stage = require_stage(config, args.stage)
    row = manifest["stages"][args.stage]
    if row["status"] != "pending":
        raise ControlError(f"stage must be pending before start: {args.stage}={row['status']}")
    ready, reason = dependency_ready(manifest, config, stage)
    if not ready:
        raise ControlError(reason or "dependencies not ready")
    expected = next_action(manifest, config)
    if expected.get("stage") != args.stage or expected.get("action") != "run":
        raise ControlError(f"stage is not the next runnable stage: {expected}")
    row["status"] = "running"
    row["attempts"] += 1
    row["reason"] = None
    row["started_at"] = utc_now()
    append_event(manifest, "start", args.stage, f"attempt={row['attempts']}")
    save_manifest(args.manifest, manifest)
    print(json.dumps({"stage": args.stage, "status": "running", "attempts": row["attempts"]}, ensure_ascii=False))
    return 0


def collect_artifacts(
    manifest: dict[str, Any], stage: dict[str, Any], extras: list[str]
) -> list[dict[str, Any]]:
    output_dir = Path(manifest["output_dir"])
    required = [str(item) for item in stage.get("required_artifacts", [])]
    requested = list(dict.fromkeys(required + extras))
    for pattern_value in stage.get("optional_artifacts", []):
        pattern = safe_relative(str(pattern_value))
        for matched in output_dir.glob(pattern.as_posix()):
            if matched.is_file():
                requested.append(matched.relative_to(output_dir).as_posix())
    requested = list(dict.fromkeys(requested))
    records: list[dict[str, Any]] = []
    for value in requested:
        candidate = Path(value)
        path = candidate.resolve() if candidate.is_absolute() else (output_dir / safe_relative(value)).resolve()
        path = ensure_within(path, output_dir, "artifact")
        if not path.is_file() or path.stat().st_size == 0:
            raise ControlError(f"required artifact is missing or empty: {path}")
        relative = path.relative_to(output_dir).as_posix()
        records.append({"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size})
    return records


def collect_explicit_artifacts(manifest: dict[str, Any], values: list[str]) -> list[dict[str, Any]]:
    if not values:
        raise ControlError("at least one --artifact is required")
    output_dir = Path(manifest["output_dir"])
    records: list[dict[str, Any]] = []
    for value in list(dict.fromkeys(values)):
        candidate = Path(value)
        path = candidate.resolve() if candidate.is_absolute() else (output_dir / safe_relative(value)).resolve()
        path = ensure_within(path, output_dir, "checkpoint artifact")
        if not path.is_file() or path.stat().st_size == 0:
            raise ControlError(f"checkpoint artifact is missing or empty: {path}")
        relative = path.relative_to(output_dir).as_posix()
        records.append({"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size})
    return records


def checkpoint_valid(
    manifest: dict[str, Any], stage: dict[str, Any], checkpoint_name: str
) -> tuple[bool, str | None]:
    stage_name = stage["name"]
    row = manifest["stages"][stage_name]
    checkpoint = row.get("checkpoints", {}).get(checkpoint_name)
    if not isinstance(checkpoint, dict):
        return False, "checkpoint_missing"
    if checkpoint.get("stage_fingerprint") != row.get("fingerprint"):
        return False, "checkpoint_fingerprint_mismatch"
    actual_levels = checkpoint.get("validation_levels", [])
    if not isinstance(actual_levels, list) or any(
        level not in VALIDATION_LEVELS for level in actual_levels
    ):
        return False, "checkpoint_validation_levels_invalid"
    required_levels = stage.get("checkpoint_validation_required", {}).get(
        checkpoint_name, []
    )
    satisfied, missing = validation_satisfies(
        normalize_validation_levels(actual_levels), required_levels
    )
    if not satisfied:
        return False, f"checkpoint_validation_missing:{','.join(missing)}"
    output_dir = Path(manifest["output_dir"])
    artifacts = checkpoint.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False, "checkpoint_artifacts_missing"
    for record in artifacts:
        try:
            relative = safe_relative(str(record["path"]))
            path = ensure_within(output_dir / relative, output_dir, "checkpoint artifact")
        except (ControlError, KeyError, TypeError) as exc:
            return False, f"checkpoint_record_invalid:{exc}"
        if not path.is_file():
            return False, f"checkpoint_artifact_missing:{relative.as_posix()}"
        if sha256_file(path) != record.get("sha256"):
            return False, f"checkpoint_hash_mismatch:{relative.as_posix()}"
    return True, None


def command_checkpoint_save(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    refresh_manifest(manifest, config)
    stage = require_stage(config, args.stage)
    row = manifest["stages"][args.stage]
    if row["status"] != "running":
        raise ControlError(f"checkpoint requires a running stage: {args.stage}={row['status']}")
    validation_levels = normalize_validation_levels(args.validation_level)
    required_levels = stage.get("checkpoint_validation_required", {}).get(args.name, [])
    satisfied, missing = validation_satisfies(validation_levels, required_levels)
    if not satisfied:
        raise ControlError(
            f"checkpoint {args.stage}/{args.name} requires validation levels: {missing}"
        )
    if args.stage == "kernel_gen" and args.name in {"review", "step7"}:
        validate_kernel_review_evidence(manifest)
    records = collect_explicit_artifacts(manifest, args.artifact)
    row.setdefault("checkpoints", {})[args.name] = {
        "stage_fingerprint": row["fingerprint"],
        "completed_at": utc_now(),
        "artifacts": records,
        "validation_levels": validation_levels,
    }
    append_event(manifest, "checkpoint_save", args.stage, f"{args.name}:artifacts={len(records)}")
    save_manifest(args.manifest, manifest)
    print(json.dumps({"stage": args.stage, "checkpoint": args.name, "reusable": True, "artifacts": records}, ensure_ascii=False, indent=2))
    return 0


def command_checkpoint_status(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    refresh_manifest(manifest, config)
    stage = require_stage(config, args.stage)
    valid, reason = checkpoint_valid(manifest, stage, args.name)
    if not valid and args.name in manifest["stages"][args.stage].get("checkpoints", {}):
        del manifest["stages"][args.stage]["checkpoints"][args.name]
        append_event(manifest, "checkpoint_reject", args.stage, f"{args.name}:{reason}")
    save_manifest(args.manifest, manifest)
    print(json.dumps({"stage": args.stage, "checkpoint": args.name, "reusable": valid, "reason": reason}, ensure_ascii=False))
    return 0 if valid else 4


def populate_cache(
    manifest: dict[str, Any],
    stage: dict[str, Any],
    records: list[dict[str, Any]],
    validation_levels: list[str],
) -> str | None:
    if not stage.get("cacheable"):
        return None
    satisfied, _ = validation_satisfies(
        validation_levels, stage.get("cache_validation_required", [])
    )
    if not satisfied:
        return None
    name = stage["name"]
    fingerprint = manifest["stages"][name]["fingerprint"]
    entry = cache_entry(manifest, name)
    if entry.exists():
        valid, reason = cache_metadata_valid(entry, stage, fingerprint)
        if not valid:
            raise ControlError(f"immutable cache entry is invalid ({reason}): {entry}")
        return f"{name}/{fingerprint}"
    output_dir = Path(manifest["output_dir"])
    entry.parent.mkdir(parents=True, exist_ok=True)
    temporary = entry.parent / f".{entry.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(exist_ok=False)
    try:
        cache_records: list[dict[str, Any]] = []
        for record in records:
            relative = safe_relative(record["path"])
            source = ensure_within(output_dir / relative, output_dir, "artifact")
            destination = ensure_within(
                temporary / "artifacts" / relative, temporary, "cached artifact"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if sha256_file(destination) != record["sha256"]:
                raise ControlError(f"cache copy verification failed: {relative.as_posix()}")
            cache_records.append(dict(record))
        write_json_atomic(
            temporary / "metadata.json",
            {
                "schema_version": 2,
                "stage_config_version": manifest["stage_config_version"],
                "stage": name,
                "fingerprint": fingerprint,
                "created_at": utc_now(),
                "input_sha256": manifest["request"]["input"]["sha256"],
                "dependency_fingerprints": {
                    dependency: manifest["stages"][dependency]["fingerprint"]
                    for dependency in stage.get("dependencies", [])
                },
                "source_snapshot": source_snapshot(stage.get("sources", [])),
                "run_context": (
                    None
                    if not stage.get("context_sensitive")
                    else {
                        key: manifest["run_context"].get(key)
                        for key in (
                            "sha256",
                            "execution_backend",
                            "hardware_key",
                            "toolchain_key",
                        )
                    }
                ),
                "mode": manifest["mode"] if stage.get("mode_sensitive") else None,
                "budget_sha256": (
                    (manifest["request"].get("budget_file") or {}).get("sha256")
                    if stage.get("budget_sensitive")
                    else None
                ),
                "validation_levels": validation_levels,
                "artifacts": cache_records,
            },
        )
        try:
            os.replace(temporary, entry)
        except OSError:
            if entry.exists():
                valid, reason = cache_metadata_valid(entry, stage, fingerprint)
                if valid:
                    shutil.rmtree(temporary, ignore_errors=True)
                    return f"{name}/{fingerprint}"
                raise ControlError(
                    f"concurrent immutable cache entry is invalid ({reason}): {entry}"
                )
            raise
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return f"{name}/{fingerprint}"


def command_complete(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    refresh_manifest(manifest, config)
    stage = require_stage(config, args.stage)
    row = manifest["stages"][args.stage]
    if row["status"] != "running":
        raise ControlError(f"stage must be running before complete: {args.stage}={row['status']}")
    validation_levels = normalize_validation_levels(args.validation_level)
    if args.stage == "kernel_gen":
        validate_kernel_review_evidence(manifest)
        validate_dispatch_evidence(manifest)
    records = collect_artifacts(manifest, stage, args.artifact or [])
    cache_key = populate_cache(manifest, stage, records, validation_levels)
    row["status"] = "complete"
    row["reason"] = None
    row["artifacts"] = records
    row["cache_key"] = cache_key
    row["validation_levels"] = validation_levels
    row["completed_at"] = utc_now()
    append_event(manifest, "complete", args.stage, f"artifacts={len(records)} cache={cache_key or 'disabled'}")
    save_manifest(args.manifest, manifest)
    print(json.dumps({"stage": args.stage, "status": "complete", "cache_key": cache_key, "artifacts": records}, ensure_ascii=False, indent=2))
    return 0


def command_restore(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    refresh_manifest(manifest, config)
    stage = require_stage(config, args.stage)
    row = manifest["stages"][args.stage]
    if row["status"] != "pending" or not stage.get("cacheable"):
        raise ControlError(f"stage is not pending/cacheable: {args.stage}={row['status']}")
    expected = next_action(manifest, config)
    if expected.get("action") != "restore" or expected.get("stage") != args.stage:
        raise ControlError(f"no valid cache restore is available: {expected}")
    entry = cache_entry(manifest, args.stage)
    valid, reason = cache_metadata_valid(entry, stage, row["fingerprint"])
    if not valid:
        raise ControlError(f"cache restore rejected: {reason}")
    metadata = read_json(entry / "metadata.json")
    output_dir = Path(manifest["output_dir"])
    records: list[dict[str, Any]] = []
    for record in metadata["artifacts"]:
        relative = safe_relative(record["path"])
        source = ensure_within(entry / "artifacts" / relative, entry, "cached artifact")
        destination = ensure_within(output_dir / relative, output_dir, "artifact")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".restore", dir=destination.parent)
        os.close(fd)
        try:
            shutil.copy2(source, temporary)
            if sha256_file(Path(temporary)) != record["sha256"]:
                raise ControlError(f"restored artifact hash mismatch: {relative.as_posix()}")
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        records.append(dict(record))
    row["status"] = "cached"
    row["reason"] = "content_addressed_cache_hit"
    row["artifacts"] = records
    row["cache_key"] = f"{args.stage}/{row['fingerprint']}"
    row["validation_levels"] = normalize_validation_levels(
        metadata.get("validation_levels", [])
    )
    row["completed_at"] = utc_now()
    append_event(manifest, "restore", args.stage, row["cache_key"])
    save_manifest(args.manifest, manifest)
    print(json.dumps({"stage": args.stage, "status": "cached", "cache_key": row["cache_key"], "artifacts": records}, ensure_ascii=False, indent=2))
    return 0


def command_fail(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    require_stage(config, args.stage)
    row = manifest["stages"][args.stage]
    if row["status"] != "running":
        raise ControlError(f"only a running stage can fail: {args.stage}={row['status']}")
    row["status"] = "failed"
    row["reason"] = args.reason
    append_event(manifest, "fail", args.stage, args.reason)
    save_manifest(args.manifest, manifest)
    print(json.dumps({"stage": args.stage, "status": "failed", "reason": args.reason}, ensure_ascii=False))
    return 0


def command_invalidate(args: argparse.Namespace) -> int:
    config = load_stage_config()
    manifest = load_manifest(args.manifest)
    require_stage(config, args.stage)
    invalidate_from(manifest, config, args.stage, args.reason)
    refresh_manifest(manifest, config)
    save_manifest(args.manifest, manifest)
    print(json.dumps(manifest_summary(manifest, config), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Create a new run manifest")
    init.add_argument("--output-dir", required=True, type=Path)
    init.add_argument("--input", required=True, help="Input file path or literal request text")
    init.add_argument("--mode", choices=MODES, default="balanced")
    init.add_argument("--budget-file", type=Path)
    init.add_argument("--cache-dir", type=Path)
    init.set_defaults(func=command_init)

    for name, help_text, recover in (
        ("status", "Validate and show current state", False),
        ("next", "Show the next run/restore action", False),
        ("resume", "Reopen an interrupted/failed stage and show the next action", True),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--manifest", required=True, type=Path)
        command.set_defaults(func=lambda args, recover=recover: command_status(args, recover=recover))

    bind = subparsers.add_parser("bind-context", help="Bind verified hardware/toolchain context")
    bind.add_argument("--manifest", required=True, type=Path)
    bind.add_argument("--context-file", required=True, type=Path)
    bind.set_defaults(func=command_bind_context)

    start = subparsers.add_parser("start", help="Mark the next stage running")
    start.add_argument("--manifest", required=True, type=Path)
    start.add_argument("--stage", required=True)
    start.set_defaults(func=command_start)

    complete = subparsers.add_parser("complete", help="Verify outputs, cache them, and complete a stage")
    complete.add_argument("--manifest", required=True, type=Path)
    complete.add_argument("--stage", required=True)
    complete.add_argument("--artifact", action="append", default=[], help="Additional output-relative artifact")
    complete.add_argument(
        "--validation-level", action="append", choices=VALIDATION_LEVELS, default=[]
    )
    complete.set_defaults(func=command_complete)

    restore = subparsers.add_parser("restore", help="Restore a verified content-addressed cache entry")
    restore.add_argument("--manifest", required=True, type=Path)
    restore.add_argument("--stage", required=True)
    restore.set_defaults(func=command_restore)

    checkpoint_save = subparsers.add_parser(
        "checkpoint-save", help="Record hash-verified inner-stage artifacts for interruption recovery"
    )
    checkpoint_save.add_argument("--manifest", required=True, type=Path)
    checkpoint_save.add_argument("--stage", required=True)
    checkpoint_save.add_argument("--name", required=True)
    checkpoint_save.add_argument("--artifact", action="append", required=True)
    checkpoint_save.add_argument(
        "--validation-level", action="append", choices=VALIDATION_LEVELS, default=[]
    )
    checkpoint_save.set_defaults(func=command_checkpoint_save)

    checkpoint_status = subparsers.add_parser(
        "checkpoint-status", help="Check whether an inner-stage checkpoint is reusable"
    )
    checkpoint_status.add_argument("--manifest", required=True, type=Path)
    checkpoint_status.add_argument("--stage", required=True)
    checkpoint_status.add_argument("--name", required=True)
    checkpoint_status.set_defaults(func=command_checkpoint_status)

    fail = subparsers.add_parser("fail", help="Record a stage failure without losing artifacts or inner state")
    fail.add_argument("--manifest", required=True, type=Path)
    fail.add_argument("--stage", required=True)
    fail.add_argument("--reason", required=True)
    fail.set_defaults(func=command_fail)

    invalidate = subparsers.add_parser("invalidate", help="Invalidate one stage and all downstream stages")
    invalidate.add_argument("--manifest", required=True, type=Path)
    invalidate.add_argument("--stage", required=True)
    invalidate.add_argument("--reason", required=True)
    invalidate.set_defaults(func=command_invalidate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (ControlError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
