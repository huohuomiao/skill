#!/usr/bin/env python3
"""Generate a deterministic p4 stage fingerprint from inputs, resources, and runtime identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


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
IGNORED_PARTS = {".git", "__pycache__", "validation-reports", "RunState"}


def _arguments() -> argparse.Namespace:
    skill_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--skill-root", type=Path, default=skill_root)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--input", action="append", default=[], help="logical-name=path")
    parser.add_argument("--dependency", action="append", default=[], help="name=version")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--hardware-model")
    parser.add_argument("--toolchain-version")
    parser.add_argument("--device-independent", action="store_true")
    parser.add_argument("--workflow-mode", choices=("full", "code-generation", "code-validation", "performance-tuning"), default="full")
    parser.add_argument("--optimization-mode", choices=("correctness", "balanced", "max-performance"), default="balanced")
    parser.add_argument("--execution-backend", choices=("local", "worker"))
    parser.add_argument("--config", action="append", default=[], help="name=JSON-value")
    parser.add_argument("--upstream", action="append", default=[], help="stage=fingerprint-json-or-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _pairs(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} requires name=value: {value}")
        key, item = value.split("=", 1)
        if not key or not item or key in result:
            raise ValueError(f"invalid or duplicate {option}: {value}")
        result[key] = item
    return result


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_hash(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ValueError(f"path is missing: {path}")
    digest = hashlib.sha256()
    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and not any(part in IGNORED_PARTS for part in item.relative_to(path).parts)
        and item.suffix not in {".pyc", ".pyo"}
    ]
    if not files:
        raise ValueError(f"directory has no hashable files: {path}")
    for item in sorted(files, key=lambda candidate: candidate.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resource_hashes(skill_root: Path, registry: dict, stage: str) -> dict[str, str]:
    resources: dict[str, str] = {}
    entries = [*registry["shared_resources"], *registry["stages"][stage]]
    for relative in entries:
        if relative in resources:
            continue
        resource = (skill_root / relative).resolve()
        try:
            resource.relative_to(skill_root)
        except ValueError as exc:
            raise ValueError(f"resource escapes skill root: {relative}") from exc
        resources[relative] = _path_hash(resource)
    return dict(sorted(resources.items()))


def _input_hashes(values: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for label, raw_path in sorted(values.items()):
        path = Path(raw_path).resolve()
        result[label] = _path_hash(path)
    return result


def _upstream_fingerprints(values: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for stage, value in sorted(values.items()):
        candidate = Path(value)
        if candidate.is_file():
            fingerprint = load_json(candidate.resolve()).get("fingerprint")
        else:
            fingerprint = value
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"invalid upstream fingerprint for {stage}")
        result[stage] = fingerprint
    return result


def _stage_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "workflow_mode": args.workflow_mode,
        "optimization_mode": args.optimization_mode,
        "execution_backend": args.execution_backend,
    }
    for key, raw_value in _pairs(args.config, "--config").items():
        try:
            config[key] = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--config value for {key} must be valid JSON") from exc
    return config


def main() -> int:
    args = _arguments()
    try:
        skill_root = args.skill_root.resolve()
        registry_path = (
            args.registry.resolve()
            if args.registry
            else skill_root / "references/cache/stage-dependencies.json"
        )
        registry = load_json(registry_path)
        registry_schema = skill_root / "references/schemas/stage-dependencies.schema.json"
        validate(registry, load_json(registry_schema), registry_schema)

        dependencies = _pairs(args.dependency, "--dependency")
        hardware_model = args.hardware_model
        toolchain_version = args.toolchain_version
        execution_backend = args.execution_backend
        if args.env_config:
            env_config_path = args.env_config.resolve()
            env = load_json(env_config_path)
            env_schema_path = skill_root / "references/schemas/env-config.schema.json"
            validate(env, load_json(env_schema_path), env_schema_path)
            hardware_model = hardware_model or env["device"]["model"]
            toolchain_version = toolchain_version or env["versions"]["toolchain"]
            execution_backend = execution_backend or env["execution_backend"]
            for name in ("triton", "torch", "toolchain"):
                dependencies.setdefault(name, env["versions"][name])
        if args.device_independent:
            hardware_model = None
            toolchain_version = None
            execution_backend = None
        elif not hardware_model or not toolchain_version:
            raise ValueError("hardware and toolchain identity are required unless --device-independent is set")
        args.execution_backend = execution_backend

        inputs = _input_hashes(_pairs(args.input, "--input"))
        resources = _resource_hashes(skill_root, registry, args.stage)
        factors = {
            "input_hash": _canonical_hash(inputs),
            "skill_version": _canonical_hash(resources),
            "dependency_versions": dict(sorted(dependencies.items())),
            "hardware_model": hardware_model,
            "toolchain_version": toolchain_version,
            "stage_config": _stage_config(args),
            "upstream_fingerprints": _upstream_fingerprints(_pairs(args.upstream, "--upstream")),
        }
        payload = {
            "schema_version": "1.0",
            "stage": args.stage,
            "fingerprint": _canonical_hash({"stage": args.stage, "factors": factors}),
            "generated_at": utc_now(),
            "factors": factors,
            "evidence": {"inputs": inputs, "resources": resources},
        }
        schema_path = skill_root / "references/schemas/stage-fingerprint.schema.json"
        validate(payload, load_json(schema_path), schema_path)
        write_json(args.output.resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except (KeyError, OSError, TypeError, ValueError, SchemaValidationError, json.JSONDecodeError) as exc:
        print(f"fingerprint-stage: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
