#!/usr/bin/env python3
"""Write canonical environment JSON and derive its human-readable report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_WORKER_URL = "http://127.0.0.1:8086/run/v1/agent/submit-task"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", required=True, choices=("local", "worker"))
    parser.add_argument("--env-check-task-id", required=True)
    parser.add_argument("--runtime-info-path", required=True)
    parser.add_argument("--worker-submit-url", default=DEFAULT_WORKER_URL)
    parser.add_argument("--device-model", default="unknown")
    parser.add_argument("--triton-version", default="unknown")
    parser.add_argument("--torch-version", default="unknown")
    parser.add_argument("--toolchain-version", default="unknown")
    parser.add_argument("--checked-at")
    return parser.parse_args()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _require_nonempty(value: str, field: str) -> str:
    if not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _checked_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("checked_at must be an ISO 8601 date-time") from exc
    if parsed.tzinfo is None:
        raise ValueError("checked_at must include a timezone")
    return value


def _build_config(args: argparse.Namespace) -> dict:
    runtime_info_path = Path(args.runtime_info_path).resolve()
    if not runtime_info_path.is_file():
        raise ValueError(f"runtime info file does not exist: {runtime_info_path}")
    if runtime_info_path.stat().st_size == 0:
        raise ValueError(f"runtime info file is empty: {runtime_info_path}")

    if args.backend == "local":
        if args.env_check_task_id != "local":
            raise ValueError("local backend requires --env-check-task-id local")
        worker_submit_url = None
    else:
        if args.env_check_task_id == "local":
            raise ValueError("worker backend requires a real task id")
        worker_submit_url = _require_nonempty(args.worker_submit_url, "worker_submit_url")

    checked_at = _checked_at(args.checked_at)
    return {
        "schema_version": "1.0",
        "status": "ready",
        "execution_backend": args.backend,
        "worker_submit_url": worker_submit_url,
        "env_check_task_id": _require_nonempty(args.env_check_task_id, "env_check_task_id"),
        "runtime_info_path": str(runtime_info_path),
        "checked_at": checked_at,
        "device": {"model": _require_nonempty(args.device_model, "device_model")},
        "versions": {
            "triton": _require_nonempty(args.triton_version, "triton_version"),
            "torch": _require_nonempty(args.torch_version, "torch_version"),
            "toolchain": _require_nonempty(args.toolchain_version, "toolchain_version"),
        },
    }


def _render_markdown(config: dict) -> str:
    return "\n".join(
        [
            "# 环境配置",
            "",
            "> 本文件由 `config.json` 生成；机器流程只能读取 `config.json`。",
            "",
            "## 执行位置",
            "",
            f"- execution_backend: {config['execution_backend']}",
            f"- env_check_task_id: {config['env_check_task_id']}",
            f"- worker_submit_url: {config['worker_submit_url'] or 'N/A'}",
            "",
            "## 环境信息",
            "",
            f"- device_model: {config['device']['model']}",
            f"- triton_version: {config['versions']['triton']}",
            f"- torch_version: {config['versions']['torch']}",
            f"- toolchain_version: {config['versions']['toolchain']}",
            f"- runtime_info: {config['runtime_info_path']}",
            f"- checked_at: {config['checked_at']}",
            "",
        ]
    )


def main() -> int:
    args = _parse_args()
    config = _build_config(args)
    env_dir = Path(args.output_dir).resolve() / "EnvConfig"
    json_path = env_dir / "config.json"
    markdown_path = env_dir / "config.md"
    _atomic_write(json_path, json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(markdown_path, _render_markdown(config))
    print(json.dumps({"config_path": str(json_path), "report_path": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
