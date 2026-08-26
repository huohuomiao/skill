#!/usr/bin/env python3
"""Submit a command to the Worker, wait for completion, and print its logs.

This is the only script BANG C skills should use for remote CNCC compile,
MLU execution, accuracy, and performance work. It posts one Task through
Agent-Service, polls until
the Task finishes, then prints the Worker's stdout/stderr logs.

Internal Agent-Service endpoints:
    POST http://127.0.0.1:8086/run/v1/agent/submit-task
    GET  http://127.0.0.1:8086/run/v1/agent/tasks/<task_id>

用法示例：
    python submit_task_to_worker.py \\
        --task-type custom \\
        --workdir /abs/path/to/repo \\
        --timeout-sec 1800 \\
        --command "python /abs/path/to/xxx.py"

退出码：
    0  Task succeeded
    1  Task failed
    2  infrastructure/input error
    3  Task canceled
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SUBMIT_URL = "http://127.0.0.1:8086/run/v1/agent/submit-task"
TASK_URL_FMT = "http://127.0.0.1:8086/run/v1/agent/tasks/{task_id}"
TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}
TRANSIENT_POLL_RETRIES = 3
TASK_TYPES = ("compile", "accuracy", "performance", "custom")


def _err(msg):
    print(f"[submit_task_to_worker][error] {msg}", file=sys.stderr)


def _info(msg):
    print(f"[submit_task_to_worker] {msg}", file=sys.stderr)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task-type", required=True, choices=TASK_TYPES)
    p.add_argument("--workdir", required=True,
                   help="absolute path the command runs in")
    p.add_argument("--command", required=True,
                   help="the exact command Worker will execute verbatim")
    p.add_argument(
        "--device-type",
        default=os.environ.get("MLU_DEVICE_TYPE", "mlu590"),
        help="Worker device type; defaults to $MLU_DEVICE_TYPE or mlu590",
    )
    p.add_argument("--job-id", default=None,
                   help="override $JOB_ID")
    p.add_argument("--env", action="append", default=[],
                   help='additional env var "KEY=VAL", repeatable')
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--timeout-sec", type=int, required=True,
                   help="runtime timeout after the Worker leases the task")
    return p.parse_args()


def _http_post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_json(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _build_payload(args, job_id):
    env_dict = {}
    for kv in args.env:
        if "=" not in kv:
            _err(f"--env must be KEY=VAL, got: {kv}")
            sys.exit(2)
        k, v = kv.split("=", 1)
        env_dict[k] = v
    return {
        "job_id": job_id,
        "task_type": args.task_type,
        "device_type": args.device_type,
        "command": args.command,
        "workdir": args.workdir,
        "timeout_sec": args.timeout_sec,
        "env": env_dict,
    }


def _parse_expires_at(expires_at):
    if not expires_at:
        return None
    try:
        text = str(expires_at)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _poll_until_terminal(task_id, poll_interval, timeout_sec, expires_at):
    expires_dt = _parse_expires_at(expires_at)
    if expires_dt is not None:
        deadline = max(time.time() + 60, expires_dt.timestamp() + 60)
    else:
        deadline = time.time() + max(timeout_sec, 30) + 60
    task_url = TASK_URL_FMT.format(task_id=task_id)
    transient_fails = 0
    last_status = None
    while time.time() < deadline:
        try:
            task = _http_get_json(task_url)
            transient_fails = 0
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            transient_fails += 1
            if transient_fails > TRANSIENT_POLL_RETRIES:
                _err(f"polling failed ({task_url}): {e}")
                sys.exit(2)
            time.sleep(poll_interval)
            continue
        last_status = task.get("status")
        if last_status in TERMINAL_STATUSES:
            return last_status
        time.sleep(poll_interval)
    _err(f"polling timeout waiting for terminal status "
         f"(last={last_status}, task_id={task_id})")
    sys.exit(2)


def _dump_task_logs(job_id, task_id, task_output_dir=None):
    if task_output_dir:
        base = Path(task_output_dir)
    elif os.environ.get("JOB_ROOT"):
        base = Path(os.environ["JOB_ROOT"]) / "tasks_info" / task_id
    else:
        base = Path("jobs_info") / job_id / "tasks_info" / task_id
    stdout_path = base / "stdout.log"
    stderr_path = base / "stderr.log"

    if stdout_path.exists():
        sys.stdout.write(stdout_path.read_text(errors="replace"))
        sys.stdout.flush()
    else:
        _err(f"stdout.log not found at {stdout_path}")

    if stderr_path.exists():
        sys.stderr.write(stderr_path.read_text(errors="replace"))
        sys.stderr.flush()
    else:
        _err(f"stderr.log not found at {stderr_path}")


def main():
    args = _parse_args()

    job_id = (args.job_id or os.environ.get("JOB_ID", "")).strip()
    if not job_id:
        _err("JOB_ID missing: set $JOB_ID or pass --job-id")
        sys.exit(2)

    if not Path(args.workdir).is_absolute():
        _err(f"--workdir must be absolute, got: {args.workdir}")
        sys.exit(2)
    if args.timeout_sec <= 0:
        _err(f"--timeout-sec must be positive, got: {args.timeout_sec}")
        sys.exit(2)

    payload = _build_payload(args, job_id)

    try:
        resp = _http_post_json(SUBMIT_URL, payload)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        _err(f"submit failed ({SUBMIT_URL}): {e}")
        sys.exit(2)

    task_id = resp.get("id") or resp.get("task_id")
    task_output_dir = resp.get("task_output_dir")
    timeout_sec = int(resp.get("timeout_sec") or args.timeout_sec)
    expires_at = resp.get("expires_at")
    if not task_id:
        _err(f"submit response missing task id: {resp}")
        sys.exit(2)
    _info(
        f"submitted task_id={task_id} task_output_dir={task_output_dir} "
        f"timeout_sec={timeout_sec} expires_at={expires_at}"
    )

    status = _poll_until_terminal(task_id, args.poll_interval, timeout_sec, expires_at)

    _dump_task_logs(job_id, task_id, task_output_dir)

    _info(f"done task_id={task_id} status={status}")
    if status == "succeeded":
        sys.exit(0)
    if status == "canceled":
        sys.exit(3)
    sys.exit(1)


if __name__ == "__main__":
    main()
