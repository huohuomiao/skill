#!/usr/bin/env python3
"""Collect NVIDIA GPU, optional PyTorch, and optional Triton facts as JSON."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from io import StringIO
from typing import Any


RTX_3090 = re.compile(r"^(?:NVIDIA\s+)?GeForce\s+RTX\s+3090$", re.IGNORECASE)
SM86_LIMITS = {
    "compute_capability": "8.6",
    "cuda_target": "sm_86",
    "warp_size": 32,
    "max_threads_per_block": 1024,
    "max_resident_blocks_per_sm": 16,
    "max_resident_warps_per_sm": 48,
    "max_resident_threads_per_sm": 1536,
    "registers_32bit_per_sm": 65536,
    "max_registers_32bit_per_thread": 255,
    "shared_memory_per_sm_bytes": 100 * 1024,
    "max_shared_memory_per_block_bytes": 99 * 1024,
    "native_fp8_tensor_core": False,
}
SMI_FIELDS = (
    "index",
    "uuid",
    "pci.bus_id",
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "temperature.gpu",
)


def clean(value: str) -> str | None:
    value = value.strip()
    return None if value in {"", "N/A", "[N/A]", "Not Supported"} else value


def integer(value: str) -> int | None:
    value = clean(value)
    if value is None:
        return None
    match = re.search(r"-?\d+", value)
    return int(match.group()) if match else None


def exact_rtx_3090(name: str | None) -> bool:
    return bool(name and RTX_3090.fullmatch(" ".join(name.split())))


def resolve_smi_selector(requested: str, logical_device: int) -> str:
    """Map a CUDA-visible logical index to its physical index/UUID when possible."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return requested
    entries = [item.strip() for item in visible.split(",")]
    if requested.lstrip("-").isdigit() and 0 <= logical_device < len(entries) and entries[logical_device]:
        return entries[logical_device]
    return requested


def run_nvidia_smi(selector: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    command = [
        executable or "nvidia-smi",
        f"--query-gpu={','.join(SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
        "-i",
        selector,
    ]
    meta: dict[str, Any] = {"available": executable is not None, "command": command}
    if executable is None:
        meta["error"] = "nvidia-smi was not found on PATH"
        return None, meta
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta
    meta.update(returncode=proc.returncode, stderr=proc.stderr.strip())
    if proc.returncode != 0:
        meta["error"] = "nvidia-smi query failed"
        return None, meta
    rows = list(csv.reader(StringIO(proc.stdout)))
    if len(rows) != 1 or len(rows[0]) != len(SMI_FIELDS):
        meta["error"] = f"expected one {len(SMI_FIELDS)}-column row, got {len(rows)} row(s)"
        return None, meta
    values = dict(zip(SMI_FIELDS, (clean(item) for item in rows[0])))
    device = {
        "index": integer(values["index"] or ""),
        "uuid": values["uuid"],
        "pci_bus_id": values["pci.bus_id"],
        "name": values["name"],
        "driver_version": values["driver_version"],
        "memory_total_mib": integer(values["memory.total"] or ""),
        "memory_used_mib": integer(values["memory.used"] or ""),
        "memory_free_mib": integer(values["memory.free"] or ""),
        "utilization_gpu_percent": integer(values["utilization.gpu"] or ""),
        "temperature_c": integer(values["temperature.gpu"] or ""),
    }
    device["is_exact_rtx_3090"] = exact_rtx_3090(device["name"])
    return device, meta


def collect_torch(logical_device: int) -> dict[str, Any]:
    result: dict[str, Any] = {"installed": False, "cuda_available": False}
    try:
        import torch
    except Exception as exc:  # PyTorch is optional for inventory.
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(
        installed=True,
        version=getattr(torch, "__version__", None),
        cuda_build=getattr(getattr(torch, "version", None), "cuda", None),
        cuda_available=bool(torch.cuda.is_available()),
        cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    )
    if not result["cuda_available"]:
        return result
    result["visible_device_count"] = torch.cuda.device_count()
    if logical_device < 0 or logical_device >= torch.cuda.device_count():
        result["error"] = f"CUDA logical device {logical_device} is outside visible range"
        return result
    try:
        props = torch.cuda.get_device_properties(logical_device)
        cc = torch.cuda.get_device_capability(logical_device)
        result["selected"] = {
            "logical_index": logical_device,
            "name": props.name,
            "is_exact_rtx_3090": exact_rtx_3090(props.name),
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "cuda_target": f"sm_{cc[0]}{cc[1]}",
            "total_memory_bytes": props.total_memory,
            "multiprocessor_count": props.multi_processor_count,
            "max_threads_per_sm": getattr(props, "max_threads_per_multi_processor", None),
            "major": props.major,
            "minor": props.minor,
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def collect_triton(logical_device: int, torch_info: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"installed": False}
    try:
        import triton
    except Exception as exc:  # Triton is optional for inventory.
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(installed=True, version=getattr(triton, "__version__", None))
    if not torch_info.get("cuda_available") or "selected" not in torch_info:
        return result
    try:
        import torch

        torch.cuda.set_device(logical_device)
        target = triton.runtime.driver.active.get_current_target()
        result["active_target"] = {
            "backend": getattr(target, "backend", None),
            "arch": str(getattr(target, "arch", "")),
        }
        utils = triton.runtime.driver.active.utils
        props = utils.get_device_properties(logical_device)
        result["backend_properties"] = {
            key: props.get(key)
            for key in ("multiprocessor_count", "max_num_regs", "max_shared_mem", "warpSize")
            if key in props
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="0",
        help="nvidia-smi selector (index/UUID); an integer also selects the PyTorch logical index",
    )
    parser.add_argument(
        "--allow-other-gpu",
        action="store_true",
        help="return success for a non-RTX-3090 device while preserving match=false in JSON",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation; use 0 for compact JSON")
    args = parser.parse_args()

    try:
        logical_device = int(args.device)
    except ValueError:
        logical_device = 0

    smi_selector = resolve_smi_selector(args.device, logical_device)
    smi_device, smi = run_nvidia_smi(smi_selector)
    torch_info = collect_torch(logical_device)
    triton_info = collect_triton(logical_device, torch_info)

    smi_match = bool(smi_device and smi_device["is_exact_rtx_3090"])
    torch_selected = torch_info.get("selected")
    torch_match = None if torch_selected is None else bool(torch_selected["is_exact_rtx_3090"])
    cc_match = None if torch_selected is None else torch_selected["compute_capability"] == "8.6"
    backend = (triton_info.get("active_target") or {}).get("backend")
    backend_match = None if backend is None else backend == "cuda"

    reasons: list[str] = []
    if smi_device is None:
        reasons.append(smi.get("error", "nvidia-smi device query failed"))
    elif not smi_match:
        reasons.append(f"nvidia-smi name is not exact RTX 3090: {smi_device.get('name')!r}")
    if torch_selected is None:
        reasons.append(
            "PyTorch CUDA validation is unavailable: "
            + str(torch_info.get("error", "no selected CUDA device"))
        )
    elif torch_match is False:
        reasons.append(f"PyTorch logical device name is not exact RTX 3090: {torch_selected['name']!r}")
    if cc_match is False:
        reasons.append(f"PyTorch compute capability is {torch_selected['compute_capability']}, expected 8.6")
    if backend is None:
        reasons.append(
            "Triton CUDA backend validation is unavailable: "
            + str(triton_info.get("error", "no active target"))
        )
    elif backend_match is False:
        reasons.append(f"Triton active backend is {backend!r}, expected 'cuda'")

    exact_match = bool(
        smi_match
        and torch_match is True
        and cc_match is True
        and backend_match is True
    )
    summary = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_device": args.device,
        "resolved_nvidia_smi_selector": smi_selector,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": smi,
        "device": smi_device,
        "pytorch": torch_info,
        "triton": triton_info,
        "target": {
            "expected_name": "NVIDIA GeForce RTX 3090",
            "expected_compute_capability": "8.6",
            "expected_cuda_target": "sm_86",
            "exact_match": exact_match,
            "reasons": reasons,
            "sm86_limits": SM86_LIMITS if exact_match else None,
        },
    }
    print(json.dumps(summary, indent=None if args.indent == 0 else args.indent, ensure_ascii=False, sort_keys=True))
    if smi_device is None:
        return 2
    if not exact_match and not args.allow_other_gpu:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
