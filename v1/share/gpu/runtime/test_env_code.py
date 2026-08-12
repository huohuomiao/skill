#!/usr/bin/env python3
"""Compile and run a CUDA Triton vector-add smoke test on an RTX 3090."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from typing import Any

try:
    import torch
    import triton
    import triton.language as tl
except Exception as exc:
    print(f"IMPORT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(2)


RTX_3090 = re.compile(r"^(?:NVIDIA\s+)?GeForce\s+RTX\s+3090$", re.IGNORECASE)


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if not (x.is_cuda and y.is_cuda):
        raise ValueError("vector_add requires CUDA tensors")
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError("inputs must have identical shape and dtype")
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(output.numel(), meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, output.numel(), BLOCK_SIZE=256, num_warps=4)
    return output


def exact_rtx_3090(name: str) -> bool:
    return bool(RTX_3090.fullmatch(" ".join(name.split())))


def fail(summary: dict[str, Any], code: int, message: str, json_output: bool) -> int:
    summary.update(ok=False, error=message)
    if json_output:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"FAIL: {message}", file=sys.stderr)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA-visible logical device index")
    parser.add_argument("--size", type=int, default=100_003, help="vector length (non-power-of-two is preferred)")
    parser.add_argument(
        "--allow-other-supported",
        action="store_true",
        help="allow another NVIDIA GPU with compute capability >= 8.0",
    )
    parser.add_argument("--json", action="store_true", help="emit one machine-readable JSON object")
    args = parser.parse_args()

    summary: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "triton_version": triton.__version__,
        "logical_device": args.device,
        "n_elements": args.size,
    }
    if args.size <= 0:
        return fail(summary, 2, "--size must be positive", args.json)
    if not torch.cuda.is_available():
        return fail(summary, 3, "torch.cuda.is_available() is false", args.json)
    if args.device < 0 or args.device >= torch.cuda.device_count():
        return fail(summary, 3, f"device index {args.device} is outside the CUDA-visible range", args.json)

    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    cc = torch.cuda.get_device_capability(args.device)
    name_match = exact_rtx_3090(props.name)
    summary.update(
        device_name=props.name,
        exact_rtx_3090=name_match,
        compute_capability=f"{cc[0]}.{cc[1]}",
        cuda_target=f"sm_{cc[0]}{cc[1]}",
        multiprocessor_count=props.multi_processor_count,
        total_memory_bytes=props.total_memory,
    )
    if cc < (8, 0):
        return fail(summary, 4, f"compute capability {cc[0]}.{cc[1]} is below Triton NVIDIA minimum 8.0", args.json)
    if name_match and cc != (8, 6):
        return fail(summary, 4, f"RTX 3090 name matched but compute capability is {cc[0]}.{cc[1]}, expected 8.6", args.json)
    if not name_match and not args.allow_other_supported:
        return fail(
            summary,
            5,
            f"device is not an exact RTX 3090: {props.name!r}; use --allow-other-supported only for non-target diagnostics",
            args.json,
        )

    try:
        target = triton.runtime.driver.active.get_current_target()
        summary["triton_target"] = {
            "backend": getattr(target, "backend", None),
            "arch": str(getattr(target, "arch", "")),
        }
    except Exception as exc:
        return fail(summary, 6, f"could not query Triton active target: {type(exc).__name__}: {exc}", args.json)
    if summary["triton_target"]["backend"] != "cuda":
        return fail(summary, 6, f"Triton backend is {summary['triton_target']['backend']!r}, expected 'cuda'", args.json)

    try:
        generator = torch.Generator(device=f"cuda:{args.device}").manual_seed(20260812)
        x = torch.randn(args.size, device=f"cuda:{args.device}", dtype=torch.float32, generator=generator)
        y = torch.randn(args.size, device=f"cuda:{args.device}", dtype=torch.float32, generator=generator)
        output = vector_add(x, y)
        torch.cuda.synchronize(args.device)
        expected = x + y
        max_abs_error = (output - expected).abs().max().item()
        torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    except Exception as exc:
        if not args.json:
            traceback.print_exc()
        return fail(summary, 7, f"Triton vector add failed: {type(exc).__name__}: {exc}", args.json)

    summary.update(
        ok=True,
        output_is_cuda=bool(output.is_cuda),
        max_abs_error=max_abs_error,
        target_match="RTX_3090_SM86" if name_match else "SUPPORTED_NON_TARGET_GPU",
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"PyTorch {torch.__version__} (CUDA build {torch.version.cuda})")
        print(f"Triton {triton.__version__}; backend=cuda")
        print(f"GPU {args.device}: {props.name}; CC {cc[0]}.{cc[1]}; sm_{cc[0]}{cc[1]}")
        if name_match:
            print("TARGET_MATCH: exact NVIDIA GeForce RTX 3090 / sm_86")
        else:
            print("TARGET_MISMATCH_ALLOWED: supported CUDA GPU, not the RTX 3090 target")
        print(f"PASS: CUDA Triton vector add ({args.size} elements), max_abs_error={max_abs_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
