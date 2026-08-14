#!/usr/bin/env python3
"""Compile and run a real BANG C/CNRT vector-add smoke test."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MAX_DIFF_RE = re.compile(r"BANGC_VECTOR_ADD_MAX_DIFF=([0-9.eE+-]+)")
PASS_MARKER = "BANGC_VECTOR_ADD_PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("bangc_vector_add.mlu"),
        help="smoke-test .mlu source",
    )
    parser.add_argument("--compile-timeout", type=float, default=180.0)
    parser.add_argument("--run-timeout", type=float, default=60.0)
    return parser.parse_args()


def executable_candidate(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path.resolve())
    return shutil.which(value)


def find_cncc() -> str | None:
    explicit = executable_candidate(os.environ.get("CNCC"))
    if explicit:
        return explicit
    found = shutil.which("cncc")
    if found:
        return found

    candidates: list[Path] = []
    for variable in ("NEUWARE_HOME", "CNTOOLKIT_HOME"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root).expanduser() / "bin" / "cncc")
    candidates.extend(
        [
            Path("/usr/local/neuware/bin/cncc"),
            Path("/usr/local/cntoolkit/bin/cncc"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return None


def candidate_roots(cncc: str) -> list[Path]:
    roots: list[Path] = []
    for variable in ("NEUWARE_HOME", "CNTOOLKIT_HOME"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    compiler_path = Path(cncc).resolve()
    if compiler_path.parent.name == "bin":
        roots.append(compiler_path.parent.parent)
    roots.extend([Path("/usr/local/neuware"), Path("/usr/local/cntoolkit")])

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def locate_layout(cncc: str) -> dict[str, str | None]:
    best: dict[str, str | None] = {
        "root": None,
        "cnrt_header": None,
        "bang_header": None,
        "libcnrt": None,
        "lib_dir": None,
    }
    for root in candidate_roots(cncc):
        cnrt_header = root / "include" / "cnrt.h"
        bang_candidates = [root / "include" / "bang.h"]
        clang_root = root / "lib" / "clang"
        if clang_root.is_dir():
            bang_candidates.extend(sorted(clang_root.glob("*/include/bang.h"), reverse=True))
        bang_header = next((p for p in bang_candidates if p.is_file()), None)

        lib_candidates = [
            root / "lib64" / "libcnrt.so",
            root / "lib" / "libcnrt.so",
        ]
        libcnrt = next((p for p in lib_candidates if p.exists()), None)
        score = int(cnrt_header.is_file()) + int(bang_header is not None) + int(libcnrt is not None)
        old_score = sum(best[key] is not None for key in ("cnrt_header", "bang_header", "libcnrt"))
        if score > old_score:
            best = {
                "root": str(root.resolve()) if root.exists() else str(root),
                "cnrt_header": str(cnrt_header.resolve()) if cnrt_header.is_file() else None,
                "bang_header": str(bang_header.resolve()) if bang_header else None,
                "libcnrt": str(libcnrt.resolve()) if libcnrt else None,
                "lib_dir": str(libcnrt.parent.resolve()) if libcnrt else None,
            }
    return best


def run(
    command: list[str],
    *,
    timeout: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def print_process(label: str, result: subprocess.CompletedProcess[str]) -> None:
    print(f"=== {label} STDOUT ===")
    if result.stdout:
        print(result.stdout.rstrip())
    print(f"=== {label} STDERR ===", file=sys.stderr)
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    print(f"{label}_EXIT_CODE={result.returncode}")


def arch_flags(cncc: str, child_env: dict[str, str]) -> tuple[list[str], str]:
    complete_flag = os.environ.get("BANGC_ARCH_FLAG", "").strip()
    if complete_flag:
        return shlex.split(complete_flag), "BANGC_ARCH_FLAG"

    arch = os.environ.get("BANGC_ARCH", "").strip()
    if not arch:
        return [], "CNCC default"

    help_result = run([cncc, "--help"], timeout=30.0, env=child_env)
    help_text = help_result.stdout + help_result.stderr
    if arch.startswith("compute_") and "--bang-arch" in help_text:
        return [f"--bang-arch={arch}"], "BANGC_ARCH + cncc --help"
    if "--bang-mlu-arch" in help_text:
        return [f"--bang-mlu-arch={arch}"], "BANGC_ARCH + cncc --help"
    raise RuntimeError(
        "BANGC_ARCH is set but current cncc --help exposes no recognized matching flag; "
        "set BANGC_ARCH_FLAG to the full verified argument"
    )


def child_environment(layout: dict[str, str | None]) -> dict[str, str]:
    env = os.environ.copy()
    root = layout.get("root")
    if root:
        env.setdefault("NEUWARE_HOME", root)
        bin_dir = str(Path(root) / "bin")
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    lib_dir = layout.get("lib_dir")
    if lib_dir:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + current if current else "")
    return env


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_file():
        print(f"ERROR: smoke source not found: {source}", file=sys.stderr)
        return 1

    cncc = find_cncc()
    if not cncc:
        print(
            "ERROR: cncc not found; checked $CNCC, PATH, configured NeuWare roots, "
            "and standard /usr/local locations",
            file=sys.stderr,
        )
        return 1

    layout = locate_layout(cncc)
    env = child_environment(layout)
    try:
        version = run([cncc, "--version"], timeout=30.0, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: could not query cncc version: {exc}", file=sys.stderr)
        return 1
    print_process("CNCC_VERSION", version)

    try:
        selected_arch_flags, arch_source = arch_flags(cncc, env)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        print(f"ERROR: architecture flag selection failed: {exc}", file=sys.stderr)
        return 1

    extra_flags = shlex.split(os.environ.get("BANGC_CNCC_EXTRA_FLAGS", ""))
    with tempfile.TemporaryDirectory(prefix="bangc_env_check_") as temporary:
        build_dir = Path(temporary)
        binary = build_dir / "bangc_vector_add"
        command = [cncc, str(source), "-o", str(binary), "-std=c++11"]
        if layout.get("cnrt_header"):
            command.extend(["-I", str(Path(layout["cnrt_header"]).parent)])
        if layout.get("lib_dir"):
            command.extend(["-L", layout["lib_dir"]])
        command.extend(selected_arch_flags)
        command.extend(extra_flags)
        command.extend(["-lcnrt", "-lstdc++", "-lm", "-lpthread"])

        print("CNCC_PATH=" + cncc)
        print("NEUWARE_ROOT=" + str(layout.get("root") or "UNRESOLVED"))
        print("CNRT_HEADER=" + str(layout.get("cnrt_header") or "UNRESOLVED"))
        print("BANG_HEADER=" + str(layout.get("bang_header") or "CNCC_RESOLVED_OR_UNRESOLVED"))
        print("LIBCNRT=" + str(layout.get("libcnrt") or "UNRESOLVED"))
        print("BANGC_ARCH_SOURCE=" + arch_source)
        print("BANGC_ARCH_FLAGS=" + (shlex.join(selected_arch_flags) if selected_arch_flags else "<CNCC default>"))
        print("BANGC_COMPILE_COMMAND=" + shlex.join(command))

        try:
            compiled = run(command, timeout=args.compile_timeout, cwd=build_dir, env=env)
        except subprocess.TimeoutExpired:
            print(f"ERROR: CNCC compilation timed out after {args.compile_timeout:g}s", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"ERROR: CNCC invocation failed: {exc}", file=sys.stderr)
            return 1
        print_process("BANGC_COMPILE", compiled)
        if compiled.returncode != 0 or not binary.is_file():
            print("ERROR: BANG C smoke test compilation failed", file=sys.stderr)
            return 1

        try:
            executed = run([str(binary)], timeout=args.run_timeout, cwd=build_dir, env=env)
        except subprocess.TimeoutExpired:
            print(f"ERROR: BANG C smoke test timed out after {args.run_timeout:g}s", file=sys.stderr)
            return 1
        except OSError as exc:
            print(f"ERROR: BANG C smoke binary failed to start: {exc}", file=sys.stderr)
            return 1
        print_process("BANGC_RUN", executed)

        max_diff_match = MAX_DIFF_RE.search(executed.stdout)
        max_diff = float(max_diff_match.group(1)) if max_diff_match else None
        passed = (
            executed.returncode == 0
            and PASS_MARKER in executed.stdout
            and max_diff is not None
            and max_diff <= 1.0e-5
        )
        payload = {
            "cncc_path": cncc,
            "cncc_version": (version.stdout or version.stderr).strip(),
            "neuware_root": layout.get("root"),
            "cnrt_header": layout.get("cnrt_header"),
            "bang_header": layout.get("bang_header") or "resolved by cncc or unavailable",
            "libcnrt": layout.get("libcnrt"),
            "arch_flags": selected_arch_flags,
            "arch_flag_source": arch_source,
            "compile_command": command,
            "compile_pass": True,
            "run_pass": executed.returncode == 0,
            "accuracy_pass": passed,
            "atol": 1.0e-5,
            "max_diff": max_diff,
        }
        print("BANGC_ENV_JSON=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if not passed:
            print(
                "ERROR: smoke binary did not produce a valid PASS marker and max_diff <= 1e-5",
                file=sys.stderr,
            )
            return 1

    print("BANGC_ENV_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
