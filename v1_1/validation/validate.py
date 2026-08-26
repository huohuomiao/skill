#!/usr/bin/env python3
"""Offline structural and behavioral validation for v1_1."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PATH = ROOT / "validation" / "expected_dispatch_metrics.json"
METRICS_SCRIPT = ROOT / "mlu-triton-code-gen" / "scripts" / "dispatch_metrics.py"
sys.dont_write_bytecode = True


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)


def read_text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def load_metrics_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dispatch_metrics", METRICS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dispatch_metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_frontmatter(v: Validation) -> None:
    for skill in ROOT.rglob("SKILL.md"):
        text = skill.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---\n", text, flags=re.DOTALL)
        v.require(match is not None, f"missing YAML frontmatter: {skill.relative_to(ROOT)}")
        if match:
            block = match.group(1)
            v.require(bool(re.search(r"^name:\s*\S+", block, flags=re.MULTILINE)), f"missing name: {skill.relative_to(ROOT)}")
            v.require(bool(re.search(r"^description:\s*\S+", block, flags=re.MULTILINE)), f"missing description: {skill.relative_to(ROOT)}")


def validate_syntax(v: Validation) -> None:
    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            v.require(False, f"Python syntax/encoding error in {path.relative_to(ROOT)}: {exc}")
        else:
            v.require(True, "")

    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            v.require(False, f"JSON error in {path.relative_to(ROOT)}: {exc}")
        else:
            v.require(True, "")


def validate_required_files(v: Validation) -> None:
    required = [
        "mlu-triton-code-gen/subagents/DesignKernel.md",
        "mlu-triton-code-gen/subagents/BuildKernel.md",
        "mlu-triton-code-gen/references/artifact-contracts.md",
        "mlu-triton-code-gen/references/dispatch-contract.json",
        "mlu-triton-code-gen/scripts/dispatch_metrics.py",
        "mlu-triton-code-review/ReviewAndFix.md",
        "validation/expected_dispatch_metrics.json",
    ]
    for relative in required:
        path = ROOT / relative
        v.require(path.is_file() and path.stat().st_size > 0, f"missing/empty: {relative}")


def validate_orchestration(v: Validation) -> None:
    main = read_text("mlu-triton-main/SKILL.md")
    codegen = read_text("mlu-triton-code-gen/SKILL.md")
    review = read_text("mlu-triton-code-review/SKILL.md")

    v.require(main.count("spawn_agent(") == 2, "Main must retain exactly two subagent calls")
    v.require(codegen.count("spawn_agent(") == 2, "Code Gen must have exactly two subagent call sites")
    v.require(review.count("spawn_agent(") == 1, "Code Review must have exactly one subagent call site")

    for required in ("DesignKernel.md", "BuildKernel.md", "dispatch_metrics.py"):
        v.require(required in codegen, f"Code Gen does not reference {required}")
    for legacy in (
        "ExtractBaseInfo.md",
        "TraceBlockMapping.md",
        "AxisFusion.md",
        "GenerateSpec.md",
        "GenerateCode.md",
        "GenTestCode.md",
    ):
        v.require(legacy not in codegen, f"active Code Gen still references legacy role {legacy}")
    for legacy in ("StaticReviewer.md", "DynamicFixer.md"):
        v.require(legacy not in review, f"active Code Review still references legacy role {legacy}")

    v.require("ReviewAndFix.md" in review, "Code Review does not reference ReviewAndFix.md")
    v.require("代码正文" in codegen and "文件路径" in codegen, "path-only context contract is missing")

    design = read_text("mlu-triton-code-gen/subagents/DesignKernel.md")
    build = read_text("mlu-triton-code-gen/subagents/BuildKernel.md")
    artifacts = (
        "step1_base_info.json",
        "step1_io_shapes.json",
        "step2_block_mapping.json",
        "step3_axis_fusion.json",
        "step4_code_spec.json",
    )
    for artifact in artifacts:
        v.require(artifact in design, f"DesignKernel missing artifact {artifact}")
    for artifact in ("step5_kernel_code.py", "step6_test_code.py"):
        v.require(artifact in build, f"BuildKernel missing artifact {artifact}")
    v.require("triton.testing.do_bench" in build, "BuildKernel missing performance-test contract")
    v.require("torch.allclose" in build, "BuildKernel missing accuracy-test contract")


def validate_contract(v: Validation) -> dict[str, Any]:
    contract = json.loads(
        read_text("mlu-triton-code-gen/references/dispatch-contract.json")
    )
    v.require(contract.get("schema_version") == 1, "dispatch contract schema_version must be 1")
    for route, outcomes in contract["routes"].items():
        for outcome, metrics in outcomes.items():
            for variant in ("baseline_dispatches", "optimized_dispatches"):
                parts = metrics[variant]
                v.require(
                    parts["total"] == parts["main"] + parts["codegen"] + parts["review"],
                    f"dispatch total mismatch: {route}/{outcome}/{variant}",
                )
    return contract


def validate_metrics(v: Validation) -> dict[str, Any]:
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    module = load_metrics_module()
    results: dict[str, Any] = {}
    for key, target in expected.items():
        route, outcome = key.split("/", 1)
        report = module.analyze(route, outcome)
        results[key] = {
            "baseline_dispatches": report["dispatches"]["baseline"]["total"],
            "optimized_dispatches": report["dispatches"]["optimized"]["total"],
            "dispatch_reduction_pct": report["dispatches"]["reduction_pct"],
            "static_context_reduction_pct": report["static_context"]["reduction_pct"],
        }
        v.require(results[key]["baseline_dispatches"] == target["baseline_dispatches"], f"baseline dispatch mismatch: {key}")
        v.require(results[key]["optimized_dispatches"] == target["optimized_dispatches"], f"optimized dispatch mismatch: {key}")
        v.require(results[key]["dispatch_reduction_pct"] >= target["minimum_dispatch_reduction_pct"], f"dispatch reduction target missed: {key}")
        v.require(results[key]["static_context_reduction_pct"] >= target["minimum_static_context_reduction_pct"], f"static-context reduction target missed: {key}")
    return results


def validate_clean_tree(v: Validation) -> None:
    generated = list(ROOT.rglob("__pycache__")) + list(ROOT.rglob("*.pyc"))
    v.require(
        not generated,
        "generated Python cache artifacts found: "
        + ", ".join(str(path.relative_to(ROOT)) for path in generated),
    )


def main() -> int:
    validation = Validation()
    validate_frontmatter(validation)
    validate_syntax(validation)
    validate_required_files(validation)
    validate_orchestration(validation)
    validate_contract(validation)
    metrics = validate_metrics(validation)
    validate_clean_tree(validation)
    report = {
        "version": "v1_1",
        "checks": validation.checks,
        "errors": validation.errors,
        "metrics": metrics,
        "status": "PASS" if not validation.errors else "FAIL",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not validation.errors else 1


if __name__ == "__main__":
    sys.exit(main())
