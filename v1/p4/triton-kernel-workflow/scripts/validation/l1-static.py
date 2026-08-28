#!/usr/bin/env python3
"""Run dependency-free L1 static validation for the Skill package."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

from validation_common import (
    finish,
    make_check,
    make_report,
    skill_fingerprint,
    strict_json_loads,
    utc_now,
)


ALLOWED_FRONTMATTER = {"name", "description", "license", "allowed-tools", "metadata"}
DIRECTIVE_WORDS = ("must", "do not", "never", "禁止", "不得", "必须", "只能", "不要")
REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "EnvConfig/config.json",
    "Extractor/requirement.md",
    "KernelGen/triton_code_fix.py",
    "Optimizer/best_so_far.json",
    "Optimizer/triton_optimized.py",
    "triton_final.py",
    "summary.md",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--warn-markdown-bytes", type=int, default=80_000)
    parser.add_argument("--max-markdown-bytes", type=int, default=120_000)
    return parser.parse_args()


def _check_frontmatter(root: Path) -> dict:
    issues: list[str] = []
    skill_path = root / "SKILL.md"
    if not skill_path.is_file():
        return make_check("frontmatter", "fail", issues=["SKILL.md is missing"])
    content = skill_path.read_text(encoding="utf-8")
    match = re.match(r"^---\r?\n(.*?)\r?\n---", content, re.DOTALL)
    if not match:
        return make_check("frontmatter", "fail", issues=["invalid YAML frontmatter delimiters"])
    frontmatter = match.group(1)
    keys = set(re.findall(r"^([A-Za-z][A-Za-z0-9_-]*):", frontmatter, re.MULTILINE))
    missing = {"name", "description"} - keys
    unexpected = keys - ALLOWED_FRONTMATTER
    if missing:
        issues.append(f"missing keys: {sorted(missing)}")
    if unexpected:
        issues.append(f"unexpected keys: {sorted(unexpected)}")
    name_match = re.search(r"^name:\s*([^\r\n]+)$", frontmatter, re.MULTILINE)
    if name_match:
        name = name_match.group(1).strip().strip("\"'")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
            issues.append(f"invalid skill name: {name}")
    description_match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
    if description_match:
        description = description_match.group(1).strip().strip("\"'")
        if len(description) > 1024 or "<" in description or ">" in description:
            issues.append("description violates length or angle-bracket constraints")
    body = content[match.end() :]
    if re.search(r"(?mi)^\s*\[TODO:[^\n]*\]\s*$", body):
        issues.append("unfinished TODO marker in SKILL.md")
    return make_check("frontmatter", "fail" if issues else "pass", issues=issues)


def _check_references(root: Path) -> dict:
    missing: list[str] = []
    checked = 0
    for markdown in root.rglob("*.md"):
        content = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\r\n]+)\)", content):
            value = target.strip().strip("<>").split("#", 1)[0]
            if not value or re.match(r"^[a-z]+://", value, re.IGNORECASE) or value.startswith("{"):
                continue
            if not re.search(r"\.(?:md|json|py|sh)$", value, re.IGNORECASE):
                continue
            checked += 1
            if not (markdown.parent / value).resolve().exists():
                missing.append(f"{markdown.relative_to(root)} -> {value}")
        for target in re.findall(r"\{skill_root\}/([A-Za-z0-9_./-]+)", content):
            value = target.rstrip(".,;:")
            checked += 1
            if not (root / value).exists():
                missing.append(f"{markdown.relative_to(root)} -> {{skill_root}}/{value}")
    for schema_path in root.rglob("*.schema.json"):
        schema = strict_json_loads(schema_path.read_text(encoding="utf-8"))
        stack = [schema]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                reference = value.get("$ref")
                if isinstance(reference, str) and not reference.startswith(("#", "http://", "https://")):
                    checked += 1
                    if not (schema_path.parent / reference).resolve().exists():
                        missing.append(f"{schema_path.relative_to(root)} -> {reference}")
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return make_check(
        "references", "fail" if missing else "pass", checked=checked, missing=sorted(set(missing))
    )


def _markdown_fence_issue(content: str) -> str | None:
    marker: str | None = None
    marker_length = 0
    for line_number, line in enumerate(content.splitlines(), start=1):
        match = re.match(r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?(`{3,}|~{3,})(.*)$", line)
        if not match:
            continue
        fence = match.group(1)
        suffix = match.group(2).strip()
        if marker is None:
            marker = fence[0]
            marker_length = len(fence)
        elif fence[0] == marker and len(fence) >= marker_length and not suffix:
            marker = None
            marker_length = 0
    return None if marker is None else f"unclosed {marker * marker_length} fence"


def _outside_fence_lines(content: str) -> list[str]:
    lines: list[str] = []
    marker: str | None = None
    marker_length = 0
    for line in content.splitlines():
        match = re.match(r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?(`{3,}|~{3,})(.*)$", line)
        if match:
            fence = match.group(1)
            suffix = match.group(2).strip()
            if marker is None:
                marker = fence[0]
                marker_length = len(fence)
            elif fence[0] == marker and len(fence) >= marker_length and not suffix:
                marker = None
                marker_length = 0
            continue
        if marker is None:
            lines.append(line)
    return lines


def _check_markdown(root: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    for path in root.rglob("*.md"):
        content = path.read_text(encoding="utf-8")
        relative = path.relative_to(root)
        fence_issue = _markdown_fence_issue(content)
        if fence_issue:
            errors.append(f"{relative}: {fence_issue}")
        body = content
        if path.name == "SKILL.md" and content.startswith("---"):
            match = re.match(r"^---\r?\n.*?\r?\n---", content, re.DOTALL)
            if match:
                body = content[match.end() :]
        structural_lines = _outside_fence_lines(body)
        first = next((line.strip() for line in structural_lines if line.strip()), "")
        if not first.startswith("# ") and "examples" not in path.relative_to(root).parts:
            warnings.append(f"{relative}: first content heading is not H1")
        levels = [
            len(match.group(1))
            for line in structural_lines
            if (match := re.match(r"^(#+)\s+", line))
        ]
        for previous, current in zip(levels, levels[1:]):
            if current > previous + 1:
                warnings.append(f"{relative}: heading jumps H{previous} -> H{current}")
                break
    status = "fail" if errors else ("warning" if warnings else "pass")
    return make_check("markdown-structure", status, errors=errors, warnings=warnings)


def _check_files_and_placeholders(root: Path) -> dict:
    empty: list[str] = []
    placeholders: list[str] = []
    pattern = re.compile(
        r"(?mi)^\s*(?:[-*]\s*)?(?:\[TODO:[^\n]*\]|TODO|TBD|PLACEHOLDER)(?:\s*[:：].*)?\s*$"
    )
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.stat().st_size == 0:
            empty.append(str(path.relative_to(root)))
            continue
        if path.suffix.lower() in {".md", ".py", ".json", ".sh"}:
            content = path.read_text(encoding="utf-8")
            if pattern.search(content):
                placeholders.append(str(path.relative_to(root)))
    return make_check(
        "empty-and-placeholders",
        "fail" if empty or placeholders else "pass",
        empty=empty,
        placeholders=placeholders,
    )


def _check_syntax(root: Path) -> dict:
    errors: list[str] = []
    parsed = {"python": 0, "json": 0, "shell": 0}
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parsed["python"] += 1
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    for path in root.rglob("*.json"):
        parsed["json"] += 1
        try:
            strict_json_loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    bash = shutil.which("bash")
    shell_warning = None
    if bash:
        for path in root.rglob("*.sh"):
            parsed["shell"] += 1
            result = subprocess.run(
                [bash, "-n", path.relative_to(root).as_posix()],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                errors.append(f"{path.relative_to(root)}: {result.stderr.strip()}")
    else:
        shell_warning = "bash unavailable; shell syntax check skipped"
    status = "fail" if errors else ("warning" if shell_warning else "pass")
    return make_check("syntax", status, parsed=parsed, errors=errors, warning=shell_warning)


def _check_duplicates_and_size(root: Path, warn_bytes: int, max_bytes: int) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    duplicates: dict[str, list[str]] = {}
    for path in root.rglob("*.md"):
        size = path.stat().st_size
        relative = str(path.relative_to(root))
        if size > max_bytes:
            errors.append(f"{relative}: {size} bytes exceeds {max_bytes}")
        elif size > warn_bytes:
            warnings.append(f"{relative}: {size} bytes exceeds warning threshold {warn_bytes}")
        directives: list[str] = []
        in_fence = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\s*(```|~~~)", line):
                in_fence = not in_fence
                continue
            normalized = re.sub(r"\s+", " ", line.strip()).lower()
            if (
                not in_fence
                and normalized.startswith(("- ", "* "))
                and len(normalized) >= 40
                and any(word in normalized for word in DIRECTIVE_WORDS)
            ):
                directives.append(normalized)
        repeated = sorted(rule for rule, count in Counter(directives).items() if count > 1)
        if repeated:
            duplicates[relative] = repeated
            warnings.append(f"{relative}: {len(repeated)} repeated directive rule(s)")
    status = "fail" if errors else ("warning" if warnings else "pass")
    return make_check(
        "duplicates-and-size", status, errors=errors, warnings=warnings, duplicates=duplicates
    )


def _check_io_contract(root: Path) -> dict:
    errors: list[str] = []
    artifact = (root / "references/contracts/artifact-layout.md").read_text(encoding="utf-8")
    full = (root / "references/workflows/full-pipeline.md").read_text(encoding="utf-8")
    validation = (root / "references/workflows/code-validation.md").read_text(encoding="utf-8")
    for token in REQUIRED_ARTIFACTS:
        if token not in artifact:
            errors.append(f"artifact-layout missing {token}")
    for token in ("Optimizer/best_so_far.json", "Optimizer/triton_optimized.py", "triton_final.py", "summary.md"):
        if token not in full:
            errors.append(f"full-pipeline missing handoff {token}")
    for token in ("xxx_fix.py", "xxx_fix.md", "仅接收文件路径"):
        if token not in validation:
            errors.append(f"code-validation missing naming rule {token}")
    return make_check("input-output-contract", "fail" if errors else "pass", errors=errors)


def _check_final_selection(root: Path) -> dict:
    errors: list[str] = []
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    tuning = (root / "references/workflows/performance-tuning.md").read_text(encoding="utf-8")
    selector = root / "scripts/state/select-best-candidate.py"
    required = {
        "SKILL.md": (skill, "scripts/state/select-best-candidate.py"),
        "performance-tuning.md selector": (tuning, "select-best-candidate.py"),
        "performance-tuning.md checkpoint": (tuning, "best_so_far.json"),
    }
    for label, (content, token) in required.items():
        if token not in content:
            errors.append(f"{label} missing {token}")
    forbidden = (
        "若步骤 3 有执行，使用 `{output_dir}/Optimizer/triton_oob_optimized.py`",
        "若步骤 3 被跳过，使用 `{output_dir}/Optimizer/triton_advanced_optimized.py`",
    )
    for phrase in forbidden:
        if phrase in tuning:
            errors.append(f"inverted final-selection rule present: {phrase}")
    if not selector.is_file():
        errors.append("selector script is missing")
    else:
        source = selector.read_text(encoding="utf-8")
        for token in ("accuracy_pass", "latency_ms", "benchmark_signature", "best_so_far.py"):
            if token not in source:
                errors.append(f"selector missing invariant token {token}")
    return make_check("final-selection-invariant", "fail" if errors else "pass", errors=errors)


def _check_stage2_contracts(root: Path) -> dict:
    errors: list[str] = []
    execution_contract = root / "references/contracts/execution-backend.md"
    platform_contract = root / "references/backend/platform-rules.md"
    legacy_platform = root / "references/backend/device-rules.md"
    codegen_path = root / "references/workflows/code-generation.md"
    manifest_schema = root / "references/schemas/run-manifest.schema.json"
    manifest_updater = root / "scripts/state/update-run-manifest.py"

    for path in (execution_contract, platform_contract, codegen_path, manifest_schema, manifest_updater):
        if not path.is_file():
            errors.append(f"stage-2 contract missing: {path.relative_to(root)}")
    if legacy_platform.exists():
        errors.append("legacy platform source still exists: references/backend/device-rules.md")

    markdown_sources = {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*.md")
    }
    for relative, content in markdown_sources.items():
        if "device-rules.md" in content:
            errors.append(f"legacy platform source referenced by {relative}")

    detailed_backend_marker = "execution_backend=worker"
    marker_owners = [
        relative for relative, content in markdown_sources.items() if detailed_backend_marker in content
    ]
    expected_owner = "references/contracts/execution-backend.md"
    if marker_owners != [expected_owner]:
        errors.append(
            f"execution detail must have one owner {expected_owner}; observed {marker_owners}"
        )

    legacy_roles = (
        "computation-analyzer.md",
        "block-mapping-analyzer.md",
        "axis-planner.md",
        "specification-builder.md",
        "kernel-generator.md",
        "test-generator.md",
    )
    for role_name in legacy_roles:
        if (root / "references/roles" / role_name).exists():
            errors.append(f"legacy Code Gen role still exists: {role_name}")
        callers = [relative for relative, content in markdown_sources.items() if role_name in content]
        if callers:
            errors.append(f"legacy Code Gen role {role_name} referenced by {callers}")

    if codegen_path.is_file():
        codegen = codegen_path.read_text(encoding="utf-8")
        dispatches = codegen.count("spawn_agent(")
        if dispatches != 2:
            errors.append(f"normal Code Gen must declare exactly 2 outer dispatches, found {dispatches}")
        for role_name in ("kernel-designer.md", "kernel-builder.md"):
            if role_name not in codegen:
                errors.append(f"code-generation.md missing merged role {role_name}")

    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    artifact_layout = (root / "references/contracts/artifact-layout.md").read_text(encoding="utf-8")
    for label, content, tokens in (
        ("SKILL.md", skill, ("execution-backend.md", "platform-rules.md", "run_manifest.json")),
        ("artifact-layout.md", artifact_layout, ("run_manifest.json", "EnvConfig/config.json")),
    ):
        for token in tokens:
            if token not in content:
                errors.append(f"{label} missing Stage 2 marker {token}")

    return make_check("stage-2-architecture", "fail" if errors else "pass", errors=errors)


def _check_p21_contracts(root: Path) -> dict:
    errors: list[str] = []
    required = (
        root / "references/roles/reduction-baseline.md",
        root / "references/examples/code-generation/code-softmax-baseline.md",
        root / "references/schemas/optimization-surface.schema.json",
        root / "scripts/validation/validate-optimization-surface.py",
        root / "scripts/validation/test-p21-contracts.py",
    )
    for path in required:
        if not path.is_file():
            errors.append(f"p2.1 contract missing: {path.relative_to(root)}")

    callers = {
        "references/workflows/code-generation.md": ("validate-optimization-surface.py", "optimization_intent"),
        "references/roles/kernel-designer.md": ("reduction-baseline.md", "optimization_intent"),
        "references/roles/kernel-builder.md": ("reduction-baseline.md", "optimization_intent"),
        "references/strategies/autotune-config.md": ("optimization_surface", "num_stages=3"),
    }
    for relative, tokens in callers.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"p2.1 caller missing: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in content:
                errors.append(f"{relative} missing p2.1 handoff marker {token}")

    return make_check("p2.1-reduction-contract", "fail" if errors else "pass", errors=errors)


def _check_p3_contracts(root: Path) -> dict:
    errors: list[str] = []
    required = (
        root / "references/contracts/tuning-policy.md",
        root / "references/schemas/strategy-plan.schema.json",
        root / "references/schemas/tuning-state.schema.json",
        root / "scripts/state/plan-strategies.py",
        root / "scripts/state/manage-tuning-budget.py",
        root / "scripts/validation/test-p3-contracts.py",
    )
    for path in required:
        if not path.is_file():
            errors.append(f"p3 contract missing: {path.relative_to(root)}")

    callers = {
        "SKILL.md": ("correctness", "balanced", "max-performance", "strategy_plan.json", "tuning_state.json"),
        "references/workflows/full-pipeline.md": ("optimization_mode", "correctness-mode", "tuning-policy.md"),
        "references/workflows/performance-tuning.md": ("plan-strategies.py", "start-round", "--budget-state", "select-best-candidate.py"),
        "references/contracts/execution-backend.md": ("--budget-state", "`4`"),
        "scripts/execution/submit-remote-task.py": ("--budget-state", "reserve-worker", "sys.exit(4)"),
        "scripts/execution/run-budgeted-local.py": ("remaining_seconds", "TimeoutExpired", "elapsed-limit"),
    }
    for relative, tokens in callers.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"p3 caller missing: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in content:
                errors.append(f"{relative} missing p3 marker {token}")

    tuning_schema_path = root / "references/schemas/tuning-state.schema.json"
    if tuning_schema_path.is_file():
        schema = strict_json_loads(tuning_schema_path.read_text(encoding="utf-8"))
        limits = schema.get("properties", {}).get("limits", {}).get("properties", {})
        expected = {
            "max_deep_rounds": 3,
            "max_worker_calls": 16,
            "max_elapsed_seconds": 1800,
        }
        observed = {key: limits.get(key, {}).get("const") for key in expected}
        if observed != expected:
            errors.append(f"p3 fixed budget changed: expected {expected}, observed {observed}")

    manifest_schema_path = root / "references/schemas/run-manifest.schema.json"
    if manifest_schema_path.is_file():
        manifest_schema = strict_json_loads(manifest_schema_path.read_text(encoding="utf-8"))
        optimization_mode = manifest_schema.get("properties", {}).get("optimization_mode", {})
        if optimization_mode.get("enum") != ["correctness", "balanced", "max-performance"]:
            errors.append("run manifest does not define the three p3 optimization modes")
        if "optimization_mode" not in manifest_schema.get("required", []):
            errors.append("run manifest does not require optimization_mode")

    candidate_schema_path = root / "references/schemas/optimization-candidate.schema.json"
    selector_path = root / "scripts/state/select-best-candidate.py"
    if candidate_schema_path.is_file():
        candidate_schema = strict_json_loads(candidate_schema_path.read_text(encoding="utf-8"))
        expected_fields = {
            "schema_version", "candidate_id", "source_stage", "code_path", "report_path",
            "accuracy_pass", "latency_ms", "bandwidth_gbps", "execution_backend",
            "hardware_model", "benchmark_signature",
        }
        observed_fields = set(candidate_schema.get("properties", {}))
        if observed_fields != expected_fields:
            errors.append(
                "candidate schema changed; p3 must preserve p2.1 single-latency fields: "
                f"{sorted(observed_fields)}"
            )
    if selector_path.is_file():
        selector = selector_path.read_text(encoding="utf-8")
        if 'eligible.sort(key=lambda item: (float(item["latency_ms"]), item["candidate_id"]))' not in selector:
            errors.append("selector no longer uses p2.1 latency_ms ordering")
        forbidden = ("latency_p20", "latency_p80", "variance", "confidence_interval", "noise_score")
        present = [token for token in forbidden if token in selector]
        if present:
            errors.append(f"noise-aware selector fields are forbidden in p3: {present}")

    return make_check("p3-mode-routing-budget-contract", "fail" if errors else "pass", errors=errors)


def _check_p4_contracts(root: Path) -> dict:
    errors: list[str] = []
    required = (
        root / "references/contracts/cache-resume.md",
        root / "references/cache/stage-dependencies.json",
        root / "references/schemas/stage-dependencies.schema.json",
        root / "references/schemas/stage-fingerprint.schema.json",
        root / "references/schemas/stage-cache-record.schema.json",
        root / "references/schemas/resume-plan.schema.json",
        root / "references/schemas/change-impact-plan.schema.json",
        root / "scripts/state/fingerprint-stage.py",
        root / "scripts/state/stage-cache.py",
        root / "scripts/state/plan-resume.py",
        root / "scripts/state/apply-resume-plan.py",
        root / "scripts/validation/plan-change-impact.py",
        root / "scripts/validation/test-p4-contracts.py",
    )
    for path in required:
        if not path.is_file():
            errors.append(f"p4 contract missing: {path.relative_to(root)}")

    callers = {
        "SKILL.md": ("cache-resume.md", "content-addressed", "upstream fingerprints"),
        "references/workflows/full-pipeline.md": ("resume_plan.json", "fingerprint-stage.py", "stage-cache.py record"),
        "references/contracts/artifact-layout.md": ("RunState", "fingerprints", "cache"),
        "references/workflows/finalization.md": ("final-summary.md", "presentation-only"),
        "references/contracts/validation-gates.md": ("plan-change-impact.py", "corrupted-cache"),
    }
    for relative, tokens in callers.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"p4 caller missing: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        for token in tokens:
            if token not in content:
                errors.append(f"{relative} missing p4 marker {token}")

    registry_path = root / "references/cache/stage-dependencies.json"
    if registry_path.is_file():
        registry = strict_json_loads(registry_path.read_text(encoding="utf-8"))
        expected_stages = {
            "environment", "requirement-extraction", "code-generation",
            "code-validation", "performance-tuning", "finalization",
        }
        observed_stages = set(registry.get("stages", {}))
        if observed_stages != expected_stages:
            errors.append(f"stage dependency registry mismatch: {sorted(observed_stages)}")
        for owner, entries in {
            "shared_resources": registry.get("shared_resources", []),
            **registry.get("stages", {}),
        }.items():
            if len(entries) != len(set(entries)):
                errors.append(f"duplicate resource entries for {owner}")
            for relative in entries:
                resource = (root / relative).resolve()
                try:
                    resource.relative_to(root)
                except ValueError:
                    errors.append(f"registry resource escapes Skill root: {relative}")
                    continue
                if not resource.exists():
                    errors.append(f"registry resource is missing: {relative}")

    fingerprint_schema_path = root / "references/schemas/stage-fingerprint.schema.json"
    if fingerprint_schema_path.is_file():
        schema = strict_json_loads(fingerprint_schema_path.read_text(encoding="utf-8"))
        factors = schema.get("properties", {}).get("factors", {}).get("required", [])
        expected_factors = {
            "input_hash", "skill_version", "dependency_versions", "hardware_model",
            "toolchain_version", "stage_config", "upstream_fingerprints",
        }
        if set(factors) != expected_factors:
            errors.append(f"stage fingerprint factors changed: {sorted(factors)}")

    manifest_schema_path = root / "references/schemas/run-manifest.schema.json"
    if manifest_schema_path.is_file():
        manifest_schema = strict_json_loads(manifest_schema_path.read_text(encoding="utf-8"))
        if "resume" not in manifest_schema.get("required", []):
            errors.append("run manifest does not require p4 resume metadata")

    return make_check("p4-cache-resume-contract", "fail" if errors else "pass", errors=errors)


def main() -> int:
    args = _arguments()
    root = args.skill_root.resolve()
    started_at = utc_now()
    started_counter = perf_counter()
    checks = [
        _check_frontmatter(root),
        _check_references(root),
        _check_markdown(root),
        _check_files_and_placeholders(root),
        _check_syntax(root),
        _check_duplicates_and_size(root, args.warn_markdown_bytes, args.max_markdown_bytes),
        _check_io_contract(root),
        _check_final_selection(root),
        _check_stage2_contracts(root),
        _check_p21_contracts(root),
        _check_p3_contracts(root),
        _check_p4_contracts(root),
    ]
    report = make_report(
        level="L1",
        started_at=started_at,
        started_counter=started_counter,
        fingerprint=skill_fingerprint(root),
        checks=checks,
    )
    if report["elapsed_seconds"] > 30:
        report["checks"].append(
            make_check("time-budget", "fail", limit_seconds=30, actual=report["elapsed_seconds"])
        )
        report["status"] = "fail"
    else:
        report["checks"].append(
            make_check("time-budget", "pass", limit_seconds=30, actual=report["elapsed_seconds"])
        )
    return finish(report, args.report.resolve() if args.report else None)


if __name__ == "__main__":
    raise SystemExit(main())
