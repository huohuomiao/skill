#!/usr/bin/env python3
"""Offline validation entrypoint for the MLU Triton skill bundle.

The validator intentionally uses only the Python standard library so that Skill
changes can be checked before an MLU runtime or third-party packages are ready.
"""

from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SKILL_DIRS = (
    "mlu-triton-main",
    "mlu-triton-code-gen",
    "mlu-triton-optimize",
    "mlu-triton-code-review",
)
SCHEMA_BY_ARTIFACT = {
    "step1_base_info.json": "step1_base_info.schema.json",
    "step1_io_shapes.json": "step1_io_shapes.schema.json",
    "step2_block_mapping.json": "step2_block_mapping.schema.json",
    "step3_axis_fusion.json": "step3_axis_fusion.schema.json",
    "step4_code_spec.json": "step4_code_spec.schema.json",
}
OPTIMIZER_SCHEMA_BY_ARTIFACT = {
    "optimization_plan.json": "optimization_plan.schema.json",
    "optimization_state.json": "optimization_state.schema.json",
}


@dataclass
class Results:
    checks: int = 0
    errors: list[str] = field(default_factory=list)

    def check(self, condition: bool, label: str, detail: str = "") -> None:
        self.checks += 1
        if not condition:
            suffix = f": {detail}" if detail else ""
            self.errors.append(f"{label}{suffix}")

    def extend(self, prefix: str, errors: list[str]) -> None:
        self.checks += 1
        for error in errors:
            self.errors.append(f"{prefix}: {error}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def json_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def validate_schema(value: Any, schema: dict[str, Any], location: str = "$") -> list[str]:
    """Validate the JSON-Schema subset used by this repository."""
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(json_type_matches(value, item) for item in expected_types):
            errors.append(f"{location} expected type {expected_types}, got {type(value).__name__}")
            return errors

    if "const" in schema and value != schema["const"]:
        errors.append(f"{location} expected constant {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{location} must be one of {schema['enum']!r}, got {value!r}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < min_length:
            errors.append(f"{location} must contain at least {min_length} character(s)")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{location} must contain at least {min_items} item(s)")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(validate_schema(item, item_schema, f"{location}[{index}]"))

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{location} missing required property {key!r}")

        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(validate_schema(value[key], child_schema, f"{location}.{key}"))

        additional = schema.get("additionalProperties", True)
        unknown_keys = set(value) - set(properties)
        if additional is False:
            for key in sorted(unknown_keys):
                errors.append(f"{location} contains unexpected property {key!r}")
        elif isinstance(additional, dict):
            for key in sorted(unknown_keys):
                errors.extend(validate_schema(value[key], additional, f"{location}.{key}"))

    return errors


def load_schemas(results: Results) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    schema_dir = ROOT / "validation" / "contracts"
    for path in sorted(schema_dir.glob("*.schema.json")):
        try:
            schema = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            results.check(False, f"schema parses: {path.relative_to(ROOT)}", str(exc))
            continue
        results.check(isinstance(schema, dict), f"schema is an object: {path.relative_to(ROOT)}")
        results.check("$schema" in schema, f"schema declares dialect: {path.relative_to(ROOT)}")
        schemas[path.name] = schema
    results.check(
        (set(SCHEMA_BY_ARTIFACT.values()) | set(OPTIMIZER_SCHEMA_BY_ARTIFACT.values())).issubset(schemas),
        "all artifact schemas exist",
        f"expected {sorted(set(SCHEMA_BY_ARTIFACT.values()) | set(OPTIMIZER_SCHEMA_BY_ARTIFACT.values()))}",
    )
    return schemas


def check_frontmatter(results: Results, skill_dir: str) -> None:
    path = ROOT / skill_dir / "SKILL.md"
    results.check(path.is_file(), f"Skill entry exists: {skill_dir}/SKILL.md")
    if not path.is_file():
        return
    text = read_text(path)
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    results.check(match is not None, f"frontmatter exists: {skill_dir}/SKILL.md")
    if match is None:
        return
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    results.check(fields.get("name") == skill_dir, f"Skill name matches folder: {skill_dir}")
    results.check(bool(fields.get("description")), f"Skill description is non-empty: {skill_dir}")


def check_source_tree() -> Results:
    results = Results()
    for skill_dir in SKILL_DIRS:
        check_frontmatter(results, skill_dir)

    required_paths = (
        ROOT / "share" / "mlu" / "runtime" / "get_device_info.py",
        ROOT / "share" / "mlu" / "runtime" / "test_env_code.py",
        ROOT / "mlu-triton-main" / "subagents" / "scripts" / "submit_task_to_worker.py",
        ROOT / "share" / "mlu" / "references" / "platform-rules.md",
    )
    for path in required_paths:
        results.check(path.is_file(), f"required path exists: {path.relative_to(ROOT)}")

    deleted_script = (
        ROOT / "mlu-triton-optimize" / "perf-analyzer" / "scripts" / "analyzer_rep.py"
    )
    results.check(not deleted_script.exists(), "deprecated analyzer_rep.py is absent")
    control_script = ROOT / "mlu-triton-optimize" / "scripts" / "optimization_control.py"
    results.check(control_script.is_file(), "optimization control script exists")

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        results.check(path.stat().st_size > 0, f"file is non-empty: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            try:
                ast.parse(read_text(path), filename=str(path))
            except (OSError, SyntaxError, UnicodeError) as exc:
                results.check(False, f"Python syntax: {path.relative_to(ROOT)}", str(exc))
            else:
                results.check(True, f"Python syntax: {path.relative_to(ROOT)}")
        if path.suffix == ".md":
            fence_count = sum(1 for line in read_text(path).splitlines() if line.lstrip().startswith("```"))
            results.check(fence_count % 2 == 0, f"Markdown fences balanced: {path.relative_to(ROOT)}")

    hardcoded_pattern = re.compile(r"\.claude/skills/[A-Za-z0-9_./-]+")
    unresolved: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        for match in hardcoded_pattern.finditer(read_text(path)):
            raw_token = match.group(0)
            relative = raw_token.removeprefix(".claude/skills/").rstrip("./-")
            token = f".claude/skills/{relative}" if relative else ".claude/skills"
            if relative and not (ROOT / relative).exists():
                unresolved.append(f"{path.relative_to(ROOT)} -> {token}")
    results.check(not unresolved, "hard-coded skill references resolve", "; ".join(unresolved[:10]))

    codegen = read_text(ROOT / "mlu-triton-code-gen" / "SKILL.md")
    optimize = read_text(ROOT / "mlu-triton-optimize" / "SKILL.md")
    review = read_text(ROOT / "mlu-triton-code-review" / "SKILL.md")
    env_config = read_text(ROOT / "mlu-triton-main" / "subagents" / "EnvConfig.md")

    results.check("并立即执行一次" not in codegen, "Code Gen does not execute tests before Code Review")
    results.check(
        'args="{step7_input_code_path}"' in codegen,
        "Code Review invocation uses the single-path contract",
    )
    results.check("仅接收文件路径" in review, "Code Review documents the single-path contract")
    results.check("correctness" in optimize and "balanced" in optimize and "max-performance" in optimize, "all optimization modes are documented")
    results.check("optimization_plan.json" in optimize and "optimization_state.json" in optimize, "plan and budget state are required")
    results.check("selected=false" in optimize, "unselected strategies are forbidden")
    results.check("optimization_state.json.best.artifact_path" in optimize, "best measured candidate has final priority")
    results.check("--resume" in optimize and "禁止再次初始化" in optimize, "Optimizer preserves budget on resume")
    main_skill = read_text(ROOT / "mlu-triton-main" / "SKILL.md")
    results.check("`correctness` 跳过步骤 4" in main_skill, "Main supports the correctness fast path")
    results.check("不得通过新建状态文件绕过预算" in main_skill, "Main preserves one global optimization budget")
    results.check(
        ".claude/skills/share/mlu/runtime/test_env_code.py" in env_config
        and ".claude/skills/share/mlu/runtime/get_device_info.py" in env_config,
        "EnvConfig points to the shared runtime scripts",
    )
    results.check(
        "| ------ | ------------------- | -------------------- | --------------------------------- | --------------------------------------------------- |"
        in codegen,
        "Code Gen Step 2 table has a complete separator row",
    )

    load_schemas(results)
    return results


def check_stage3_control() -> Results:
    results = Results()
    control_path = ROOT / "mlu-triton-optimize" / "scripts" / "optimization_control.py"
    results.check(control_path.is_file(), "stage3 control module exists")
    if not control_path.is_file():
        return results
    spec = importlib.util.spec_from_file_location("optimization_control_validation", control_path)
    results.check(spec is not None and spec.loader is not None, "stage3 control module can be loaded")
    if spec is None or spec.loader is None:
        return results
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    reduction_source = """
@triton.jit
def reduce_kernel(x, y, n: tl.constexpr, BLOCK_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    value = tl.load(x + offsets)
    result = tl.sum(value, axis=0)
    tl.store(y + pid, result)

def wrapper(x, y):
    TOTAL_CORE_NUM = 16
    grid = (min(16, TOTAL_CORE_NUM),)
    reduce_kernel[grid](x, y, 16, BLOCK_M=256)

def test_accuracy():
    torch.testing.assert_close(wrapper(x, y), ref)

def benchmark():
    return triton.testing.do_bench(lambda: wrapper(x, y))
"""
    math_source = """
@triton.jit
def math_kernel(x, y, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(x + offsets)
    tl.store(y + offsets, tl.exp(value) / 3.0)

def wrapper(x, y, n):
    grid = (triton.cdiv(n, 256),)
    math_kernel[grid](x, y, n, BLOCK_SIZE=256)

def test_accuracy():
    torch.testing.assert_close(wrapper(x, y, n), ref)

def benchmark():
    return triton.testing.do_bench(lambda: wrapper(x, y, n))
"""
    test_only_division_source = """
@triton.jit
def copy_kernel(x, y, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(x + offsets)
    tl.store(y + offsets, value)

def wrapper(x, y, n):
    grid = (triton.cdiv(n, 256),)
    copy_kernel[grid](x, y, n, BLOCK_SIZE=256)

def test_accuracy():
    reference = x / 2.0
    torch.testing.assert_close(wrapper(x, y, n), reference)
"""

    correctness = module.build_plan(reduction_source, "reduce.py", "correctness")
    results.check(not correctness["selected_oob_strategies"], "correctness selects no OOB strategy")
    results.check(not correctness["advanced"]["enabled"], "correctness disables advanced optimization")

    balanced = module.build_plan(reduction_source, "reduce.py", "balanced")
    results.check("reduce-opt" in balanced["selected_oob_strategies"], "reduction selects reduce-opt")
    results.check("retiling" in balanced["selected_oob_strategies"], "reduction selects retiling")
    results.check("modify-grid" not in balanced["selected_oob_strategies"], "core-capped 1D grid skips modify-grid")
    results.check(not balanced["advanced"]["enabled"], "balanced disables advanced optimization")

    maximum = module.build_plan(math_source, "math.py", "max-performance")
    results.check(maximum["advanced"]["enabled"], "max-performance enables advanced optimization")
    for strategy in ("libdevice-opt", "config-tuner", "div-to-mul"):
        results.check(strategy in maximum["advanced"]["candidates"], f"max-performance routes {strategy}")
    results.check("reduce-opt" not in maximum["selected_oob_strategies"], "elementwise math skips reduce-opt")
    results.check("modify-grid" in maximum["selected_oob_strategies"], "uncapped grid selects modify-grid")

    test_only = module.build_plan(test_only_division_source, "copy.py", "max-performance")
    results.check("div-to-mul" not in test_only["advanced"]["candidates"], "division in test code does not route kernel optimization")

    incomplete = module.build_plan("def f(): return 1", "bad.py", "balanced")
    results.check(incomplete["manual_review_required"], "incomplete Triton input requires manual review")

    state = module.build_state(maximum, Path("optimization_plan.json"))
    allowed, reason = module.budget_decision(state, "advanced")
    results.check(allowed and reason is None, "fresh max-performance state allows advanced work")

    subagent_exhausted = copy.deepcopy(state)
    subagent_exhausted["usage"]["subagent_calls"] = subagent_exhausted["limits"]["max_subagent_calls"]
    allowed, reason = module.budget_decision(subagent_exhausted, "oob")
    results.check(not allowed and reason == "max_subagent_calls_reached", "subagent hard limit stops work")

    patience_state = copy.deepcopy(state)
    for _ in range(2):
        module.record_event(
            patience_state,
            "advanced_iteration",
            "advanced",
            "config-tuner",
            "completed",
            True,
            10.0,
            9.9,
            "/tmp/candidate.py",
        )
    allowed, reason = module.budget_decision(patience_state, "advanced")
    results.check(not allowed and reason == "advanced_patience_exhausted", "advanced patience stops low-yield iterations")

    improvement_state = copy.deepcopy(state)
    improvement_state["advanced_no_improvement"] = 1
    module.record_event(
        improvement_state,
        "advanced_iteration",
        "advanced",
        "config-tuner",
        "completed",
        True,
        10.0,
        9.7,
        "/tmp/better.py",
    )
    results.check(improvement_state["advanced_no_improvement"] == 0, "meaningful improvement resets patience")

    def quiet_plan(arguments: argparse.Namespace) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return int(module.command_plan(arguments))

    with tempfile.TemporaryDirectory(prefix="mlu-triton-v3-1-") as temporary:
        temp_root = Path(temporary)
        input_path = temp_root / "kernel.py"
        input_path.write_text(math_source, encoding="utf-8")
        output_dir = temp_root / "Optimizer"
        plan_args = argparse.Namespace(
            input=input_path,
            output_dir=output_dir,
            mode="balanced",
            budget_file=None,
            resume=False,
        )
        results.check(quiet_plan(plan_args) == 0, "first plan initializes budget state")
        state_path = output_dir / "optimization_state.json"
        persisted = module.read_json(state_path)
        persisted["usage"]["subagent_calls"] = 3
        persisted["advanced_no_improvement"] = 1
        persisted["history"].append({"event": "fixture_consumption"})
        module.write_json_atomic(state_path, persisted)

        plan_args.resume = True
        results.check(quiet_plan(plan_args) == 0, "compatible plan resumes")
        resumed = module.read_json(state_path)
        results.check(resumed["usage"]["subagent_calls"] == 3, "resume preserves consumed budget")
        results.check(resumed["advanced_no_improvement"] == 1, "resume preserves patience")
        results.check(resumed["history"] == persisted["history"], "resume preserves history")

        plan_args.resume = False
        results.check(quiet_plan(plan_args) == 2, "plain plan refuses to overwrite existing state")

        plan_args.resume = True
        input_path.write_text(math_source.replace("tl.exp(value)", "tl.log(value)"), encoding="utf-8")
        results.check(quiet_plan(plan_args) == 2, "resume rejects changed input hash")

        input_path.write_text(math_source, encoding="utf-8")
        state_path.unlink()
        results.check(quiet_plan(plan_args) == 2, "resume rejects a partial plan/state pair")
    return results


def check_behavior_cases() -> Results:
    results = Results()
    schemas = load_schemas(results)
    case_path = ROOT / "validation" / "cases" / "behavior_cases.json"
    try:
        manifest = load_json(case_path)
    except (OSError, json.JSONDecodeError) as exc:
        results.check(False, "behavior case manifest parses", str(exc))
        return results

    results.check(manifest.get("version") == 1, "behavior manifest version is supported")
    cases = manifest.get("cases")
    results.check(isinstance(cases, list) and len(cases) >= 6, "at least six behavior cases exist")
    if isinstance(cases, list):
        ids: list[str] = []
        for index, case in enumerate(cases):
            label = f"behavior case #{index + 1}"
            results.check(isinstance(case, dict), f"{label} is an object")
            if not isinstance(case, dict):
                continue
            case_id = case.get("id")
            results.check(isinstance(case_id, str) and bool(case_id), f"{label} has an id")
            if isinstance(case_id, str):
                ids.append(case_id)
            expected = case.get("expected_route")
            results.check(isinstance(expected, dict), f"{label} has expected_route")
            if isinstance(expected, dict):
                must_run = expected.get("must_run", [])
                must_skip = expected.get("must_skip", [])
                results.check(isinstance(must_run, list), f"{label} must_run is a list")
                results.check(isinstance(must_skip, list), f"{label} must_skip is a list")
                if isinstance(must_run, list) and isinstance(must_skip, list):
                    results.check(
                        set(must_run).isdisjoint(must_skip),
                        f"{label} does not both run and skip a stage",
                    )
            assertions = case.get("source_assertions", [])
            results.check(isinstance(assertions, list) and bool(assertions), f"{label} has source assertions")
            if isinstance(assertions, list):
                for assertion in assertions:
                    if not isinstance(assertion, dict):
                        results.check(False, f"{label} assertion is an object")
                        continue
                    source = ROOT / str(assertion.get("file", ""))
                    results.check(source.is_file(), f"{label} assertion source exists", str(source))
                    if not source.is_file():
                        continue
                    text = read_text(source)
                    for token in assertion.get("contains", []):
                        results.check(token in text, f"{label} source contains invariant", token)
                    for token in assertion.get("not_contains", []):
                        results.check(token not in text, f"{label} source excludes invalid behavior", token)
        results.check(len(ids) == len(set(ids)), "behavior case ids are unique")

    valid_dir = ROOT / "validation" / "fixtures" / "valid" / "KernelGen"
    for artifact, schema_name in SCHEMA_BY_ARTIFACT.items():
        path = valid_dir / artifact
        results.check(path.is_file(), f"valid fixture exists: {artifact}")
        if path.is_file() and schema_name in schemas:
            errors = validate_schema(load_json(path), schemas[schema_name])
            results.extend(f"valid fixture matches {schema_name}", errors)

    base_info = load_json(valid_dir / "step1_base_info.json")
    io_shapes = load_json(valid_dir / "step1_io_shapes.json")
    results.check(base_info.get("io_shapes") == io_shapes, "step1 io_shapes fixture is consistent")

    invalid_path = ROOT / "validation" / "fixtures" / "invalid" / "step1_missing_required.json"
    results.check(invalid_path.is_file(), "negative fixture exists")
    if invalid_path.is_file() and "step1_base_info.schema.json" in schemas:
        errors = validate_schema(load_json(invalid_path), schemas["step1_base_info.schema.json"])
        results.check(bool(errors), "negative fixture is rejected by the schema")
    optimizer_valid_dir = ROOT / "validation" / "fixtures" / "valid" / "Optimizer"
    for artifact, schema_name in OPTIMIZER_SCHEMA_BY_ARTIFACT.items():
        path = optimizer_valid_dir / artifact
        results.check(path.is_file(), f"valid optimizer fixture exists: {artifact}")
        if path.is_file() and schema_name in schemas:
            errors = validate_schema(load_json(path), schemas[schema_name])
            results.extend(f"valid optimizer fixture matches {schema_name}", errors)

    stage3 = check_stage3_control()
    results.checks += stage3.checks
    results.errors.extend(stage3.errors)
    return results


def check_output_artifacts(output_dir: Path, require_complete: bool) -> Results:
    results = Results()
    output_dir = output_dir.resolve()
    results.check(output_dir.is_dir(), "output directory exists", str(output_dir))
    if not output_dir.is_dir():
        return results

    schemas = load_schemas(results)
    validated_count = 0
    kernel_dir = output_dir / "KernelGen"
    for artifact, schema_name in SCHEMA_BY_ARTIFACT.items():
        path = kernel_dir / artifact
        if not path.exists():
            continue
        validated_count += 1
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            results.check(False, f"artifact parses: {path}", str(exc))
            continue
        errors = validate_schema(value, schemas[schema_name])
        results.extend(f"artifact matches {schema_name}", errors)
    optimizer_dir = output_dir / "Optimizer"
    plan_path = optimizer_dir / "optimization_plan.json"
    state_path = optimizer_dir / "optimization_state.json"
    if plan_path.is_file() and state_path.is_file():
        plan_value = load_json(plan_path)
        state_value = load_json(state_path)
        results.check(plan_value.get("mode") == state_value.get("mode"), "optimization plan and state modes match")
        results.check(plan_value.get("limits") == state_value.get("limits"), "optimization plan and state limits match")
        selected_from_rows = [
            item.get("name")
            for item in plan_value.get("oob_strategies", [])
            if isinstance(item, dict) and item.get("selected") is True
        ]
        results.check(selected_from_rows == plan_value.get("selected_oob_strategies"), "selected OOB list matches routed rows")
    fast_path_source = output_dir / "Extractor" / "original_code.py"
    results.check(
        validated_count > 0 or fast_path_source.is_file(),
        "known JSON artifacts exist, or the run used the Triton fast path",
    )

    for artifact, schema_name in OPTIMIZER_SCHEMA_BY_ARTIFACT.items():
        path = optimizer_dir / artifact
        if not path.exists():
            continue
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            results.check(False, f"artifact parses: {path}", str(exc))
            continue
        errors = validate_schema(value, schemas[schema_name])
        results.extend(f"artifact matches {schema_name}", errors)

    base_path = kernel_dir / "step1_base_info.json"
    shapes_path = kernel_dir / "step1_io_shapes.json"
    if base_path.is_file() and shapes_path.is_file():
        results.check(
            load_json(base_path).get("io_shapes") == load_json(shapes_path),
            "step1_base_info.io_shapes equals step1_io_shapes",
        )

    config_path = output_dir / "EnvConfig" / "config.md"
    if config_path.is_file():
        config_text = read_text(config_path)
        results.check(
            re.search(r"execution_backend\s*[:：]\s*(local|worker)", config_text) is not None,
            "EnvConfig records a valid execution_backend",
        )
    elif require_complete:
        results.check(False, "complete output contains EnvConfig/config.md")

    if require_complete:
        required_outputs = [
            output_dir / "KernelGen" / "triton_code_fix.py",
            output_dir / "KernelGen" / "triton_report.md",
            output_dir / "triton_final.py",
            output_dir / "summary.md",
        ]
        if optimizer_dir.is_dir():
            required_outputs.extend(
                [
                    optimizer_dir / "triton_optimized.py",
                    optimizer_dir / "triton_optimized.md",
                    optimizer_dir / "optimization_plan.json",
                    optimizer_dir / "optimization_state.json",
                ]
            )
        for path in required_outputs:
            results.check(
                path.is_file() and path.stat().st_size > 0,
                f"complete output is non-empty: {path.relative_to(output_dir)}",
            )
        summary_path = output_dir / "summary.md"
        if summary_path.is_file():
            summary = read_text(summary_path)
            for token in ("accuracy_pass", "atol", "rtol", "Code Gen", "Optimize", "triton_final.py"):
                results.check(token in summary, "summary contains required evidence", token)
    return results


def print_results(level: str, results: Results) -> int:
    if results.errors:
        print(f"[FAIL] {level}: {len(results.errors)} error(s), {results.checks} check(s)")
        for error in results.errors:
            print(f"  - {error}")
        return 1
    print(f"[PASS] {level}: {results.checks} check(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("l1", help="Run source-tree and contract checks")
    subparsers.add_parser("l2", help="Run offline behavior-contract checks")
    subparsers.add_parser("all", help="Run L1 and L2; never requires MLU")
    artifacts = subparsers.add_parser("artifacts", help="Validate artifacts from a real run")
    artifacts.add_argument("--output-dir", required=True, type=Path)
    artifacts.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    exit_code = 0
    if args.command in {"l1", "all"}:
        exit_code |= print_results("L1", check_source_tree())
    if args.command in {"l2", "all"}:
        exit_code |= print_results("L2", check_behavior_cases())
    if args.command == "artifacts":
        exit_code |= print_results(
            "artifacts",
            check_output_artifacts(args.output_dir, args.require_complete),
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
