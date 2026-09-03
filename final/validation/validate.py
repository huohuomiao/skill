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
import shutil
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
REVIEW_SCHEMA_BY_ARTIFACT = {
    "review_result.json": "review_result.schema.json",
}
RUNTIME_SCHEMA_BY_ARTIFACT = {
    "run_context.json": "run_context.schema.json",
    "run_manifest.json": "run_manifest.schema.json",
    "regression_result.json": "regression_result.schema.json",
    "regression_report.json": "regression_report.schema.json",
    "regression_policy.json": "regression_policy.schema.json",
}
CONTROL_SCHEMA_NAMES = {
    "cache_metadata.schema.json",
    "review_result.schema.json",
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
    schema_dir = ROOT / "share" / "contracts"
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
        (
            set(SCHEMA_BY_ARTIFACT.values())
            | set(OPTIMIZER_SCHEMA_BY_ARTIFACT.values())
            | set(REVIEW_SCHEMA_BY_ARTIFACT.values())
            | set(RUNTIME_SCHEMA_BY_ARTIFACT.values())
            | CONTROL_SCHEMA_NAMES
        ).issubset(schemas),
        "all artifact schemas exist",
        f"expected {sorted(set(SCHEMA_BY_ARTIFACT.values()) | set(OPTIMIZER_SCHEMA_BY_ARTIFACT.values()) | set(REVIEW_SCHEMA_BY_ARTIFACT.values()) | set(RUNTIME_SCHEMA_BY_ARTIFACT.values()) | CONTROL_SCHEMA_NAMES)}",
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
        ROOT / "share" / "manifest.json",
        ROOT / "share" / "contracts" / "cache_metadata.schema.json",
        ROOT / "share" / "contracts" / "review_result.schema.json",
        ROOT / "mlu-triton-code-gen" / "subagents" / "DesignKernel.md",
        ROOT / "mlu-triton-code-gen" / "subagents" / "BuildKernel.md",
        ROOT / "mlu-triton-code-review" / "ReviewAndFix.md",
    )
    for path in required_paths:
        results.check(path.is_file(), f"required path exists: {path.relative_to(ROOT)}")

    deleted_script = (
        ROOT / "mlu-triton-optimize" / "perf-analyzer" / "scripts" / "analyzer_rep.py"
    )
    results.check(not deleted_script.exists(), "deprecated analyzer_rep.py is absent")
    control_script = ROOT / "mlu-triton-optimize" / "scripts" / "optimization_control.py"
    results.check(control_script.is_file(), "optimization control script exists")
    run_control = ROOT / "mlu-triton-main" / "scripts" / "run_control.py"
    regression_control = ROOT / "validation" / "regression.py"
    stage_sources = ROOT / "mlu-triton-main" / "references" / "stage-sources.json"
    results.check(run_control.is_file(), "run/cache/resume control script exists")
    results.check(regression_control.is_file(), "continuous regression comparator exists")
    results.check(stage_sources.is_file(), "stage source boundary config exists")

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
        'args="{output_dir}/KernelGen/step6_test_code.py"' in codegen,
        "Code Review invocation uses the single-path contract",
    )
    results.check("Accept exactly one absolute" in review, "Code Review documents the single-path contract")
    results.check("correctness" in optimize and "balanced" in optimize and "max-performance" in optimize, "all optimization modes are documented")
    results.check("optimization_plan.json" in optimize and "optimization_state.json" in optimize, "plan and budget state are required")
    results.check("selected=false" in optimize, "unselected strategies are forbidden")
    results.check("optimization_state.json.best.artifact_path" in optimize, "best measured candidate has final priority")
    main_skill = read_text(ROOT / "mlu-triton-main" / "SKILL.md")
    results.check("`correctness` 跳过步骤 4" in main_skill, "Main supports the correctness fast path")
    results.check("不得通过新建状态文件绕过预算" in main_skill, "Main preserves one global optimization budget")
    results.check("run_manifest.json" in main_skill and "run_control.py resume" in main_skill, "Main requires resumable run control")
    results.check("checkpoint-status" in codegen and "checkpoint-save" in codegen, "Code Gen uses inner checkpoints")
    results.check("--resume" in optimize and "禁止再次初始化" in optimize, "Optimizer resumes without resetting budget")
    results.check(
        ".claude/skills/share/mlu/runtime/test_env_code.py" in env_config
        and ".claude/skills/share/mlu/runtime/get_device_info.py" in env_config,
        "EnvConfig points to the shared runtime scripts",
    )
    results.check("DesignKernel -> BuildKernel" in codegen, "Code Gen uses the merged design/build route")
    results.check("review_result.json" in codegen and "review_result.json" in review, "Code Review emits machine-readable L3 evidence")
    results.check("--validation-level l1" in codegen, "Code Gen checkpoints record validation levels")

    load_schemas(results)
    return results


def check_dispatch_control() -> Results:
    results = Results()
    metrics_path = ROOT / "mlu-triton-code-gen" / "scripts" / "dispatch_metrics.py"
    results.check(metrics_path.is_file(), "dispatch metrics script exists")
    if not metrics_path.is_file():
        return results
    try:
        metrics = load_python_module("dispatch_metrics_validation", metrics_path)
    except Exception as exc:
        results.check(False, "dispatch metrics module loads", str(exc))
        return results
    results.check(True, "dispatch metrics module loads")
    for route in ("normal", "triton-fast"):
        for outcome in ("direct-pass", "repair"):
            try:
                report = metrics.analyze(route, outcome)
            except Exception as exc:
                results.check(False, f"dispatch metrics analyze {route}/{outcome}", str(exc))
                continue
            results.check(True, f"dispatch metrics analyze {route}/{outcome}")
            if route == "normal":
                results.check(
                    report["dispatches"]["reduction_pct"] >= 50.0,
                    f"normal dispatch reduction retained for {outcome}",
                )
                results.check(
                    report["static_context"]["reduction_pct"] >= 50.0,
                    f"normal static-context reduction retained for {outcome}",
                )
    return results


def check_stage2_control() -> Results:
    results = Results()
    control_path = ROOT / "mlu-triton-optimize" / "scripts" / "optimization_control.py"
    results.check(control_path.is_file(), "stage2 control module exists")
    if not control_path.is_file():
        return results
    spec = importlib.util.spec_from_file_location("optimization_control_validation", control_path)
    results.check(spec is not None and spec.loader is not None, "stage2 control module can be loaded")
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
    return results


def load_python_module(label: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(label, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quiet_call(function: Any, *args: Any, **kwargs: Any) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return int(function(*args, **kwargs))


def write_fixture_run_context(output_dir: Path) -> Path:
    env_dir = output_dir / "EnvConfig"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "config.md").write_text(
        "# Environment\n\nexecution_backend: local\n", encoding="utf-8"
    )
    (env_dir / "runtime_info.txt").write_text("verified MLU fixture\n", encoding="utf-8")
    context_path = env_dir / "run_context.json"
    shutil.copy2(
        ROOT / "validation" / "fixtures" / "valid" / "RunControl" / "run_context.json",
        context_path,
    )
    return context_path


def check_stage4_control() -> Results:
    results = Results()
    run_path = ROOT / "mlu-triton-main" / "scripts" / "run_control.py"
    regression_path = ROOT / "validation" / "regression.py"
    optimization_path = ROOT / "mlu-triton-optimize" / "scripts" / "optimization_control.py"
    try:
        run = load_python_module("run_control_validation", run_path)
        regression = load_python_module("regression_validation", regression_path)
        optimization = load_python_module("optimization_resume_validation", optimization_path)
    except Exception as exc:
        results.check(False, "stage4 modules load", str(exc))
        return results
    results.check(True, "stage4 modules load")

    schemas = load_schemas(results)
    context_fixture = ROOT / "validation" / "fixtures" / "valid" / "RunControl" / "run_context.json"
    results.extend(
        "run context fixture matches schema",
        validate_schema(load_json(context_fixture), schemas["run_context.schema.json"]),
    )

    with tempfile.TemporaryDirectory(prefix="mlu-triton-v4-") as temporary:
        root = Path(temporary)
        cache_dir = root / "cache"
        output_one = root / "run-one"
        manifest_one = output_one / "run_manifest.json"
        init_args = argparse.Namespace(
            output_dir=output_one,
            input="deterministic reduce-sum request",
            mode="balanced",
            budget_file=None,
            cache_dir=cache_dir,
        )
        results.check(quiet_call(run.command_init, init_args) == 0, "run manifest initializes")
        config = run.load_stage_config()
        satisfied, missing = run.validation_satisfies(["l1"], ["l1", "l2"])
        results.check(
            not satisfied and missing == ["l2"],
            "cache promotion detects missing validation levels",
        )
        low_output = root / "run-insufficient-validation"
        low_manifest = low_output / "run_manifest.json"
        results.check(
            quiet_call(
                run.command_init,
                argparse.Namespace(
                    output_dir=low_output,
                    input="validation-gating request",
                    mode="correctness",
                    budget_file=None,
                    cache_dir=cache_dir,
                ),
            )
            == 0,
            "insufficient-validation fixture initializes",
        )
        results.check(
            quiet_call(
                run.command_start,
                argparse.Namespace(manifest=low_manifest, stage="env_config"),
            )
            == 0,
            "insufficient-validation EnvConfig starts",
        )
        write_fixture_run_context(low_output)
        results.check(
            quiet_call(
                run.command_complete,
                argparse.Namespace(
                    manifest=low_manifest,
                    stage="env_config",
                    artifact=[],
                    validation_level=[],
                ),
            )
            == 0,
            "insufficient-validation EnvConfig completes",
        )
        results.check(
            quiet_call(
                run.command_start,
                argparse.Namespace(manifest=low_manifest, stage="extractor"),
            )
            == 0,
            "insufficient-validation Extractor starts",
        )
        low_extractor = low_output / "Extractor"
        low_extractor.mkdir(parents=True)
        (low_extractor / "requirement.md").write_text(
            "# L1-only requirement\n", encoding="utf-8"
        )
        results.check(
            quiet_call(
                run.command_complete,
                argparse.Namespace(
                    manifest=low_manifest,
                    stage="extractor",
                    artifact=[],
                    validation_level=["l1"],
                ),
            )
            == 0,
            "stage may complete without cache promotion",
        )
        low_state = run.load_manifest(low_manifest)["stages"]["extractor"]
        results.check(
            low_state["status"] == "complete"
            and low_state["cache_key"] is None
            and low_state["validation_levels"] == ["l1"],
            "insufficient validation never publishes a cache entry",
        )
        manifest = run.load_manifest(manifest_one)
        results.check(run.next_action(manifest, config)["stage"] == "env_config", "env_config is first")
        results.check(
            run.next_action(manifest, config)["action"] == "run",
            "env_config is never restored from cache",
        )

        results.check(
            quiet_call(run.command_start, argparse.Namespace(manifest=manifest_one, stage="env_config")) == 0,
            "env_config starts transactionally",
        )
        context_one = write_fixture_run_context(output_one)
        results.check(
            quiet_call(
                run.command_complete,
                argparse.Namespace(
                    manifest=manifest_one,
                    stage="env_config",
                    artifact=[],
                    validation_level=[],
                ),
            )
            == 0,
            "env_config completes only with required artifacts",
        )
        results.check(
            quiet_call(
                run.command_bind_context,
                argparse.Namespace(manifest=manifest_one, context_file=context_one),
            )
            == 0,
            "verified runtime context binds",
        )

        results.check(
            quiet_call(run.command_start, argparse.Namespace(manifest=manifest_one, stage="extractor")) == 0,
            "extractor starts",
        )
        extractor_dir = output_one / "Extractor"
        extractor_dir.mkdir(parents=True)
        (extractor_dir / "requirement.md").write_text("# deterministic requirement\n", encoding="utf-8")
        results.check(
            quiet_call(
                run.command_complete,
                argparse.Namespace(
                    manifest=manifest_one,
                    stage="extractor",
                    artifact=[],
                    validation_level=["l1", "l2"],
                ),
            )
            == 0,
            "extractor populates content-addressed cache",
        )

        results.check(
            quiet_call(run.command_start, argparse.Namespace(manifest=manifest_one, stage="kernel_gen")) == 0,
            "kernel_gen starts after context binding",
        )
        kernel_dir = output_one / "KernelGen"
        kernel_dir.mkdir(parents=True)
        shutil.copy2(
            ROOT / "validation" / "fixtures" / "valid" / "KernelGen" / "step1_base_info.json",
            kernel_dir / "step1_base_info.json",
        )
        shutil.copy2(
            ROOT / "validation" / "fixtures" / "valid" / "KernelGen" / "step1_io_shapes.json",
            kernel_dir / "step1_io_shapes.json",
        )
        checkpoint_args = argparse.Namespace(
            manifest=manifest_one,
            stage="kernel_gen",
            name="step1",
            artifact=["KernelGen/step1_base_info.json", "KernelGen/step1_io_shapes.json"],
            validation_level=["l1", "l2"],
        )
        insufficient_checkpoint_args = copy.deepcopy(checkpoint_args)
        insufficient_checkpoint_args.validation_level = ["l1"]
        try:
            quiet_call(run.command_checkpoint_save, insufficient_checkpoint_args)
        except run.ControlError:
            results.check(True, "checkpoint rejects insufficient validation evidence")
        else:
            results.check(False, "checkpoint rejects insufficient validation evidence")
        results.check(quiet_call(run.command_checkpoint_save, checkpoint_args) == 0, "inner checkpoint saves")
        results.check(
            quiet_call(
                run.command_status,
                argparse.Namespace(manifest=manifest_one),
                recover=True,
            )
            == 0,
            "interrupted stage reopens",
        )
        manifest = run.load_manifest(manifest_one)
        checkpoint_ok, checkpoint_reason = run.checkpoint_valid(
            manifest, run.config_by_name(config)["kernel_gen"], "step1"
        )
        results.check(checkpoint_ok, "valid inner checkpoint survives interruption", str(checkpoint_reason))
        results.check(manifest["stages"]["kernel_gen"]["attempts"] == 1, "resume preserves attempt history")
        results.check(
            quiet_call(run.command_start, argparse.Namespace(manifest=manifest_one, stage="kernel_gen")) == 0,
            "reopened kernel_gen starts a new attempt",
        )
        (kernel_dir / "triton_code_fix.py").write_text("# verified triton fixture\n", encoding="utf-8")
        (kernel_dir / "triton_report.md").write_text("# verified report\n", encoding="utf-8")
        (kernel_dir / "step6_test_code_fix.py").write_text("# reviewed fixture\n", encoding="utf-8")
        (kernel_dir / "step6_test_code_fix.md").write_text("# review fixture\n", encoding="utf-8")
        failing_review = load_json(
            ROOT / "validation" / "fixtures" / "valid" / "Review" / "review_result.json"
        )
        failing_review["status"] = "failed"
        failing_review["accuracy"]["pass"] = False
        (kernel_dir / "review_result.json").write_text(
            json.dumps(failing_review, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dispatch = load_python_module(
            "dispatch_metrics_run_control_validation",
            ROOT / "mlu-triton-code-gen" / "scripts" / "dispatch_metrics.py",
        )
        (kernel_dir / "dispatch_metrics.json").write_text(
            json.dumps(dispatch.analyze("normal", "direct-pass"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            quiet_call(
                run.command_complete,
                argparse.Namespace(
                    manifest=manifest_one,
                    stage="kernel_gen",
                    artifact=[],
                    validation_level=["l1", "l2", "l3"],
                ),
            )
        except run.ControlError:
            results.check(True, "kernel_gen rejects false L3 evidence")
        else:
            results.check(False, "kernel_gen rejects false L3 evidence")
        shutil.copy2(
            ROOT / "validation" / "fixtures" / "valid" / "Review" / "review_result.json",
            kernel_dir / "review_result.json",
        )
        results.check(
            quiet_call(
                run.command_complete,
                argparse.Namespace(
                    manifest=manifest_one,
                    stage="kernel_gen",
                    artifact=[],
                    validation_level=["l1", "l2", "l3"],
                ),
            )
            == 0,
            "kernel_gen completes and caches final handoff",
        )
        generated_manifest = run.load_manifest(manifest_one)
        kernel_cache_entry = run.cache_entry(generated_manifest, "kernel_gen")
        cache_metadata = load_json(kernel_cache_entry / "metadata.json")
        results.extend(
            "kernel cache metadata matches schema",
            validate_schema(cache_metadata, schemas["cache_metadata.schema.json"]),
        )
        results.check(
            cache_metadata.get("validation_levels") == ["l1", "l2", "l3"],
            "kernel cache records L1/L2/L3 promotion evidence",
        )

        output_two = root / "run-two"
        manifest_two = output_two / "run_manifest.json"
        second_init = argparse.Namespace(
            output_dir=output_two,
            input="deterministic reduce-sum request",
            mode="balanced",
            budget_file=None,
            cache_dir=cache_dir,
        )
        quiet_call(run.command_init, second_init)
        quiet_call(run.command_start, argparse.Namespace(manifest=manifest_two, stage="env_config"))
        context_two = write_fixture_run_context(output_two)
        quiet_call(
            run.command_complete,
            argparse.Namespace(
                manifest=manifest_two,
                stage="env_config",
                artifact=[],
                validation_level=[],
            ),
        )
        quiet_call(
            run.command_bind_context,
            argparse.Namespace(manifest=manifest_two, context_file=context_two),
        )
        manifest = run.load_manifest(manifest_two)
        results.check(
            run.next_action(manifest, config)["action"] == "restore"
            and run.next_action(manifest, config)["stage"] == "extractor",
            "identical request restores extractor cache",
        )
        results.check(
            quiet_call(run.command_restore, argparse.Namespace(manifest=manifest_two, stage="extractor")) == 0,
            "extractor cache restores with hashes",
        )
        manifest = run.load_manifest(manifest_two)
        results.check(
            run.next_action(manifest, config)["action"] == "restore"
            and run.next_action(manifest, config)["stage"] == "kernel_gen",
            "same hardware/toolchain restores kernel cache",
        )
        results.check(
            quiet_call(run.command_restore, argparse.Namespace(manifest=manifest_two, stage="kernel_gen")) == 0,
            "kernel cache restores atomically",
        )
        results.check(
            (output_two / "KernelGen" / "step1_base_info.json").is_file(),
            "kernel cache restores optional intermediate audit artifacts",
        )

        restored_code = output_two / "KernelGen" / "triton_code_fix.py"
        restored_code.write_text("corrupted output\n", encoding="utf-8")
        manifest = run.load_manifest(manifest_two)
        run.refresh_manifest(manifest, config)
        results.check(
            manifest["stages"]["kernel_gen"]["status"] == "pending"
            and manifest["stages"]["optimizer"]["status"] == "pending",
            "artifact corruption invalidates stage and downstream",
        )
        results.check(
            run.next_action(manifest, config)["action"] == "restore",
            "valid immutable cache can repair corrupted output",
        )

        output_three = root / "run-other-context"
        manifest_three = output_three / "run_manifest.json"
        quiet_call(
            run.command_init,
            argparse.Namespace(
                output_dir=output_three,
                input="deterministic reduce-sum request",
                mode="balanced",
                budget_file=None,
                cache_dir=cache_dir,
            ),
        )
        quiet_call(run.command_start, argparse.Namespace(manifest=manifest_three, stage="env_config"))
        context_three = write_fixture_run_context(output_three)
        other_context = load_json(context_three)
        other_context["hardware_key"] = "MLU590-M9:cluster-24:memory-64GB"
        run.write_json_atomic(context_three, other_context)
        quiet_call(
            run.command_complete,
            argparse.Namespace(
                manifest=manifest_three,
                stage="env_config",
                artifact=[],
                validation_level=[],
            ),
        )
        quiet_call(
            run.command_bind_context,
            argparse.Namespace(manifest=manifest_three, context_file=context_three),
        )
        quiet_call(run.command_restore, argparse.Namespace(manifest=manifest_three, stage="extractor"))
        other_manifest = run.load_manifest(manifest_three)
        results.check(
            run.next_action(other_manifest, config)["action"] == "run"
            and run.next_action(other_manifest, config)["stage"] == "kernel_gen",
            "different hardware context cannot reuse kernel runtime cache",
        )

        output_four = root / "run-correctness-complete"
        manifest_four = output_four / "run_manifest.json"
        quiet_call(
            run.command_init,
            argparse.Namespace(
                output_dir=output_four,
                input="deterministic reduce-sum request",
                mode="correctness",
                budget_file=None,
                cache_dir=cache_dir,
            ),
        )
        quiet_call(run.command_start, argparse.Namespace(manifest=manifest_four, stage="env_config"))
        context_four = write_fixture_run_context(output_four)
        quiet_call(
            run.command_complete,
            argparse.Namespace(
                manifest=manifest_four,
                stage="env_config",
                artifact=[],
                validation_level=[],
            ),
        )
        quiet_call(
            run.command_bind_context,
            argparse.Namespace(manifest=manifest_four, context_file=context_four),
        )
        quiet_call(run.command_restore, argparse.Namespace(manifest=manifest_four, stage="extractor"))
        quiet_call(run.command_restore, argparse.Namespace(manifest=manifest_four, stage="kernel_gen"))
        correctness_manifest = run.load_manifest(manifest_four)
        results.check(
            correctness_manifest["stages"]["optimizer"]["status"] == "skipped"
            and run.next_action(correctness_manifest, config)["stage"] == "finalize",
            "correctness reuses mode-independent cache and skips optimizer",
        )
        quiet_call(run.command_start, argparse.Namespace(manifest=manifest_four, stage="finalize"))
        (output_four / "triton_final.py").write_text("# verified final code\n", encoding="utf-8")
        (output_four / "summary.md").write_text(
            "# summary\naccuracy_pass: true\natol: 1e-4\nrtol: 1e-4\n"
            "Code Gen\nOptimize: N/A\ntriton_final.py\nrun_manifest\n"
            "regression_result.json\nhardware_key\ntoolchain_key\ncache hit\n",
            encoding="utf-8",
        )
        regression_result = load_json(
            ROOT / "validation" / "fixtures" / "valid" / "Regression" / "current_pass.json"
        )
        regression_result["run_id"] = run.load_manifest(manifest_four)["run_id"]
        run.write_json_atomic(output_four / "regression_result.json", regression_result)
        quiet_call(
            run.command_complete,
            argparse.Namespace(
                manifest=manifest_four,
                stage="finalize",
                artifact=[],
                validation_level=["l1", "l2", "l3"],
            ),
        )
        correctness_manifest = run.load_manifest(manifest_four)
        results.check(run.next_action(correctness_manifest, config)["action"] == "done", "complete run reaches done")
        complete_artifacts = check_output_artifacts(output_four, require_complete=True)
        results.checks += complete_artifacts.checks
        results.errors.extend(f"synthetic complete output: {error}" for error in complete_artifacts.errors)

        manifest["stages"]["extractor"]["fingerprint"] = "0" * 64
        run.refresh_manifest(manifest, config)
        results.check(
            manifest["stages"]["extractor"]["status"] == "pending"
            and manifest["stages"]["kernel_gen"]["status"] == "pending",
            "fingerprint drift invalidates only the stage and downstream",
        )
        try:
            run.safe_relative("../escape.py")
        except run.ControlError:
            results.check(True, "cache path traversal is rejected")
        else:
            results.check(False, "cache path traversal is rejected")

        generated_manifest = run.load_manifest(manifest_one)
        results.extend(
            "generated manifest matches schema",
            validate_schema(generated_manifest, schemas["run_manifest.schema.json"]),
        )
        base_fingerprints = run.compute_fingerprints(generated_manifest, config)
        changed_mode = copy.deepcopy(generated_manifest)
        changed_mode["mode"] = "max-performance"
        mode_fingerprints = run.compute_fingerprints(changed_mode, config)
        results.check(
            base_fingerprints["extractor"] == mode_fingerprints["extractor"]
            and base_fingerprints["kernel_gen"] == mode_fingerprints["kernel_gen"],
            "mode changes retain mode-independent extractor/kernel cache keys",
        )
        results.check(
            base_fingerprints["optimizer"] != mode_fingerprints["optimizer"],
            "mode changes invalidate optimizer cache key",
        )
        kernel_cache_entry = run.cache_entry(generated_manifest, "kernel_gen")
        cached_code = kernel_cache_entry / "artifacts" / "KernelGen" / "triton_code_fix.py"
        cached_code.write_text("corrupted immutable cache\n", encoding="utf-8")
        cache_valid, cache_reason = run.cache_metadata_valid(
            kernel_cache_entry,
            run.config_by_name(config)["kernel_gen"],
            generated_manifest["stages"]["kernel_gen"]["fingerprint"],
        )
        results.check(
            not cache_valid and bool(cache_reason and cache_reason.startswith("cache_hash_mismatch")),
            "corrupted cache entry is rejected instead of silently restored",
            str(cache_reason),
        )

        optimization_input = root / "optimizer_input.py"
        optimization_input.write_text(
            "@triton.jit\ndef kernel(x):\n    pass\n\ndef wrapper(x):\n    kernel[(1,)](x)\n\ndef test_accuracy():\n    pass\n\ndef benchmark():\n    pass\n",
            encoding="utf-8",
        )
        optimizer_dir = root / "optimizer-state"
        plan_args = argparse.Namespace(
            input=optimization_input,
            output_dir=optimizer_dir,
            mode="balanced",
            budget_file=None,
            resume=False,
        )
        results.check(quiet_call(optimization.command_plan, plan_args) == 0, "optimizer plan initializes once")
        state_path = optimizer_dir / "optimization_state.json"
        state = optimization.read_json(state_path)
        state["usage"]["subagent_calls"] = 3
        optimization.write_json_atomic(state_path, state)
        plan_args.resume = True
        results.check(quiet_call(optimization.command_plan, plan_args) == 0, "compatible optimizer state resumes")
        results.check(
            optimization.read_json(state_path)["usage"]["subagent_calls"] == 3,
            "optimizer resume preserves consumed budget",
        )
        plan_args.resume = False
        results.check(
            quiet_call(optimization.command_plan, plan_args) == 2,
            "optimizer refuses accidental state reset",
        )

    regression_base = load_json(ROOT / "validation" / "fixtures" / "valid" / "Regression" / "baseline.json")
    regression_pass = load_json(ROOT / "validation" / "fixtures" / "valid" / "Regression" / "current_pass.json")
    regression_fail = load_json(ROOT / "validation" / "fixtures" / "invalid" / "Regression" / "current_fail.json")
    regression_mismatch = load_json(
        ROOT / "validation" / "fixtures" / "invalid" / "Regression" / "current_context_mismatch.json"
    )
    policy = load_json(ROOT / "validation" / "regression_policy.json")
    results.extend(
        "regression policy matches schema",
        validate_schema(policy, schemas["regression_policy.schema.json"]),
    )
    for label, fixture in (
        ("baseline", regression_base),
        ("current pass", regression_pass),
        ("current fail", regression_fail),
        ("context mismatch", regression_mismatch),
    ):
        results.extend(
            f"{label} regression fixture matches schema",
            validate_schema(fixture, schemas["regression_result.schema.json"]),
        )
    pass_report = regression.compare_results(regression_base, regression_pass, policy)
    fail_report = regression.compare_results(regression_base, regression_fail, policy)
    mismatch_report = regression.compare_results(regression_base, regression_mismatch, policy)
    missing_resources = copy.deepcopy(regression_pass)
    del missing_resources["cases"][0]["resources"]["token_count"]
    missing_resource_report = regression.compare_results(regression_base, missing_resources, policy)
    results.check(pass_report["passed"], "regression fixture inside thresholds passes")
    results.check(not fail_report["passed"], "accuracy/performance/resource regression fails")
    fail_violations = {
        violation for row in fail_report["cases"] for violation in row["violations"]
    }
    results.check("accuracy_failed" in fail_violations, "regression detects accuracy failure")
    results.check("latency_regression_exceeded" in fail_violations, "regression detects latency slowdown")
    results.check(
        not mismatch_report["passed"] and not mismatch_report["context_compatible"],
        "hardware/toolchain mismatch is not reported as comparable performance",
    )
    results.check(
        not missing_resource_report["passed"]
        and any(
            "token_count_missing_or_invalid" in row["violations"]
            for row in missing_resource_report["cases"]
        ),
        "required token telemetry cannot silently disappear",
    )
    results.extend(
        "generated regression report matches schema",
        validate_schema(pass_report, schemas["regression_report.schema.json"]),
    )
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

    review_valid_dir = ROOT / "validation" / "fixtures" / "valid" / "Review"
    for artifact, schema_name in REVIEW_SCHEMA_BY_ARTIFACT.items():
        path = review_valid_dir / artifact
        results.check(path.is_file(), f"valid review fixture exists: {artifact}")
        if path.is_file() and schema_name in schemas:
            errors = validate_schema(load_json(path), schemas[schema_name])
            results.extend(f"valid review fixture matches {schema_name}", errors)

    dispatch = check_dispatch_control()
    results.checks += dispatch.checks
    results.errors.extend(dispatch.errors)
    stage2 = check_stage2_control()
    results.checks += stage2.checks
    results.errors.extend(stage2.errors)
    stage4 = check_stage4_control()
    results.checks += stage4.checks
    results.errors.extend(stage4.errors)
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

    for artifact, schema_name in REVIEW_SCHEMA_BY_ARTIFACT.items():
        path = kernel_dir / artifact
        if not path.exists():
            if require_complete:
                results.check(False, f"complete output contains {artifact}")
            continue
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            results.check(False, f"artifact parses: {path}", str(exc))
            continue
        errors = validate_schema(value, schemas[schema_name])
        results.extend(f"artifact matches {schema_name}", errors)
        results.check(
            value.get("status") in {"passed", "repaired"}
            and value.get("accuracy", {}).get("pass") is True,
            "review result carries passing L3 evidence",
        )
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
    context_path = output_dir / "EnvConfig" / "run_context.json"
    if config_path.is_file():
        config_text = read_text(config_path)
        results.check(
            re.search(r"execution_backend\s*[:：]\s*(local|worker)", config_text) is not None,
            "EnvConfig records a valid execution_backend",
        )
    elif require_complete:
        results.check(False, "complete output contains EnvConfig/config.md")

    if context_path.is_file():
        try:
            context_value = load_json(context_path)
        except (OSError, json.JSONDecodeError) as exc:
            results.check(False, "run context parses", str(exc))
        else:
            results.extend(
                "run context matches run_context.schema.json",
                validate_schema(context_value, schemas["run_context.schema.json"]),
            )
    elif require_complete:
        results.check(False, "complete output contains EnvConfig/run_context.json")

    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest_value = load_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            results.check(False, "run manifest parses", str(exc))
        else:
            results.extend(
                "run manifest matches run_manifest.schema.json",
                validate_schema(manifest_value, schemas["run_manifest.schema.json"]),
            )
            if require_complete:
                stage_statuses = {
                    name: row.get("status")
                    for name, row in manifest_value.get("stages", {}).items()
                    if isinstance(row, dict)
                }
                results.check(
                    stage_statuses.get("finalize") == "complete",
                    "complete output has a completed finalize stage",
                )
    elif require_complete:
        results.check(False, "complete output contains run_manifest.json")

    regression_result_path = output_dir / "regression_result.json"
    if regression_result_path.is_file():
        try:
            regression_value = load_json(regression_result_path)
        except (OSError, json.JSONDecodeError) as exc:
            results.check(False, "regression result parses", str(exc))
        else:
            results.extend(
                "regression result matches regression_result.schema.json",
                validate_schema(regression_value, schemas["regression_result.schema.json"]),
            )
            if context_path.is_file():
                context_value = load_json(context_path)
                results.check(
                    regression_value.get("hardware_key") == context_value.get("hardware_key")
                    and regression_value.get("toolchain_key") == context_value.get("toolchain_key"),
                    "regression result context matches EnvConfig",
                )
            if manifest_path.is_file():
                results.check(
                    regression_value.get("run_id") == load_json(manifest_path).get("run_id"),
                    "regression result run_id matches run manifest",
                )
    elif require_complete:
        results.check(False, "complete output contains regression_result.json")

    if require_complete:
        required_outputs = [
            output_dir / "KernelGen" / "triton_code_fix.py",
            output_dir / "KernelGen" / "triton_report.md",
            output_dir / "KernelGen" / "review_result.json",
            output_dir / "KernelGen" / "dispatch_metrics.json",
            output_dir / "triton_final.py",
            output_dir / "summary.md",
            output_dir / "regression_result.json",
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
            for token in (
                "accuracy_pass",
                "atol",
                "rtol",
                "Code Gen",
                "Optimize",
                "triton_final.py",
                "run_manifest",
                "regression_result.json",
                "hardware_key",
                "toolchain_key",
                "cache",
            ):
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
