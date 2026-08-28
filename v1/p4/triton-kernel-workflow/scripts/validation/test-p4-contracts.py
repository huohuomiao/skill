#!/usr/bin/env python3
"""Offline regression tests for p4 fingerprints, cache snapshots, and resume planning."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT = SKILL_ROOT / "scripts/state/fingerprint-stage.py"
CACHE = SKILL_ROOT / "scripts/state/stage-cache.py"
PLAN = SKILL_ROOT / "scripts/state/plan-resume.py"
APPLY = SKILL_ROOT / "scripts/state/apply-resume-plan.py"
IMPACT = SKILL_ROOT / "scripts/validation/plan-change-impact.py"


def _run(script: Path, *arguments: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _dummy_fingerprint(root: Path, stage: str, seed: str) -> Path:
    path = root / "RunState" / "fingerprints" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    factors = {
        "input_hash": _sha(f"input:{seed}"),
        "skill_version": _sha(f"skill:{stage}"),
        "dependency_versions": {},
        "hardware_model": None,
        "toolchain_version": None,
        "stage_config": {},
        "upstream_fingerprints": {},
    }
    payload = {
        "schema_version": "1.0",
        "stage": stage,
        "fingerprint": _canonical_hash({"stage": stage, "factors": factors}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "factors": factors,
        "evidence": {"inputs": {}, "resources": {}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record_many(root: Path, fingerprint: Path, artifacts: dict[str, Path]) -> dict:
    arguments = [
        "record",
        "--output-dir", str(root),
        "--fingerprint-file", str(fingerprint),
    ]
    for key, target in sorted(artifacts.items()):
        arguments.extend(("--artifact", f"{key}={target}"))
    result = _run(CACHE, *arguments)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)["record"]


def _write_artifacts(root: Path, stage: str) -> dict[str, Path]:
    stage_root = root / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    if stage == "environment":
        runtime = stage_root / "runtime_info.txt"
        report = stage_root / "report.md"
        config = stage_root / "config.json"
        runtime.write_text("device=test-device\n", encoding="utf-8")
        report.write_text("# environment\n", encoding="utf-8")
        config.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "ready",
                    "execution_backend": "local",
                    "env_check_task_id": "local",
                    "runtime_info_path": str(runtime),
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "device": {"model": "test-device"},
                    "versions": {"triton": "test", "torch": "test", "toolchain": "test"},
                }
            ),
            encoding="utf-8",
        )
        return {"config": config, "runtime_info": runtime, "report": report}
    if stage == "requirement-extraction":
        requirement = stage_root / "requirement.md"
        requirement.write_text("# requirement\n", encoding="utf-8")
        return {"requirement": requirement}
    if stage in {"code-generation", "code-validation"}:
        code = stage_root / "code.py"
        report = stage_root / "report.md"
        code.write_text("def cached_kernel():\n    pass\n", encoding="utf-8")
        report.write_text(f"# {stage}\n", encoding="utf-8")
        return {"code": code, "report": report}
    if stage == "finalization":
        code = stage_root / "final.py"
        summary = stage_root / "summary.md"
        code.write_text("def final_kernel():\n    pass\n", encoding="utf-8")
        summary.write_text("# summary\n", encoding="utf-8")
        return {"final_code": code, "summary": summary}
    raise ValueError(f"unsupported helper stage: {stage}")


class P4ContractTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_stage_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied_skill = root / "skill"
            shutil.copytree(SKILL_ROOT, copied_skill)
            request = root / "request.txt"
            request.write_text("softmax requirement\n", encoding="utf-8")

            def generate(stage: str, name: str) -> dict:
                output = root / f"{name}.json"
                result = _run(
                    FINGERPRINT,
                    "--stage", stage,
                    "--skill-root", str(copied_skill),
                    "--input", f"request={request}",
                    "--device-independent",
                    "--output", str(output),
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(output.read_text(encoding="utf-8"))

            requirement_before = generate("requirement-extraction", "requirement-before")
            requirement_again = generate("requirement-extraction", "requirement-again")
            final_before = generate("finalization", "final-before")
            self.assertEqual(requirement_before["fingerprint"], requirement_again["fingerprint"])

            summary_template = copied_skill / "references/templates/final-summary.md"
            summary_template.write_text(
                summary_template.read_text(encoding="utf-8") + "\nAdditional presentation line.\n",
                encoding="utf-8",
            )
            requirement_after = generate("requirement-extraction", "requirement-after")
            final_after = generate("finalization", "final-after")
            self.assertEqual(requirement_before["fingerprint"], requirement_after["fingerprint"])
            self.assertNotEqual(final_before["fingerprint"], final_after["fingerprint"])

    def test_cache_restores_deleted_artifact_and_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fingerprint = _dummy_fingerprint(root, "code-validation", "validated")
            target = root / "KernelGen/triton_code_fix.py"
            target.parent.mkdir(parents=True)
            expected = "def kernel():\n    return 1\n"
            target.write_text(expected, encoding="utf-8")
            report = root / "KernelGen/triton_code_fix.md"
            report.write_text("# validation\n", encoding="utf-8")
            record = _record_many(root, fingerprint, {"code": target, "report": report})
            record_path = root / "RunState/cache/code-validation" / record["fingerprint"] / "record.json"

            target.unlink()
            restored = _run(CACHE, "restore", "--output-dir", str(root), "--record", str(record_path))
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), expected)

            fingerprint_snapshot = root / record["fingerprint_path"]
            original_fingerprint = fingerprint_snapshot.read_text(encoding="utf-8")
            tampered = json.loads(original_fingerprint)
            tampered["factors"]["stage_config"]["tampered"] = True
            fingerprint_snapshot.write_text(json.dumps(tampered), encoding="utf-8")
            invalid_fingerprint = _run(
                CACHE, "verify", "--output-dir", str(root), "--record", str(record_path)
            )
            self.assertEqual(invalid_fingerprint.returncode, 2)
            self.assertIn("digest is invalid", invalid_fingerprint.stderr)
            fingerprint_snapshot.write_text(original_fingerprint, encoding="utf-8")

            snapshot = root / record["artifacts"]["code"]["cache_path"]
            snapshot.write_text("corrupted\n", encoding="utf-8")
            invalid = _run(CACHE, "verify", "--output-dir", str(root), "--record", str(record_path))
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("hash mismatch", invalid.stderr)

            repaired = _run(
                CACHE,
                "record",
                "--output-dir",
                str(root),
                "--fingerprint-file",
                str(fingerprint),
                "--artifact",
                f"code={target}",
                "--artifact",
                f"report={report}",
            )
            self.assertEqual(repaired.returncode, 0, repaired.stderr)
            repaired_payload = json.loads(repaired.stdout)
            quarantine = Path(repaired_payload["quarantined_path"])
            self.assertTrue(quarantine.is_dir())
            self.assertTrue((quarantine / "record.json").is_file())
            verified = _run(
                CACHE, "verify", "--output-dir", str(root), "--record", str(record_path)
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)

    def test_relocated_dynamic_cache_is_rejected_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            original = temporary_root / "original"
            original.mkdir()
            fingerprint = _dummy_fingerprint(original, "environment", "identity")
            record = _record_many(
                original, fingerprint, _write_artifacts(original, "environment")
            )
            relocated = temporary_root / "relocated"
            shutil.copytree(original, relocated)
            relocated_record = (
                relocated
                / "RunState/cache/environment"
                / record["fingerprint"]
                / "record.json"
            )
            verified = _run(
                CACHE,
                "verify",
                "--output-dir",
                str(relocated),
                "--record",
                str(relocated_record),
            )
            self.assertEqual(verified.returncode, 2)
            self.assertIn("current output root", verified.stderr)

    def test_first_miss_invalidates_every_active_downstream_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stages = (
                "environment",
                "requirement-extraction",
                "code-generation",
                "code-validation",
                "finalization",
            )
            fingerprints = {stage: _dummy_fingerprint(root, stage, "desired") for stage in stages}
            for stage in ("environment", "code-generation", "code-validation", "finalization"):
                _record_many(root, fingerprints[stage], _write_artifacts(root, stage))

            plan_path = root / "RunState/resume_plan.json"
            arguments = [
                "--output-dir", str(root),
                "--mode", "full",
                "--optimization-mode", "correctness",
                "--output", str(plan_path),
            ]
            for stage, path in fingerprints.items():
                arguments.extend(("--fingerprint", f"{stage}={path}"))
            result = _run(PLAN, *arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            actions = {item["stage"]: item["action"] for item in plan["stages"]}
            reasons = {item["stage"]: item["reason"] for item in plan["stages"]}
            self.assertEqual(actions["environment"], "reuse")
            self.assertEqual(actions["requirement-extraction"], "rerun")
            self.assertEqual(actions["performance-tuning"], "skip")
            for stage in ("code-generation", "code-validation", "finalization"):
                self.assertEqual(actions[stage], "rerun")
                self.assertEqual(reasons[stage], "upstream-invalidated")
            self.assertEqual(plan["resume_from"], "requirement-extraction")

    def test_downstream_fingerprints_may_be_deferred_after_first_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = _dummy_fingerprint(root, "environment", "new-run")
            plan_path = root / "RunState/resume_plan.json"
            result = _run(
                PLAN,
                "--output-dir", str(root),
                "--mode", "full",
                "--optimization-mode", "balanced",
                "--fingerprint", f"environment={environment}",
                "--output", str(plan_path),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["resume_from"], "environment")
            self.assertTrue(all(
                item["action"] == "rerun" for item in plan["stages"]
            ))
            self.assertIsNone(plan["stages"][1]["fingerprint"])

    def test_apply_plan_restores_artifact_and_updates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fingerprint = _dummy_fingerprint(root, "code-validation", "resume")
            report = root / "KernelGen/validation.md"
            report.parent.mkdir(parents=True)
            report.write_text("# passed\n", encoding="utf-8")
            code = root / "KernelGen/validated.py"
            code.write_text("def validated():\n    pass\n", encoding="utf-8")
            _record_many(root, fingerprint, {"code": code, "report": report})
            report.unlink()

            plan_path = root / "RunState/resume_plan.json"
            planned = _run(
                PLAN,
                "--output-dir", str(root),
                "--mode", "code-validation",
                "--optimization-mode", "balanced",
                "--fingerprint", f"code-validation={fingerprint}",
                "--output", str(plan_path),
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            applied = _run(APPLY, "--plan", str(plan_path))
            self.assertEqual(applied.returncode, 0, applied.stderr)
            self.assertTrue(report.is_file())
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["workflow_status"], "completed")
            self.assertEqual(manifest["stages"]["code-validation"]["metadata"]["cache_status"], "restored")
            self.assertEqual(manifest["resume"]["plan_path"], str(plan_path.resolve()))

    def test_performance_cache_restores_selected_checkpoint_without_new_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fingerprint = _dummy_fingerprint(root, "performance-tuning", "measured")
            optimizer = root / "Optimizer"
            optimizer.mkdir(parents=True)
            code = optimizer / "triton_optimized.py"
            report = optimizer / "triton_optimized.md"
            best = optimizer / "best_so_far.json"
            strategy_plan = optimizer / "strategy_plan.json"
            tuning_state = optimizer / "tuning_state.json"
            code.write_text("def optimized():\n    return 1\n", encoding="utf-8")
            report.write_text("# measured result\n", encoding="utf-8")
            now = datetime.now(timezone.utc).isoformat()
            strategy_plan.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "optimization_mode": "max-performance",
                        "input_path": str(code),
                        "generated_at": now,
                        "detected": {
                            "jit_kernels": ["optimized"],
                            "has_reduction": False,
                            "grid_bounded": False,
                            "has_complex_index": False,
                            "tunable_parameters": [],
                            "has_device_math_pattern": False,
                            "has_division": False,
                            "has_tiling_surface": False,
                        },
                        "strategies": [
                            {
                                "order": index,
                                "name": f"strategy-{index}",
                                "phase": "oob" if index <= 5 else "advanced",
                                "decision": "skip",
                                "reason": "offline fixture",
                                "strategy_doc": str(report),
                            }
                            for index in range(1, 9)
                        ],
                        "dynamic_execution": "serial",
                    }
                ),
                encoding="utf-8",
            )
            tuning_state.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "optimization_mode": "max-performance",
                        "status": "completed",
                        "started_at": now,
                        "updated_at": now,
                        "limits": {
                            "max_deep_rounds": 3,
                            "max_worker_calls": 16,
                            "max_elapsed_seconds": 1800,
                        },
                        "usage": {"deep_rounds_started": 0, "worker_calls": 0},
                        "stop_reason": "planned-work-complete",
                        "events": [
                            {"at": now, "event": "initialized", "label": "max-performance"},
                            {"at": now, "event": "completed", "label": "planned-work-complete"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            best.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "selected_at": now,
                        "selected_candidate": {
                            "schema_version": "1.0",
                            "candidate_id": "cached-best",
                            "source_stage": "baseline",
                            "code_path": str(code),
                            "report_path": str(report),
                            "accuracy_pass": True,
                            "latency_ms": 0.25,
                            "bandwidth_gbps": None,
                            "execution_backend": "local",
                            "hardware_model": "test-device",
                            "benchmark_signature": "offline-fixture",
                        },
                        "compared_candidates": ["cached-best"],
                        "best_code_path": str(code),
                        "final_code_path": str(code),
                    }
                ),
                encoding="utf-8",
            )
            recorded = _run(
                CACHE,
                "record",
                "--output-dir", str(root),
                "--fingerprint-file", str(fingerprint),
                "--artifact", f"best={best}",
                "--artifact", f"code={code}",
                "--artifact", f"report={report}",
                "--artifact", f"strategy_plan={strategy_plan}",
                "--artifact", f"tuning_state={tuning_state}",
            )
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            for path in (best, code, report, strategy_plan, tuning_state):
                path.unlink()

            plan_path = root / "RunState/resume_plan.json"
            planned = _run(
                PLAN,
                "--output-dir", str(root),
                "--mode", "performance-tuning",
                "--optimization-mode", "max-performance",
                "--fingerprint", f"performance-tuning={fingerprint}",
                "--output", str(plan_path),
            )
            self.assertEqual(planned.returncode, 0, planned.stderr)
            applied = _run(APPLY, "--plan", str(plan_path))
            self.assertEqual(applied.returncode, 0, applied.stderr)
            manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_checkpoint"]["candidate_id"], "cached-best")
            self.assertEqual(manifest["selected_checkpoint"]["latency_ms"], 0.25)
            self.assertEqual(Path(manifest["selected_checkpoint"]["code_path"]), code)
            restored_budget = json.loads((optimizer / "tuning_state.json").read_text(encoding="utf-8"))
            self.assertEqual(restored_budget["status"], "completed")
            self.assertEqual(restored_budget["usage"]["worker_calls"], 0)

    def test_change_impact_routes_the_four_documented_cases(self) -> None:
        cases = (
            ("references/templates/final-summary.md", False, ["finalization"], [], []),
            ("references/strategies/reduction.md", True, ["performance-tuning", "finalization"], ["reduction"], []),
            ("scripts/execution/submit-remote-task.py", True, ["environment"], [], ["submission", "failure-recovery"]),
            ("references/backend/platform-rules.md", True, list(("environment", "requirement-extraction", "code-generation", "code-validation", "performance-tuning", "finalization")), ["elementwise", "reduction", "layout"], ["submission", "failure-recovery"]),
        )
        for changed, hardware, stages, l3_cases, workers in cases:
            with self.subTest(changed=changed):
                result = _run(IMPACT, "--skill-root", str(SKILL_ROOT), "--changed", changed)
                self.assertEqual(result.returncode, 0, result.stderr)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["hardware_required"], hardware)
                self.assertEqual(plan["affected_stages"], stages)
                self.assertEqual(plan["l3_cases"], l3_cases)
                self.assertEqual(plan["worker_checks"], workers)


if __name__ == "__main__":
    unittest.main()
