#!/usr/bin/env python3
"""Offline regression tests for p3 mode, static routing, and fixed budgets."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2]
ROUTER = SKILL_ROOT / "scripts/state/plan-strategies.py"
BUDGET = SKILL_ROOT / "scripts/state/manage-tuning-budget.py"
MANIFEST = SKILL_ROOT / "scripts/state/update-run-manifest.py"
LOCAL_RUNNER = SKILL_ROOT / "scripts/execution/run-budgeted-local.py"
REMOTE_RUNNER = SKILL_ROOT / "scripts/execution/submit-remote-task.py"
ARTIFACTS = SKILL_ROOT / "references/evals/artifacts"


def _run(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


class P3ContractTests(unittest.TestCase):
    def _plan(self, source: str, mode: str) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "strategy_plan.json"
            result = _run(
                ROUTER,
                "--input", str(ARTIFACTS / source),
                "--output", str(output),
                "--mode", mode,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(output.read_text(encoding="utf-8"))

    def test_balanced_routes_only_applicable_oob_strategies(self) -> None:
        plan = self._plan("router-softmax.py", "balanced")
        decisions = {item["name"]: item["decision"] for item in plan["strategies"]}
        self.assertEqual(
            decisions,
            {
                "retiling": "apply",
                "reduce-opt": "apply",
                "modify-grid": "skip",
                "index-computation-simplify": "skip",
                "gen-autotune-config": "apply",
                "libdevice-opt": "skip",
                "config-tuner": "skip",
                "div-to-mul": "skip",
            },
        )
        self.assertEqual(plan["dynamic_execution"], "serial")

    def test_max_performance_admits_detected_advanced_strategies(self) -> None:
        plan = self._plan("router-softmax.py", "max-performance")
        decisions = {item["name"]: item["decision"] for item in plan["strategies"]}
        for name in ("libdevice-opt", "config-tuner", "div-to-mul"):
            self.assertEqual(decisions[name], "apply")
        self.assertEqual(decisions["modify-grid"], "skip")
        self.assertEqual(decisions["index-computation-simplify"], "skip")

    def test_elementwise_router_skips_absent_patterns(self) -> None:
        plan = self._plan("router-elementwise.py", "balanced")
        decisions = {item["name"]: item["decision"] for item in plan["strategies"]}
        self.assertEqual(decisions["reduce-opt"], "skip")
        self.assertEqual(decisions["index-computation-simplify"], "skip")
        self.assertEqual(decisions["modify-grid"], "apply")
        self.assertEqual(decisions["retiling"], "apply")

    def test_shape_constexpr_is_not_misclassified_as_tunable(self) -> None:
        plan = self._plan("router-shape-only.py", "max-performance")
        decisions = {item["name"]: item["decision"] for item in plan["strategies"]}
        self.assertEqual(plan["detected"]["tunable_parameters"], [])
        self.assertEqual(decisions["gen-autotune-config"], "skip")
        self.assertEqual(decisions["config-tuner"], "skip")
        self.assertEqual(decisions["retiling"], "skip")

    def test_correctness_disables_every_performance_strategy(self) -> None:
        plan = self._plan("router-softmax.py", "correctness")
        self.assertTrue(all(item["decision"] == "skip" for item in plan["strategies"]))

    def test_worker_call_limit_is_exactly_sixteen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "tuning_state.json"
            init = _run(BUDGET, "init", "--state", str(state), "--mode", "balanced")
            self.assertEqual(init.returncode, 0, init.stderr)
            for index in range(16):
                reserved = _run(
                    BUDGET,
                    "reserve-worker",
                    "--state", str(state),
                    "--label", f"worker-{index + 1}",
                )
                self.assertEqual(reserved.returncode, 0, reserved.stderr)
            blocked = _run(BUDGET, "reserve-worker", "--state", str(state), "--label", "worker-17")
            self.assertEqual(blocked.returncode, 3, blocked.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["usage"]["worker_calls"], 16)
            self.assertEqual(payload["stop_reason"], "worker-call-limit")

    def test_deep_round_limit_is_exactly_three(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "tuning_state.json"
            init = _run(BUDGET, "init", "--state", str(state), "--mode", "max-performance")
            self.assertEqual(init.returncode, 0, init.stderr)
            for index in range(3):
                started = _run(
                    BUDGET,
                    "start-round",
                    "--state", str(state),
                    "--label", f"iter_{index + 1}",
                )
                self.assertEqual(started.returncode, 0, started.stderr)
            blocked = _run(BUDGET, "start-round", "--state", str(state), "--label", "iter_4")
            self.assertEqual(blocked.returncode, 3, blocked.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["usage"]["deep_rounds_started"], 3)
            self.assertEqual(payload["stop_reason"], "deep-round-limit")

    def test_elapsed_limit_is_thirty_minutes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "tuning_state.json"
            init = _run(BUDGET, "init", "--state", str(state), "--mode", "balanced")
            self.assertEqual(init.returncode, 0, init.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            payload["started_at"] = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
            state.write_text(json.dumps(payload), encoding="utf-8")
            blocked = _run(BUDGET, "check", "--state", str(state))
            self.assertEqual(blocked.returncode, 3, blocked.stderr)
            stopped = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(stopped["limits"]["max_elapsed_seconds"], 1800)
            self.assertEqual(stopped["stop_reason"], "elapsed-limit")

            local = _run(
                LOCAL_RUNNER,
                "--budget-state", str(state),
                "--workdir", str(Path(temporary)),
                "--label", "blocked-local",
                "--", sys.executable, "-c", "raise SystemExit(99)",
            )
            self.assertEqual(local.returncode, 4, local.stderr)
            remote = _run(
                REMOTE_RUNNER,
                "--task-type", "performance",
                "--workdir", str(Path(temporary).resolve()),
                "--timeout-sec", "30",
                "--job-id", "offline-budget-test",
                "--budget-state", str(state),
                "--command", "this-command-must-not-be-submitted",
            )
            self.assertEqual(remote.returncode, 4, remote.stderr)

    def test_manifest_optimization_mode_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "run_manifest.json"
            first = _run(
                MANIFEST,
                "--manifest", str(manifest),
                "--mode", "full",
                "--optimization-mode", "correctness",
                "--stage", "environment",
                "--status", "running",
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            changed = _run(
                MANIFEST,
                "--manifest", str(manifest),
                "--optimization-mode", "max-performance",
                "--stage", "environment",
                "--status", "running",
            )
            self.assertEqual(changed.returncode, 2)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["optimization_mode"], "correctness")


if __name__ == "__main__":
    unittest.main()
