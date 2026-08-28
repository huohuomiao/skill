#!/usr/bin/env python3
"""Regression tests for p1 layered validation and L3 gating."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = SKILL_ROOT / "scripts" / "validation"
L1 = VALIDATION_DIR / "l1-static.py"
L2 = VALIDATION_DIR / "l2-offline.py"
L3 = VALIDATION_DIR / "l3-integration.py"


def _run(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


class LayeredValidationTests(unittest.TestCase):
    def test_l1_passes_current_skill_without_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "l1.json"
            result = _run(
                L1,
                "--skill-root",
                str(SKILL_ROOT),
                "--report",
                str(report),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            self.assertFalse(payload["hardware_evidence"])
            self.assertLess(payload["elapsed_seconds"], 30)

    def test_l2_fixed_scenarios_pass_without_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "l2.json"
            result = _run(
                L2,
                "--skill-root",
                str(SKILL_ROOT),
                "--report",
                str(report),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "pass")
            scenario_checks = [
                item for item in payload["checks"] if item["name"].startswith("scenario:")
            ]
            self.assertGreaterEqual(len(scenario_checks), 18)
            self.assertTrue(all(item["status"] == "pass" for item in scenario_checks))

    def test_l3_does_not_execute_when_lower_gate_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "command-ran.txt"
            command = [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
            ]
            suite = {
                "schema_version": "1.0",
                "suite_id": "gate-test",
                "kind": "hardware",
                "hardware_model": "gate-test-device",
                "timeout_seconds": 10,
                "cases": [
                    {
                        "id": operator_class,
                        "operator_class": operator_class,
                        "workdir": ".",
                        "commands": {
                            "compile": command,
                            "accuracy": command,
                            "performance": command,
                        },
                    }
                    for operator_class in ("elementwise", "reduction", "layout")
                ],
                "worker_checks": {"submission": command, "failure_recovery": command},
            }
            suite_path = root / "suite.json"
            suite_path.write_text(json.dumps(suite), encoding="utf-8")
            fingerprint = "0" * 64
            base_report = {
                "level": "L1",
                "status": "fail",
                "elapsed_seconds": 1.0,
                "skill_fingerprint": fingerprint,
            }
            l1_path = root / "l1.json"
            l2_path = root / "l2.json"
            l1_path.write_text(json.dumps(base_report), encoding="utf-8")
            base_report.update({"level": "L2", "status": "pass"})
            l2_path.write_text(json.dumps(base_report), encoding="utf-8")
            l3_report = root / "l3.json"
            result = _run(
                L3,
                "--skill-root",
                str(SKILL_ROOT),
                "--l1-report",
                str(l1_path),
                "--l2-report",
                str(l2_path),
                "--suite",
                str(suite_path),
                "--report",
                str(l3_report),
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(marker.exists())
            payload = json.loads(l3_report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "fail")
            self.assertFalse(payload["hardware_evidence"])


if __name__ == "__main__":
    unittest.main()
