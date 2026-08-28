#!/usr/bin/env python3
"""Offline regression tests for p2 dispatch and compact-state contracts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2]
UPDATE_MANIFEST = SKILL_ROOT / "scripts" / "state" / "update-run-manifest.py"


def _update(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(UPDATE_MANIFEST), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


class Stage2ContractTests(unittest.TestCase):
    def test_partial_codegen_mode_completes_after_build_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "run_manifest.json"
            code = root / "triton_code_fix.py"
            report = root / "triton_report.md"
            validation = root / "validation.md"
            for path, content in (
                (code, "def kernel():\n    pass\n"),
                (report, "# report\n"),
                (validation, "# validation\n"),
            ):
                path.write_text(content, encoding="utf-8")

            first = _update(
                "--manifest", str(manifest),
                "--mode", "code-generation",
                "--stage", "code-generation",
                "--status", "completed",
                "--artifact", f"code={code}",
                "--artifact", f"report={report}",
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            interim = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(interim["workflow_status"], "running")

            second = _update(
                "--manifest", str(manifest),
                "--stage", "code-validation",
                "--status", "completed",
                "--artifact", f"report={validation}",
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            completed = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(completed["workflow_status"], "completed")
            self.assertEqual(completed["mode"], "code-generation")

    def test_completed_stage_rejects_a_missing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "run_manifest.json"
            result = _update(
                "--manifest", str(manifest),
                "--stage", "environment",
                "--status", "completed",
                "--artifact", f"config={Path(temporary) / 'missing.json'}",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("missing or empty", result.stderr)
            self.assertFalse(manifest.exists())

    def test_selected_checkpoint_requires_real_code_and_positive_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "run_manifest.json"
            code = root / "best.py"
            report = root / "best.json"
            code.write_text("pass\n", encoding="utf-8")
            report.write_text("{}\n", encoding="utf-8")
            result = _update(
                "--manifest", str(manifest),
                "--mode", "performance-tuning",
                "--stage", "performance-tuning",
                "--status", "completed",
                "--artifact", f"best={report}",
                "--selected-candidate-id", "best-valid",
                "--selected-code-path", str(code),
                "--selected-latency-ms", "0.8",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["workflow_status"], "completed")
            self.assertEqual(payload["selected_checkpoint"]["candidate_id"], "best-valid")
            self.assertEqual(payload["selected_checkpoint"]["latency_ms"], 0.8)


if __name__ == "__main__":
    unittest.main()
