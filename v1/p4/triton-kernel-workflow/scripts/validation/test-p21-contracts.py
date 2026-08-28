#!/usr/bin/env python3
"""Offline regression tests for the p2.1 reduction-generation correction."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = SKILL_ROOT / "scripts" / "validation" / "validate-optimization-surface.py"
ARTIFACTS = SKILL_ROOT / "references" / "evals" / "artifacts"
BASE_INFO = ARTIFACTS / "base-info.softmax.json"


def _validate(spec_name: str, intent: str = "handoff-to-tuning") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--base-info",
            str(BASE_INFO),
            "--spec",
            str(ARTIFACTS / spec_name),
            "--intent",
            intent,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


class P21ReductionContractTests(unittest.TestCase):
    def test_softmax_tuning_handoff_retains_loops_and_candidates(self) -> None:
        result = _validate("code-spec.softmax.valid.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"operator_pattern": "softmax-style"', result.stdout)

    def test_direct_vectorized_handoff_is_rejected(self) -> None:
        result = _validate("code-spec.softmax.vectorized-invalid.json")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(
            "requires reduce_loop" in result.stderr
            or "below the full extent" in result.stderr,
            result.stderr,
        )

    def test_standalone_generation_does_not_require_a_tuning_surface(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--base-info",
                str(BASE_INFO),
                "--spec",
                str(ARTIFACTS / "code-spec.reduction.standalone.json"),
                "--intent",
                "standalone",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("standalone-reduction", result.stdout)


if __name__ == "__main__":
    unittest.main()
