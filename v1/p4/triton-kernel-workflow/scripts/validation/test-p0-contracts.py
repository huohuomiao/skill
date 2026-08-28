#!/usr/bin/env python3
"""Offline regression tests for the P0 contracts carried into p1."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[2]
WRITE_ENV = SKILL_ROOT / "scripts" / "state" / "write-env-config.py"
SELECT_BEST = SKILL_ROOT / "scripts" / "state" / "select-best-candidate.py"


class P0ContractTests(unittest.TestCase):
    def test_environment_json_is_canonical_and_report_is_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            runtime_info = output_dir / "EnvConfig" / "runtime_info.txt"
            runtime_info.parent.mkdir(parents=True)
            runtime_info.write_text("device probe passed\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(WRITE_ENV),
                    "--output-dir",
                    str(output_dir),
                    "--backend",
                    "local",
                    "--env-check-task-id",
                    "local",
                    "--runtime-info-path",
                    str(runtime_info),
                    "--device-model",
                    "test-device",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((output_dir / "EnvConfig" / "config.json").read_text())
            report = (output_dir / "EnvConfig" / "config.md").read_text(encoding="utf-8")
            self.assertEqual(config["execution_backend"], "local")
            self.assertIsNone(config["worker_submit_url"])
            self.assertIn("由 `config.json` 生成", report)

    def test_fastest_accuracy_passing_candidate_wins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_root = root / "candidates"
            output_dir = root / "Optimizer"
            for candidate_id, source_stage, latency, accuracy in (
                ("baseline", "baseline", 1.0, True),
                ("fast-invalid", "oob", 0.5, False),
                ("best-valid", "oob", 0.8, True),
                ("last-slower", "advanced", 1.2, True),
            ):
                workdir = candidate_root / candidate_id
                workdir.mkdir(parents=True)
                (workdir / "code.py").write_text(
                    f"SELECTED = {candidate_id!r}\n", encoding="utf-8"
                )
                (workdir / "report.md").write_text("report\n", encoding="utf-8")
                manifest = {
                    "schema_version": "1.0",
                    "candidate_id": candidate_id,
                    "source_stage": source_stage,
                    "code_path": "code.py",
                    "report_path": "report.md",
                    "accuracy_pass": accuracy,
                    "latency_ms": latency,
                    "bandwidth_gbps": None,
                    "execution_backend": "local",
                    "hardware_model": "test-device",
                    "benchmark_signature": "fixture-v1",
                }
                (workdir / "candidate.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SELECT_BEST),
                    "--candidate-root",
                    str(candidate_root),
                    "--output-dir",
                    str(output_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((output_dir / "best_so_far.json").read_text())
            self.assertEqual(metadata["selected_candidate"]["candidate_id"], "best-valid")
            self.assertEqual(
                (output_dir / "best_so_far.py").read_bytes(),
                (output_dir / "triton_optimized.py").read_bytes(),
            )
            self.assertIn("best-valid", (output_dir / "triton_optimized.py").read_text())

    def test_incompatible_benchmarks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, signature in enumerate(("shape-a", "shape-b")):
                workdir = root / str(index)
                workdir.mkdir()
                (workdir / "code.py").write_text("pass\n", encoding="utf-8")
                (workdir / "report.md").write_text("report\n", encoding="utf-8")
                (workdir / "candidate.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "candidate_id": str(index),
                            "source_stage": "oob",
                            "code_path": "code.py",
                            "report_path": "report.md",
                            "accuracy_pass": True,
                            "latency_ms": 1.0 + index,
                            "execution_backend": "local",
                            "hardware_model": "test-device",
                            "benchmark_signature": signature,
                        }
                    ),
                    encoding="utf-8",
                )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SELECT_BEST),
                    "--candidate-root",
                    str(root),
                    "--output-dir",
                    str(root / "out"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("incompatible", result.stderr)

    def test_duplicate_candidate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(2):
                workdir = root / str(index)
                workdir.mkdir()
                (workdir / "code.py").write_text("pass\n", encoding="utf-8")
                (workdir / "report.md").write_text("report\n", encoding="utf-8")
                (workdir / "candidate.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "1.0",
                            "candidate_id": "duplicate",
                            "source_stage": "oob",
                            "code_path": "code.py",
                            "report_path": "report.md",
                            "accuracy_pass": True,
                            "latency_ms": 1.0 + index,
                            "execution_backend": "local",
                            "hardware_model": "test-device",
                            "benchmark_signature": "fixture-v1",
                        }
                    ),
                    encoding="utf-8",
                )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SELECT_BEST),
                    "--candidate-root",
                    str(root),
                    "--output-dir",
                    str(root / "out"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("duplicate candidate_id", result.stderr)

    def test_non_finite_latency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "code.py").write_text("pass\n", encoding="utf-8")
            (root / "report.md").write_text("report\n", encoding="utf-8")
            (root / "candidate.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "candidate_id": "not-finite",
                        "source_stage": "oob",
                        "code_path": "code.py",
                        "report_path": "report.md",
                        "accuracy_pass": True,
                        "latency_ms": float("nan"),
                        "execution_backend": "local",
                        "hardware_model": "test-device",
                        "benchmark_signature": "fixture-v1",
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SELECT_BEST),
                    "--candidate-root",
                    str(root),
                    "--output-dir",
                    str(root / "out"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("finite and positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
