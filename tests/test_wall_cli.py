"""Tests for wall analysis orchestration CLI."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from serve_analyzer.wall_serve import main
from wall_test_helpers import generate_wall_impact_video


class TestWallAnalysisCli(unittest.TestCase):
    """Contract tests for the wall analysis orchestration CLI."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_calibration_json(self, tmpdir: str) -> Path:
        """Write a minimal valid WallCalibration JSON with non-collinear reference points."""
        # 4 corner-style points spanning the frame for a valid homography.
        cal = {
            "setup": {
                "serve_contact_distance_m": 6.11,
                "camera_wall_distance_m": 1.57,
                "serve_contact_height_m": 2.80,
                "wall_reference_points": [
                    {"name": "bottom_left", "pixel": [100, 400], "wall_m": [-4.0, 0.0]},
                    {"name": "bottom_right", "pixel": [540, 400], "wall_m": [4.0, 0.0]},
                    {"name": "top_left", "pixel": [100, 80], "wall_m": [-4.0, 3.0]},
                    {"name": "top_right", "pixel": [540, 80], "wall_m": [4.0, 3.0]},
                ],
                "hook_reference": {"pixel": [320, 60], "height_m": 2.45},
                "chair_references": [{"pixel": [320, 180], "height_m": 1.0}],
            }
        }
        path = Path(tmpdir) / "calibration.json"
        path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
        return path

    def _run_main(self, argv: list[str]) -> int:
        """Invoke main(argv) capturing any SystemExit."""
        try:
            return main(argv)
        except SystemExit as exc:
            return exc.code if isinstance(exc.code, int) else 1

    # ------------------------------------------------------------------
    # Acceptance tests
    # ------------------------------------------------------------------

    def test_help_exits_zero(self):
        """--help prints usage and exits 0."""
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_synthetic_end_to_end_outputs_all_artifacts(self):
        """Synthetic video produces all required artifacts: JSON, CSV, annotated MP4, plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = self._make_calibration_json(tmpdir)
            video_path = Path(tmpdir) / "synthetic_serve.mp4"
            generate_wall_impact_video(str(video_path), width=640, height=480, wall_x_px=480, impact_frame=30, ball_speed_px_per_frame=8.0)
            output_dir = Path(tmpdir) / "results"

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = self._run_main(
                    [
                        "--video",
                        str(video_path),
                        "--metadata",
                        str(cal_path),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            finally:
                captured = sys.stderr.getvalue()
                sys.stderr = old_stderr

            self.assertEqual(exit_code, 0, f"CLI failed. stderr: {captured}")

            video_stem = video_path.stem
            video_out = output_dir / video_stem
            self.assertTrue(video_out.exists())

            # --- JSON with 6 sections and non-null wall meters ---
            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())
            raw = json.loads(result_json.read_text(encoding="utf-8"))
            for section in ("measured", "inferred", "assumed", "confidence", "warnings", "artifacts"):
                self.assertIn(section, raw)

            # Wall-meter coordinates must be populated for calibrated impacts
            self.assertIsNotNone(
                raw["measured"].get("wall_x_m"),
                "wall_x_m should be non-null for calibrated impact",
            )
            self.assertIsNotNone(
                raw["measured"].get("wall_y_m"),
                "wall_y_m should be non-null for calibrated impact",
            )
            # --- CSV ---
            result_csv = video_out / "result.csv"
            self.assertTrue(result_csv.exists())
            lines = result_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)  # header + 1 row

            # --- Annotated video at deterministic path ---
            annotated = video_out / f"{video_stem}_annotated.mp4"
            self.assertTrue(
                annotated.exists(),
                f"Annotated MP4 must exist at {annotated}. stderr: {captured}",
            )

            # --- Plot PNGs under plots/ subdir ---
            plots_dir = video_out / "plots"
            self.assertTrue(plots_dir.exists(), "plots/ directory must exist")
            plot_pngs = list(plots_dir.glob("*.png"))
            self.assertGreaterEqual(
                len(plot_pngs), 3,
                f"Expected >= 3 plot PNGs, got {len(plot_pngs)}: {plot_pngs}",
            )

    def test_no_video_no_plots_flags_skip_artifacts(self):
        """--no-video and --no-plots skip heavy artifacts but still produce JSON/CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = self._make_calibration_json(tmpdir)
            video_path = Path(tmpdir) / "synthetic_serve.mp4"
            generate_wall_impact_video(str(video_path), width=640, height=480, wall_x_px=480, impact_frame=30, ball_speed_px_per_frame=8.0)
            output_dir = Path(tmpdir) / "results"

            exit_code = self._run_main(
                [
                    "--video",
                    str(video_path),
                    "--metadata",
                    str(cal_path),
                    "--output-dir",
                    str(output_dir),
                    "--no-video",
                    "--no-plots",
                ]
            )
            self.assertEqual(exit_code, 0)

            video_stem = video_path.stem
            video_out = output_dir / video_stem
            self.assertTrue(video_out.exists())

            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())

            result_csv = video_out / "result.csv"
            self.assertTrue(result_csv.exists())
            lines = result_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            # No MP4 or PNGs should be present
            mp4_files = list(video_out.glob("*.mp4"))
            png_files = list(video_out.glob("*.png"))
            plot_dir = video_out / "plots"
            if plot_dir.exists():
                png_files.extend(plot_dir.glob("*.png"))
            self.assertEqual(len(mp4_files), 0)
            self.assertEqual(len(png_files), 0)

    def test_documented_synthetic_workflow_command(self):
        """The exact command pattern from README works with synthetic fixtures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = self._make_calibration_json(tmpdir)
            video_path = Path(tmpdir) / "synthetic_serve.MOV"
            generate_wall_impact_video(
                str(video_path),
                width=640,
                height=480,
                wall_x_px=480,
                impact_frame=30,
                ball_speed_px_per_frame=8.0,
            )
            output_dir = Path(tmpdir) / "results"

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = self._run_main(
                    [
                        "--batch",
                        str(Path(tmpdir) / "*.MOV"),
                        "--metadata",
                        str(cal_path),
                        "--output-dir",
                        str(output_dir),
                        "--no-video",
                        "--no-plots",
                    ]
                )
            finally:
                sys.stderr = old_stderr

            self.assertEqual(exit_code, 0)

            video_stem = video_path.stem
            video_out = output_dir / video_stem
            self.assertTrue(video_out.exists())

            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())
            raw = json.loads(result_json.read_text(encoding="utf-8"))
            self.assertIn("measured", raw)
            self.assertIn("inferred", raw)

            result_csv = video_out / "result.csv"
            self.assertTrue(result_csv.exists())
            lines = result_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)  # header + 1 row

            aggregate_csv = output_dir / "all_serves.csv"
            self.assertTrue(aggregate_csv.exists())
            agg_lines = aggregate_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(agg_lines), 2)  # header + 1 row

    def test_real_wall_video_examples_are_documented_only(self):
        """Real wall video paths appear in docs but never in test code."""
        repo_root = Path(__file__).resolve().parent.parent
        readme_path = repo_root / "README.md"
        wall_serve_path = repo_root / "serve_analyzer" / "wall_serve.py"

        readme_text = readme_path.read_text(encoding="utf-8")
        wall_serve_text = wall_serve_path.read_text(encoding="utf-8")

        self.assertIn("videos/wall/", readme_text)
        self.assertIn("videos/wall/", wall_serve_text)

        test_files = list((repo_root / "tests").glob("test_wall_*.py"))
        self.assertTrue(
            len(test_files) > 0,
            "Expected at least one test_wall_*.py file",
        )

        _video_pattern = "videos/wall/" + "IMG"
        for test_file in test_files:
            text = test_file.read_text(encoding="utf-8")
            matches = [
                line
                for line in text.splitlines()
                if _video_pattern in line
            ]
            self.assertEqual(
                len(matches),
                0,
                f"{test_file.name} references real wall video: {matches}",
            )



if __name__ == "__main__":
    unittest.main()
