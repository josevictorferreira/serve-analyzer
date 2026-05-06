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
        """Write a minimal valid WallCalibration JSON using synthetic ground-truth positions."""
        # Ground-truth positions from WallFixtureGroundTruth (320x240, wall at x=240):
        # We use the wall_x_px=240 and frame dimensions to place reference points.
        # Wall frame: x along wall (camera-right positive), y up (vertical).
        # We'll place 4 points at the wall plane.
        cal = {
            "setup": {
                "serve_contact_distance_m": 6.11,
                "camera_wall_distance_m": 1.57,
                "serve_contact_height_m": 2.80,
                "wall_reference_points": [
                    {"name": "bottom_left", "pixel": [240, 200], "wall_m": [-4.0, 0.0]},
                    {"name": "bottom_right", "pixel": [240, 200], "wall_m": [4.0, 0.0]},
                    {"name": "top_left", "pixel": [240, 40], "wall_m": [-4.0, 3.0]},
                    {"name": "top_right", "pixel": [240, 40], "wall_m": [4.0, 3.0]},
                ],
                "hook_reference": {"pixel": [240, 60], "height_m": 2.45},
                "chair_references": [{"pixel": [240, 180], "height_m": 1.0}],
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
        """Synthetic video produces result.json and result.csv; tolerates missing T10 artifacts."""
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

            self.assertEqual(exit_code, 0)

            video_stem = video_path.stem
            video_out = output_dir / video_stem
            self.assertTrue(video_out.exists())

            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())
            raw = json.loads(result_json.read_text(encoding="utf-8"))
            self.assertIn("measured", raw)
            self.assertIn("inferred", raw)
            self.assertIn("assumed", raw)
            self.assertIn("confidence", raw)
            self.assertIn("warnings", raw)
            self.assertIn("artifacts", raw)

            result_csv = video_out / "result.csv"
            self.assertTrue(result_csv.exists())
            lines = result_csv.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)  # header + 1 row

            # T10 artifacts are not yet implemented; their absence is tolerated with warnings.
            if not (video_out / f"{video_stem}_annotated.mp4").exists():
                self.assertIn("wall_artifacts", captured.lower())

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


if __name__ == "__main__":
    unittest.main()
