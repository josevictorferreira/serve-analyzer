"""Tests for wall calibration CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from serve_analyzer.wall_calibration import (
    WallCalibration,
    main,
)


class TestWallCalibrationCli(unittest.TestCase):
    """Contract tests for the wall calibration CLI."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wall_points_flag(self, count: int = 4) -> str:
        """Return a --wall-points string with *count* reference pairs."""
        pairs = [
            "100,500,-4.0,0.0",
            "700,500,4.0,0.0",
            "100,100,-4.0,3.0",
            "700,100,4.0,3.0",
            "400,300,0.0,1.5",
            "200,200,-2.0,2.0",
        ]
        return ";".join(pairs[:count])

    # ------------------------------------------------------------------
    # Acceptance tests
    # ------------------------------------------------------------------

    def test_help_exits_zero(self):
        """--help prints usage and exits 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use subprocess so argparse handles --help internally
            result = subprocess.run(
                [sys.executable, "-m", "serve_analyzer.wall_calibration", "--help"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            self.assertEqual(result.returncode, 0)
            self.assertIn("setup", result.stdout.lower())
            self.assertIn("override", result.stdout.lower())

    def test_noninteractive_writes_valid_setup_json(self):
        """setup mode with flags writes JSON that from_dict() accepts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "setup.json"
            wall_pts = self._wall_points_flag(4)
            exit_code = main(
                [
                    "--mode",
                    "setup",
                    "--output",
                    str(out),
                    "--serve-contact-height",
                    "2.80",
                    "--wall-points",
                    wall_pts,
                    "--hook-point",
                    "400,150",
                    "--chair-point",
                    "200,450",
                    "--chair-point",
                    "600,450",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out.exists())

            raw = json.loads(out.read_text())
            cal = WallCalibration.from_dict(raw)

            self.assertEqual(cal.serve_contact_height_m, 2.80)
            self.assertEqual(cal.serve_contact_distance_m, 6.11)
            self.assertEqual(cal.camera_wall_distance_m, 1.57)
            self.assertEqual(len(cal.wall_reference_points), 4)

            self.assertIsNotNone(cal.hook_reference)
            self.assertEqual(cal.hook_reference.height_m, 2.45)  # type: ignore[union-attr]

            self.assertEqual(len(cal.chair_references), 2)
            self.assertEqual(cal.chair_references[0].height_m, 1.0)
            self.assertEqual(cal.chair_references[1].height_m, 1.0)

    def test_insufficient_wall_points_returns_error(self):
        """Only 3 wall points → nonzero exit, structured stderr, no traceback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "setup.json"
            wall_pts = self._wall_points_flag(3)

            # Capture stderr by redirecting sys.stderr
            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = main(
                    [
                        "--mode",
                        "setup",
                        "--output",
                        str(out),
                        "--serve-contact-height",
                        "2.80",
                        "--wall-points",
                        wall_pts,
                    ]
                )
            finally:
                captured = sys.stderr.getvalue()
                sys.stderr = old_stderr

            self.assertNotEqual(exit_code, 0)
            self.assertFalse(out.exists())

            # Structured JSON error
            err = json.loads(captured)
            self.assertIn("wall", err["error"].lower())
            self.assertIn("4", err["error"])
            # No Python traceback
            self.assertNotIn("Traceback", captured)

    def test_missing_contact_height_returns_error(self):
        """No --serve-contact-height → nonzero exit, structured message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "setup.json"
            wall_pts = self._wall_points_flag(4)

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = main(
                    [
                        "--mode",
                        "setup",
                        "--output",
                        str(out),
                        "--wall-points",
                        wall_pts,
                    ]
                )
            finally:
                captured = sys.stderr.getvalue()
                sys.stderr = old_stderr

            self.assertNotEqual(exit_code, 0)
            self.assertFalse(out.exists())

            err = json.loads(captured)
            self.assertIn("serve_contact_height_m", err["error"])
            self.assertNotIn("Traceback", captured)

    def test_override_mode_writes_minimal_per_video_json(self):
        """override mode writes a JSON with video_override keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "override.json"
            exit_code = main(
                [
                    "--mode",
                    "override",
                    "--output",
                    str(out),
                    "--serve-contact-height",
                    "2.95",
                    "--camera-wall-distance",
                    "1.80",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(out.exists())

            raw = json.loads(out.read_text())
            self.assertIn("video_override", raw)
            vo = raw["video_override"]
            self.assertEqual(vo["serve_contact_height_m"], 2.95)
            self.assertEqual(vo["camera_wall_distance_m"], 1.80)
            # Keys that were NOT supplied must not appear
            self.assertNotIn("serve_contact_distance_m", vo)

    def test_override_mode_with_wall_points_and_refs(self):
        """override mode accepts wall points, hook, and chairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "override.json"
            wall_pts = self._wall_points_flag(4)
            exit_code = main(
                [
                    "--mode",
                    "override",
                    "--output",
                    str(out),
                    "--wall-points",
                    wall_pts,
                    "--hook-point",
                    "400,150",
                    "--chair-point",
                    "200,450",
                ]
            )
            self.assertEqual(exit_code, 0)
            raw = json.loads(out.read_text())
            vo = raw["video_override"]
            self.assertEqual(len(vo["wall_reference_points"]), 4)
            self.assertEqual(vo["hook_reference"]["height_m"], 2.45)
            self.assertEqual(len(vo["chair_references"]), 1)

    def test_default_mode_is_setup(self):
        """Omitting --mode defaults to setup mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "setup.json"
            wall_pts = self._wall_points_flag(4)
            exit_code = main(
                [
                    "--output",
                    str(out),
                    "--serve-contact-height",
                    "2.80",
                    "--wall-points",
                    wall_pts,
                ]
            )
            self.assertEqual(exit_code, 0)
            raw = json.loads(out.read_text())
            self.assertIn("setup", raw)
            cal = WallCalibration.from_dict(raw)
            self.assertEqual(cal.serve_contact_height_m, 2.80)

    def test_intrinsics_source_not_emitted_by_cli(self):
        """CLI does not emit intrinsics block because matrix data is unavailable via flags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "setup.json"
            wall_pts = self._wall_points_flag(4)
            exit_code = main(
                [
                    "--mode",
                    "setup",
                    "--output",
                    str(out),
                    "--serve-contact-height",
                    "2.80",
                    "--wall-points",
                    wall_pts,
                ]
            )
            self.assertEqual(exit_code, 0)
            raw = json.loads(out.read_text())
            # Intrinsics block is absent because the CLI cannot supply
            # camera_matrix / dist_coeffs required by from_dict().
            self.assertNotIn("intrinsics", raw)

    def test_interactive_flag_defaults_off(self):
        """--interactive is not required and defaults to False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "setup.json"
            wall_pts = self._wall_points_flag(4)
            # Should succeed without any display requirement
            exit_code = main(
                [
                    "--mode",
                    "setup",
                    "--output",
                    str(out),
                    "--serve-contact-height",
                    "2.80",
                    "--wall-points",
                    wall_pts,
                ]
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
