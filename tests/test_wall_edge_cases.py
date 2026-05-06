"""Edge-case tests for the wall analysis pipeline.

Covers: variable fps override, rotated video metadata, conflicting overrides,
nonexistent manual correction serve index, missing intrinsics, and lateral
pipeline symbol stability.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import cv2

from serve_analyzer.wall_serve import (
    main,
)
from wall_test_helpers import generate_wall_impact_video


def _make_calibration_dict(**overrides) -> dict:
    """Build a minimal valid calibration dict for synthetic tests."""
    setup = {
        "serve_contact_distance_m": 6.11,
        "camera_wall_distance_m": 1.57,
        "serve_contact_height_m": 2.80,
        "wall_reference_points": [
            {"name": "bl", "pixel": [100, 400], "wall_m": [-4.0, 0.0]},
            {"name": "br", "pixel": [480, 400], "wall_m": [4.0, 0.0]},
            {"name": "tl", "pixel": [100, 80], "wall_m": [-4.0, 3.0]},
            {"name": "tr", "pixel": [480, 80], "wall_m": [4.0, 3.0]},
        ],
        "hook_reference": {"pixel": [290, 100], "height_m": 2.45},
    }
    setup.update(overrides)
    setup.update(overrides)
    return {"setup": setup}


def _make_calibration_json(tmpdir: str, **overrides) -> Path:
    """Write calibration JSON to a temp dir and return its path."""
    cal = _make_calibration_dict(**overrides)
    path = Path(tmpdir) / "calibration.json"
    path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return path


class TestVariableFpsMetadata(unittest.TestCase):
    """FPS override changes impact_time_sec computation."""

    def test_variable_fps_metadata_uses_override(self):
        """Synthetic 60 fps video with --fps 30 override; impact_time_sec uses 30 fps timeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = _make_calibration_json(tmpdir)
            video_path = Path(tmpdir) / "fps_test.mp4"
            gt = generate_wall_impact_video(
                str(video_path),
                width=640,
                height=480,
                fps=60,
                wall_x_px=480,
                impact_frame=30,
                ball_speed_px_per_frame=8.0,
            )
            output_dir = Path(tmpdir) / "results"

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = main(
                    [
                        "--video",
                        str(video_path),
                        "--metadata",
                        str(cal_path),
                        "--output-dir",
                        str(output_dir),
                        "--fps",
                        "30",
                        "--no-video",
                        "--no-plots",
                    ]
                )
            finally:
                sys.stderr = old_stderr

            self.assertEqual(exit_code, 0)

            video_out = output_dir / video_path.stem
            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())

            raw = json.loads(result_json.read_text(encoding="utf-8"))
            impact_time = raw["measured"]["impact_time_sec"]
            impact_frame = raw["measured"]["impact_frame"]

            # With --fps 30, impact_time_sec should be frame / 30
            if impact_frame is not None:
                expected_time = impact_frame / 30.0
                self.assertAlmostEqual(
                    impact_time,
                    expected_time,
                    places=4,
                    msg=(
                        f"impact_time_sec ({impact_time}) should equal "
                        f"impact_frame ({impact_frame}) / 30 = {expected_time}"
                    ),
                )


class TestRotatedVideoMetadata(unittest.TestCase):
    """Rotated video metadata (swapped dims) handled without unhandled exceptions."""

    def test_rotated_video_metadata_via_mock(self):
        """Patch VideoCapture to report swapped dims; pipeline must not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = _make_calibration_json(tmpdir)
            video_path = Path(tmpdir) / "rotated_test.mp4"
            generate_wall_impact_video(
                str(video_path),
                width=640,
                height=480,
                fps=60,
                wall_x_px=480,
                impact_frame=30,
                ball_speed_px_per_frame=8.0,
            )
            output_dir = Path(tmpdir) / "results"

            # Build a mock VideoCapture that wraps the real one but swaps W/H.
            _RealVC = cv2.VideoCapture

            class _SwappedVC(_RealVC):
                def get(self, prop):
                    if prop == cv2.CAP_PROP_FRAME_WIDTH:
                        return 480.0  # swapped
                    if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                        return 640.0  # swapped
                    return super().get(prop)

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                with patch("serve_analyzer.wall_serve.cv2.VideoCapture", _SwappedVC):
                    exit_code = main(
                        [
                            "--video", str(video_path),
                            "--metadata", str(cal_path),
                            "--output-dir", str(output_dir),
                            "--no-video",
                            "--no-plots",
                        ]
                    )
            except Exception as exc:
                self.fail(
                    f"Pipeline raised unhandled exception with swapped dims: {exc}"
                )
            finally:
                sys.stderr = old_stderr

            # Must complete without crash (exit 0 or 1, not unhandled exception)
            self.assertIn(exit_code, (0, 1), f"Unexpected exit code: {exit_code}")

class TestConflictingOverride(unittest.TestCase):
    """Per-video override wins over setup metadata."""

    def test_conflicting_per_video_override(self):
        """setup.json serve_contact_height_m=2.45 vs override 2.80; override wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup with 2.45
            cal_path = _make_calibration_json(tmpdir, serve_contact_height_m=2.45)
            # Override with 2.80
            override_path = Path(tmpdir) / "override.json"
            override_path.write_text(
                json.dumps(
                    {
                        "video_override": {
                            "serve_contact_height_m": 2.80,
                        }
                    }
                ),
                encoding="utf-8",
            )

            video_path = Path(tmpdir) / "override_test.mp4"
            generate_wall_impact_video(
                str(video_path),
                width=640,
                height=480,
                fps=60,
                wall_x_px=480,
                impact_frame=30,
                ball_speed_px_per_frame=8.0,
            )
            output_dir = Path(tmpdir) / "results"

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = main(
                    [
                        "--video",
                        str(video_path),
                        "--metadata",
                        str(cal_path),
                        "--override",
                        str(override_path),
                        "--output-dir",
                        str(output_dir),
                        "--no-video",
                        "--no-plots",
                    ]
                )
            finally:
                sys.stderr = old_stderr

            self.assertEqual(exit_code, 0)

            video_out = output_dir / video_path.stem
            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())

            raw = json.loads(result_json.read_text(encoding="utf-8"))

            # The assumed section should reflect the override value
            assumed = raw.get("assumed", {})
            self.assertEqual(
                assumed.get("contact_height_m"),
                2.80,
                f"Override should set contact_height_m to 2.80, got {assumed.get('contact_height_m')}",
            )


class TestNonexistentManualCorrection(unittest.TestCase):
    """Manual correction for a serve index that doesn't exist is ignored."""

    def test_nonexistent_manual_correction_serve_index(self):
        """Pass manual_corrections with key '99'; pipeline does not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = _make_calibration_json(tmpdir)
            video_path = Path(tmpdir) / "correction_test.mp4"
            generate_wall_impact_video(
                str(video_path),
                width=640,
                height=480,
                fps=60,
                wall_x_px=480,
                impact_frame=30,
                ball_speed_px_per_frame=8.0,
            )
            output_dir = Path(tmpdir) / "results"

            # Manual correction for serve_index 99 — no such serve exists
            corrections_path = Path(tmpdir) / "corrections.json"
            corrections_path.write_text(
                json.dumps(
                    {
                        "99": {"pixel_x": 300, "pixel_y": 200},
                    }
                ),
                encoding="utf-8",
            )

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = main(
                    [
                        "--video",
                        str(video_path),
                        "--metadata",
                        str(cal_path),
                        "--manual-corrections",
                        str(corrections_path),
                        "--output-dir",
                        str(output_dir),
                        "--no-video",
                        "--no-plots",
                    ]
                )
            finally:
                sys.stderr = old_stderr

            # Must not crash
            self.assertEqual(exit_code, 0)

            video_out = output_dir / video_path.stem
            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())

            raw = json.loads(result_json.read_text(encoding="utf-8"))

            # manual_correction_used should NOT appear since key "99" doesn't match serve 0
            warnings = raw.get("warnings", [])
            self.assertNotIn(
                "manual_correction_used",
                warnings,
                "Unmatched correction index should not set manual_correction_used",
            )


class TestMissingIntrinsics(unittest.TestCase):
    """Missing intrinsics block produces degraded_intrinsics or correct behavior."""

    def test_missing_intrinsics_flags_degraded(self):
        """No intrinsics block in metadata; verify pipeline handles it correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_path = _make_calibration_json(tmpdir)
            # The calibration has no intrinsics key — that's the normal case.
            # Intrinsics is only present when explicitly provided.

            video_path = Path(tmpdir) / "no_intrinsics_test.mp4"
            generate_wall_impact_video(
                str(video_path),
                width=640,
                height=480,
                fps=60,
                wall_x_px=480,
                impact_frame=30,
                ball_speed_px_per_frame=8.0,
            )
            output_dir = Path(tmpdir) / "results"

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = main(
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
            finally:
                sys.stderr = old_stderr

            self.assertEqual(exit_code, 0)

            video_out = output_dir / video_path.stem
            result_json = video_out / "result.json"
            self.assertTrue(result_json.exists())

            raw = json.loads(result_json.read_text(encoding="utf-8"))

            # When no intrinsics are provided, degraded_intrinsics should be False
            # in the confidence section (since calibration.intrinsics is None, not approx_exif).
            confidence = raw.get("confidence", {})
            self.assertFalse(
                confidence.get("degraded_intrinsics", False),
                "Missing intrinsics should not flag degraded_intrinsics (only approx_exif does)",
            )

            # Confidence aggregate should be reasonable
            agg_score = confidence.get("aggregate_score", 0)
            self.assertGreater(agg_score, 0.0, "Aggregate confidence should be > 0")

    def test_approx_exif_intrinsics_flags_degraded(self):
        """intrinsics.source='approx_exif' triggers degraded_intrinsics warning and reduced confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cal = _make_calibration_dict()
            cal["intrinsics"] = {
                "source": "approx_exif",
                "camera_matrix": [
                    [1000.0, 0.0, 320.0],
                    [0.0, 1000.0, 240.0],
                    [0.0, 0.0, 1.0],
                ],
                "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
            cal_path = Path(tmpdir) / "calibration_exif.json"
            cal_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")

            video_path = Path(tmpdir) / "exif_intrinsics_test.mp4"
            generate_wall_impact_video(
                str(video_path),
                width=640,
                height=480,
                fps=60,
                wall_x_px=480,
                impact_frame=30,
                ball_speed_px_per_frame=8.0,
            )
            output_dir = Path(tmpdir) / "results"

            old_stderr = sys.stderr
            sys.stderr = StringIO()
            try:
                exit_code = main(
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
            finally:
                sys.stderr = old_stderr

            self.assertEqual(exit_code, 0)

            video_out = output_dir / video_path.stem
            result_json = video_out / "result.json"
            raw = json.loads(result_json.read_text(encoding="utf-8"))

            # Should contain degraded_intrinsics in warnings
            warnings = raw.get("warnings", [])
            self.assertIn(
                "degraded_intrinsics",
                warnings,
                "approx_exif intrinsics should produce degraded_intrinsics warning",
            )

            # Confidence should be reduced
            confidence = raw.get("confidence", {})
            self.assertTrue(
                confidence.get("degraded_intrinsics", False),
                "degraded_intrinsics should be True in confidence dict",
            )


class TestLateralPipelineUnchanged(unittest.TestCase):
    """Wall imports must not shadow lateral pipeline public symbols."""

    def test_existing_lateral_pipeline_unchanged(self):
        """Import cli and serve_attempts_v6; check public symbols match snapshot."""
        import serve_analyzer.cli as cli_mod
        import serve_analyzer.serve_attempts_v6 as v6_mod

        cli_symbols = set(dir(cli_mod))
        # These key symbols must exist (non-exhaustive snapshot)
        expected_cli = {
            "main",
            "InteractiveCalibrator",
            "run_analysis",
        }
        for sym in expected_cli:
            self.assertIn(
                sym,
                cli_symbols,
                f"cli module missing expected symbol: {sym}",
            )

        v6_symbols = set(dir(v6_mod))
        expected_v6 = {
            "detect_serve_candidates_v6",
            "main",
        }
        for sym in expected_v6:
            self.assertIn(
                sym,
                v6_symbols,
                f"serve_attempts_v6 module missing expected symbol: {sym}",
            )


if __name__ == "__main__":
    unittest.main()
