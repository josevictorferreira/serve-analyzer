"""Tests for autonomous wall-impact detection with manual-correction overlay.

Uses the synthetic video helper from ``tests/wall_test_helpers.py`` to
generate deterministic fixtures, then asserts that
:func:`detect_wall_impact` recovers the known impact frame and pixel
within tolerance.
"""

import tempfile
import unittest
from pathlib import Path

from serve_analyzer.wall_calibration import WallCalibration, WallReferencePoint
from serve_analyzer.wall_serve import detect_wall_impact

from wall_test_helpers import generate_wall_impact_video


def _make_calibration(wall_x_px: float, height_px: int = 240) -> WallCalibration:
    """Build a minimal WallCalibration for synthetic-video tests.

    Four reference points arranged in a rectangle whose right edge sits at
    *wall_x_px* and spans the frame height.
    """
    return WallCalibration(
        serve_contact_height_m=2.8,
        wall_reference_points=[
            WallReferencePoint(
                name="tl", pixel=(wall_x_px - 80, 20), wall_m=(0.0, 2.0)
            ),
            WallReferencePoint(name="tr", pixel=(wall_x_px, 20), wall_m=(0.5, 2.0)),
            WallReferencePoint(
                name="bl",
                pixel=(wall_x_px - 80, height_px - 20),
                wall_m=(0.0, 0.0),
            ),
            WallReferencePoint(
                name="br",
                pixel=(wall_x_px, height_px - 20),
                wall_m=(0.5, 0.0),
            ),
        ],
    )


class TestWallImpactDetection(unittest.TestCase):
    """Core wall-impact detection contract tests."""

    def test_detects_known_impact_frame_and_point(self):
        """Generate synthetic video with impact_frame=60, wall_x_px=240;
        assert detected impact frame within ±1 of 60 AND detected pixel x
        within ±5 px of wall_x_px.  No manual_correction_used in warnings.
        """
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "impact_test.mp4")
            gt = generate_wall_impact_video(
                video_path,
                width=640,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=60,
                ball_radius=6,
                ball_speed_px_per_frame=3.0,
                wall_x_px=240,
            )

            calibration = _make_calibration(wall_x_px=gt.wall_x_px, height_px=240)
            result = detect_wall_impact(video_path, calibration)

            # Impact frame must be within ±1 of ground truth.
            self.assertIsNotNone(result.impact_frame)
            self.assertLessEqual(
                abs(result.impact_frame - gt.impact_frame),
                1,
                f"impact_frame {result.impact_frame} not within ±1 of {gt.impact_frame}",
            )

            # Impact pixel x must be within ±5 px of wall_x_px.
            self.assertIsNotNone(result.impact_pixel)
            self.assertLessEqual(
                abs(result.impact_pixel[0] - gt.wall_x_px),
                5,
                f"impact_pixel x {result.impact_pixel[0]:.1f} not within ±5 of wall_x_px={gt.wall_x_px}",
            )

            # Autonomous values must match final values (no manual correction).
            self.assertEqual(result.impact_frame, result.autonomous_frame)
            self.assertEqual(result.impact_pixel, result.autonomous_pixel)

            # No manual_correction_used warning.
            self.assertNotIn("manual_correction_used", result.warnings)

    def test_manual_correction_overrides_autonomous_detection(self):
        """Pass manual_correction={"impact_frame": 55, "impact_pixel": (230, 120)};
        assert impact_frame==55 AND impact_pixel==(230,120); assert
        autonomous_frame still populated and DIFFERENT from 55 (or preserved);
        warnings include 'manual_correction_used'.
        """
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "manual_test.mp4")
            gt = generate_wall_impact_video(
                video_path,
                width=640,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=60,
                ball_speed_px_per_frame=3.0,
                wall_x_px=240,
            )

            calibration = _make_calibration(wall_x_px=gt.wall_x_px, height_px=240)
            result = detect_wall_impact(
                video_path,
                calibration,
                manual_correction={
                    "impact_frame": 55,
                    "impact_pixel": (230.0, 120.0),
                },
            )

            # Final values must equal the manual correction.
            self.assertEqual(result.impact_frame, 55)
            self.assertEqual(result.impact_pixel, (230.0, 120.0))

            # Autonomous candidate must still be populated.
            self.assertIsNotNone(result.autonomous_frame)
            self.assertIsNotNone(result.autonomous_pixel)

            # Autonomous should differ from the manual override.
            self.assertNotEqual(result.autonomous_frame, 55)

            # Warning must include manual_correction_used.
            self.assertIn("manual_correction_used", result.warnings)

    def test_insufficient_track_returns_warning(self):
        """Generate video where ball never reaches wall due to heavy blur;
        assert impact_frame is None and warnings include
        'insufficient_track'.
        """
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "noimpact_test.mp4")
            # Use heavy blur (sigma=10) to make the ball undetectable
            # by the brightness-threshold detector.
            gt = generate_wall_impact_video(
                video_path,
                width=640,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=30,
                ball_speed_px_per_frame=4.0,
                wall_x_px=240,
                blur_sigma=10.0,
            )

            calibration = _make_calibration(wall_x_px=gt.wall_x_px, height_px=240)
            result = detect_wall_impact(video_path, calibration)

            self.assertIsNone(result.impact_frame)
            self.assertIsNone(result.impact_pixel)
            self.assertIn("insufficient_track", result.warnings)


if __name__ == "__main__":
    unittest.main()
