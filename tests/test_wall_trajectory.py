"""Tests for pre-wall speed estimation with uncertainty.

Builds synthetic ``WallImpactResult`` fixtures and asserts that
:func:`estimate_pre_wall_speed` returns correct speeds, refuses when
insufficient track, and inflates uncertainty for degraded intrinsics.
"""

from __future__ import annotations

import unittest
from typing import List, Tuple


from serve_analyzer.wall_calibration import (
    WallCalibration,
    WallReferencePoint,
    Intrinsics,
)
from serve_analyzer.wall_serve import (
    WallImpactResult,
    PreWallSpeedResult,
    estimate_pre_wall_speed,
)


def _make_calibration(
    wall_x_px: float = 240.0,
    height_px: int = 240,
    intrinsics: Intrinsics | None = None,
) -> WallCalibration:
    """Build a minimal WallCalibration with a known pixel→meter scale.

    Four reference points arranged in a rectangle whose right edge sits at
    *wall_x_px* and spans the frame height.  The wall frame is chosen so
    that 1 pixel ≈ 0.01 m on the wall (80 px ↔ 0.8 m horizontally,
    220 px ↔ 2.2 m vertically).
    """
    return WallCalibration(
        serve_contact_height_m=2.8,
        wall_reference_points=[
            WallReferencePoint(
                name="tl", pixel=(wall_x_px - 80, 20), wall_m=(0.0, 2.2)
            ),
            WallReferencePoint(name="tr", pixel=(wall_x_px, 20), wall_m=(0.8, 2.2)),
            WallReferencePoint(
                name="bl",
                pixel=(wall_x_px - 80, height_px - 20),
                wall_m=(0.0, 0.0),
            ),
            WallReferencePoint(
                name="br",
                pixel=(wall_x_px, height_px - 20),
                wall_m=(0.8, 0.0),
            ),
        ],
        intrinsics=intrinsics,
    )


def _make_synthetic_track(
    *,
    fps: float = 60.0,
    impact_frame: int = 60,
    ball_speed_px_per_frame: float = 3.0,
    wall_x_px: float = 240.0,
    height_px: int = 240,
    total_frames: int = 90,
) -> WallImpactResult:
    """Build a WallImpactResult with a uniformly-moving synthetic track.

    The ball moves left-to-right horizontally at constant speed.  Positions
    are generated for every frame from 0 up to and including *impact_frame*.
    """
    start_x = wall_x_px - ball_speed_px_per_frame * impact_frame
    y = height_px // 2

    candidate_track: List[Tuple[int, float, float]] = []
    for t in range(total_frames):
        x = start_x + ball_speed_px_per_frame * t
        if x > wall_x_px:
            break
        candidate_track.append((t, float(x), float(y)))

    # Impact pixel is the last point (exactly at wall on impact_frame).
    impact_pixel = (float(wall_x_px), float(y))

    return WallImpactResult(
        impact_frame=impact_frame,
        impact_pixel=impact_pixel,
        autonomous_frame=impact_frame,
        autonomous_pixel=impact_pixel,
        candidate_track=candidate_track,
        warnings=[],
        confidence={"track_length": len(candidate_track)},
    )


class TestPreWallSpeed(unittest.TestCase):
    """Core pre-wall speed estimation contract tests."""

    def test_estimates_known_synthetic_speed(self):
        """Synthetic ball at 3 px/frame, 60 fps, scale ≈ 0.01 m/px.

        Expected speed ≈ 3 * 60 * 0.01 = 1.8 m/s.
        Assert speed_m_s within 10 %, km_h and mph match conversions.
        """
        calibration = _make_calibration()
        impact = _make_synthetic_track(
            fps=60.0,
            impact_frame=60,
            ball_speed_px_per_frame=3.0,
            wall_x_px=240.0,
            height_px=240,
        )

        result = estimate_pre_wall_speed(impact, calibration, fps=60.0, min_samples=4)

        self.assertIsInstance(result, PreWallSpeedResult)
        self.assertIsNotNone(result.speed_m_s)
        self.assertIsNotNone(result.speed_km_h)
        self.assertIsNotNone(result.speed_mph)

        expected_m_s = 3.0 * 60.0 * 0.01  # 1.8 m/s
        self.assertAlmostEqual(
            result.speed_m_s,
            expected_m_s,
            delta=expected_m_s * 0.10,
            msg=f"speed_m_s {result.speed_m_s} not within 10% of {expected_m_s}",
        )

        self.assertAlmostEqual(
            result.speed_km_h,
            result.speed_m_s * 3.6,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            result.speed_mph,
            result.speed_m_s * 2.2369362920544,
            delta=1e-6,
        )

        self.assertGreaterEqual(result.samples_used, 4)
        self.assertGreaterEqual(result.uncertainty_m_s, 0.0)
        self.assertIn("velocity_vector_wall_m_s", result.metadata)

    def test_refuses_speed_with_too_few_points(self):
        """candidate_track length < 4 → all speed fields None,
        'insufficient_track' in warnings, no exception.
        """
        calibration = _make_calibration()
        # Only 2 pre-impact points.
        impact = WallImpactResult(
            impact_frame=2,
            impact_pixel=(240.0, 120.0),
            autonomous_frame=2,
            autonomous_pixel=(240.0, 120.0),
            candidate_track=[(0, 234.0, 120.0), (1, 237.0, 120.0)],
            warnings=[],
            confidence={},
        )

        result = estimate_pre_wall_speed(impact, calibration, fps=60.0, min_samples=4)

        self.assertIsNone(result.speed_m_s)
        self.assertIsNone(result.speed_km_h)
        self.assertIsNone(result.speed_mph)
        self.assertIn("insufficient_track", result.warnings)
        self.assertEqual(result.samples_used, 2)

    def test_degraded_intrinsics_inflates_uncertainty(self):
        """Same fixture with intrinsics source 'approx_exif' → larger
        uncertainty_m_s than baseline AND 'degraded_intrinsics' in warnings.
        """
        baseline_cal = _make_calibration()
        degraded_cal = _make_calibration(
            intrinsics=Intrinsics(
                source="approx_exif",
                camera_matrix=[
                    [800.0, 0.0, 320.0],
                    [0.0, 800.0, 120.0],
                    [0.0, 0.0, 1.0],
                ],
                dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
            )
        )

        impact = _make_synthetic_track(
            fps=60.0,
            impact_frame=60,
            ball_speed_px_per_frame=3.0,
        )

        baseline = estimate_pre_wall_speed(
            impact, baseline_cal, fps=60.0, min_samples=4
        )
        degraded = estimate_pre_wall_speed(
            impact, degraded_cal, fps=60.0, min_samples=4
        )

        self.assertIn("degraded_intrinsics", degraded.warnings)
        self.assertNotIn("degraded_intrinsics", baseline.warnings)

        self.assertGreater(
            degraded.uncertainty_m_s,
            baseline.uncertainty_m_s,
            "degraded intrinsics should inflate uncertainty",
        )


if __name__ == "__main__":
    unittest.main()
