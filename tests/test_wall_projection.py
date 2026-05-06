"""Tests for regulation-court landing projection (gravity-only, no wall).

Builds synthetic ``PreWallSpeedResult`` fixtures and asserts that
:func:`project_to_court` returns correct landing coordinates, refuses
when speed is unavailable, and includes sensitivity analysis.
"""

from __future__ import annotations

import math
import unittest

from serve_analyzer.wall_calibration import (
    WallCalibration,
)
from serve_analyzer.wall_serve import (
    CourtProjectionResult,
    PreWallSpeedResult,
    project_to_court,
)


def _make_speed_result(
    speed_m_s: float | None,
    vx: float = 0.0,
    vy: float = 0.0,
    warnings: list[str] | None = None,
) -> PreWallSpeedResult:
    """Build a minimal PreWallSpeedResult for projection testing."""
    return PreWallSpeedResult(
        speed_m_s=speed_m_s,
        speed_km_h=speed_m_s * 3.6 if speed_m_s is not None else None,
        speed_mph=speed_m_s * 2.2369362920544 if speed_m_s is not None else None,
        uncertainty_m_s=0.5 if speed_m_s is not None else 0.0,
        samples_used=8 if speed_m_s is not None else 0,
        warnings=warnings or [],
        metadata={"velocity_vector_wall_m_s": (vx, vy)},
    )


def _make_calibration(
    contact_height_m: float = 2.5,
    contact_distance_m: float = 6.11,
) -> WallCalibration:
    """Build a minimal WallCalibration for projection testing."""
    return WallCalibration(
        serve_contact_height_m=contact_height_m,
        serve_contact_distance_m=contact_distance_m,
    )


class TestCourtProjection(unittest.TestCase):
    """Core court-projection contract tests."""

    def test_projects_known_projectile_landing(self):
        """Hand-set projectile: h0=2.5m, vz=-5 m/s (toward wall), vy=0, vx=0.

        Gravity-only closed-form:
            t* = sqrt(2 * g * h0) / g  (when vy0 = 0)
            z_rel = vz0 * t*
            landing_z = z_contact + z_rel

        With g=9.81, h0=2.5, vz0=-5, z_contact=6.11:
            t* = sqrt(49.05) / 9.81 ≈ 0.7139 s
            z_rel ≈ -3.569
            landing_z ≈ 2.540  → inside service box [0, 6.40]

        Assert landing_z within 0.5 m of closed-form, in_service_box=True.
        """
        g = 9.81
        h0 = 2.5
        vz = 5.0  # toward wall; speed = 5 m/s with vx=vy=0
        z_contact = 6.11

        # Closed-form expected values
        t_flight = math.sqrt(2 * g * h0) / g
        expected_z_rel = -vz * t_flight
        expected_landing_z = z_contact + expected_z_rel

        speed_result = _make_speed_result(speed_m_s=vz, vx=0.0, vy=0.0)
        calibration = _make_calibration(
            contact_height_m=h0,
            contact_distance_m=z_contact,
        )

        result = project_to_court(speed_result, calibration)

        self.assertIsInstance(result, CourtProjectionResult)
        self.assertIsNotNone(result.landing_z_m)
        self.assertIsNotNone(result.landing_x_m)
        self.assertAlmostEqual(
            result.landing_z_m,
            expected_landing_z,
            delta=0.5,
            msg=(
                f"landing_z_m {result.landing_z_m:.3f} not within 0.5 m "
                f"of closed-form {expected_landing_z:.3f}"
            ),
        )
        self.assertAlmostEqual(result.landing_x_m, 0.0, delta=0.01)
        self.assertTrue(result.in_service_box)
        self.assertEqual(result.service_box_side, "deuce")

        # Assumptions check
        self.assertEqual(result.assumptions["model"], "gravity_only")
        self.assertTrue(result.assumptions["no_wall_continuation"])
        self.assertTrue(result.assumptions["wall_aligned_with_net"])
        self.assertNotIn("projection_refused", result.warnings)

    def test_refuses_projection_without_speed(self):
        """PreWallSpeedResult with speed_m_s=None and 'insufficient_track'
        warning → all landing fields None, 'projection_refused' in warnings,
        no exception.
        """
        speed_result = _make_speed_result(
            speed_m_s=None,
            warnings=["insufficient_track"],
        )
        calibration = _make_calibration()

        result = project_to_court(speed_result, calibration)

        self.assertIsNone(result.landing_x_m)
        self.assertIsNone(result.landing_z_m)
        self.assertIsNone(result.in_service_box)
        self.assertIsNone(result.service_box_side)
        self.assertIn("projection_refused", result.warnings)

    def test_uncertainty_sensitivity_present(self):
        """Result includes landing_z_sensitivity_m and landing_x_sensitivity_m
        as positive floats when projection succeeds.
        """
        speed_result = _make_speed_result(speed_m_s=30.0, vx=1.0, vy=-2.0)
        calibration = _make_calibration(contact_height_m=2.8)

        result = project_to_court(speed_result, calibration)

        self.assertIn("landing_z_sensitivity_m", result.uncertainty)
        self.assertIn("landing_x_sensitivity_m", result.uncertainty)
        self.assertGreater(result.uncertainty["landing_z_sensitivity_m"], 0.0)
        self.assertGreater(result.uncertainty["landing_x_sensitivity_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
