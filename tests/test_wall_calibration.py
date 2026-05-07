"""Tests for wall calibration schema, validation, and constants."""

import unittest

import numpy as np

from serve_analyzer.wall_calibration import (
    COURT_LENGTH_M,
    CSV_COLUMNS,
    DOUBLES_WIDTH_M,
    NET_HEIGHT_M,
    SERVICE_BOX_DEPTH_M,
    SERVICE_BOX_WIDTH_M,
    SINGLES_WIDTH_M,
    Intrinsics,
    WARNING_CODES,
    WallCalibration,
    WallCalibrationError,
    compute_reprojection_rms,
    compute_wall_homography,
    pixel_to_wall,
    wall_to_pixel,
)


class TestWallMetadataSchema(unittest.TestCase):
    """Contract tests for the wall-serve calibration schema."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_valid_dict(self) -> dict:
        """Return a well-formed calibration dict with 4 wall points."""
        return {
            "setup": {
                "serve_contact_distance_m": 6.11,
                "camera_wall_distance_m": 1.57,
                "serve_contact_height_m": 2.80,
                "wall_reference_points": [
                    {
                        "name": "left_base",
                        "pixel": [100.0, 500.0],
                        "wall_m": [-4.0, 0.0],
                    },
                    {
                        "name": "right_base",
                        "pixel": [700.0, 500.0],
                        "wall_m": [4.0, 0.0],
                    },
                    {
                        "name": "left_top",
                        "pixel": [100.0, 100.0],
                        "wall_m": [-4.0, 3.0],
                    },
                    {
                        "name": "right_top",
                        "pixel": [700.0, 100.0],
                        "wall_m": [4.0, 3.0],
                    },
                ],
                "hook_reference": {
                    "pixel": [400.0, 150.0],
                    "height_m": 2.45,
                },
                "chair_references": [
                    {
                        "pixel": [200.0, 450.0],
                        "height_m": 1.0,
                    },
                ],
            },
        }

    # ------------------------------------------------------------------
    # Acceptance tests
    # ------------------------------------------------------------------

    def test_valid_reusable_setup(self):
        """Accept a well-formed dict with all required fields."""
        d = self._make_valid_dict()
        cal = WallCalibration.from_dict(d)

        self.assertEqual(cal.serve_contact_distance_m, 6.11)
        self.assertEqual(cal.camera_wall_distance_m, 1.57)
        self.assertEqual(cal.serve_contact_height_m, 2.80)
        self.assertEqual(len(cal.wall_reference_points), 4)

        hook = cal.hook_reference
        self.assertIsNotNone(hook)
        self.assertEqual(hook.height_m, 2.45)

        self.assertEqual(len(cal.chair_references), 1)
        self.assertEqual(cal.chair_references[0].height_m, 1.0)

    def test_missing_contact_height_rejected(self):
        """Missing serve_contact_height_m raises WallCalibrationError."""
        d = self._make_valid_dict()
        del d["setup"]["serve_contact_height_m"]

        with self.assertRaises(WallCalibrationError) as ctx:
            WallCalibration.from_dict(d)

        self.assertIn("serve_contact_height_m", str(ctx.exception))

    def test_fewer_than_four_wall_points_rejected(self):
        """Fewer than 4 wall_reference_points raises WallCalibrationError."""
        d = self._make_valid_dict()
        d["setup"]["wall_reference_points"] = d["setup"]["wall_reference_points"][:3]

        with self.assertRaises(WallCalibrationError) as ctx:
            WallCalibration.from_dict(d)

        self.assertIn("4", str(ctx.exception))

    def test_optional_intrinsics_accepted(self):
        """All valid intrinsics_source values are accepted."""
        for source in {"none", "approx_exif", "opencv_chessboard", "opencv_charuco"}:
            with self.subTest(source=source):
                d = self._make_valid_dict()
                intrinsics: dict = {"source": source}
                if source != "none":
                    intrinsics["camera_matrix"] = [
                        [1000.0, 0.0, 640.0],
                        [0.0, 1000.0, 360.0],
                        [0.0, 0.0, 1.0],
                    ]
                    intrinsics["dist_coeffs"] = [0.1, -0.2, 0.0, 0.0, 0.0]
                d["intrinsics"] = intrinsics

                cal = WallCalibration.from_dict(d)
                self.assertIsNotNone(cal.intrinsics)
                self.assertEqual(cal.intrinsics.source, source)

    def test_regulation_court_constants_exposed(self):
        """Module exposes regulation court dimensions in meters."""
        self.assertEqual(COURT_LENGTH_M, 23.77)
        self.assertEqual(SINGLES_WIDTH_M, 8.23)
        self.assertEqual(DOUBLES_WIDTH_M, 10.97)
        self.assertEqual(SERVICE_BOX_DEPTH_M, 6.40)
        self.assertEqual(SERVICE_BOX_WIDTH_M, 4.115)
        self.assertEqual(NET_HEIGHT_M, 0.914)

    def test_csv_field_names_exposed(self):
        """Module exposes CSV_COLUMNS in the exact required order."""
        expected = (
            "impact_index",
            "video",
            "serve_index",
            "impact_time_sec",
            "impact_frame",
            "wall_x_m",
            "wall_y_m",
            "speed_m_s",
            "speed_km_h",
            "speed_mph",
            "landing_x_m",
            "landing_z_m",
            "in_service_box",
            "confidence_score",
            "warning_codes",
        )
        self.assertEqual(CSV_COLUMNS, expected)

    def test_warning_codes_frozenset(self):
        """WARNING_CODES contains the minimum required codes."""
        minimum = {
            "degraded_intrinsics",
            "insufficient_track",
            "manual_correction_used",
            "projection_refused",
            "low_calibration_confidence",
        }
        self.assertTrue(minimum.issubset(WARNING_CODES))

    def test_intrinsics_none_does_not_require_matrix(self):
        """source='none' is accepted without camera_matrix or dist_coeffs."""
        d = self._make_valid_dict()
        d["intrinsics"] = {"source": "none"}

        cal = WallCalibration.from_dict(d)
        self.assertIsNotNone(cal.intrinsics)
        self.assertEqual(cal.intrinsics.source, "none")
        self.assertIsNone(cal.intrinsics.camera_matrix)
        self.assertIsNone(cal.intrinsics.dist_coeffs)

    def test_intrinsics_non_none_requires_matrix(self):
        """Non-'none' source without camera_matrix raises WallCalibrationError."""
        d = self._make_valid_dict()
        d["intrinsics"] = {"source": "approx_exif"}

        with self.assertRaises(WallCalibrationError) as ctx:
            WallCalibration.from_dict(d)

        self.assertIn("camera_matrix", str(ctx.exception))

    def test_intrinsics_non_none_requires_dist_coeffs(self):
        """Non-'none' source without dist_coeffs raises WallCalibrationError."""
        d = self._make_valid_dict()
        d["intrinsics"] = {
            "source": "opencv_chessboard",
            "camera_matrix": [
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
        }

        with self.assertRaises(WallCalibrationError) as ctx:
            WallCalibration.from_dict(d)

        self.assertIn("dist_coeffs", str(ctx.exception))

    def test_to_dict_roundtrip(self):
        """to_dict() produces a dict that from_dict() can reconstruct."""
        d = self._make_valid_dict()
        cal1 = WallCalibration.from_dict(d)
        cal2 = WallCalibration.from_dict(cal1.to_dict())

        self.assertEqual(cal1.serve_contact_distance_m, cal2.serve_contact_distance_m)
        self.assertEqual(cal1.camera_wall_distance_m, cal2.camera_wall_distance_m)
        self.assertEqual(cal1.serve_contact_height_m, cal2.serve_contact_height_m)
        self.assertEqual(
            len(cal1.wall_reference_points), len(cal2.wall_reference_points)
        )

    def test_manual_corrections_optional(self):
        """manual_corrections is accepted as an optional dict keyed by serve_index."""
        d = self._make_valid_dict()
        d["manual_corrections"] = {"0": {"impact_frame": 42}}

        cal = WallCalibration.from_dict(d)
        self.assertEqual(cal.manual_corrections, {"0": {"impact_frame": 42}})

    def test_video_override_optional(self):
        """video_override is accepted as an optional per-video dict."""
        d = self._make_valid_dict()
        d["video_override"] = {"start_frame": 100}

        cal = WallCalibration.from_dict(d)
        self.assertEqual(cal.video_override, {"start_frame": 100})


class TestWallHomography(unittest.TestCase):
    """Tests for deterministic wall-plane homography primitives."""

    def test_round_trip_synthetic_points(self):
        """Synthetic wall points round-trip pixel→wall→pixel within 1 px RMS."""
        wall_points = np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 2.45],
                [0.0, 2.45],
            ],
            dtype=np.float64,
        )
        synthetic_h = np.array(
            [
                [145.0, 12.0, 220.0],
                [-8.0, -150.0, 520.0],
                [0.006, -0.004, 1.0],
            ],
            dtype=np.float64,
        )
        image_points = wall_to_pixel(synthetic_h, wall_points)

        fitted_h, residuals = compute_wall_homography(image_points, wall_points)
        fitted_h_inv = np.linalg.inv(fitted_h)
        recovered_wall = pixel_to_wall(fitted_h_inv, image_points)
        round_trip_px = wall_to_pixel(fitted_h, recovered_wall)
        round_trip_rms = float(
            np.sqrt(np.mean(np.linalg.norm(round_trip_px - image_points, axis=1) ** 2))
        )

        self.assertLessEqual(round_trip_rms, 1.0)
        self.assertLessEqual(
            compute_reprojection_rms(image_points, wall_points, fitted_h), 1.0
        )
        np.testing.assert_allclose(recovered_wall, wall_points, atol=1e-6)
        self.assertEqual(residuals["inlier_count"], 4)
        self.assertEqual(residuals["point_count"], 4)

    def test_collinear_points_rejected(self):
        """Four collinear references raise structured calibration failure."""
        image_points = np.array(
            [[100.0, 500.0], [200.0, 500.0], [300.0, 500.0], [400.0, 500.0]],
            dtype=np.float64,
        )
        wall_points = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=np.float64,
        )

        with self.assertRaises(WallCalibrationError) as ctx:
            compute_wall_homography(image_points, wall_points)

        self.assertIn("calibration_degenerate", str(ctx.exception))

    def test_approx_intrinsics_marks_degraded(self):
        """Approximate EXIF intrinsics mark calibration residuals degraded."""
        wall_points = np.array(
            [[0.0, 0.0], [4.0, 0.0], [4.0, 2.45], [0.0, 2.45]],
            dtype=np.float64,
        )
        image_points = np.array(
            [[220.0, 520.0], [790.0, 500.0], [820.0, 160.0], [200.0, 140.0]],
            dtype=np.float64,
        )
        intrinsics = Intrinsics(
            source="approx_exif",
            camera_matrix=[
                [1000.0, 0.0, 640.0],
                [0.0, 1000.0, 360.0],
                [0.0, 0.0, 1.0],
            ],
            dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

        _, residuals = compute_wall_homography(
            image_points, wall_points, intrinsics=intrinsics
        )

        self.assertTrue(residuals["degraded_intrinsics"])


if __name__ == "__main__":
    unittest.main()
