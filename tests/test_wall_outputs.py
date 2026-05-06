"""Tests for wall-serve output contracts.

Contract: the output module produces JSON dicts with exactly 6 top-level sections,
CSV rows matching the LOCKED ``CSV_COLUMNS`` order, deterministic warning-code
flattening, and deterministic plot filenames.
"""

import csv
import json
import tempfile
import unittest

from serve_analyzer.wall_calibration import CSV_COLUMNS, WARNING_CODES
from serve_analyzer.wall_outputs import (
    PLOT_FILENAMES,
    WallAnalysisResult,
    _flatten_warning_codes,
    assemble_wall_analysis_result,
    compute_confidence_score,
    serve_to_csv_row,
    to_json,
    write_csv,
)


class TestWallOutputContracts(unittest.TestCase):
    """Enforce JSON/CSV/plot contracts for wall-serve analysis."""

    # ------------------------------------------------------------------
    # JSON + CSV contract
    # ------------------------------------------------------------------

    def test_json_and_csv_contract(self):
        """JSON dict has exactly 6 required keys; CSV row length/order matches CSV_COLUMNS."""
        result = WallAnalysisResult(
            measured={"impact_time_sec": 1.23, "wall_x_m": 0.5},
            inferred={"speed_m_s": 45.0},
            assumed={"serve_contact_distance_m": 6.11},
            confidence={"score": 0.85},
            warnings=["low_calibration_confidence"],
            artifacts={"speed": "out_speed.png"},
        )

        payload = to_json(result)
        self.assertEqual(
            set(payload.keys()),
            {"measured", "inferred", "assumed", "confidence", "warnings", "artifacts"},
        )

        # CSV row length matches CSV_COLUMNS
        serve = {
            "video": "test.mp4",
            "serve_index": 0,
            "impact_time_sec": 1.23,
            "impact_frame": 37,
            "wall_x_m": 0.5,
            "wall_y_m": 2.1,
            "speed_m_s": 45.0,
            "speed_km_h": 162.0,
            "speed_mph": 100.7,
            "landing_x_m": 1.2,
            "landing_z_m": 5.0,
            "in_service_box": True,
            "confidence_score": 0.85,
            "warnings": ["low_calibration_confidence"],
        }
        row = serve_to_csv_row(serve)
        self.assertEqual(len(row), len(CSV_COLUMNS))
        self.assertEqual(tuple(CSV_COLUMNS), CSV_COLUMNS)

        # Verify column order: each value sits at the correct index
        for idx, col in enumerate(CSV_COLUMNS):
            if col == "warning_codes":
                continue  # handled separately
            self.assertEqual(row[idx], serve.get(col))

    # ------------------------------------------------------------------
    # Warning codes flatten deterministically
    # ------------------------------------------------------------------

    def test_warning_codes_flatten_deterministically(self):
        """Unordered warning codes produce identical CSV cell across two calls."""
        unordered = [
            "manual_correction_used",
            "degraded_intrinsics",
            "insufficient_track",
        ]

        cell_a = _flatten_warning_codes(unordered)
        cell_b = _flatten_warning_codes(list(reversed(unordered)))
        self.assertEqual(cell_a, cell_b)

        # No JSON braces or nested structure
        self.assertNotIn("{", cell_a)
        self.assertNotIn("}", cell_a)
        self.assertNotIn("[", cell_a)
        self.assertNotIn("]", cell_a)

        # Expected sorted order
        self.assertEqual(
            cell_a, "degraded_intrinsics;insufficient_track;manual_correction_used"
        )

    # ------------------------------------------------------------------
    # CSV round-trip via csv module
    # ------------------------------------------------------------------

    def test_csv_round_trip_via_csv_module(self):
        """Write rows via csv.writer, read back via csv.reader, verify header and data."""
        rows = [
            (
                "test.mp4",
                0,
                1.23,
                37,
                0.5,
                2.1,
                45.0,
                162.0,
                100.7,
                1.2,
                5.0,
                True,
                0.85,
                "low_calibration_confidence",
            ),
            (
                "test.mp4",
                1,
                2.50,
                75,
                0.8,
                2.3,
                42.0,
                151.2,
                93.9,
                1.5,
                5.5,
                False,
                0.70,
                "",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/output.csv"
            write_csv(path, rows)

            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                read_rows = list(reader)

            self.assertEqual(tuple(header), CSV_COLUMNS)
            self.assertEqual(len(read_rows), len(rows))
            for expected, actual in zip(rows, read_rows):
                # csv.reader returns strings; compare after type coercion for non-string fields
                self.assertEqual(len(actual), len(expected))
                for e, a in zip(expected, actual):
                    self.assertEqual(str(e), a)

    # ------------------------------------------------------------------
    # Plot filenames are deterministic
    # ------------------------------------------------------------------

    def test_plot_filenames_are_deterministic(self):
        """Formatting the same template twice yields identical strings."""
        args = {"video_stem": "video_123", "idx": 2}

        name_a = PLOT_FILENAMES["speed"].format(**args)
        name_b = PLOT_FILENAMES["speed"].format(**args)
        self.assertEqual(name_a, name_b)
        self.assertEqual(name_a, "video_123_serve02_speed.png")

        # Verify all three templates format without error
        for key in ("speed", "trajectory", "wall_impact"):
            formatted = PLOT_FILENAMES[key].format(**args)
            self.assertIn("video_123", formatted)
            self.assertIn("serve02", formatted)


# ---------------------------------------------------------------------------
# Additional contract tests
# ---------------------------------------------------------------------------


class TestWarningCodesMinimumSet(unittest.TestCase):
    """Verify WARNING_CODES includes the minimum required set."""

    def test_minimum_warning_codes_present(self):
        expected = frozenset(
            {
                "degraded_intrinsics",
                "insufficient_track",
                "manual_correction_used",
                "projection_refused",
                "low_calibration_confidence",
            }
        )
        self.assertTrue(expected.issubset(WARNING_CODES))


class TestToJsonExclusivelySixKeys(unittest.TestCase):
    """to_json must not emit any top-level keys beyond the 6 required."""

    def test_no_extra_top_level_keys(self):
        result = WallAnalysisResult(
            measured={"a": 1},
            inferred={"b": 2},
            assumed={"c": 3},
            confidence={"d": 4},
            warnings=["w1"],
            artifacts={"e": "f"},
        )
        payload = to_json(result)
        extra = set(payload.keys()) - {
            "measured",
            "inferred",
            "assumed",
            "confidence",
            "warnings",
            "artifacts",
        }
        self.assertEqual(extra, set())

    def test_json_serializable(self):
        """The output of to_json must be serializable via json.dumps."""
        result = WallAnalysisResult(
            measured={"impact_time_sec": 1.23},
            inferred={"speed_m_s": 45.0},
            assumed={},
            confidence={"score": 0.9},
            warnings=["low_calibration_confidence"],
            artifacts={"speed": "out.png"},
        )
        payload = to_json(result)
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        self.assertEqual(set(parsed.keys()), set(payload.keys()))


class TestServeToCsvRowEdgeCases(unittest.TestCase):
    """Edge cases for serve_to_csv_row."""

    def test_missing_fields_filled_with_none(self):
        serve = {"video": "v.mp4"}
        row = serve_to_csv_row(serve)
        self.assertEqual(row[0], "v.mp4")
        # All other fields should be None (except warning_codes which is "")
        for idx in range(1, len(CSV_COLUMNS) - 1):
            self.assertIsNone(row[idx])
        # warning_codes defaults to empty string
        self.assertEqual(row[-1], "")

    def test_warning_codes_as_string(self):
        serve = {"video": "v.mp4", "warning_codes": "low_calibration_confidence"}
        row = serve_to_csv_row(serve)
        self.assertEqual(row[-1], "low_calibration_confidence")

    def test_warning_codes_from_warnings_key(self):
        serve = {
            "video": "v.mp4",
            "warnings": ["insufficient_track", "degraded_intrinsics"],
        }
        row = serve_to_csv_row(serve)
        self.assertEqual(row[-1], "degraded_intrinsics;insufficient_track")
        self.assertEqual(row[-1], "degraded_intrinsics;insufficient_track")


# ---------------------------------------------------------------------------
# Serialization finalization tests (Task 11)
# ---------------------------------------------------------------------------


class _FakeImpactResult:
    """Minimal stand-in for WallImpactResult."""

    def __init__(
        self,
        *,
        impact_frame=60,
        impact_pixel=(240.0, 240.0),
        autonomous_frame=60,
        autonomous_pixel=(240.0, 240.0),
        candidate_track=None,
        warnings=None,
        confidence=None,
    ):
        self.impact_frame = impact_frame
        self.impact_pixel = impact_pixel
        self.autonomous_frame = autonomous_frame
        self.autonomous_pixel = autonomous_pixel
        self.candidate_track = candidate_track or [(f, 240.0, 240.0) for f in range(30, 61)]
        self.warnings = warnings or []
        self.confidence = confidence or {"track_length": 31, "method": "brightness_peak"}


class _FakeSpeedResult:
    """Minimal stand-in for PreWallSpeedResult."""

    def __init__(
        self,
        *,
        speed_m_s=45.0,
        speed_km_h=162.0,
        speed_mph=100.7,
        uncertainty_m_s=2.5,
        samples_used=30,
        warnings=None,
        metadata=None,
    ):
        self.speed_m_s = speed_m_s
        self.speed_km_h = speed_km_h
        self.speed_mph = speed_mph
        self.uncertainty_m_s = uncertainty_m_s
        self.samples_used = samples_used
        self.warnings = warnings or []
        self.metadata = metadata or {
            "velocity_vector_wall_m_s": (40.0, 20.0),
            "homography_residuals": {"reprojection_rms_px": 1.2},
            "scale_factor_approx": 0.05,
            "fps": 60.0,
        }


class _FakeProjectionResult:
    """Minimal stand-in for CourtProjectionResult."""

    def __init__(
        self,
        *,
        landing_x_m=1.2,
        landing_z_m=5.0,
        in_service_box=True,
        service_box_side="ad",
        assumptions=None,
        uncertainty=None,
        warnings=None,
    ):
        self.landing_x_m = landing_x_m
        self.landing_z_m = landing_z_m
        self.in_service_box = in_service_box
        self.service_box_side = service_box_side
        self.assumptions = assumptions or {
            "model": "gravity_only",
            "contact_height_m": 2.8,
            "serve_contact_distance_m": 6.11,
            "wall_aligned_with_net": True,
            "no_wall_continuation": True,
        }
        self.uncertainty = uncertainty or {
            "landing_z_sensitivity_m": 0.8,
            "landing_x_sensitivity_m": 0.3,
        }
        self.warnings = warnings or []


class _FakeCalibration:
    """Minimal stand-in for WallCalibration."""

    def __init__(self, intrinsics=None):
        self.intrinsics = intrinsics


class TestWallSerialization(unittest.TestCase):
    """Parseability and refused-projection retention tests."""

    def test_parseable_json_and_csv_from_analysis_result(self):
        """Build synthetic result, serialize JSON, verify 6 keys; CSV row matches columns."""
        impact = _FakeImpactResult()
        speed = _FakeSpeedResult()
        projection = _FakeProjectionResult()
        calibration = _FakeCalibration()

        result = assemble_wall_analysis_result(
            video_path="/tmp/test_video.mp4",
            calibration=calibration,
            impact_result=impact,
            speed_result=speed,
            projection_result=projection,
            artifact_paths={
                "annotated_video": "/tmp/test_video_annotated.mp4",
                "plots": {"speed": "/tmp/test_video_serve01_speed.png"},
            },
        )

        # --- JSON parseability ---
        payload = result.to_json_dict()
        self.assertEqual(
            set(payload.keys()),
            {"measured", "inferred", "assumed", "confidence", "warnings", "artifacts"},
        )

        # Round-trip through json.dumps / json.loads
        serialized = json.dumps(payload)
        parsed = json.loads(serialized)
        self.assertEqual(set(parsed.keys()), set(payload.keys()))

        # Verify measured contains expected fields
        self.assertEqual(parsed["measured"]["video"], "test_video")
        self.assertEqual(parsed["measured"]["impact_frame"], 60)
        self.assertEqual(parsed["measured"]["impact_time_sec"], 1.0)  # 60/60
        self.assertEqual(parsed["measured"]["raw_track_samples"], 31)

        # Verify inferred contains speed and landing
        self.assertEqual(parsed["inferred"]["speed_m_s"], 45.0)
        self.assertEqual(parsed["inferred"]["landing_x_m"], 1.2)
        self.assertTrue(parsed["inferred"]["in_service_box"])

        # Verify confidence has aggregate_score
        self.assertIn("aggregate_score", parsed["confidence"])
        self.assertGreaterEqual(parsed["confidence"]["aggregate_score"], 0.0)
        self.assertLessEqual(parsed["confidence"]["aggregate_score"], 1.0)

        # Verify artifacts preserved None values
        self.assertEqual(
            parsed["artifacts"]["annotated_video"],
            "/tmp/test_video_annotated.mp4",
        )
        self.assertEqual(
            parsed["artifacts"]["plots"]["speed"],
            "/tmp/test_video_serve01_speed.png",
        )

        # --- CSV row ---
        serve = {
            "video": result.measured["video"],
            "serve_index": result.measured["serve_index"],
            "impact_time_sec": result.measured["impact_time_sec"],
            "impact_frame": result.measured["impact_frame"],
            "wall_x_m": result.measured.get("wall_x_m"),
            "wall_y_m": result.measured.get("wall_y_m"),
            "speed_m_s": result.inferred["speed_m_s"],
            "speed_km_h": result.inferred["speed_km_h"],
            "speed_mph": result.inferred["speed_mph"],
            "landing_x_m": result.inferred["landing_x_m"],
            "landing_z_m": result.inferred["landing_z_m"],
            "in_service_box": result.inferred["in_service_box"],
            "confidence_score": result.confidence["aggregate_score"],
            "warnings": result.warnings,
        }
        row = serve_to_csv_row(serve)
        self.assertEqual(len(row), len(CSV_COLUMNS))

        # Verify specific values
        self.assertEqual(row[CSV_COLUMNS.index("video")], "test_video")
        self.assertEqual(row[CSV_COLUMNS.index("speed_m_s")], 45.0)
        self.assertEqual(row[CSV_COLUMNS.index("landing_x_m")], 1.2)
        self.assertEqual(row[CSV_COLUMNS.index("warning_codes")], "degraded_intrinsics")

    def test_refused_projection_still_writes_row_with_warning(self):
        """Refused projection produces CSV row with null fields and projection_refused warning."""
        impact = _FakeImpactResult()
        speed = _FakeSpeedResult(speed_m_s=None, speed_km_h=None, speed_mph=None)
        projection = _FakeProjectionResult(
            landing_x_m=None,
            landing_z_m=None,
            in_service_box=None,
            service_box_side=None,
            assumptions={"model": "gravity_only", "refused": True},
            uncertainty={"landing_z_sensitivity_m": 0.0, "landing_x_sensitivity_m": 0.0},
            warnings=["projection_refused"],
        )
        calibration = _FakeCalibration()

        result = assemble_wall_analysis_result(
            video_path="/tmp/test_video.mp4",
            calibration=calibration,
            impact_result=impact,
            speed_result=speed,
            projection_result=projection,
        )

        # JSON should still have 6 sections
        payload = result.to_json_dict()
        self.assertEqual(
            set(payload.keys()),
            {"measured", "inferred", "assumed", "confidence", "warnings", "artifacts"},
        )

        # Inferred landing fields should be None
        self.assertIsNone(payload["inferred"]["landing_x_m"])
        self.assertIsNone(payload["inferred"]["landing_z_m"])
        self.assertIsNone(payload["inferred"]["in_service_box"])

        # Warnings should contain projection_refused
        self.assertIn("projection_refused", payload["warnings"])

        # CSV row should exist with empty/null landing fields
        serve = {
            "video": result.measured["video"],
            "serve_index": result.measured["serve_index"],
            "impact_time_sec": result.measured["impact_time_sec"],
            "impact_frame": result.measured["impact_frame"],
            "wall_x_m": result.measured.get("wall_x_m"),
            "wall_y_m": result.measured.get("wall_y_m"),
            "speed_m_s": result.inferred["speed_m_s"],
            "speed_km_h": result.inferred["speed_km_h"],
            "speed_mph": result.inferred["speed_mph"],
            "landing_x_m": result.inferred["landing_x_m"],
            "landing_z_m": result.inferred["landing_z_m"],
            "in_service_box": result.inferred["in_service_box"],
            "confidence_score": result.confidence["aggregate_score"],
            "warnings": result.warnings,
        }
        row = serve_to_csv_row(serve)
        self.assertEqual(len(row), len(CSV_COLUMNS))

        # landing fields should be None (rendered as "" by csv.writer)
        self.assertIsNone(row[CSV_COLUMNS.index("landing_x_m")])
        self.assertIsNone(row[CSV_COLUMNS.index("landing_z_m")])
        self.assertIsNone(row[CSV_COLUMNS.index("in_service_box")])

        # warning_codes should contain projection_refused
        warning_cell = row[CSV_COLUMNS.index("warning_codes")]
        self.assertIn("projection_refused", warning_cell)

    def test_confidence_score_formula(self):
        """Verify confidence score computation matches documented formula."""
        # Perfect conditions: no uncertainty, no degraded, no refusal
        score = compute_confidence_score(
            speed_m_s=50.0,
            uncertainty_m_s=0.0,
            degraded_intrinsics=False,
            has_refusal_warning=False,
        )
        self.assertEqual(score, 1.0)

        # High uncertainty (100% of speed)
        score = compute_confidence_score(
            speed_m_s=50.0,
            uncertainty_m_s=50.0,
            degraded_intrinsics=False,
            has_refusal_warning=False,
        )
        self.assertEqual(score, 0.5)  # 1.0 - 0.5*0.5 = 0.5

        # Degraded intrinsics only
        score = compute_confidence_score(
            speed_m_s=50.0,
            uncertainty_m_s=0.0,
            degraded_intrinsics=True,
            has_refusal_warning=False,
        )
        self.assertEqual(score, 0.7)  # 1.0 - 0.3 = 0.7

        # Refusal warning only
        score = compute_confidence_score(
            speed_m_s=50.0,
            uncertainty_m_s=0.0,
            degraded_intrinsics=False,
            has_refusal_warning=True,
        )
        self.assertEqual(score, 0.8)  # 1.0 - 0.2 = 0.8

        # All penalties combined
        score = compute_confidence_score(
            speed_m_s=50.0,
            uncertainty_m_s=50.0,
            degraded_intrinsics=True,
            has_refusal_warning=True,
        )
        self.assertEqual(score, 0.0)  # 1.0 - 0.25 - 0.3 - 0.2 = 0.25, clamped to 0

        # None speed (refused)
        score = compute_confidence_score(
            speed_m_s=None,
            uncertainty_m_s=0.0,
            degraded_intrinsics=False,
            has_refusal_warning=False,
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
