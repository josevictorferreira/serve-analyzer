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


if __name__ == "__main__":
    unittest.main()
