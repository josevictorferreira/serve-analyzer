"""Post-hoc evaluation tests for timestamp parsing and target-candidate matching.

Contract: the evaluator owns all timestamp-related behaviour — parsing human
annotation files, matching detected candidates to target timestamps, and
producing the combined summary.  Functions live in serve_evaluation.py.
"""

import io
import json
import os
import tempfile
import unittest

from serve_analyzer.serve_evaluation import (
    evaluate_from_files,
    load_target_timestamps,
    main,
    match_targets_to_candidates,
    parse_timestamp_lines,
    summarize_serve_attempts,
)

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


class TestTimestampParsing(unittest.TestCase):
    def test_parse_timestamp_lines_supports_mm_ss_and_hh_mm_ss(self):
        timestamps = parse_timestamp_lines(
            [
                "# comment",
                "Serve 1 - 00:01.250",
                "Serve 2 - 01:02.500",
                "Serve 3 - 00:01:05.125",
                "",
            ]
        )

        self.assertEqual(timestamps, [1.25, 62.5, 65.125])

    def test_load_target_timestamps_raises_on_invalid_line(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("Serve 1 - not-a-time\n")
            path = handle.name

        try:
            with self.assertRaises(ValueError):
                load_target_timestamps(path)
        finally:
            os.remove(path)

    def test_parse_timestamp_lines_supports_written_seconds_format(self):
        timestamps = parse_timestamp_lines(
            [
                "- 1st serve - 13 seconds",
                "- 2nd serve - 19 seconds",
                "- 8th serve - 1 minute and 2 seconds",
            ]
        )

        self.assertEqual(timestamps, [13.0, 19.0, 62.0])

    def test_parse_empty_lines_after_comments_is_allowed(self):
        """Lines that are comments-only should not raise."""
        timestamps = parse_timestamp_lines(
            [
                "# only comment",
                "Serve 1 - 00:05.000",
            ]
        )
        self.assertEqual(timestamps, [5.0])

    def test_load_target_timestamps_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_target_timestamps("/nonexistent/path/timestamps.txt")

    def test_parse_timestamp_lines_raises_on_no_timestamps(self):
        with self.assertRaises(ValueError):
            parse_timestamp_lines(["# only comments", "# nothing else"])


# ---------------------------------------------------------------------------
# Target-candidate matching
# ---------------------------------------------------------------------------


class TestMatchTargetsToCandidates(unittest.TestCase):
    """Direct tests on the greedy ordered matching algorithm."""

    def test_exact_match(self):
        matches = match_targets_to_candidates([10.0], [10.0], tolerance_sec=0.5)
        self.assertEqual(matches, [0])

    def test_within_tolerance(self):
        matches = match_targets_to_candidates([10.0], [10.3], tolerance_sec=0.5)
        self.assertEqual(matches, [0])

    def test_outside_tolerance_is_unmatched(self):
        matches = match_targets_to_candidates([10.0], [10.6], tolerance_sec=0.5)
        self.assertEqual(matches, [None])

    def test_multiple_targets_ordered(self):
        matches = match_targets_to_candidates(
            [10.0, 20.0], [10.1, 20.2], tolerance_sec=0.5
        )
        self.assertEqual(matches, [0, 1])

    def test_candidate_consumed_once(self):
        matches = match_targets_to_candidates([10.0, 10.5], [10.2], tolerance_sec=0.5)
        self.assertEqual(matches, [0, None])

    def test_no_candidates_all_unmatched(self):
        matches = match_targets_to_candidates([10.0, 20.0], [], tolerance_sec=1.0)
        self.assertEqual(matches, [None, None])

    def test_negative_tolerance_raises(self):
        with self.assertRaises(ValueError):
            match_targets_to_candidates([10.0], [10.0], tolerance_sec=-1.0)

    def test_zero_tolerance_exact_only(self):
        matches = match_targets_to_candidates([10.0], [10.0], tolerance_sec=0.0)
        self.assertEqual(matches, [0])

    def test_zero_tolerance_no_fuzzy(self):
        matches = match_targets_to_candidates([10.0], [10.001], tolerance_sec=0.0)
        self.assertEqual(matches, [None])


# ---------------------------------------------------------------------------
# Summarize serve attempts
# ---------------------------------------------------------------------------


class TestSummarizeServeAttempts(unittest.TestCase):
    def test_matches_candidates_with_tolerance_and_preserves_velocity_fields(self):
        result = summarize_serve_attempts(
            candidates=[
                {
                    "contact_time_sec": 10.2,
                    "post_contact_max_kmh": 181.4,
                    "post_contact_mean_kmh": 170.0,
                    "post_contact_max_mps": 50.4,
                    "post_contact_mean_mps": 47.2,
                },
                {
                    "contact_time_sec": 19.95,
                    "post_contact_max_kmh": 176.8,
                    "post_contact_mean_kmh": 165.3,
                },
            ],
            target_timestamps=[10.0, 19.3, 30.0],
            tolerance_sec=0.75,
        )

        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["unmatched_candidate_count"], 0)
        self.assertEqual(len(result["attempts"]), 3)

        first = result["attempts"][0]
        self.assertTrue(first["matched"])
        self.assertAlmostEqual(first["target_time_sec"], 10.0)
        self.assertAlmostEqual(first["detected_time_sec"], 10.2)
        self.assertAlmostEqual(first["delta_sec"], 0.2)
        self.assertAlmostEqual(first["post_contact_max_kmh"], 181.4)
        self.assertAlmostEqual(first["post_contact_mean_mps"], 47.2)

        second = result["attempts"][1]
        self.assertTrue(second["matched"])
        self.assertAlmostEqual(second["delta_sec"], 0.65)
        self.assertAlmostEqual(second["post_contact_mean_kmh"], 165.3)

        third = result["attempts"][2]
        self.assertFalse(third["matched"])
        self.assertIsNone(third["detected_time_sec"])
        self.assertIsNone(third["delta_sec"])
        self.assertIsNone(third["post_contact_max_kmh"])

    def test_duplicate_nearby_candidates_only_match_once(self):
        result = summarize_serve_attempts(
            candidates=[
                {"contact_time_sec": 9.95, "post_contact_max_kmh": 175.0},
                {"contact_time_sec": 10.2, "post_contact_max_kmh": 169.0},
                {"contact_time_sec": 20.1, "post_contact_max_kmh": 180.0},
            ],
            target_timestamps=[10.0, 20.0],
            tolerance_sec=0.3,
        )

        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["unmatched_candidate_count"], 1)
        self.assertEqual(result["attempts"][0]["candidate_index"], 0)
        self.assertAlmostEqual(result["attempts"][0]["detected_time_sec"], 9.95)
        self.assertEqual(len(result["unmatched_candidates"]), 1)
        self.assertAlmostEqual(
            result["unmatched_candidates"][0]["contact_time_sec"], 10.2
        )

    def test_no_candidates_all_unmatched(self):
        result = summarize_serve_attempts(
            candidates=[], target_timestamps=[10.0, 20.0], tolerance_sec=1.0
        )
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(result["candidate_count"], 0)
        self.assertFalse(result["attempts"][0]["matched"])

    def test_attempt_record_has_expected_keys(self):
        result = summarize_serve_attempts(
            candidates=[{"contact_time_sec": 5.0, "post_contact_max_kmh": 150.0}],
            target_timestamps=[5.0],
            tolerance_sec=0.5,
        )
        attempt = result["attempts"][0]
        expected_keys = sorted(
            [
                "candidate_index",
                "delta_sec",
                "detected_time_sec",
                "matched",
                "post_contact_max_kmh",
                "post_contact_max_mps",
                "post_contact_mean_kmh",
                "post_contact_mean_mps",
                "serve_number",
                "target_time_sec",
            ]
        )
        self.assertEqual(sorted(attempt.keys()), expected_keys)


# ---------------------------------------------------------------------------
# Evaluator CLI (timestamps required)
# ---------------------------------------------------------------------------


class TestEvaluatorCLI(unittest.TestCase):
    """Evaluator CLI reads detection JSON + timestamps, produces match summary."""

    def test_main_outputs_json_summary_shape(self):
        candidates = [
            {
                "contact_time_sec": 10.1,
                "post_contact_max_kmh": 182.0,
                "post_contact_mean_kmh": 171.5,
                "post_contact_max_mps": 50.6,
                "post_contact_mean_mps": 47.6,
            },
            {
                "contact_time_sec": 20.2,
                "post_contact_max_kmh": 178.0,
                "post_contact_mean_kmh": 168.0,
                "post_contact_max_mps": 49.4,
                "post_contact_mean_mps": 46.7,
            },
        ]
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json", encoding="utf-8"
        ) as dj:
            json.dump(candidates, dj)
            detection_json_path = dj.name

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("Serve 1 - 00:10.000\nServe 2 - 00:20.000\n")
            timestamps_path = handle.name

        stdout = io.StringIO()
        try:
            import sys

            saved_stdout = sys.stdout
            sys.stdout = stdout
            try:
                exit_code = main(
                    [
                        "--detection-json",
                        detection_json_path,
                        "--timestamps-file",
                        timestamps_path,
                        "--tolerance-sec",
                        "0.5",
                    ]
                )
            finally:
                sys.stdout = saved_stdout

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertIn("detection_json", payload)
            self.assertEqual(payload["matched_count"], 2)
            self.assertEqual(payload["target_count"], 2)
            self.assertEqual(payload["candidate_count"], 2)
            self.assertEqual(len(payload["attempts"]), 2)
            self.assertEqual(
                sorted(payload["attempts"][0].keys()),
                [
                    "candidate_index",
                    "delta_sec",
                    "detected_time_sec",
                    "matched",
                    "post_contact_max_kmh",
                    "post_contact_max_mps",
                    "post_contact_mean_kmh",
                    "post_contact_mean_mps",
                    "serve_number",
                    "target_time_sec",
                ],
            )
        finally:
            os.remove(detection_json_path)
            os.remove(timestamps_path)

    def test_main_with_output_file(self):
        candidates = [
            {
                "contact_time_sec": 5.0,
                "post_contact_max_kmh": 175.0,
                "post_contact_mean_kmh": 165.0,
                "post_contact_max_mps": 48.6,
                "post_contact_mean_mps": 45.8,
            },
        ]
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json", encoding="utf-8"
        ) as dj:
            json.dump(candidates, dj)
            detection_json_path = dj.name

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("Serve 1 - 00:05.000\n")
            timestamps_path = handle.name

        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json"
        ) as out_handle:
            output_path = out_handle.name

        try:
            exit_code = main(
                [
                    "--detection-json",
                    detection_json_path,
                    "--timestamps-file",
                    timestamps_path,
                    "--output",
                    output_path,
                ]
            )

            self.assertEqual(exit_code, 0)
            with open(output_path, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["matched_count"], 1)
        finally:
            os.remove(detection_json_path)
            os.remove(timestamps_path)
            os.remove(output_path)


class TestEvaluatorSourceSelection(unittest.TestCase):
    def test_evaluate_from_files_can_use_selected_serves(self):
        payload = {
            "candidates": [
                {"contact_time_sec": 1.0, "post_contact_max_kmh": 100.0},
            ],
            "selected_serves": [
                {"contact_time_sec": 5.0, "post_contact_max_kmh": 120.0},
            ],
        }
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json", encoding="utf-8"
        ) as detection_file:
            json.dump(payload, detection_file)
            detection_json_path = detection_file.name

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("Serve 1 - 00:05.000\n")
            timestamps_path = handle.name

        try:
            result = evaluate_from_files(
                detection_json_path,
                timestamps_path,
                tolerance_sec=0.5,
                source="selected_serves",
            )

            self.assertEqual(result["source"], "selected_serves")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["matched_count"], 1)
            self.assertAlmostEqual(result["attempts"][0]["detected_time_sec"], 5.0)
        finally:
            os.remove(detection_json_path)
            os.remove(timestamps_path)

    def test_empty_selected_serves_does_not_fall_back_to_attempts(self):
        payload = {
            "selected_serves": [],
            "attempts": [
                {"contact_time_sec": 5.0, "post_contact_max_kmh": 120.0},
            ],
        }
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json", encoding="utf-8"
        ) as detection_file:
            json.dump(payload, detection_file)
            detection_json_path = detection_file.name

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("Serve 1 - 00:05.000\n")
            timestamps_path = handle.name

        try:
            result = evaluate_from_files(
                detection_json_path,
                timestamps_path,
                tolerance_sec=0.5,
                source="selected_serves",
            )

            self.assertEqual(result["source"], "selected_serves")
            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["matched_count"], 0)
        finally:
            os.remove(detection_json_path)
            os.remove(timestamps_path)

    def test_evaluate_from_files_rejects_unknown_source(self):
        with self.assertRaises(ValueError):
            evaluate_from_files("missing.json", "missing.txt", 3.0, source="bad")


if __name__ == "__main__":
    unittest.main()
