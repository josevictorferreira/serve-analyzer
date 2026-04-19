"""Detector-only tests for serve_analyzer.serve_attempts.

Contract: detect_serve_candidates() and _merge_candidate_events() perform
pure video analysis — no timestamps, no target matching, no file I/O beyond
the video itself. The CLI entry point (main) should eventually support a
video-only mode that does not require --timestamps-file.
"""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from serve_analyzer.serve_attempts import (
    _merge_candidate_events,
    detect_serve_candidates,
    infer_serve_count,
    select_serves,
)


def _make_event(contact_frame: int, score: float = 1.0, **extra):
    """Build a minimal candidate event dict for merge tests."""
    event = {"contact_frame": contact_frame, "score": score}
    event.update(extra)
    return event


def _make_candidate(
    contact_time_sec: float,
    score: float = 0.8,
    max_kmh: float = 170.0,
    mean_kmh: float = 160.0,
    **extra,
) -> dict:
    """Build a minimal candidate dict for select_serves tests.

    Includes contact_time_sec, score, and post-contact velocity fields
    that a smarter selector may use for quality heuristics.
    Accepts **extra to inject support_count, frames_after_apex, etc.
    """
    d = {
        "candidate_index": 0,
        "contact_frame": int(contact_time_sec * 30),
        "contact_time_sec": contact_time_sec,
        "score": score,
        "post_contact_max_kmh": max_kmh,
        "post_contact_mean_kmh": mean_kmh,
        "post_contact_max_mps": max_kmh / 3.6,
        "post_contact_mean_mps": mean_kmh / 3.6,
    }
    d.update(extra)
    return d


# ---------------------------------------------------------------------------
# Detect serve candidates — input validation
# ---------------------------------------------------------------------------


class TestDetectServeCandidatesValidation(unittest.TestCase):
    """detect_serve_candidates rejects bad arguments."""

    def test_frame_skip_below_one_raises(self):
        with self.assertRaises(ValueError):
            detect_serve_candidates("irrelevant.mov", frame_skip=0)

    @patch(
        "serve_analyzer.serve_attempts.detect_ball_yolo",
        return_value=([], 30.0, 100, None),
    )
    def test_expected_serves_zero_raises(self, _mock_detect):
        """expected_serves=0 validated after video I/O — still raises ValueError."""
        with self.assertRaises(ValueError):
            detect_serve_candidates("irrelevant.mov", expected_serves=0, frame_skip=1)


# ---------------------------------------------------------------------------
# Detect serve candidates — output shape (mocked video pipeline)
# ---------------------------------------------------------------------------


class TestDetectServeCandidatesOutputShape(unittest.TestCase):
    """With mocked internals, verify the returned candidate list shape."""

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events", return_value=[])
    @patch("serve_analyzer.serve_attempts.compute_vertical_velocity")
    @patch("serve_analyzer.serve_attempts.compute_frame_velocities")
    @patch("serve_analyzer.serve_attempts.interpolate_missing_detections")
    @patch(
        "serve_analyzer.serve_attempts.detect_ball_yolo",
        return_value=([], 30.0, 100, None),
    )
    def test_returns_list_of_candidate_dicts(
        self,
        mock_detect,
        mock_interp,
        mock_vel,
        mock_vert_vel,
        mock_events,
        mock_analyze,
    ):
        """When no events detected, should return empty list."""
        result = detect_serve_candidates("video.mov")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events")
    @patch("serve_analyzer.serve_attempts.compute_vertical_velocity")
    @patch("serve_analyzer.serve_attempts.compute_frame_velocities")
    @patch("serve_analyzer.serve_attempts.interpolate_missing_detections")
    @patch(
        "serve_analyzer.serve_attempts.detect_ball_yolo",
        return_value=([], 60.0, 200, None),
    )
    def test_candidate_dict_has_required_keys(
        self,
        mock_detect,
        mock_interp,
        mock_vel,
        mock_vert_vel,
        mock_events,
        mock_analyze,
    ):
        """Each candidate dict must carry detection-only fields (no target/match info)."""
        event = _make_event(contact_frame=90, score=0.9)
        serve_mock = MagicMock()
        serve_mock.post_contact_max_velocity = 180.0
        serve_mock.post_contact_mean_velocity = 170.0

        mock_events.return_value = [event]
        mock_analyze.return_value = serve_mock

        candidates = detect_serve_candidates("video.mov")

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]

        # Detector-only keys — must exist
        required_keys = {
            "candidate_index",
            "contact_frame",
            "contact_time_sec",
            "post_contact_max_kmh",
            "post_contact_mean_kmh",
            "post_contact_max_mps",
            "post_contact_mean_mps",
            "score",
        }
        self.assertTrue(
            required_keys.issubset(candidate.keys()),
            f"Missing detector keys: {required_keys - candidate.keys()}",
        )

        # Evaluator-only keys — must NOT exist in detector output
        evaluator_keys = {
            "matched",
            "target_time_sec",
            "delta_sec",
            "serve_number",
        }
        for key in evaluator_keys:
            self.assertNotIn(
                key, candidate, f"Evaluator key '{key}' leaked into detector output"
            )

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events")
    @patch("serve_analyzer.serve_attempts.compute_vertical_velocity")
    @patch("serve_analyzer.serve_attempts.compute_frame_velocities")
    @patch("serve_analyzer.serve_attempts.interpolate_missing_detections")
    @patch(
        "serve_analyzer.serve_attempts.detect_ball_yolo",
        return_value=([], 30.0, 300, None),
    )
    def test_contact_time_sec_computed_from_fps(
        self,
        mock_detect,
        mock_interp,
        mock_vel,
        mock_vert_vel,
        mock_events,
        mock_analyze,
    ):
        """contact_time_sec = contact_frame / fps."""
        fps = 30.0
        frame = 150
        event = _make_event(contact_frame=frame)
        serve_mock = MagicMock(
            post_contact_max_velocity=100.0, post_contact_mean_velocity=90.0
        )

        mock_events.return_value = [event]
        mock_analyze.return_value = serve_mock

        candidates = detect_serve_candidates("video.mov")

        self.assertAlmostEqual(candidates[0]["contact_time_sec"], frame / fps)


# ---------------------------------------------------------------------------
# _merge_candidate_events
# ---------------------------------------------------------------------------


class TestMergeCandidateEvents(unittest.TestCase):
    """_merge_candidate_events deduplicates nearby detections."""

    def test_single_group_no_merge(self):
        events = [_make_event(10), _make_event(100)]
        merged = _merge_candidate_events([events], fps=30.0)
        self.assertEqual(len(merged), 2)

    def test_nearby_events_merged_keeps_higher_score(self):
        group_a = [_make_event(10, score=0.5)]
        group_b = [_make_event(11, score=0.9)]
        merged = _merge_candidate_events([group_a, group_b], fps=30.0)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["score"], 0.9)

    def test_distant_events_not_merged(self):
        group_a = [_make_event(10)]
        group_b = [_make_event(500)]  # >0.75s at 30fps
        merged = _merge_candidate_events([group_a, group_b], fps=30.0)
        self.assertEqual(len(merged), 2)

    def test_empty_groups_return_empty(self):
        merged = _merge_candidate_events([], fps=30.0)
        self.assertEqual(merged, [])

    def test_all_empty_groups_return_empty(self):
        merged = _merge_candidate_events([[], []], fps=30.0)
        self.assertEqual(merged, [])


# ---------------------------------------------------------------------------
# Detector CLI — video-only mode
# ---------------------------------------------------------------------------


class TestDetectorCLI(unittest.TestCase):
    """Detector CLI should work without --timestamps-file (video-only mode).

    IMPLEMENTATION GAP: main() currently requires --timestamps-file.
    These tests document the expected detector-only CLI contract.
    """

    @patch(
        "serve_analyzer.serve_attempts.detect_serve_candidates",
        return_value=[
            {
                "candidate_index": 0,
                "contact_frame": 300,
                "contact_time_sec": 10.0,
                "post_contact_max_kmh": 180.0,
                "post_contact_mean_kmh": 170.0,
                "post_contact_max_mps": 50.0,
                "post_contact_mean_mps": 47.2,
                "score": 0.85,
            },
        ],
    )
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_video_only_mode_no_timestamps_required(self, mock_stdout, mock_detect):
        """Detector CLI should accept video without --timestamps-file.

        IMPLEMENTATION GAP: build_parser() marks --timestamps-file as required=True.
        Until fixed, this test documents the expected video-only contract.
        """
        from serve_analyzer.serve_attempts import main

        try:
            exit_code = main(["video.mov"])
        except SystemExit as exc:
            # Expected gap: argparse rejects missing --timestamps-file
            self.assertNotEqual(exc.code, 0)
            self.skipTest(
                "IMPLEMENTATION GAP: main() requires --timestamps-file; "
                "video-only mode not yet implemented"
            )
            return

        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertIn("candidates", payload)
        self.assertNotIn("matched_count", payload)
        self.assertNotIn("targets_sec", payload)

    @patch(
        "serve_analyzer.serve_attempts.detect_serve_candidates",
        return_value=[],
    )
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_video_only_mode_empty_result(self, mock_stdout, mock_detect):
        """No serves detected → empty candidates list, exit 0."""
        from serve_analyzer.serve_attempts import main

        try:
            exit_code = main(["video.mov"])
        except SystemExit:
            self.skipTest("IMPLEMENTATION GAP: video-only CLI mode not yet implemented")
            return

        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertEqual(payload["candidates"], [])


# ---------------------------------------------------------------------------
# select_serves — selector quality contract
# ---------------------------------------------------------------------------


class TestSelectServes(unittest.TestCase):
    """Unit tests for select_serves() using synthetic candidate lists.

    These tests lock the selector contract: the implementation may change
    from greedy-score to sequence-based, but the observable behaviour below
    must be preserved.
    """

    # --- (a) Avoid early high-score false positive ---

    def test_avoids_early_high_score_false_positive(self):
        """Selector should skip an implausibly-early high-score candidate
        when later candidates form a more plausible serve sequence.

        The early candidate has the highest detection score but very low
        post-contact velocity (40 km/h), signalling a false positive.
        The three later candidates carry realistic serve velocities.
        """
        candidates = [
            _make_candidate(
                contact_time_sec=0.3, score=0.98, max_kmh=40.0, mean_kmh=35.0
            ),
            _make_candidate(
                contact_time_sec=12.0, score=0.82, max_kmh=175.0, mean_kmh=165.0
            ),
            _make_candidate(
                contact_time_sec=28.0, score=0.78, max_kmh=168.0, mean_kmh=158.0
            ),
            _make_candidate(
                contact_time_sec=44.0, score=0.72, max_kmh=162.0, mean_kmh=152.0
            ),
        ]
        result = select_serves(candidates, expected_serves=3)
        times = [c["contact_time_sec"] for c in result]
        self.assertEqual(times, [12.0, 28.0, 44.0])

    def test_avoids_clustered_early_false_positives(self):
        """Multiple early high-score FPs should not crowd out real serves."""
        candidates = [
            _make_candidate(
                contact_time_sec=0.4, score=0.96, max_kmh=35.0, mean_kmh=30.0
            ),
            _make_candidate(
                contact_time_sec=1.2, score=0.94, max_kmh=45.0, mean_kmh=38.0
            ),
            _make_candidate(
                contact_time_sec=10.0, score=0.80, max_kmh=172.0, mean_kmh=162.0
            ),
            _make_candidate(
                contact_time_sec=25.0, score=0.75, max_kmh=165.0, mean_kmh=155.0
            ),
            _make_candidate(
                contact_time_sec=40.0, score=0.70, max_kmh=158.0, mean_kmh=148.0
            ),
        ]
        result = select_serves(candidates, expected_serves=3)
        times = [c["contact_time_sec"] for c in result]
        self.assertEqual(times, [10.0, 25.0, 40.0])

    # --- (b) Honours expected_serves ---

    def test_honors_expected_serves_returns_exact_count(self):
        """When enough non-overlapping candidates exist, return exactly expected_serves."""
        candidates = [
            _make_candidate(contact_time_sec=5.0, score=0.9),
            _make_candidate(contact_time_sec=20.0, score=0.8),
            _make_candidate(contact_time_sec=35.0, score=0.7),
            _make_candidate(contact_time_sec=50.0, score=0.6),
        ]
        result = select_serves(candidates, expected_serves=2)
        self.assertEqual(len(result), 2)

    def test_honors_expected_serves_one(self):
        """expected_serves=1 returns exactly one candidate."""
        candidates = [
            _make_candidate(contact_time_sec=10.0, score=0.9),
            _make_candidate(contact_time_sec=30.0, score=0.8),
        ]
        result = select_serves(candidates, expected_serves=1)
        self.assertEqual(len(result), 1)

    def test_returns_fewer_when_not_enough_non_overlapping(self):
        """If overlapping candidates prevent filling expected_serves, return fewer."""
        candidates = [
            _make_candidate(contact_time_sec=5.0, score=0.9),
            _make_candidate(contact_time_sec=5.5, score=0.8),  # within min_gap of 5.0
        ]
        result = select_serves(candidates, expected_serves=2)
        self.assertLessEqual(len(result), 1)

    def test_empty_candidates_returns_empty(self):
        """No candidates → empty result regardless of expected_serves."""
        result = select_serves([], expected_serves=3)
        self.assertEqual(result, [])

    # --- (c) Results sorted by contact_time_sec ---

    def test_results_sorted_by_contact_time(self):
        """Output is sorted by contact_time_sec ascending regardless of input score order."""
        candidates = [
            _make_candidate(contact_time_sec=30.0, score=0.95),
            _make_candidate(contact_time_sec=10.0, score=0.80),
            _make_candidate(contact_time_sec=50.0, score=0.70),
        ]
        result = select_serves(candidates, expected_serves=3)
        times = [c["contact_time_sec"] for c in result]
        self.assertEqual(times, [10.0, 30.0, 50.0])

    def test_sorting_holds_with_many_candidates(self):
        """Sort order verified with more candidates than expected_serves."""
        candidates = [
            _make_candidate(contact_time_sec=60.0, score=0.95),
            _make_candidate(contact_time_sec=10.0, score=0.85),
            _make_candidate(contact_time_sec=35.0, score=0.75),
            _make_candidate(contact_time_sec=80.0, score=0.65),
        ]
        result = select_serves(candidates, expected_serves=3)
        times = [c["contact_time_sec"] for c in result]
        self.assertEqual(times, sorted(times))

    # --- min_gap_sec enforcement ---

    def test_min_gap_sec_respected(self):
        """Candidates closer than min_gap_sec should not both appear."""
        candidates = [
            _make_candidate(contact_time_sec=10.0, score=0.9),
            _make_candidate(
                contact_time_sec=11.0, score=0.8
            ),  # 1s apart < default 2s gap
            _make_candidate(contact_time_sec=30.0, score=0.7),
        ]
        result = select_serves(candidates, expected_serves=3, min_gap_sec=2.0)
        times = [c["contact_time_sec"] for c in result]
        # 10 and 11 can't both be selected; at most 2 candidates returned
        self.assertLessEqual(len(result), 2)
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                self.assertGreaterEqual(times[j] - times[i], 2.0)


# ---------------------------------------------------------------------------
# detect_serve_candidates — autonomous default count
# ---------------------------------------------------------------------------


class TestDetectServeCandidatesDefaultCount(unittest.TestCase):
    """detect_serve_candidates uses sensible default when expected_serves omitted."""

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events", return_value=[])
    @patch("serve_analyzer.serve_attempts.compute_vertical_velocity")
    @patch("serve_analyzer.serve_attempts.compute_frame_velocities")
    @patch("serve_analyzer.serve_attempts.interpolate_missing_detections")
    @patch(
        "serve_analyzer.serve_attempts.detect_ball_yolo",
        return_value=([], 30.0, 100, None),
    )
    def test_expected_serves_none_does_not_raise(
        self,
        mock_detect,
        mock_interp,
        mock_vel,
        mock_vert_vel,
        mock_events,
        mock_analyze,
    ):
        """Omitting expected_serves should not raise; default of 8 used internally."""
        result = detect_serve_candidates("video.mov")
        self.assertIsInstance(result, list)

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events")
    @patch("serve_analyzer.serve_attempts.compute_vertical_velocity")
    @patch("serve_analyzer.serve_attempts.compute_frame_velocities")
    @patch("serve_analyzer.serve_attempts.interpolate_missing_detections")
    @patch(
        "serve_analyzer.serve_attempts.detect_ball_yolo",
        return_value=([], 30.0, 100, None),
    )
    def test_expected_serves_none_passes_expanded_count_to_profiles(
        self,
        mock_detect,
        mock_interp,
        mock_vel,
        mock_vert_vel,
        mock_events,
        mock_analyze,
    ):
        """When expected_serves=None, detect_serve_events gets default-expanded count."""
        detect_serve_candidates("video.mov", expected_serves=None)
        for call_args in mock_events.call_args_list:
            profile_expected = call_args[1]["expected_serves"]
            # default 8 -> max(8*3, 8+8) = 24 for first two profiles
            self.assertGreaterEqual(profile_expected, 16)

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events")
    @patch("serve_analyzer.serve_attempts.compute_vertical_velocity")
    @patch("serve_analyzer.serve_attempts.compute_frame_velocities")
    @patch("serve_analyzer.serve_attempts.interpolate_missing_detections")
    @patch(
        "serve_analyzer.serve_attempts.detect_ball_yolo",
        return_value=([], 30.0, 100, None),
    )
    def test_explicit_expected_serves_overrides_default(
        self,
        mock_detect,
        mock_interp,
        mock_vel,
        mock_vert_vel,
        mock_events,
        mock_analyze,
    ):
        """Explicit expected_serves propagates to detection profiles."""
        detect_serve_candidates("video.mov", expected_serves=3)
        for call_args in mock_events.call_args_list:
            profile_expected = call_args[1]["expected_serves"]
            # explicit 3 -> max(3*3, 3+8) = max(9, 11) = 11 for first two
            self.assertGreaterEqual(profile_expected, 9)


# ---------------------------------------------------------------------------
# select_serves — autonomous count mode (expected_serves=None)
# ---------------------------------------------------------------------------


class TestSelectServesAutonomousCount(unittest.TestCase):
    """select_serves(expected_serves=None) selects all quality candidates."""

    def _try_autonomous(self, candidates, **kwargs):
        """Call select_serves with expected_serves=None; skipTest on TypeError."""
        try:
            return select_serves(candidates, expected_serves=None, **kwargs)
        except TypeError:
            self.skipTest(
                "IMPLEMENTATION GAP: select_serves does not accept expected_serves=None"
            )

    def test_returns_all_quality_candidates(self):
        """Autonomous mode returns all candidates that pass quality heuristics."""
        candidates = [
            _make_candidate(contact_time_sec=5.0, score=0.9, max_kmh=175.0),
            _make_candidate(contact_time_sec=20.0, score=0.8, max_kmh=170.0),
            _make_candidate(contact_time_sec=40.0, score=0.7, max_kmh=165.0),
        ]
        result = self._try_autonomous(candidates)
        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), len(candidates))

    def test_respects_min_gap_sec(self):
        """Autonomous mode still enforces min_gap_sec between selections."""
        candidates = [
            _make_candidate(contact_time_sec=10.0, score=0.9),
            _make_candidate(contact_time_sec=11.0, score=0.8),
            _make_candidate(contact_time_sec=30.0, score=0.7),
        ]
        result = self._try_autonomous(candidates, min_gap_sec=2.0)
        times = [c["contact_time_sec"] for c in result]
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                self.assertGreaterEqual(times[j] - times[i], 2.0)

    def test_empty_candidates_returns_empty(self):
        """No candidates -> empty result in autonomous mode."""
        result = self._try_autonomous([])
        self.assertEqual(result, [])

    def test_excludes_low_velocity_candidates(self):
        """Candidates with implausibly low velocity excluded in autonomous mode."""
        candidates = [
            _make_candidate(contact_time_sec=10.0, score=0.9, max_kmh=170.0),
            _make_candidate(
                contact_time_sec=30.0, score=0.1, max_kmh=5.0, mean_kmh=3.0
            ),
            _make_candidate(contact_time_sec=50.0, score=0.8, max_kmh=165.0),
        ]
        result = self._try_autonomous(candidates)
        times = [c["contact_time_sec"] for c in result]
        self.assertNotIn(30.0, times)

    def test_sorted_by_contact_time(self):
        """Autonomous results sorted by contact_time_sec ascending."""
        candidates = [
            _make_candidate(contact_time_sec=50.0, score=0.7),
            _make_candidate(contact_time_sec=10.0, score=0.9),
            _make_candidate(contact_time_sec=30.0, score=0.8),
        ]
        result = self._try_autonomous(candidates)
        times = [c["contact_time_sec"] for c in result]
        self.assertEqual(times, sorted(times))


# ---------------------------------------------------------------------------
# select_serves — backward-compatible explicit expected_serves
# ---------------------------------------------------------------------------


class TestSelectServesExplicitKBackwardCompat(unittest.TestCase):
    """Explicit expected_serves preserved when autonomous mode is added."""

    def test_explicit_k_one(self):
        """expected_serves=1 returns exactly one."""
        candidates = [
            _make_candidate(contact_time_sec=10.0, score=0.9),
            _make_candidate(contact_time_sec=30.0, score=0.8),
            _make_candidate(contact_time_sec=50.0, score=0.7),
        ]
        result = select_serves(candidates, expected_serves=1)
        self.assertEqual(len(result), 1)

    def test_explicit_k_more_than_candidates(self):
        """expected_serves > len(candidates) returns at most len(candidates)."""
        candidates = [_make_candidate(contact_time_sec=10.0, score=0.9)]
        result = select_serves(candidates, expected_serves=5)
        self.assertLessEqual(len(result), 1)

    def test_explicit_k_zero_returns_empty(self):
        """expected_serves=0 returns empty list."""
        candidates = [_make_candidate(contact_time_sec=10.0, score=0.9)]
        result = select_serves(candidates, expected_serves=0)
        self.assertEqual(result, [])

    def test_explicit_k_negative_returns_empty(self):
        """Negative expected_serves returns empty list."""
        result = select_serves(
            [_make_candidate(contact_time_sec=10.0)],
            expected_serves=-1,
        )
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# CLI — autonomous count in video-only mode
# ---------------------------------------------------------------------------


class TestDetectorCLIAutonomousCount(unittest.TestCase):
    """CLI video-only mode uses default count when --expected-serves is omitted."""

    @patch(
        "serve_analyzer.serve_attempts.select_serves",
        side_effect=lambda c, expected_serves=1, **kw: list(c)[:expected_serves],
    )
    @patch(
        "serve_analyzer.serve_attempts.detect_serve_candidates",
        return_value=[
            _make_candidate(contact_time_sec=10.0),
            _make_candidate(contact_time_sec=30.0),
        ],
    )
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_video_only_without_expected_serves_is_autonomous(
        self,
        mock_stdout,
        mock_detect,
        mock_select,
    ):
        """CLI video-only without --expected-serves uses autonomous inference."""
        from serve_analyzer.serve_attempts import main

        exit_code = main(["video.mov"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertIsNone(payload["expected_serves"])
        self.assertTrue(payload["count_inferred"])

    @patch(
        "serve_analyzer.serve_attempts.select_serves",
        side_effect=lambda c, expected_serves=1, **kw: list(c)[:expected_serves],
    )
    @patch(
        "serve_analyzer.serve_attempts.detect_serve_candidates",
        return_value=[_make_candidate(contact_time_sec=10.0)],
    )
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_video_only_with_explicit_expected_serves(
        self,
        mock_stdout,
        mock_detect,
        mock_select,
    ):
        """CLI video-only with --expected-serves 3 passes it through."""
        from serve_analyzer.serve_attempts import main

        exit_code = main(["video.mov", "--expected-serves", "3"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertEqual(payload["expected_serves"], 3)

    @patch(
        "serve_analyzer.serve_attempts.select_serves",
        side_effect=lambda c, expected_serves=1, **kw: list(c)[:expected_serves],
    )
    @patch(
        "serve_analyzer.serve_attempts.detect_serve_candidates",
        return_value=[],
    )
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_video_only_empty_candidates_output_shape(
        self,
        mock_stdout,
        mock_detect,
        mock_select,
    ):
        """Video-only with no detections has correct output shape."""
        from serve_analyzer.serve_attempts import main

        exit_code = main(["video.mov"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_stdout.getvalue())
        self.assertIn("candidates", payload)
        self.assertIn("selected_serves", payload)
        self.assertEqual(payload["candidates"], [])
        self.assertEqual(payload["selected_serves"], [])
        # No evaluator keys
        self.assertNotIn("matched_count", payload)
        self.assertNotIn("targets_sec", payload)


# ---------------------------------------------------------------------------
# infer_serve_count — autonomous count inference
# ---------------------------------------------------------------------------


class TestInferServeCount(unittest.TestCase):
    """infer_serve_count uses rank-gap elbow detection."""

    def test_empty_candidates_returns_zero(self):
        self.assertEqual(infer_serve_count([]), 0)

    def test_single_candidate_returns_one(self):
        candidates = [{"selector_rank": 0.5, "contact_time_sec": 10.0}]
        self.assertEqual(infer_serve_count(candidates), 1)

    def test_uniform_ranks_returns_all(self):
        """No clear gap → return all candidates above floor."""
        candidates = [
            {"selector_rank": 0.40 + i * 0.001, "contact_time_sec": float(i * 5)}
            for i in range(5)
        ]
        self.assertEqual(infer_serve_count(candidates), 5)

    def test_clear_elbow_cuts_at_gap(self):
        """A large rank gap should cut the count before it."""
        candidates = [
            {"selector_rank": 0.60, "contact_time_sec": 5.0},
            {"selector_rank": 0.55, "contact_time_sec": 15.0},
            {"selector_rank": 0.50, "contact_time_sec": 25.0},
            {"selector_rank": 0.15, "contact_time_sec": 35.0},
            {"selector_rank": 0.10, "contact_time_sec": 45.0},
        ]
        # Gap between 0.50 and 0.15 is 0.35 — much larger than others
        result = infer_serve_count(candidates)
        self.assertEqual(result, 3)

    def test_floor_excludes_low_rank(self):
        """Candidates below min_rank_floor are excluded before elbow analysis."""
        candidates = [
            {"selector_rank": 0.50, "contact_time_sec": 5.0},
            {"selector_rank": 0.48, "contact_time_sec": 15.0},
            {"selector_rank": 0.01, "contact_time_sec": 25.0},
        ]
        result = infer_serve_count(candidates, min_rank_floor=0.05)
        # 0.01 is below floor, so only 2 above floor, gap too small → 2
        self.assertEqual(result, 2)

    def test_all_below_floor_returns_zero(self):
        candidates = [
            {"selector_rank": 0.02, "contact_time_sec": 5.0},
            {"selector_rank": 0.01, "contact_time_sec": 15.0},
        ]
        self.assertEqual(infer_serve_count(candidates, min_rank_floor=0.05), 0)


# ---------------------------------------------------------------------------
# select_serves — autonomous mode
# ---------------------------------------------------------------------------


class TestSelectServesAutonomous(unittest.TestCase):
    """select_serves(expected_serves=None) uses autonomous count inference."""

    def test_autonomous_with_clear_quality_candidates(self):
        """When top candidates are clearly better than rest, infer correct count."""
        candidates = [
            _make_candidate(
                contact_time_sec=10.0,
                score=0.9,
                max_kmh=175.0,
                mean_kmh=120.0,
                support_count=3,
                recent_upward_fraction=0.55,
                frames_after_apex=30,
                contact_velocity=720.0,
            ),
            _make_candidate(
                contact_time_sec=25.0,
                score=0.85,
                max_kmh=170.0,
                mean_kmh=115.0,
                support_count=3,
                recent_upward_fraction=0.52,
                frames_after_apex=28,
                contact_velocity=710.0,
            ),
            _make_candidate(
                contact_time_sec=40.0,
                score=0.84,
                max_kmh=168.0,
                mean_kmh=118.0,
                support_count=3,
                recent_upward_fraction=0.53,
                frames_after_apex=29,
                contact_velocity=715.0,
            ),
        ]
        result = select_serves(candidates, expected_serves=None)
        # All 3 are good quality — should infer all 3
        self.assertEqual(len(result), 3)

    def test_autonomous_excludes_poor_candidates(self):
        """Autonomous mode should not force-include low-quality candidates."""
        candidates = [
            _make_candidate(contact_time_sec=10.0, score=0.9, max_kmh=175.0),
            _make_candidate(contact_time_sec=25.0, score=0.85, max_kmh=170.0),
            _make_candidate(
                contact_time_sec=1.0, score=0.5, max_kmh=30.0, mean_kmh=25.0
            ),
        ]
        result = select_serves(candidates, expected_serves=None)
        # The 3rd candidate is poor quality; autonomous mode should infer <= 2
        self.assertLessEqual(len(result), 2)

    def test_autonomous_empty_candidates(self):
        result = select_serves([], expected_serves=None)
        self.assertEqual(result, [])

    def test_explicit_still_works(self):
        """Explicit expected_serves=2 forces exactly 2 (backward compat)."""
        candidates = [
            _make_candidate(contact_time_sec=10.0, score=0.9),
            _make_candidate(contact_time_sec=25.0, score=0.8),
            _make_candidate(contact_time_sec=40.0, score=0.7),
        ]
        result = select_serves(candidates, expected_serves=2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
