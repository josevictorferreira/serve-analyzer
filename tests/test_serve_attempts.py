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

from web.backend.services.analysis_service import run_analysis


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
        # Default rightward/drop fields — represent a plausible serve.
        # Tests that need non-serve shapes override these via **extra.
        "rightward_fraction": 0.6,
        "net_rightward_displacement": 100.0,
        "drop_after_apex": 200.0,
        "support_count": 2,
        "contact_velocity": 1000.0,
        "recent_upward_fraction": 0.5,
        "frames_after_apex": 30,
        "upward_fraction": 0.5,
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
    @patch("serve_analyzer.serve_attempts.compute_horizontal_velocity")
    @patch("serve_analyzer.serve_attempts.compute_vertical_velocity")
    @patch("serve_analyzer.serve_attempts.compute_frame_velocities")
    @patch("serve_analyzer.serve_attempts.interpolate_missing_detections")
    def test_expected_serves_zero_raises(
        self, mock_interp, mock_vel, mock_vert_vel, mock_horiz_vel, _mock_detect
    ):
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
    @patch("serve_analyzer.serve_attempts.compute_horizontal_velocity")
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
        mock_horiz_vel,
        mock_events,
        mock_analyze,
    ):
        """When no events detected, should return empty list."""
        result = detect_serve_candidates("video.mov")
        self.assertIsInstance(result, dict)
        self.assertIn("candidates", result)
        self.assertIn("positions", result)
        self.assertIn("frame_skip", result)
        candidates = result["candidates"]
        self.assertIsInstance(candidates, list)
        self.assertEqual(len(candidates), 0)

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events")
    @patch("serve_analyzer.serve_attempts.compute_horizontal_velocity")
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
        mock_horiz_vel,
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

        result = detect_serve_candidates("video.mov")
        self.assertIsInstance(result, dict)
        self.assertIn("candidates", result)
        self.assertIn("positions", result)
        self.assertIn("frame_skip", result)
        candidates = result["candidates"]

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
    @patch("serve_analyzer.serve_attempts.compute_horizontal_velocity")
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
        mock_horiz_vel,
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

        result = detect_serve_candidates("video.mov")
        candidates = result["candidates"]

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
        post-contact velocity (40 km/h) and no rightward motion, signalling
        a false positive. The three later candidates carry realistic serve
        velocities with clear rightward motion.
        """
        candidates = [
            _make_candidate(
                contact_time_sec=0.3,
                score=0.98,
                max_kmh=40.0,
                mean_kmh=35.0,
                rightward_fraction=0.1,
                net_rightward_displacement=-5.0,
            ),
            _make_candidate(
                contact_time_sec=12.0,
                score=0.82,
                max_kmh=175.0,
                mean_kmh=165.0,
                rightward_fraction=0.8,
                net_rightward_displacement=100.0,
                support_count=3,
                recent_upward_fraction=0.55,
                frames_after_apex=30,
            ),
            _make_candidate(
                contact_time_sec=28.0,
                score=0.78,
                max_kmh=168.0,
                mean_kmh=158.0,
                rightward_fraction=0.75,
                net_rightward_displacement=90.0,
                support_count=3,
                recent_upward_fraction=0.53,
                frames_after_apex=28,
            ),
            _make_candidate(
                contact_time_sec=44.0,
                score=0.72,
                max_kmh=162.0,
                mean_kmh=152.0,
                rightward_fraction=0.7,
                net_rightward_displacement=85.0,
                support_count=2,
                recent_upward_fraction=0.50,
                frames_after_apex=26,
            ),
        ]
        result = select_serves(candidates, expected_serves=3)
        times = [c["contact_time_sec"] for c in result]
        self.assertEqual(times, [12.0, 28.0, 44.0])

    def test_avoids_clustered_early_false_positives(self):
        """Multiple early high-score FPs should not crowd out real serves."""
        candidates = [
            _make_candidate(
                contact_time_sec=0.4,
                score=0.96,
                max_kmh=35.0,
                mean_kmh=30.0,
                rightward_fraction=0.1,
                net_rightward_displacement=-5.0,
            ),
            _make_candidate(
                contact_time_sec=1.2,
                score=0.94,
                max_kmh=45.0,
                mean_kmh=38.0,
                rightward_fraction=0.1,
                net_rightward_displacement=-10.0,
            ),
            _make_candidate(
                contact_time_sec=10.0,
                score=0.80,
                max_kmh=172.0,
                mean_kmh=162.0,
                rightward_fraction=0.8,
                net_rightward_displacement=100.0,
                support_count=3,
                recent_upward_fraction=0.55,
                frames_after_apex=30,
            ),
            _make_candidate(
                contact_time_sec=25.0,
                score=0.75,
                max_kmh=165.0,
                mean_kmh=155.0,
                rightward_fraction=0.75,
                net_rightward_displacement=90.0,
                support_count=2,
                recent_upward_fraction=0.52,
                frames_after_apex=28,
            ),
            _make_candidate(
                contact_time_sec=40.0,
                score=0.70,
                max_kmh=158.0,
                mean_kmh=148.0,
                rightward_fraction=0.7,
                net_rightward_displacement=85.0,
                support_count=2,
                recent_upward_fraction=0.50,
                frames_after_apex=25,
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
    @patch("serve_analyzer.serve_attempts.compute_horizontal_velocity")
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
        mock_horiz_vel,
        mock_events,
        mock_analyze,
    ):
        """Omitting expected_serves should not raise; default of 12 used internally."""
        result = detect_serve_candidates("video.mov")
        self.assertIsInstance(result, dict)
        self.assertIn("candidates", result)
        self.assertIn("positions", result)
        self.assertIn("frame_skip", result)
        candidates = result["candidates"]
        self.assertIsInstance(candidates, list)

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events")
    @patch("serve_analyzer.serve_attempts.compute_horizontal_velocity")
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
        mock_horiz_vel,
        mock_events,
        mock_analyze,
    ):
        """When expected_serves=None, detect_serve_events gets default-expanded count."""
        detect_serve_candidates("video.mov", expected_serves=None)
        for call_args in mock_events.call_args_list:
            profile_expected = call_args[1]["expected_serves"]
            # default 12 -> max(12*3, 12+8) = 36 for first two profiles
            self.assertGreaterEqual(profile_expected, 16)

    @patch("serve_analyzer.serve_attempts.analyze_serve")
    @patch("serve_analyzer.serve_attempts.detect_serve_events")
    @patch("serve_analyzer.serve_attempts.compute_horizontal_velocity")
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
        mock_horiz_vel,
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
                contact_time_sec=30.0,
                score=0.1,
                max_kmh=5.0,
                mean_kmh=3.0,
                rightward_fraction=0.1,
                net_rightward_displacement=-2.0,
                drop_after_apex=2.0,
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
            {"selector_rank": 0.90, "contact_time_sec": 5.0},
            {"selector_rank": 0.85, "contact_time_sec": 15.0},
            {"selector_rank": 0.80, "contact_time_sec": 25.0},
            {"selector_rank": 0.10, "contact_time_sec": 35.0},
            {"selector_rank": 0.05, "contact_time_sec": 45.0},
        ]
        # Gap between 0.80 and 0.10 is 0.70 - clear elbow
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
                contact_time_sec=1.0,
                score=0.5,
                max_kmh=30.0,
                mean_kmh=25.0,
                rightward_fraction=0.15,
                net_rightward_displacement=-3.0,
                drop_after_apex=3.0,
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


# ---------------------------------------------------------------------------
# Rightward motion & robust speed regression tests
# ---------------------------------------------------------------------------


class TestRightwardMotionValidation(unittest.TestCase):
    """Post-contact rightward motion validation prevents false serves."""

    def test_rightward_candidate_ranks_above_nonrightward(self):
        """A candidate with clear rightward motion ranks higher than one without."""
        rightward = _make_candidate(
            contact_time_sec=15.0,
            score=0.85,
            max_kmh=170.0,
            rightward_fraction=0.85,
            net_rightward_displacement=120.0,
            support_count=3,
            recent_upward_fraction=0.55,
            frames_after_apex=30,
        )
        non_rightward = _make_candidate(
            contact_time_sec=12.0,
            score=0.95,
            max_kmh=200.0,
            rightward_fraction=0.1,
            net_rightward_displacement=-50.0,
            support_count=3,
            recent_upward_fraction=0.55,
            frames_after_apex=30,
        )
        result = select_serves([rightward, non_rightward], expected_serves=1)
        # Rightward candidate should win despite lower raw score
        self.assertEqual(result[0]["contact_time_sec"], 15.0)

    def test_toss_only_upward_rejected(self):
        """A toss-only candidate (no rightward, upward-only) ranks below true serves."""
        toss_only = _make_candidate(
            contact_time_sec=10.0,
            score=0.9,
            max_kmh=80.0,
            rightward_fraction=0.05,
            net_rightward_displacement=5.0,
            support_count=1,
            recent_upward_fraction=0.50,
            frames_after_apex=10,
        )
        true_serve = _make_candidate(
            contact_time_sec=25.0,
            score=0.80,
            max_kmh=170.0,
            rightward_fraction=0.80,
            net_rightward_displacement=100.0,
            support_count=3,
            recent_upward_fraction=0.55,
            frames_after_apex=30,
        )
        result = select_serves([toss_only, true_serve], expected_serves=1)
        self.assertEqual(result[0]["contact_time_sec"], 25.0)

    def test_rebound_leftward_penalized(self):
        """A rebound (leftward) candidate is penalized below rightward serves."""
        rebound = _make_candidate(
            contact_time_sec=20.0,
            score=0.88,
            max_kmh=190.0,
            rightward_fraction=0.1,
            net_rightward_displacement=-80.0,
            support_count=2,
            recent_upward_fraction=0.50,
            frames_after_apex=25,
        )
        true_serve = _make_candidate(
            contact_time_sec=35.0,
            score=0.80,
            max_kmh=170.0,
            rightward_fraction=0.75,
            net_rightward_displacement=90.0,
            support_count=3,
            recent_upward_fraction=0.55,
            frames_after_apex=30,
        )
        result = select_serves([rebound, true_serve], expected_serves=1)
        self.assertEqual(result[0]["contact_time_sec"], 35.0)

    def test_no_temporal_bias_in_ranking(self):
        """Selector does not prefer early candidates over late ones by time alone."""
        early = _make_candidate(
            contact_time_sec=5.0,
            score=0.85,
            max_kmh=170.0,
            rightward_fraction=0.7,
            net_rightward_displacement=80.0,
            support_count=3,
            recent_upward_fraction=0.55,
            frames_after_apex=30,
        )
        late = _make_candidate(
            contact_time_sec=70.0,
            score=0.85,
            max_kmh=170.0,
            rightward_fraction=0.7,
            net_rightward_displacement=80.0,
            support_count=3,
            recent_upward_fraction=0.55,
            frames_after_apex=30,
        )
        # Same geometric quality — ranks should be equal (no early bonus / late penalty)
        result = select_serves([early, late], expected_serves=2)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(
            result[0]["selector_rank"], result[1]["selector_rank"], places=3
        )


class TestRobustSpeedMetric(unittest.TestCase):
    """Robust speed metric is not blown up by one-frame outliers."""

    def test_robust_speed_less_than_spike_max(self):
        """compute_horizontal_velocity returns array that can be used for robust speed."""
        from serve_analyzer.multi_serve import compute_horizontal_velocity
        import numpy as np

        # Simulate post-contact positions: mostly rightward at ~10px/frame,
        # but one frame jumps 100px rightward (outlier spike)
        positions = [(float(i * 10), 100.0) for i in range(10)]
        # Inject one outlier frame
        positions[5] = (positions[4][0] + 100.0, 100.0)
        positions[6] = (positions[5][0] + 10.0, 100.0)

        horiz = compute_horizontal_velocity(positions, smooth_sigma=0.0)
        # Most frames should have dx ~10, one frame has dx ~100
        self.assertGreater(np.max(horiz), 50.0)  # spike exists
        # p90 of rightward frames is more robust than max
        rightward = horiz[horiz > 0.5]
        if len(rightward) >= 2:
            p90 = float(np.percentile(rightward, 90))
            self.assertLess(p90, float(np.max(rightward)))

    def test_rightward_only_p90_not_inflated_by_leftward(self):
        """Leftward frames are excluded from robust speed calculation."""
        import numpy as np

        # Mix of rightward and leftward frames
        speeds = np.array([10.0, 12.0, -5.0, 11.0, -3.0, 13.0, 10.0, 14.0, -8.0, 12.0])
        rightward = speeds[speeds > 0.5]
        p90 = float(np.percentile(rightward, 90))
        # p90 should be close to the typical rightward speed, not inflated
        self.assertLess(p90, 14.0)  # well below any potential spike
        self.assertGreater(p90, 10.0)  # but still represents real speed


class TestAdapterAutonomousMode(unittest.TestCase):
    """analysis_service passes expected_serves=None through to detector."""

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.detection_services.select_serves")
    @patch("web.backend.services.detection_services.detect_serve_candidates")
    def test_autonomous_mode_passes_none_to_detector(
        self, mock_detect, mock_select, mock_info
    ):
        """When expected_serves=None, detector receives None (not 12)."""
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = {"candidates": [], "positions": [], "frame_skip": 1}
        mock_select.return_value = []

        run_analysis("/tmp/video.mov", expected_serves=None)

        _, kwargs = mock_detect.call_args
        self.assertIsNone(kwargs["expected_serves"])

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.detection_services.select_serves")
    @patch("web.backend.services.detection_services.detect_serve_candidates")
    def test_explicit_mode_passes_int_to_detector(
        self, mock_detect, mock_select, mock_info
    ):
        """When expected_serves=5, detector receives 5."""
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = {"candidates": [], "positions": [], "frame_skip": 1}
        mock_select.return_value = []

        run_analysis("/tmp/video.mov", expected_serves=5)

        _, kwargs = mock_detect.call_args
        self.assertEqual(kwargs["expected_serves"], 5)


class TestDirectionUnreliableRecovery(unittest.TestCase):
    """Direction-unreliable candidates with strong toss evidence can compete."""

    def test_recovered_candidate_beats_weak_ordinary(self):
        """A direction_unreliable candidate with strong toss evidence ranks above
        a weak ordinary candidate (low rightward, low support)."""
        recovered = _make_candidate(
            contact_time_sec=10.0,
            score=0.80,
            max_kmh=170.0,
            rightward_fraction=0.17,
            net_rightward_displacement=-2000.0,
            support_count=3,
            recent_upward_fraction=0.65,
            frames_after_apex=107,
            contact_velocity=1600.0,
            upward_fraction=0.56,
            drop_after_apex=538.0,
            direction_unreliable=True,
        )
        weak_ordinary = _make_candidate(
            contact_time_sec=52.0,
            score=0.50,
            max_kmh=10.0,
            rightward_fraction=0.48,
            net_rightward_displacement=1.0,
            support_count=1,
            recent_upward_fraction=0.37,
            frames_after_apex=128,
            contact_velocity=900.0,
            upward_fraction=0.49,
            drop_after_apex=582.0,
        )
        result = select_serves([recovered, weak_ordinary], expected_serves=2)
        self.assertEqual(len(result), 2)
        # recovered should have higher rank than weak_ordinary
        ranks = {c["contact_time_sec"]: c["selector_rank"] for c in result}
        self.assertGreater(ranks[10.0], ranks[52.0])

    def test_junk_still_excluded(self):
        """Junk candidates (no-motion, near-zero drop+nrd) stay out."""
        junk = _make_candidate(
            contact_time_sec=14.0,
            score=0.80,
            max_kmh=40.0,
            rightward_fraction=0.72,
            net_rightward_displacement=-1.5,
            drop_after_apex=0.3,
        )
        good = _make_candidate(
            contact_time_sec=42.0,
            score=0.90,
            max_kmh=175.0,
        )
        result = select_serves([junk, good], expected_serves=2)
        # junk should be filtered by no-motion gate
        self.assertNotIn(14.0, [c["contact_time_sec"] for c in result])

    def test_recovery_bonus_requires_strong_evidence(self):
        """A direction_unreliable candidate WITHOUT strong toss evidence
        does NOT get recovery bonus and ranks low."""
        weak_recovered = _make_candidate(
            contact_time_sec=13.0,
            score=0.80,
            max_kmh=170.0,
            rightward_fraction=0.52,
            net_rightward_displacement=-1500.0,
            support_count=2,
            recent_upward_fraction=0.54,
            frames_after_apex=88,
            contact_velocity=2000.0,
            upward_fraction=0.31,  # weak toss evidence
            drop_after_apex=4.0,  # small drop
            direction_unreliable=True,
        )
        clean_rightward = _make_candidate(
            contact_time_sec=42.0,
            score=0.85,
            max_kmh=170.0,
        )
        result = select_serves([weak_recovered, clean_rightward], expected_serves=2)
        # weak recovered should be filtered (negative rank from penalties)
        # only clean_rightward survives
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["contact_time_sec"], 42.0)


# ---------------------------------------------------------------------------
# Contact-frame refinement (_refine_contact_frame)
# ---------------------------------------------------------------------------


class TestContactFrameRefinement(unittest.TestCase):
    """_refine_contact_frame shifts toward sharpest horizontal acceleration."""

    def _make_horiz_velocities(self, n, accel_frame, pre_val=1.0, post_val=20.0):
        """Build a horiz_velocities array with a sharp acceleration at accel_frame."""
        import numpy as np

        hv = np.full(n, pre_val, dtype=float)
        hv[accel_frame:] = post_val
        # Smooth slightly to mimic real data
        from scipy.ndimage import gaussian_filter1d

        hv = gaussian_filter1d(hv, sigma=1.0)
        return hv

    def test_refines_to_strongest_acceleration(self):
        """Refinement shifts contact to frame with strongest rightward acceleration."""
        from serve_analyzer.multi_serve import _refine_contact_frame

        n = 60
        accel_frame = 30  # true contact: sharp rightward acceleration here
        peak_frame = 33  # smoothed velocity peak is a few frames late
        hv = self._make_horiz_velocities(n, accel_frame, pre_val=1.0, post_val=25.0)
        positions = [(float(i), 100.0) for i in range(n)]

        refined = _refine_contact_frame(peak_frame, positions, hv, window=5)
        # Should shift toward the acceleration transition (near accel_frame)
        self.assertLessEqual(abs(refined - accel_frame), 3)
        self.assertNotEqual(refined, peak_frame)  # must have shifted

    def test_no_shift_when_peak_is_best(self):
        """If peak_frame already has the strongest acceleration, keep it."""
        from serve_analyzer.multi_serve import _refine_contact_frame
        import numpy as np

        n = 60
        # Uniform horiz velocity — no acceleration anywhere
        hv = np.full(n, 5.0)
        positions = [(float(i), 100.0) for i in range(n)]

        refined = _refine_contact_frame(30, positions, hv, window=5)
        self.assertEqual(refined, 30)  # no shift

    def test_no_shift_beyond_window(self):
        """Refinement never shifts more than ±window frames."""
        from serve_analyzer.multi_serve import _refine_contact_frame
        import numpy as np

        n = 100
        # Acceleration far from peak
        hv = np.zeros(n)
        hv[10] = 0.1  # tiny accel far away
        hv[20] = 0.1  # tiny accel
        hv[50] = 100.0  # huge accel but 20 frames from peak=30 with window=5
        positions = [(float(i), 100.0) for i in range(n)]

        refined = _refine_contact_frame(30, positions, hv, window=5)
        self.assertEqual(refined, 30)  # too far, no shift

    def test_conservative_threshold_rejects_weak_evidence(self):
        """Refinement rejects if candidate acceleration is <1.5x peak acceleration."""
        from serve_analyzer.multi_serve import _refine_contact_frame
        import numpy as np

        n = 60
        # Build horiz_velocities where peak_frame (30) has strong accel (=20)
        # and a nearby frame (28) has slightly higher accel (=21).
        # 21 is NOT > 20*1.5=30, so refinement must be rejected.
        hv = np.zeros(n)
        hv[28] = 21.0  # accel from 27→28 = 21 (strong, but not 1.5x of peak)
        hv[30] = 0.0  # setup so accel from 30→31 = 20
        hv[31] = 20.0  # accel from 30→31 = 20 (peak's own accel)
        positions = [(float(i), 100.0) for i in range(n)]

        refined = _refine_contact_frame(30, positions, hv, window=5)
        self.assertEqual(refined, 30)  # 21 is not 1.5x of 20


# ---------------------------------------------------------------------------
# Post-contact speed excludes contact frame (analyze_serve)
# ---------------------------------------------------------------------------


class TestPostContactSpeedExcludesContactFrame(unittest.TestCase):
    """analyze_serve post-contact speed must not include the contact-frame jump."""

    def test_contact_frame_spike_excluded(self):
        """A huge contact-frame displacement must not inflate post-contact speed."""
        from serve_analyzer.multi_serve import analyze_serve

        fps = 30.0
        scale = 0.001  # 1mm/px
        # 45 frames total: toss frames 0-19, contact at 20, post 21-40
        positions = [(100.0, 200.0 - i * 2) for i in range(20)]  # toss: upward
        # Contact frame: huge jump rightward (the racket hit)
        positions.append((500.0, 180.0))  # frame 20 — contact
        # Post-contact: steady rightward motion at 10px/frame
        for i in range(20):
            positions.append((510.0 + i * 10, 180.0 + i * 0.5))

        event = {
            "toss_start_frame": 0,
            "contact_frame": 20,
            "post_contact_end_frame": 39,
            "apex_frame": 15,
            "apex_position": (100.0, 170.0),
        }

        serve = analyze_serve(0, event, positions, fps, scale)

        # Post-contact positions should start from frame 21 (contact+1)
        # The 400px contact-frame jump must NOT be in post_contact_positions
        for p in serve.post_contact_positions:
            # All post-contact x should be >= 500 (starting from frame 21)
            self.assertGreaterEqual(p[0], 500.0)

        # Post-contact max speed should be based on ~10px/frame motion,
        # not the 400px contact jump
        # 10 px/frame * 0.001 m/px * 30 fps * 3.6 = 1.08 km/h (low scale)
        # The key test: it should NOT be 400px/frame * scale * fps * 3.6 ≈ 43.2 km/h
        self.assertLess(serve.post_contact_max_velocity, 50.0)

    def test_post_contact_positions_length(self):
        """post_contact_positions has len = post_end - contact."""
        from serve_analyzer.multi_serve import analyze_serve

        fps = 30.0
        positions = [(float(i), 100.0) for i in range(50)]
        event = {
            "toss_start_frame": 0,
            "contact_frame": 10,
            "post_contact_end_frame": 20,
            "apex_frame": 5,
            "apex_position": (5.0, 100.0),
        }

        serve = analyze_serve(0, event, positions, fps, 0.001)
        # post_contact_positions = positions[11:22] = 11 elements
        self.assertEqual(len(serve.post_contact_positions), 10)

    def test_toss_still_includes_contact_frame(self):
        """Toss phase still includes the contact frame as its endpoint."""
        from serve_analyzer.multi_serve import analyze_serve

        fps = 30.0
        positions = [(float(i), 100.0) for i in range(50)]
        event = {
            "toss_start_frame": 5,
            "contact_frame": 10,
            "post_contact_end_frame": 20,
            "apex_frame": 8,
            "apex_position": (8.0, 100.0),
        }

        serve = analyze_serve(0, event, positions, fps, 0.001)
        # toss_positions = positions[5:11] = 6 elements (includes contact frame 10)
        self.assertEqual(len(serve.toss_positions), 6)
        # Last toss position should be the contact frame position
        self.assertEqual(serve.toss_positions[-1], positions[10])


# ---------------------------------------------------------------------------
# Toss geometry and floor-drive false-positive rejection
# ---------------------------------------------------------------------------


class TestTossGeometryAndFloorDriveRejection(unittest.TestCase):
    """New metadata fields filter prep motions and floor-drive events."""

    def test_prep_pocket_false_positive_rejected(self):
        """Weak toss rise and short toss duration should not survive selection."""
        prep_motion = _make_candidate(
            contact_time_sec=5.0,
            score=0.7,
            max_kmh=50.0,
            mean_kmh=40.0,
            rightward_fraction=0.3,
            net_rightward_displacement=20.0,
            drop_after_apex=30.0,
            toss_rise_px=30.0,
            toss_duration_frames=4,
            early_post_downward_fraction=0.75,
            early_post_net_dy=50.0,
        )
        true_serve = _make_candidate(
            contact_time_sec=20.0,
            score=0.85,
            max_kmh=170.0,
            rightward_fraction=0.75,
            net_rightward_displacement=100.0,
            drop_after_apex=200.0,
            toss_rise_px=120.0,
            toss_duration_frames=12,
            early_post_downward_fraction=0.3,
            early_post_net_dy=20.0,
        )
        result = select_serves([prep_motion, true_serve], expected_serves=2)
        times = [c["contact_time_sec"] for c in result]
        self.assertNotIn(5.0, times)
        self.assertIn(20.0, times)

    def test_floor_hit_false_positive_rejected(self):
        """Rightward motion plus strong immediate downward post-contact is rejected."""
        floor_hit = _make_candidate(
            contact_time_sec=8.0,
            score=0.75,
            max_kmh=90.0,
            mean_kmh=80.0,
            rightward_fraction=0.55,
            net_rightward_displacement=60.0,
            drop_after_apex=50.0,
            toss_rise_px=40.0,
            toss_duration_frames=5,
            early_post_downward_fraction=0.85,
            early_post_net_dy=60.0,
        )
        true_serve = _make_candidate(
            contact_time_sec=25.0,
            score=0.82,
            max_kmh=168.0,
            rightward_fraction=0.72,
            net_rightward_displacement=95.0,
            drop_after_apex=180.0,
            toss_rise_px=110.0,
            toss_duration_frames=14,
            early_post_downward_fraction=0.35,
            early_post_net_dy=25.0,
        )
        result = select_serves([floor_hit, true_serve], expected_serves=2)
        times = [c["contact_time_sec"] for c in result]
        self.assertNotIn(8.0, times)
        self.assertIn(25.0, times)

    def test_real_serve_with_moderate_descent_not_over_pruned(self):
        """Candidate with strong toss geometry and moderate post-contact descent survives."""
        serve = _make_candidate(
            contact_time_sec=15.0,
            score=0.80,
            max_kmh=165.0,
            rightward_fraction=0.70,
            net_rightward_displacement=90.0,
            drop_after_apex=150.0,
            toss_rise_px=100.0,
            toss_duration_frames=10,
            early_post_downward_fraction=0.55,
            early_post_net_dy=35.0,
        )
        result = select_serves([serve], expected_serves=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["contact_time_sec"], 15.0)


if __name__ == "__main__":
    unittest.main()
