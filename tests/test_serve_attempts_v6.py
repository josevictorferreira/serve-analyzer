"""Tests for the autonomous v6 serve detector."""

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve_analyzer.serve_attempts_v6 import (
    DetectionVote,
    _build_speed_rescue_candidates,
    _combine_votes,
    _merge_windows,
    build_parser,
    detect_serve_candidates_v6,
    select_serves_v6,
)


def _make_candidate(contact_frame: int = 10, contact_time_sec: float = 1.0) -> dict:
    return {
        "candidate_index": 0,
        "contact_frame": contact_frame,
        "contact_time_sec": contact_time_sec,
        "score": 500.0,
        "post_contact_max_kmh": 80.0,
        "post_contact_mean_kmh": 50.0,
        "post_contact_max_mps": 22.2,
        "post_contact_mean_mps": 13.9,
        "support_count": 2,
        "rightward_fraction": 0.8,
        "net_rightward_displacement": 120.0,
        "recent_upward_fraction": 0.7,
        "drop_after_apex": 120.0,
    }


class TestV6AutonomousContract(unittest.TestCase):
    def test_public_detector_signature_has_no_expected_serves(self) -> None:
        signature = inspect.signature(detect_serve_candidates_v6)

        self.assertNotIn("expected_serves", signature.parameters)

    def test_parser_has_no_expected_serves_option(self) -> None:
        help_text = build_parser().format_help()

        self.assertNotIn("--expected-serves", help_text)

    def test_output_always_reports_autonomous_count(self) -> None:
        positions = [[float(index), float(index)] for index in range(20)]
        payload = {"candidates": [], "raw_positions": positions}
        candidate = _make_candidate()

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "detections.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

            with (
                patch(
                    "serve_analyzer.serve_attempts_v6._extract_video_metadata",
                    return_value=(30.0, 20, 1280, 720),
                ),
                patch(
                    "serve_analyzer.serve_attempts_v6._build_candidates_from_positions",
                    return_value={
                        "candidates": [candidate],
                        "positions": positions,
                        "raw_positions": positions,
                        "frame_skip": 4,
                        "detector": "v6",
                    },
                ),
                patch(
                    "serve_analyzer.serve_attempts_v6._refine_contact_hybrid",
                    side_effect=lambda candidates, *_args, **_kwargs: list(candidates),
                ),
                patch("serve_analyzer.serve_attempts_v6._recompute_peak_velocities"),
                patch(
                    "serve_analyzer.serve_attempts_v6.select_serves_v6",
                    return_value=[candidate],
                ),
            ):
                result = detect_serve_candidates_v6(
                    "video.mov",
                    input_detections=str(cache_path),
                    fine_window_sec=0.0,
                )

        self.assertIsNone(result["expected_serves"])
        self.assertTrue(result["count_inferred"])
        self.assertEqual(result["inferred_count"], 1)
        self.assertEqual(result["detector"], "v6")


class TestV6WindowsAndVotes(unittest.TestCase):
    def test_merge_windows_combines_overlaps_and_sources(self) -> None:
        result = _merge_windows(
            [
                {"start_frame": 10, "end_frame": 20, "source": "coarse"},
                {"start_frame": 18, "end_frame": 30, "source": "rescue"},
                {"start_frame": 50, "end_frame": 60, "sources": ["coarse"]},
            ],
            total_frames=55,
        )

        self.assertEqual(
            result,
            [
                {
                    "start_frame": 10,
                    "end_frame": 30,
                    "sources": ["coarse", "rescue"],
                },
                {"start_frame": 50, "end_frame": 54, "sources": ["coarse"]},
            ],
        )

    def test_combine_votes_accepts_agreeing_sources(self) -> None:
        position, stats = _combine_votes(
            [
                DetectionVote("yolo", 100.0, 200.0, 0.7),
                DetectionVote("motion_hsv", 104.0, 203.0, 0.5),
            ],
            radius_px=10.0,
            min_vote_count=2,
        )

        self.assertIsNotNone(position)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["vote_count"], 2)
        self.assertEqual(stats["sources"], ["motion_hsv", "yolo"])
        self.assertTrue(stats["accepted"])

    def test_combine_votes_marks_single_strong_vote_pending(self) -> None:
        position, stats = _combine_votes(
            [DetectionVote("yolo", 100.0, 200.0, 0.8)],
            radius_px=10.0,
            min_vote_count=2,
        )

        self.assertIsNone(position)
        self.assertEqual(stats["pending_position"], [100.0, 200.0])
        self.assertFalse(stats["accepted"])

    def test_speed_rescue_adds_candidate_for_uncovered_rescue_window(self) -> None:
        positions = []
        for frame in range(80):
            if frame < 40:
                positions.append((100.0 + frame, 300.0 - frame * 3.0))
            else:
                positions.append((300.0 + (frame - 40) * 20.0, 180.0 + frame - 40))
        diagnostics = [None] * len(positions)
        diagnostics[40] = {
            "vote_count": 2,
            "sources": ["motion_hsv", "yolo"],
            "accepted": True,
        }

        rescued = _build_speed_rescue_candidates(
            [],
            positions,
            fps=10.0,
            scale_factor=0.001,
            windows=[
                {
                    "start_frame": 38,
                    "end_frame": 45,
                    "sources": ["motion_hsv_rescue"],
                }
            ],
            diagnostics=diagnostics,
        )

        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0]["v6_rescue_source"], "motion_hsv_speed")
        self.assertGreaterEqual(rescued[0]["rightward_fraction"], 0.2)


class TestV6Selection(unittest.TestCase):
    def test_selection_rejects_short_toss_downward_artifact(self) -> None:
        artifact = _make_candidate(contact_frame=10, contact_time_sec=1.0)
        artifact.update(
            {
                "toss_duration_frames": 4,
                "early_post_net_dy": 450.0,
                "v6_fine_confirmed": False,
            }
        )
        serve = _make_candidate(contact_frame=60, contact_time_sec=6.0)

        selected = select_serves_v6([artifact, serve])

        self.assertEqual([item["contact_time_sec"] for item in selected], [6.0])


if __name__ == "__main__":
    unittest.main()
