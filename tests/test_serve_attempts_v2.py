"""Detector-only tests for serve_analyzer.serve_attempts_v2."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve_analyzer.serve_attempts_v2 import (
    continuity_gate_positions,
    detect_serve_candidates_v2,
)


class TestContinuityGatePositions(unittest.TestCase):
    """Ball continuity gate rejects short-gap jumps and allows re-acquisition."""

    def test_rejects_large_jump_after_short_gap(self):
        detections = [(10.0, 10.0), (12.0, 10.0), (500.0, 500.0)]
        filtered, stats = continuity_gate_positions(
            detections, max_jump_px=50.0, max_missing_frames=3
        )

        self.assertIsNone(filtered[2])
        self.assertEqual(stats["rejected_jumps"], 1)

    def test_allows_large_jump_after_long_gap(self):
        detections = [(10.0, 10.0), None, None, None, (500.0, 500.0)]
        filtered, stats = continuity_gate_positions(
            detections, max_jump_px=50.0, max_missing_frames=2
        )

        self.assertEqual(filtered[4], (500.0, 500.0))
        self.assertEqual(stats["rejected_jumps"], 0)


class TestDetectServeCandidatesV2(unittest.TestCase):
    """V2 detector output stays detector-only and JSON-safe."""

    @patch("serve_analyzer.serve_attempts_v2.extract_motion_cues", return_value={})
    @patch("serve_analyzer.serve_attempts_v2._extract_fps", return_value=30.0)
    def test_cached_detection_output_has_no_evaluator_keys(
        self, _mock_fps, _mock_motion
    ):
        payload = {
            "candidates": [
                {
                    "candidate_index": 0,
                    "contact_frame": 30,
                    "contact_time_sec": 1.0,
                    "score": 10.0,
                    "post_contact_max_kmh": 100.0,
                    "post_contact_mean_kmh": 90.0,
                    "post_contact_max_mps": 27.7,
                    "post_contact_mean_mps": 25.0,
                    "rightward_fraction": 0.7,
                    "net_rightward_displacement": 100.0,
                    "drop_after_apex": 100.0,
                    "support_count": 2,
                    "contact_velocity": 200.0,
                    "recent_upward_fraction": 0.5,
                    "frames_after_apex": 20,
                    "upward_fraction": 0.5,
                }
            ],
            "raw_positions": [[float(index), float(index)] for index in range(60)],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "detections.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            result = detect_serve_candidates_v2(
                "video.mov", expected_serves=1, input_detections=str(cache_path)
            )

        self.assertEqual(result["detector"], "v2")
        self.assertEqual(len(result["selected_serves"]), 1)
        candidate = result["selected_serves"][0]
        for key in {"matched", "target_time_sec", "delta_sec", "serve_number"}:
            self.assertNotIn(key, candidate)
        self.assertIn("v2_original_contact_frame", candidate)
        self.assertIn("v2_contact_score", candidate)


if __name__ == "__main__":
    unittest.main()
