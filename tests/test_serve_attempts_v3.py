"""Detector-only tests for serve_analyzer.serve_attempts_v3."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve_analyzer.serve_attempts_v3 import (
    _recompute_peak_velocities,
    detect_serve_candidates_v3,
)


def _make_candidate(contact_frame=30, contact_time_sec=1.0):
    """Minimal candidate dict matching v1 detector output shape."""
    return {
        "candidate_index": 0,
        "contact_frame": contact_frame,
        "contact_time_sec": contact_time_sec,
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


class TestRecomputePeakVelocities(unittest.TestCase):
    """`_recompute_peak_velocities` augments candidates with `peak_kmh`."""

    def test_emits_peak_kmh_with_real_track(self):
        # Synthetic constant-velocity track: 10 px/frame.
        positions = [(float(i * 10), 100.0) for i in range(60)]
        candidates = [_make_candidate(contact_frame=20)]

        _recompute_peak_velocities(
            candidates,
            positions=positions,
            fps=30.0,
            scale_factor=0.01,  # 1 px = 1 cm
            post_contact_sec=1.0,
            top_k=5,
        )

        cand = candidates[0]
        self.assertIn("peak_kmh", cand)
        self.assertIsNotNone(cand["peak_kmh"])
        self.assertGreater(cand["peak_kmh"], 0.0)
        # 10 px/frame * 0.01 m/px * 30 fps * 3.6 = 10.8 km/h (steady-state)
        self.assertAlmostEqual(cand["peak_kmh"], 10.8, delta=0.5)

    def test_no_track_leaves_peak_kmh_none(self):
        positions = [None] * 60
        candidates = [_make_candidate()]

        _recompute_peak_velocities(
            candidates,
            positions=positions,
            fps=30.0,
            scale_factor=0.01,
        )

        self.assertIn("peak_kmh", candidates[0])
        self.assertIsNone(candidates[0]["peak_kmh"])
        # Redundant parameter-echo field must not exist.
        self.assertNotIn("peak_kmh_top_k", candidates[0])

    def test_top_k_is_outlier_robust(self):
        # 1 huge spike inside otherwise-flat track. Top-K mean should not equal max.
        positions = [(float(i * 10), 100.0) for i in range(60)]
        # Inject one outlier displacement at frame 25 -> 26 of 1000 px.
        positions[26] = (positions[25][0] + 1000.0, 100.0)
        candidates = [_make_candidate(contact_frame=20)]

        _recompute_peak_velocities(
            candidates,
            positions=positions,
            fps=30.0,
            scale_factor=0.01,
            post_contact_sec=1.0,
            top_k=5,
        )

        cand = candidates[0]
        self.assertIsNotNone(cand["peak_kmh"])
        self.assertIsNotNone(cand.get("v3_max_kmh_smoothed"))
        # Top-K mean should be well below smoothed-max because the spike is one-off.
        self.assertLess(cand["peak_kmh"], cand["v3_max_kmh_smoothed"])


class TestDetectServeCandidatesV3(unittest.TestCase):
    """V3 detector output stays detector-only and JSON-safe."""

    @patch(
        "serve_analyzer.serve_attempts_v3.detect_onsets",
        return_value=[],
    )
    @patch(
        "serve_analyzer.serve_attempts_v3.extract_motion_cues",
        return_value={},
    )
    @patch("serve_analyzer.serve_attempts_v3._extract_fps", return_value=30.0)
    def test_cached_detection_output_has_no_evaluator_keys(
        self, _mock_fps, _mock_motion, _mock_onsets
    ):
        payload = {
            "candidates": [_make_candidate(contact_frame=30, contact_time_sec=1.0)],
            "raw_positions": [[float(i), float(i)] for i in range(60)],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "detections.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            result = detect_serve_candidates_v3(
                "video.mov",
                expected_serves=1,
                input_detections=str(cache_path),
                use_audio=False,
            )

        self.assertEqual(result["detector"], "v3")
        self.assertEqual(len(result["selected_serves"]), 1)

        candidate = result["selected_serves"][0]
        for key in {"matched", "target_time_sec", "delta_sec", "serve_number"}:
            self.assertNotIn(key, candidate)

        # V3-specific enrichments.
        self.assertIn("v3_contact_score", candidate)
        self.assertIn("peak_kmh", candidate)
        self.assertNotIn("peak_kmh_top_k", candidate)

        # Frame-indexed tracks required by clip_service.
        self.assertIn("positions", result)
        self.assertIn("raw_positions", result)

        # JSON-roundtrip safety (catches numpy scalar leaks).
        json.dumps(result)

    @patch(
        "serve_analyzer.serve_attempts_v3.detect_onsets",
        return_value=[1.0],  # onset right at the candidate's contact_time_sec
    )
    @patch(
        "serve_analyzer.serve_attempts_v3.extract_motion_cues",
        return_value={},
    )
    @patch("serve_analyzer.serve_attempts_v3._extract_fps", return_value=30.0)
    def test_audio_match_emits_bonus_and_delta(
        self, _mock_fps, _mock_motion, _mock_onsets
    ):
        # Track with a velocity spike at frame 30 so refine_candidate_contacts_v3
        # picks frame 30 (matching the audio onset at t=1.0s).
        positions = []
        for i in range(60):
            if i == 0:
                positions.append((0.0, 100.0))
            elif i < 30 or i >= 35:
                positions.append((positions[-1][0] + 5.0, 100.0))
            else:
                # Contact spike: 5x faster for 5 frames after contact.
                positions.append((positions[-1][0] + 25.0, 100.0))
        payload = {
            "candidates": [_make_candidate(contact_frame=30, contact_time_sec=1.0)],
            "raw_positions": [[p[0], p[1]] for p in positions],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "detections.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            result = detect_serve_candidates_v3(
                "video.mov",
                expected_serves=1,
                input_detections=str(cache_path),
                use_audio=True,
                audio_match_tolerance_sec=0.25,
            )

        self.assertEqual(result["v3_audio_onset_count"], 1)
        self.assertEqual(result["v3_audio_matched_serves"], 1)

        candidate = result["selected_serves"][0]
        self.assertIn("v3_audio_match_delta_sec", candidate)
        self.assertIsNotNone(candidate["v3_audio_match_delta_sec"])
        # Onset-based refinement may shift contact frame slightly; delta should be well within audio tolerance.
        self.assertLess(abs(candidate["v3_audio_match_delta_sec"]), 0.10)


if __name__ == "__main__":
    unittest.main()
