"""Tests for serve_attempts_v5."""

import json
import tempfile
import unittest
from pathlib import Path

from serve_analyzer.serve_attempts_v5 import (
    _adaptive_apex_offset,
    _quality_gate_candidates,
    _refine_contact_hybrid,
    detect_serve_candidates_v5,
)


def _make_candidate(contact_frame: int = 100, contact_time_sec: float = 5.0, **kwargs) -> dict:
    base = {
        "candidate_index": 0,
        "contact_frame": contact_frame,
        "contact_time_sec": contact_time_sec,
        "score": 500.0,
        "post_contact_max_kmh": 100.0,
        "post_contact_mean_kmh": 80.0,
        "post_contact_max_mps": 27.8,
        "post_contact_mean_mps": 22.2,
        "support_count": 3,
        "contact_velocity": 200.0,
        "upward_fraction": 0.6,
        "recent_upward_fraction": 0.7,
        "drop_after_apex": 200.0,
        "frames_after_apex": 10,
        "rightward_fraction": 0.8,
        "net_rightward_displacement": 300.0,
        "direction_unreliable": False,
        "toss_rise_px": 150.0,
        "toss_duration_frames": 20,
        "early_post_downward_fraction": 0.9,
        "early_post_net_dy": 50.0,
        "selector_rank": 0.8,
    }
    base.update(kwargs)
    return base


class TestAdaptiveApexOffset(unittest.TestCase):
    def test_low_toss_returns_min_offset(self) -> None:
        self.assertEqual(_adaptive_apex_offset(10.0), 2)

    def test_medium_toss_returns_scaled_offset(self) -> None:
        self.assertEqual(_adaptive_apex_offset(200.0), 10)

    def test_high_toss_capped_at_max(self) -> None:
        self.assertEqual(_adaptive_apex_offset(400.0), 15)


class TestQualityGate(unittest.TestCase):
    def test_filters_low_speed_candidates(self) -> None:
        candidates = [
            _make_candidate(contact_frame=100, post_contact_max_kmh=50.0),
            _make_candidate(contact_frame=200, post_contact_max_kmh=20.0),
        ]
        result = _quality_gate_candidates(candidates, min_post_contact_kmh=30.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["contact_frame"], 100)

    def test_filters_low_rightward_candidates(self) -> None:
        candidates = [
            _make_candidate(contact_frame=100, rightward_fraction=0.5),
            _make_candidate(contact_frame=200, rightward_fraction=0.1),
        ]
        result = _quality_gate_candidates(candidates, min_rightward_fraction=0.2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["contact_frame"], 100)

    def test_passes_qualified_candidates(self) -> None:
        candidates = [
            _make_candidate(contact_frame=100),
            _make_candidate(contact_frame=200),
        ]
        result = _quality_gate_candidates(candidates)
        self.assertEqual(len(result), 2)

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(_quality_gate_candidates([]), [])


class TestRefineContactHybrid(unittest.TestCase):
    def test_keeps_v1_frame_when_close_to_apex(self) -> None:
        positions: list = []
        for i in range(50):
            if i <= 20:
                y = 400 - i * 8
            else:
                y = 240 + (i - 20) * 12
            positions.append((250.0, y))

        candidates = [_make_candidate(contact_frame=25, contact_time_sec=25 / 30.0)]
        refined = _refine_contact_hybrid(candidates, positions, fps=30.0, search_backward_frames=30, max_backward_shift=10)

        self.assertEqual(len(refined), 1)
        self.assertIn("v5_refined_frame_delta", refined[0])

    def test_caps_excessive_backward_shift(self) -> None:
        positions: list = []
        for i in range(100):
            if i <= 30:
                y = 500 - i * 10
            else:
                y = 200 + (i - 30) * 15
            positions.append((250.0, y))

        candidates = [_make_candidate(contact_frame=90, contact_time_sec=90 / 30.0, toss_rise_px=300.0)]
        refined = _refine_contact_hybrid(candidates, positions, fps=30.0, search_backward_frames=60, max_backward_shift=10)

        self.assertEqual(len(refined), 1)
        apex = refined[0]["v5_apex_frame"]
        refined_frame = refined[0]["contact_frame"]
        self.assertLessEqual(refined_frame - apex, 10)

    def test_empty_candidates_returns_empty(self) -> None:
        result = _refine_contact_hybrid([], [], fps=30.0)
        self.assertEqual(result, [])


class TestDetectServeCandidatesV5(unittest.TestCase):
    def test_cached_detection_output_has_no_evaluator_keys(self) -> None:
        evaluator_keys = {"matched", "target_time_sec", "delta_sec", "serve_number"}

        positions: list = []
        for i in range(100):
            if i <= 30:
                y = 300 - i * 5
            else:
                y = 150 + (i - 30) * 8
            positions.append((200.0, y))

        payload = {
            "candidates": [_make_candidate(contact_frame=50, contact_time_sec=1.67)],
            "raw_positions": [[p[0], p[1]] if p else None for p in positions],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "detections.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            result = detect_serve_candidates_v5(
                "video.mov",
                expected_serves=1,
                input_detections=str(cache_path),
            )

        output_keys = set()
        for candidate in result["selected_serves"]:
            output_keys.update(candidate.keys())

        leaked = output_keys & evaluator_keys
        self.assertEqual(leaked, set(), f"Evaluator keys leaked: {leaked}")

    def test_v5_detector_label(self) -> None:
        positions: list = [(200.0, 200.0) for _ in range(50)]
        payload = {
            "candidates": [_make_candidate(contact_frame=30, contact_time_sec=1.0)],
            "raw_positions": [[p[0], p[1]] if p else None for p in positions],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "detections.json"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            result = detect_serve_candidates_v5(
                "video.mov",
                expected_serves=1,
                input_detections=str(cache_path),
            )

        self.assertEqual(result["detector"], "v5")


if __name__ == "__main__":
    unittest.main()
