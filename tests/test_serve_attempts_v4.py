"""Tests for serve_attempts_v4."""

import json
import tempfile
import unittest
from pathlib import Path

from serve_analyzer.serve_attempts_v4 import (
    _find_direction_change,
    _refine_contact_to_apex,
    detect_serve_candidates_v4,
)


def _make_candidate(
    contact_frame: int = 100, contact_time_sec: float = 5.0, **kwargs
) -> dict:
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


class TestDirectionChangeDetection(unittest.TestCase):
    """Test _find_direction_change."""

    def test_finds_apex_in_toss(self) -> None:
        positions: list = []
        # Simulate toss: ball moves up (y decreasing) then down (y increasing)
        # Frames 0-10: going up, frame 10: apex, frames 11-20: going down
        for i in range(21):
            if i <= 10:
                y = 300 - i * 10  # going up
            else:
                y = 200 + (i - 10) * 15  # going down
            positions.append((200.0, y))

        apex = _find_direction_change(positions, contact_frame=20, search_backward=20)
        self.assertLessEqual(apex, 12)  # Should be near frame 10

    def test_returns_contact_frame_when_no_apex(self) -> None:
        positions: list = [(100.0 + i * 5, 200.0) for i in range(30)]
        apex = _find_direction_change(positions, contact_frame=25, search_backward=20)
        self.assertLessEqual(apex, 25)


class TestRefineContactToApex(unittest.TestCase):
    """Test _refine_contact_to_apex."""

    def test_refines_contact_frame(self) -> None:
        positions: list = []
        for i in range(50):
            if i <= 20:
                y = 400 - i * 8
            else:
                y = 240 + (i - 20) * 12
            positions.append((250.0, y))

        candidates = [_make_candidate(contact_frame=40, contact_time_sec=40 / 30.0)]
        refined = _refine_contact_to_apex(
            candidates, positions, fps=30.0, search_backward_frames=30
        )

        self.assertEqual(len(refined), 1)
        self.assertIn("v4_original_contact_frame", refined[0])
        self.assertIn("v4_apex_frame", refined[0])
        self.assertIn("v4_refined_frame_delta", refined[0])

    def test_empty_candidates_returns_empty(self) -> None:
        result = _refine_contact_to_apex([], [], fps=30.0)
        self.assertEqual(result, [])


class TestDetectServeCandidatesV4(unittest.TestCase):
    """Test detect_serve_candidates_v4 output shape."""

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
            result = detect_serve_candidates_v4(
                "video.mov",
                expected_serves=1,
                input_detections=str(cache_path),
                use_audio=False,
            )

        output_keys = set()
        for candidate in result["selected_serves"]:
            output_keys.update(candidate.keys())

        leaked = output_keys & evaluator_keys
        self.assertEqual(leaked, set(), f"Evaluator keys leaked: {leaked}")


if __name__ == "__main__":
    unittest.main()
