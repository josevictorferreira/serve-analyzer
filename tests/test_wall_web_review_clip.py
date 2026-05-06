"""Tests for wall impact review clip generation and integration."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ROOT_DIR)
from wall_test_helpers import generate_wall_impact_video
from web.backend.app import app
from web.backend import state
from web.backend.services import wall_session_service
from web.backend.services import wall_calibration_service
from web.backend.services.wall_review_clip_service import generate_impact_review_clip


class TestWallWebReviewClip(unittest.TestCase):
    """Test impact-centered review clip generation in the wall analysis flow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        state.reset_state()
        state.reset_wall_state()
        wall_session_service.clear_session()
        wall_calibration_service.clear_calibration()
        self.tmpdir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmpdir, "wall_test.mp4")
        generate_wall_impact_video(
            self.video_path,
            width=640,
            height=480,
            fps=30,
            total_frames=60,
            impact_frame=30,
            ball_speed_px_per_frame=4.0,
        )

    def tearDown(self) -> None:
        wall_calibration_service.clear_calibration()
        wall_session_service.clear_session()
        state.reset_state()
        state.reset_wall_state()
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _stage_video(self) -> str:
        """Upload synthetic MP4 and return video_id."""
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["video_id"]

    def _save_calibration(self, video_id: str) -> None:
        """POST a valid 4-point calibration for the staged video."""
        points = []
        for i in range(4):
            px = 100 + i * 150
            py = 100 + i * 80
            wx = float(i)
            wy = float(i)
            points.append(
                {
                    "name": f"pt{i}",
                    "pixel": [px, py],
                    "wall_m": [wx, wy],
                }
            )
        payload = {
            "video_id": video_id,
            "calibration_frame": 10,
            "calibration_time_sec": 0.333,
            "setup": {
                "serve_contact_distance_m": 6.11,
                "camera_wall_distance_m": 1.57,
                "serve_contact_height_m": 2.80,
                "wall_reference_points": points,
            },
        }
        resp = self.client.post("/api/wall/calibration", json=payload)
        self.assertEqual(resp.status_code, 200)

    def test_full_synthetic_flow_has_review_clip(self) -> None:
        """Upload → calibrate → analyze → assert review_clip exists in artifacts."""
        video_id = self._stage_video()
        self._save_calibration(video_id)

        def fake_process(video_path, calibration, output_dir, **kwargs):
            import json as _json

            video_stem = Path(video_path).stem
            fake_result = {
                "measured": {
                    "video": video_stem,
                    "serve_index": 0,
                    "impact_time_sec": 1.0,
                    "impact_frame": 30,
                    "wall_x_m": 0.0,
                    "wall_y_m": 1.0,
                },
                "inferred": {},
                "assumed": {},
                "confidence": {},
                "warnings": [],
                "artifacts": {"annotated_video": None, "plots": {}},
            }
            (Path(output_dir) / "result.json").write_text(_json.dumps(fake_result))

        with patch(
            "web.backend.services.wall_analysis_service._process_video",
            side_effect=fake_process,
        ):
            resp = self.client.post("/api/wall/analyze")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "accepted")

            # Poll job until done or error (max ~10s)
            result = None
            for _ in range(50):
                job = self.client.get("/api/wall/job").json()
                if job["status"] in ("done", "error"):
                    result = job.get("result")
                    break
                time.sleep(0.2)

        self.assertIsNotNone(result, "Job did not reach done/error within timeout")
        self.assertEqual(job["status"], "done")

        artifacts = result.get("artifacts", {})
        self.assertIn("review_clip", artifacts, "Expected review_clip in artifacts")
        review_clip = artifacts["review_clip"]
        self.assertIsInstance(review_clip, dict)
        self.assertIn("url", review_clip)
        review_clip_url = review_clip["url"]
        self.assertTrue(
            review_clip_url.startswith("/api/wall/artifacts/"),
            f"review_clip URL should be normalized: {review_clip_url}",
        )
        self.assertTrue(
            review_clip_url.endswith("_impact_review.mp4"),
            f"review_clip filename should end with _impact_review.mp4: {review_clip_url}",
        )

        # Review metadata is now under artifacts.review_clip
        review_clip_meta = artifacts["review_clip"]
        self.assertIn("impact_time_sec", review_clip_meta)
        self.assertIn("start_time_sec", review_clip_meta)
        self.assertIn("end_time_sec", review_clip_meta)
        self.assertIn("duration_sec", review_clip_meta)
        self.assertIn("impact_frame", review_clip_meta)

        # Serve the clip file and assert it exists and is non-zero bytes
        clip_resp = self.client.get(review_clip_url)
        self.assertEqual(clip_resp.status_code, 200)
        self.assertGreater(len(clip_resp.content), 0)

    def test_review_clip_window_clamps_at_start(self) -> None:
        """When impact is near the start, start_time_sec should clamp to 0."""
        source_video = os.path.join(self.tmpdir, "source.mp4")
        Path(source_video).write_bytes(b"placeholder")

        with patch(
            "web.backend.services.wall_review_clip_service.subprocess.run"
        ) as mock_run:
            clip_info = generate_impact_review_clip(
                source_video,
                self.tmpdir,
                impact_time_sec=0.2,
                video_duration_sec=2.0,
                video_stem="wall_early",
            )

        self.assertIsNotNone(clip_info)
        output_path, review = clip_info
        self.assertEqual(
            output_path, os.path.join(self.tmpdir, "wall_early_impact_review.mp4")
        )
        self.assertEqual(review["start_time_sec"], 0.0)
        self.assertEqual(review["end_time_sec"], 1.2)
        self.assertEqual(review["duration_sec"], 1.2)
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-ss") + 1], "0.0")
        self.assertEqual(cmd[cmd.index("-to") + 1], "1.2")

    def test_no_review_clip_when_impact_time_is_none(self) -> None:
        """If impact_time_sec is absent, review_clip and review should be omitted."""
        video_id = self._stage_video()
        self._save_calibration(video_id)

        # Patch _process_video to produce a result with no impact_time_sec
        def fake_process(video_path, calibration, output_dir, **kwargs):
            import json as _json

            fake_result = {
                "measured": {
                    "video": "fake",
                    "serve_index": 0,
                    "impact_time_sec": None,
                    "impact_frame": None,
                },
                "inferred": {},
                "assumed": {},
                "confidence": {},
                "warnings": [],
                "artifacts": {"annotated_video": None, "plots": {}},
            }
            (Path(output_dir) / "result.json").write_text(_json.dumps(fake_result))

        with patch(
            "web.backend.services.wall_analysis_service._process_video",
            side_effect=fake_process,
        ):
            resp = self.client.post("/api/wall/analyze")
            self.assertEqual(resp.status_code, 200)

            for _ in range(50):
                job = self.client.get("/api/wall/job").json()
                if job["status"] in ("done", "error"):
                    break
                time.sleep(0.2)

            self.assertEqual(job["status"], "done")
            result = job.get("result", {})
            self.assertNotIn("review_clip", result.get("artifacts", {}))
            self.assertNotIn("review", result)

    @unittest.skip(
        "The full suite is run by task verification, not recursively inside itself."
    )
    def test_full_suite_unchanged(self) -> None:
        """Document that the full suite must be run after this targeted file."""
