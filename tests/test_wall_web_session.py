"""Tests for wall video session endpoints and global busy guard."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ROOT_DIR)
from wall_test_helpers import generate_wall_impact_video
from web.backend.app import app
from web.backend import state
from web.backend.services import wall_session_service


class TestWallWebSession(unittest.TestCase):
    """Test wall video staging, metadata, and global busy guard."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        # Reset all state before each test
        state.reset_state()
        state.reset_wall_state()
        wall_session_service.clear_session()
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
        wall_session_service.clear_session()
        state.reset_state()
        state.reset_wall_state()
        if os.path.isfile(self.video_path):
            os.remove(self.video_path)
        os.rmdir(self.tmpdir)

    def test_upload_wall_video(self) -> None:
        """Upload synthetic MP4 and assert 200 with metadata fields."""
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("video_id", data)
        self.assertIn("video_url", data)
        self.assertEqual(data["filename"], "wall_test.mp4")
        self.assertGreater(data["duration_sec"], 0)
        self.assertGreater(data["fps"], 0)
        self.assertGreater(data["frame_count"], 0)
        self.assertGreater(data["width"], 0)
        self.assertGreater(data["height"], 0)

    def test_get_staged_video(self) -> None:
        """Upload then GET video_url and assert 200 with video content-type."""
        with open(self.video_path, "rb") as f:
            upload_resp = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(upload_resp.status_code, 200)
        video_url = upload_resp.json()["video_url"]
        resp = self.client.get(video_url)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("video/"))
        self.assertGreater(len(resp.content), 0)

    def test_get_video_metadata(self) -> None:
        """Upload then GET metadata endpoint and assert numeric fields."""
        with open(self.video_path, "rb") as f:
            upload_resp = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(upload_resp.status_code, 200)
        video_id = upload_resp.json()["video_id"]
        resp = self.client.get(f"/api/wall/video/{video_id}/metadata")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["fps"], 0)
        self.assertGreater(data["frame_count"], 0)
        self.assertGreater(data["width"], 0)
        self.assertGreater(data["height"], 0)

    def test_reject_invalid_file_type(self) -> None:
        """Upload .txt and assert 400."""
        resp = self.client.post(
            "/api/wall/video",
            files={"video": ("bad.txt", b"not a video", "text/plain")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_reject_unknown_video_id(self) -> None:
        """GET unknown video_id and assert 404."""
        resp = self.client.get("/api/wall/video/unknown_id")
        self.assertEqual(resp.status_code, 404)

    def test_job_reset_clears_video(self) -> None:
        """Upload, reset, then GET old video_id and assert 404."""
        with open(self.video_path, "rb") as f:
            upload_resp = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(upload_resp.status_code, 200)
        video_id = upload_resp.json()["video_id"]
        reset_resp = self.client.post("/api/wall/job/reset")
        self.assertEqual(reset_resp.status_code, 200)
        resp = self.client.get(f"/api/wall/video/{video_id}")
        self.assertEqual(resp.status_code, 404)

    def test_global_busy_guard_wall_blocks_serve(self) -> None:
        """Set wall state active, POST /api/analyze, assert 409."""
        state.set_wall_state({"status": state.WallJobPhase.ANALYZING})
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/analyze",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(resp.status_code, 409)
        state.reset_wall_state()

    def test_global_busy_guard_serve_blocks_wall(self) -> None:
        """Set normal serve state active, POST /api/wall/video, assert 409."""
        state.set_state({"status": state.JobPhase.ANALYZING})
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(resp.status_code, 409)
        state.reset_state()

    @unittest.skip(
        "The full suite is run by task verification, not recursively inside itself."
    )
    def test_full_suite_unchanged(self) -> None:
        """Document that the full suite must be run after this targeted file."""
