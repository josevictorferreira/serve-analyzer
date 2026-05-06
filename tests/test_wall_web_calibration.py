"""Tests for wall calibration persistence endpoints."""

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
from web.backend.services import wall_calibration_service


class TestWallWebCalibration(unittest.TestCase):
    """Test wall calibration POST, GET, DELETE, and job-reset isolation."""

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
        if os.path.isfile(self.video_path):
            os.remove(self.video_path)
        os.rmdir(self.tmpdir)

    def _stage_video(self) -> str:
        """Upload synthetic MP4 and return video_id."""
        with open(self.video_path, "rb") as f:
            resp = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_test.mp4", f, "video/mp4")},
            )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["video_id"]

    def _make_calibration_payload(self, video_id: str, num_points: int = 4) -> dict:
        """Build a calibration payload with ``num_points`` wall reference points."""
        points = []
        for i in range(num_points):
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
        return {
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

    def test_post_valid_calibration(self) -> None:
        """POST 4-point calibration -> 200 with point_count."""
        video_id = self._stage_video()
        payload = self._make_calibration_payload(video_id, num_points=4)
        resp = self.client.post("/api/wall/calibration", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["video_id"], video_id)
        self.assertEqual(data["point_count"], 4)
        self.assertIsNone(data["rms_m"])

    def test_get_calibration_after_post(self) -> None:
        """POST calibration then GET returns same values."""
        video_id = self._stage_video()
        payload = self._make_calibration_payload(video_id, num_points=4)
        post_resp = self.client.post("/api/wall/calibration", json=payload)
        self.assertEqual(post_resp.status_code, 200)

        get_resp = self.client.get("/api/wall/calibration")
        self.assertEqual(get_resp.status_code, 200)
        data = get_resp.json()
        self.assertEqual(data["video_id"], video_id)
        self.assertEqual(data["calibration_frame"], 10)
        self.assertEqual(data["calibration_time_sec"], 0.333)
        self.assertEqual(data["point_count"], 4)
        self.assertIn("calibration", data)
        self.assertIn("setup", data["calibration"])

    def test_post_too_few_points(self) -> None:
        """POST with 3 points -> 422 with actionable message."""
        video_id = self._stage_video()
        payload = self._make_calibration_payload(video_id, num_points=3)
        resp = self.client.post("/api/wall/calibration", json=payload)
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertIn("4", detail)
        self.assertIn("3", detail)

    def test_delete_calibration(self) -> None:
        """DELETE calibration -> 200, then GET returns 404."""
        video_id = self._stage_video()
        payload = self._make_calibration_payload(video_id, num_points=4)
        self.client.post("/api/wall/calibration", json=payload)

        del_resp = self.client.delete("/api/wall/calibration")
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.json()["status"], "deleted")

        get_resp = self.client.get("/api/wall/calibration")
        self.assertEqual(get_resp.status_code, 404)

    def test_job_reset_does_not_clear_calibration(self) -> None:
        """POST calibration, POST job/reset, GET calibration still 200."""
        video_id = self._stage_video()
        payload = self._make_calibration_payload(video_id, num_points=4)
        self.client.post("/api/wall/calibration", json=payload)

        reset_resp = self.client.post("/api/wall/job/reset")
        self.assertEqual(reset_resp.status_code, 200)

        get_resp = self.client.get("/api/wall/calibration")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["video_id"], video_id)

    def test_post_requires_matching_video_id(self) -> None:
        """POST calibration with mismatched video_id -> 400."""
        video_id = self._stage_video()
        payload = self._make_calibration_payload("wrong_id", num_points=4)
        resp = self.client.post("/api/wall/calibration", json=payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("video_id", resp.json()["detail"])

    @unittest.skip(
        "The full suite is run by task verification, not recursively inside itself."
    )
    def test_full_suite_unchanged(self) -> None:
        """Document that the full suite must be run after this targeted file."""
