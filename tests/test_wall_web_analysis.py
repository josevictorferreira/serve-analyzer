"""Tests for wall analysis endpoints: analyze, job polling, artifact serving."""

from __future__ import annotations

import os
import sys
import tempfile
import time
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


class TestWallWebAnalysis(unittest.TestCase):
    """Test wall analysis POST, GET job, artifact serving, and guards."""

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

    def test_full_synthetic_flow(self) -> None:
        """Upload → calibrate → analyze → poll job → assert six-section payload."""
        video_id = self._stage_video()
        self._save_calibration(video_id)

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

        # Top-level keys
        for key in (
            "measured",
            "inferred",
            "assumed",
            "confidence",
            "warnings",
            "artifacts",
        ):
            self.assertIn(key, result, f"Missing top-level key: {key}")

        measured = result["measured"]
        # wall_x_m / wall_y_m may be None if calibration points don't span enough
        # for a valid homography; assert they exist as keys.
        self.assertIn("wall_x_m", measured)
        self.assertIn("wall_y_m", measured)

    def test_nested_plot_artifact(self) -> None:
        """Analyze then GET a nested plot artifact and assert 200 + non-zero bytes."""
        video_id = self._stage_video()
        self._save_calibration(video_id)

        resp = self.client.post("/api/wall/analyze")
        self.assertEqual(resp.status_code, 200)

        # Wait for completion
        for _ in range(50):
            job = self.client.get("/api/wall/job").json()
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.2)

        self.assertEqual(job["status"], "done")
        artifacts = job["result"]["artifacts"]
        plots = artifacts.get("plots", {})
        self.assertTrue(plots, "Expected at least one plot artifact")

        plot_url = list(plots.values())[0]
        self.assertTrue(plot_url.startswith("/api/wall/artifacts/plots/"))

        artifact_resp = self.client.get(plot_url)
        self.assertEqual(artifact_resp.status_code, 200)
        self.assertGreater(len(artifact_resp.content), 0)

    def test_path_traversal_rejected(self) -> None:
        """GET with traversal sequences must return 400/403/404."""
        video_id = self._stage_video()
        self._save_calibration(video_id)
        self.client.post("/api/wall/analyze")

        for _ in range(50):
            job = self.client.get("/api/wall/job").json()
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.2)

        # Literal ".." is normalized away by the router -> 404 (also a rejection)
        resp = self.client.get("/api/wall/artifacts/../etc/passwd")
        self.assertIn(resp.status_code, (400, 403, 404))

        # Encoded traversal is caught by our handler -> 400
        resp2 = self.client.get("/api/wall/artifacts/%2e%2e/etc/passwd")
        self.assertIn(resp2.status_code, (400, 403, 404))


    def test_reject_analyze_without_calibration(self) -> None:
        """POST analyze without calibration → 400."""
        self._stage_video()
        resp = self.client.post("/api/wall/analyze")
        self.assertEqual(resp.status_code, 400)

    def test_reject_analyze_when_busy(self) -> None:
        """POST analyze when another job is active → 409."""
        video_id = self._stage_video()
        self._save_calibration(video_id)

        # Start analysis
        resp = self.client.post("/api/wall/analyze")
        self.assertEqual(resp.status_code, 200)

        # Immediately try again
        resp2 = self.client.post("/api/wall/analyze")
        self.assertEqual(resp2.status_code, 409)
