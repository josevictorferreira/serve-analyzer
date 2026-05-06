"""End-to-end regression coverage for the wall web analysis flow."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from typing import Any

from fastapi.testclient import TestClient

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, ROOT_DIR)
from wall_test_helpers import generate_wall_impact_video
from web.backend import state
from web.backend.app import app
from web.backend.services import wall_calibration_service, wall_session_service


class TestWallWebE2E(unittest.TestCase):
    """Exercise upload, calibration, analysis, artifacts, and reset together."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        state.reset_state()
        state.reset_wall_state()
        wall_session_service.clear_session()
        wall_calibration_service.clear_calibration()

        self.tmpdir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.tmpdir, "wall_e2e.mp4")
        self.metadata_path = os.path.join(self.tmpdir, "wall_e2e_metadata.json")
        self.ground_truth = generate_wall_impact_video(
            self.video_path,
            width=640,
            height=480,
            fps=30,
            total_frames=60,
            impact_frame=30,
            ball_speed_px_per_frame=8.0,
            wall_x_px=540,
        )
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fps": self.ground_truth.fps,
                    "total_frames": self.ground_truth.total_frames,
                    "impact_frame": self.ground_truth.impact_frame,
                    "impact_pixel": list(self.ground_truth.impact_pixel),
                },
                f,
            )

    def tearDown(self) -> None:
        wall_calibration_service.clear_calibration()
        wall_session_service.clear_session()
        state.reset_state()
        state.reset_wall_state()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_wall_web_flow(self) -> None:
        """Upload → calibrate → analyze → poll → artifacts → reset."""
        video_id = self._upload_video()
        self._assert_video_metadata(video_id)
        self._post_calibration(video_id)

        persisted = self.client.get("/api/wall/calibration")
        self.assertEqual(persisted.status_code, 200)
        self.assertEqual(persisted.json()["video_id"], video_id)

        analyze = self.client.post("/api/wall/analyze")
        self.assertIn(analyze.status_code, (200, 202))
        self.assertEqual(analyze.json()["status"], "accepted")

        job = self._poll_job_until_terminal()
        self.assertEqual(job["status"], "done", job.get("error"))
        result = job["result"]
        self._assert_result_contract(result)
        self._assert_artifacts(result["artifacts"])

        reset = self.client.post("/api/wall/job/reset")
        self.assertEqual(reset.status_code, 200)

        calibration_after_reset = self.client.get("/api/wall/calibration")
        self.assertEqual(calibration_after_reset.status_code, 200)
        self.assertEqual(calibration_after_reset.json()["video_id"], video_id)

    def _upload_video(self) -> str:
        """Upload the synthetic video and return the staged video id."""
        with open(self.video_path, "rb") as f:
            response = self.client.post(
                "/api/wall/video",
                files={"video": ("wall_e2e.mp4", f, "video/mp4")},
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("video_id", data)
        return data["video_id"]

    def _assert_video_metadata(self, video_id: str) -> None:
        """Assert staged-video metadata matches the generated fixture."""
        response = self.client.get(f"/api/wall/video/{video_id}/metadata")
        self.assertEqual(response.status_code, 200)
        metadata = response.json()
        self.assertEqual(metadata["fps"], self.ground_truth.fps)
        self.assertEqual(metadata["frame_count"], self.ground_truth.total_frames)
        self.assertEqual(metadata["width"], 640)
        self.assertEqual(metadata["height"], 480)

    def _post_calibration(self, video_id: str) -> None:
        """Save a valid rectangular four-point wall calibration."""
        points = [
            (100, 100, 0.0, 0.0),
            (540, 100, 4.0, 0.0),
            (540, 380, 4.0, 2.0),
            (100, 380, 0.0, 2.0),
        ]
        payload = {
            "video_id": video_id,
            "calibration_frame": 0,
            "calibration_time_sec": 0.0,
            "setup": {
                "serve_contact_distance_m": 6.11,
                "camera_wall_distance_m": 1.57,
                "serve_contact_height_m": 2.80,
                "wall_reference_points": [
                    {
                        "name": f"corner_{idx}",
                        "pixel": [pixel_x, pixel_y],
                        "wall_m": [wall_x_m, wall_y_m],
                    }
                    for idx, (pixel_x, pixel_y, wall_x_m, wall_y_m) in enumerate(points)
                ],
            },
        }
        response = self.client.post("/api/wall/calibration", json=payload)
        self.assertEqual(response.status_code, 200)

    def _poll_job_until_terminal(self) -> dict[str, Any]:
        """Poll the wall job endpoint for up to 30 seconds."""
        deadline = time.time() + 30.0
        last_job: dict[str, Any] = {}
        while time.time() < deadline:
            response = self.client.get("/api/wall/job")
            self.assertEqual(response.status_code, 200)
            last_job = response.json()
            if last_job["status"] in ("done", "error"):
                return last_job
            time.sleep(0.2)
        self.fail(f"Wall analysis did not finish within 30 seconds: {last_job}")

    def _assert_result_contract(self, result: dict[str, Any]) -> None:
        """Assert required top-level sections and core numeric outputs."""
        for key in (
            "measured",
            "inferred",
            "assumed",
            "confidence",
            "warnings",
            "artifacts",
        ):
            self.assertIn(key, result)

        measured = result["measured"]
        self.assertIsInstance(measured["wall_x_m"], (int, float))
        self.assertIsInstance(measured["wall_y_m"], (int, float))

        speed_km_h = result["inferred"]["speed_km_h"]
        self.assertIsInstance(speed_km_h, (int, float))
        self.assertGreater(speed_km_h, 0)

    def _assert_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Assert artifact URLs are relative and fetchable."""
        artifact_urls = self._collect_artifact_urls(artifacts)
        for url in artifact_urls:
            self.assertTrue(url.startswith("/api/wall/artifacts/"), url)
            self.assertNotIn("://", url)
            self.assertFalse(os.path.isabs(url.removeprefix("/api/wall/artifacts/")))

        result_json = self.client.get("/api/wall/artifacts/result.json")
        self.assertEqual(result_json.status_code, 200)

        result_csv = self.client.get("/api/wall/artifacts/result.csv")
        self.assertEqual(result_csv.status_code, 200)

        for plot_url in artifacts.get("plots", {}).values():
            plot = self.client.get(plot_url)
            self.assertEqual(plot.status_code, 200)
            self.assertTrue(plot.headers["content-type"].startswith("image/png"))

    def _collect_artifact_urls(self, artifacts: dict[str, Any]) -> list[str]:
        """Flatten artifact URL strings from the result artifact section."""
        urls: list[str] = []
        for value in artifacts.values():
            if isinstance(value, str):
                urls.append(value)
            elif isinstance(value, dict):
                for nested_value in value.values():
                    if isinstance(nested_value, str):
                        urls.append(nested_value)
                    elif isinstance(nested_value, dict):
                        nested_url = nested_value.get("url")
                        if isinstance(nested_url, str):
                            urls.append(nested_url)
        return urls


if __name__ == "__main__":
    unittest.main()
