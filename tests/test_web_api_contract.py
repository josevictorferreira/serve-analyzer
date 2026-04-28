"""Contract tests for web backend API."""

import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web.backend.app import app
from web.backend.paths import get_annotation_session_dir
from web.backend.state import reset_state


class TestWebApiContract(unittest.TestCase):
    """Verify API endpoints match expected contracts."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        reset_state()

    def test_get_job_idle(self) -> None:
        """GET /api/job returns idle state with correct shape."""
        response = self.client.get("/api/job")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "idle")
        self.assertIsNone(data["phase"])
        self.assertIsNone(data["error"])
        self.assertEqual(data["clips"], [])
        self.assertEqual(data["selected_serves"], [])
        self.assertEqual(data["candidates"], [])
        self.assertIsNone(data["count_inferred"])
        self.assertIsNone(data["inferred_count"])

    def test_post_analyze_conflict(self) -> None:
        """Concurrent POST /api/analyze returns 409."""
        # Mock the analysis thread target to prevent state from changing.
        # This ensures the second POST deterministically sees ANALYZING state.
        with (
            patch("web.backend.app._run_analysis_thread"),
            patch(
                "web.backend.services.analysis_service.estimate_analysis_duration",
                return_value=1.0,
            ),
        ):
            fake_video = io.BytesIO(b"fake video data")
            response1 = self.client.post(
                "/api/analyze",
                files={"video": ("test.mp4", fake_video, "video/mp4")},
            )
            self.assertEqual(response1.status_code, 202)

            fake_video2 = io.BytesIO(b"fake video data 2")
            response2 = self.client.post(
                "/api/analyze",
                files={"video": ("test2.mp4", fake_video2, "video/mp4")},
            )
            self.assertEqual(response2.status_code, 409)

    def test_post_reset(self) -> None:
        """POST /api/job/reset returns to idle and cleans artifacts."""
        with (
            patch("web.backend.app._run_analysis_thread"),
            patch(
                "web.backend.services.analysis_service.estimate_analysis_duration",
                return_value=1.0,
            ),
        ):
            fake_video = io.BytesIO(b"fake video data")
            self.client.post(
                "/api/analyze",
                files={"video": ("test.mp4", fake_video, "video/mp4")},
            )

        response = self.client.post("/api/job/reset")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "reset")

        job = self.client.get("/api/job").json()
        self.assertEqual(job["status"], "idle")

    def test_post_reset_preserves_annotation_sessions(self) -> None:
        """POST /api/job/reset does not delete annotation artifacts."""
        with (
            tempfile.TemporaryDirectory() as analysis_tmp,
            patch.dict(
                os.environ,
                {"SERVE_ANALYZER_TEMP": analysis_tmp},
                clear=False,
            ),
        ):
            os.environ.pop("SERVE_ANALYZER_ANNOTATION_DIR", None)
            session_dir = get_annotation_session_dir("persist")
            marker_path = os.path.join(session_dir, "session.json")
            with open(marker_path, "w", encoding="utf-8") as marker:
                marker.write("{}")

            response = self.client.post("/api/job/reset")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(os.path.isfile(marker_path))
            job = self.client.get("/api/job").json()
            self.assertEqual(job["status"], "idle")
            shutil.rmtree(session_dir, ignore_errors=True)

    def test_post_analyze_rejects_non_video(self) -> None:
        """POST /api/analyze with a text file returns 400."""
        fake_text = io.BytesIO(b"not a video")
        response = self.client.post(
            "/api/analyze",
            files={"video": ("test.txt", fake_text, "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    @patch("web.backend.app.annotation_service.create_annotation_session")
    def test_post_annotation_session_returns_session_without_mutating_job(
        self, mock_create
    ) -> None:
        """POST /api/annotation/sessions creates a separate annotation session."""
        mock_create.return_value = {"id": "abc123", "frames": [], "progress": {}}

        fake_video = io.BytesIO(b"fake video data")
        response = self.client.post(
            "/api/annotation/sessions?max_frames=12&prelabel=false",
            files={"video": ("serve.mp4", fake_video, "video/mp4")},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["session"]["id"], "abc123")
        _, _, kwargs = mock_create.mock_calls[0]
        self.assertEqual(kwargs["max_frames"], 12)
        self.assertFalse(kwargs["prelabel"])
        job = self.client.get("/api/job").json()
        self.assertEqual(job["status"], "idle")

    def test_get_clip_path_traversal(self) -> None:
        """GET /clips with path traversal returns 400."""
        response = self.client.get("/clips/passwd..txt")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
