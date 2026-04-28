"""Tests for the web backend clip service.

Includes mock-based unit tests for window calculation, naming, and URL
formatting, plus an integration test that generates a real tiny video with
ffmpeg and verifies clip extraction.
"""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from web.backend.services import clip_service


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in for generate_clips tests."""

    def __init__(self, path):
        self.path = path
        self.position = 0
        self.set_calls = []
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 900.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480.0
        return 0.0

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)

    def read(self):
        self.position += 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class _FakeWriter:
    """Minimal cv2.VideoWriter stand-in for generate_clips tests."""

    instances = []

    def __init__(self, path, fourcc, fps, size):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.write_count = 0
        self.released = False
        self.instances.append(self)

    def write(self, frame):
        self.write_count += 1

    def release(self):
        self.released = True


class TestClipServiceUnit(unittest.TestCase):
    """Mock-based tests for generate_clips logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="serve_analyzer_clip_unit_")
        self.env_patcher = patch.dict(
            os.environ,
            {"SERVE_ANALYZER_TEMP": self.temp_dir},
        )
        self.env_patcher.start()
        _FakeWriter.instances = []

    def tearDown(self):
        self.env_patcher.stop()
        if os.path.isdir(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def _run_with_fakes(self, serves):
        fake_capture = _FakeCapture("/tmp/video.mp4")
        positions = [(320.0, 240.0)] * 900
        with (
            patch(
                "web.backend.services.clip_service.cv2.VideoCapture",
                return_value=fake_capture,
            ),
            patch("web.backend.services.clip_service.cv2.VideoWriter", _FakeWriter),
            patch(
                "web.backend.services.clip_service.cv2.VideoWriter_fourcc",
                return_value=1,
            ),
            patch("web.backend.services.clip_service.subprocess.run"),
            patch("web.backend.services.clip_service.os.rename"),
            patch("web.backend.services.clip_service.os.remove"),
        ):
            result = clip_service.generate_clips(
                "/tmp/video.mp4",
                serves,
                positions,
                detection_frame_skip=1,
            )
        return result, fake_capture

    def test_clip_naming_and_url(self):
        """Files are named serve-01.mp4, serve-02.mp4 with matching url_path."""
        serves = [
            {"contact_frame": 300, "contact_time_sec": 10.0},
            {"contact_frame": 765, "contact_time_sec": 25.5},
        ]
        result, _ = self._run_with_fakes(serves)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["filename"], "serve-01.mp4")
        self.assertEqual(result[0]["url_path"], "/clips/serve-01.mp4")
        self.assertEqual(result[1]["filename"], "serve-02.mp4")
        self.assertEqual(result[1]["url_path"], "/clips/serve-02.mp4")

    def test_window_calculation(self):
        """Clip window is max(contact - 2.25, 0) to contact + 1.75."""
        serves = [
            {"contact_frame": 300, "contact_time_sec": 10.0},
            {"contact_frame": 30, "contact_time_sec": 1.0},
        ]
        _, fake_capture = self._run_with_fakes(serves)

        start_frames = [value for prop, value in fake_capture.set_calls]
        # 2.25s * 30fps truncates to 67 frames; 1.75s truncates to 52.
        self.assertEqual(start_frames, [233, 0])

    def test_duration_field(self):
        """Duration equals end - start."""
        serves = [{"contact_frame": 150, "contact_time_sec": 5.0}]
        result, _ = self._run_with_fakes(serves)
        # start = 83, end = 202, duration = 119/30 = 3.967s.
        self.assertEqual(result[0]["duration"], 3.967)

    def test_serve_index_field(self):
        """serve_index is 1-indexed."""
        serves = [
            {"contact_frame": 30, "contact_time_sec": 1.0},
            {"contact_frame": 60, "contact_time_sec": 2.0},
        ]
        result, _ = self._run_with_fakes(serves)
        self.assertEqual(result[0]["serve_index"], 1)
        self.assertEqual(result[1]["serve_index"], 2)

    def test_empty_selected_serves(self):
        """Empty input returns empty metadata list, no writer work."""
        result, fake_capture = self._run_with_fakes([])
        self.assertEqual(result, [])
        self.assertEqual(_FakeWriter.instances, [])
        self.assertTrue(fake_capture.released)

    def test_contact_time_passed_through(self):
        """contact_time_sec is preserved in metadata."""
        serves = [{"contact_frame": 800, "contact_time_sec": 42.5}]
        result, _ = self._run_with_fakes(serves)
        self.assertEqual(result[0]["contact_time_sec"], 42.5)


class TestClipServiceIntegration(unittest.TestCase):
    """Integration test with a real tiny video generated by ffmpeg."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="serve_analyzer_test_")
        self.video_path = os.path.join(self.temp_dir, "test_video.mp4")
        # Generate a 5-second 640x480 test video at 30 fps
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=5:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
                self.video_path,
            ],
            capture_output=True,
            check=True,
        )
        os.environ["SERVE_ANALYZER_TEMP"] = self.temp_dir

    def tearDown(self):
        if os.path.isfile(self.video_path):
            os.remove(self.video_path)
        if os.path.isdir(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)
        os.environ.pop("SERVE_ANALYZER_TEMP", None)

    def test_generate_clips_creates_mp4_files(self):
        """Clips are extracted, scaled, and written as valid MP4 files."""
        serves = [
            {"contact_frame": 60, "contact_time_sec": 2.0},
            {"contact_frame": 120, "contact_time_sec": 4.0},
        ]
        positions = [(320.0, 240.0)] * 150
        result = clip_service.generate_clips(
            self.video_path,
            serves,
            positions,
            detection_frame_skip=1,
        )

        self.assertEqual(len(result), 2)
        for meta in result:
            path = os.path.join(self.temp_dir, "clips", meta["filename"])
            self.assertTrue(
                os.path.isfile(path),
                f"Expected clip file {path} to exist",
            )
            self.assertGreater(
                os.path.getsize(path),
                0,
                f"Clip file {path} should not be empty",
            )

    def test_cleanup_clips_removes_files(self):
        """cleanup_clips removes all files from the clips directory."""
        serves = [{"contact_frame": 60, "contact_time_sec": 2.0}]
        positions = [(320.0, 240.0)] * 150
        clip_service.generate_clips(
            self.video_path,
            serves,
            positions,
            detection_frame_skip=1,
        )
        clip_service.cleanup_clips()
        clips_dir = os.path.join(self.temp_dir, "clips")
        if os.path.isdir(clips_dir):
            self.assertEqual(os.listdir(clips_dir), [])


if __name__ == "__main__":
    unittest.main()
