"""Tests for the web backend analysis service adapter.

These tests verify that the adapter correctly wraps the detector stack,
normalizes the result payload, and respects expected_serves semantics.
"""

import unittest
from unittest.mock import patch

from web.backend.services.analysis_service import run_analysis


class TestAnalysisServiceShape(unittest.TestCase):
    """Adapter returns correct keys and excludes evaluator-only keys."""

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_result_has_required_keys(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = [
            {"contact_time_sec": 10.0, "score": 0.9},
        ]
        mock_select.return_value = [
            {"contact_time_sec": 10.0, "score": 0.9, "selector_rank": 0.5},
        ]

        result = run_analysis("/tmp/video.mov")

        required_keys = {
            "video_path",
            "expected_serves",
            "count_inferred",
            "inferred_count",
            "selected_serves",
            "candidates",
        }
        self.assertTrue(
            required_keys.issubset(result.keys()),
            f"Missing keys: {required_keys - result.keys()}",
        )

        evaluator_keys = {
            "matched",
            "target_time_sec",
            "delta_sec",
            "serve_number",
        }
        for key in evaluator_keys:
            self.assertNotIn(key, result, f"Evaluator key '{key}' leaked into result")

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_empty_candidates_shape(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = []
        mock_select.return_value = []

        result = run_analysis("/tmp/video.mov")

        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["selected_serves"], [])
        self.assertTrue(result["count_inferred"])
        self.assertEqual(result["inferred_count"], 0)


class TestAnalysisServiceCountInference(unittest.TestCase):
    """count_inferred reflects expected_serves=None vs explicit int."""

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_none_means_inferred(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = [
            {"contact_time_sec": 10.0, "score": 0.9},
        ]
        mock_select.return_value = [
            {"contact_time_sec": 10.0, "score": 0.9, "selector_rank": 0.5},
        ]

        result = run_analysis("/tmp/video.mov", expected_serves=None)
        self.assertTrue(result["count_inferred"])
        self.assertEqual(result["inferred_count"], 1)

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_explicit_int_not_inferred(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = [
            {"contact_time_sec": 10.0, "score": 0.9},
            {"contact_time_sec": 30.0, "score": 0.8},
        ]
        mock_select.return_value = [
            {"contact_time_sec": 10.0, "score": 0.9, "selector_rank": 0.5},
        ]

        result = run_analysis("/tmp/video.mov", expected_serves=1)
        self.assertFalse(result["count_inferred"])
        self.assertIsNone(result["inferred_count"])

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_pool_size_passes_none_in_autonomous_mode(
        self, mock_detect, mock_select, mock_info
    ):
        """When expected_serves=None, detect_serve_candidates receives None."""
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = []
        mock_select.return_value = []

        run_analysis("/tmp/video.mov", expected_serves=None)
        mock_detect.assert_called_once()
        _, kwargs = mock_detect.call_args
        self.assertIsNone(kwargs["expected_serves"])

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_pool_size_passed_through_when_explicit(
        self, mock_detect, mock_select, mock_info
    ):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = []
        mock_select.return_value = []

        run_analysis("/tmp/video.mov", expected_serves=5)
        _, kwargs = mock_detect.call_args
        self.assertEqual(kwargs["expected_serves"], 5)


class TestAnalysisServiceProgress(unittest.TestCase):
    """Progress callback receives expected phase strings."""

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_callback_receives_analyzing_and_done(
        self, mock_detect, mock_select, mock_info
    ):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = []
        mock_select.return_value = []

        phases = []
        run_analysis("/tmp/video.mov", on_progress=phases.append)

        self.assertIn("analyzing", phases)
        self.assertIn("done", phases)

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_callback_receives_clipping_between_analyzing_and_done(
        self, mock_detect, mock_select, mock_info
    ):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = []
        mock_select.return_value = []

        phases = []
        run_analysis("/tmp/video.mov", on_progress=phases.append)

        self.assertEqual(phases, ["analyzing", "clipping", "done"])

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_no_callback_when_none_provided(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = []
        mock_select.return_value = []

        # Should not raise
        result = run_analysis("/tmp/video.mov")
        self.assertEqual(result["candidates"], [])


class TestAnalysisServiceAdaptiveFrameSkip(unittest.TestCase):
    """Adapter picks frame_skip based on video metadata."""

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_4k_uses_frame_skip_4(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 3840, "height": 2160, "frame_count": 4133}
        mock_detect.return_value = []
        mock_select.return_value = []

        run_analysis("/tmp/video.mov")
        _, kwargs = mock_detect.call_args
        self.assertEqual(kwargs["frame_skip"], 4)

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_1080p_uses_frame_skip_2(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 1920, "height": 1080, "frame_count": 2000}
        mock_detect.return_value = []
        mock_select.return_value = []

        run_analysis("/tmp/video.mov")
        _, kwargs = mock_detect.call_args
        self.assertEqual(kwargs["frame_skip"], 2)

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_small_video_uses_frame_skip_1(self, mock_detect, mock_select, mock_info):
        mock_info.return_value = {"width": 1280, "height": 720, "frame_count": 500}
        mock_detect.return_value = []
        mock_select.return_value = []

        run_analysis("/tmp/video.mov")
        _, kwargs = mock_detect.call_args
        self.assertEqual(kwargs["frame_skip"], 1)

    @patch("web.backend.services.analysis_service.get_video_info")
    @patch("web.backend.services.analysis_service.select_serves")
    @patch("web.backend.services.analysis_service.detect_serve_candidates")
    def test_autonomous_expected_serves_none_passed_through(
        self, mock_detect, mock_select, mock_info
    ):
        mock_info.return_value = {"width": 3840, "height": 2160, "frame_count": 4133}
        mock_detect.return_value = []
        mock_select.return_value = []

        run_analysis("/tmp/video.mov", expected_serves=None)
        _, kwargs = mock_detect.call_args
        self.assertIsNone(kwargs["expected_serves"])
        self.assertEqual(kwargs["frame_skip"], 4)


if __name__ == "__main__":
    unittest.main()
