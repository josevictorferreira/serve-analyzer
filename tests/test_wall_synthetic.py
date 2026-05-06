"""
Unit tests for synthetic wall-impact video fixtures.

Verifies that ``wall_test_helpers.generate_wall_impact_video`` produces
readable, deterministic MP4s with known ground-truth properties.
"""

import os
import tempfile
import unittest

import cv2

from wall_test_helpers import generate_wall_impact_video, WallFixtureGroundTruth


class TestSyntheticWallVideo(unittest.TestCase):
    """Tests for synthetic wall-impact video generation."""

    def test_generates_readable_known_impact_video(self):
        """Generated video is readable by OpenCV and matches ground truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wall_impact.mp4")
            gt = generate_wall_impact_video(
                path,
                width=640,
                height=480,
                fps=60,
                total_frames=90,
                impact_frame=30,
                ball_speed_px_per_frame=4.0,
                wall_x_px=240,
            )

            self.assertIsInstance(gt, WallFixtureGroundTruth)
            self.assertTrue(os.path.exists(path))

            cap = cv2.VideoCapture(path)
            self.assertTrue(
                cap.isOpened(), "cv2.VideoCapture failed to open generated MP4"
            )

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.assertEqual(frame_count, gt.total_frames)

            fps_read = cap.get(cv2.CAP_PROP_FPS)
            self.assertAlmostEqual(fps_read, float(gt.fps), places=0)

            # Verify impact frame contains the ball near the expected pixel.
            cap.set(cv2.CAP_PROP_POS_FRAMES, gt.impact_frame)
            ret, frame = cap.read()
            self.assertTrue(ret, "Failed to read impact frame")

            expected_x, expected_y = gt.impact_pixel
            # The ball is white; sample the expected centre.
            # Allow a small tolerance because of anti-aliasing / rounding.
            roi = frame[
                expected_y - 2 : expected_y + 3,
                expected_x - 2 : expected_x + 3,
            ]
            # White ball should have high intensity in all channels.
            max_intensity = roi.max()
            self.assertGreater(
                max_intensity,
                200,
                f"Impact frame pixel near ({expected_x},{expected_y}) is not bright enough",
            )

            cap.release()

    def test_blur_variant_records_expected_uncertainty(self):
        """Blur sigma increases expected uncertainty; video still readable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wall_impact_blur.mp4")
            gt = generate_wall_impact_video(
                path,
                width=640,
                height=480,
                fps=60,
                total_frames=90,
                impact_frame=30,
                blur_sigma=2.0,
                ball_speed_px_per_frame=4.0,
                wall_x_px=240,
            )

            self.assertGreater(
                gt.expected_uncertainty_px,
                1.0,
                "Expected uncertainty should exceed baseline when blur_sigma > 0",
            )
            self.assertTrue(os.path.exists(path))

            cap = cv2.VideoCapture(path)
            self.assertTrue(cap.isOpened())
            cap.release()

    def test_ground_truth_positions_are_deterministic(self):
        """Two calls with identical arguments produce identical ball_positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "a.mp4")
            path_b = os.path.join(tmpdir, "b.mp4")

            gt_a = generate_wall_impact_video(
                path_a,
                width=640,
                height=480,
                fps=60,
                total_frames=90,
                impact_frame=30,
                ball_speed_px_per_frame=4.0,
                wall_x_px=240,
            )
            gt_b = generate_wall_impact_video(
                path_b,
                width=640,
                height=480,
                fps=60,
                total_frames=90,
                impact_frame=30,
                ball_speed_px_per_frame=4.0,
                wall_x_px=240,
            )

            self.assertEqual(gt_a.ball_positions, gt_b.ball_positions)
            self.assertEqual(gt_a.impact_pixel, gt_b.impact_pixel)
            self.assertEqual(gt_a.impact_frame, gt_b.impact_frame)
            self.assertEqual(gt_a.wall_x_px, gt_b.wall_x_px)


if __name__ == "__main__":
    unittest.main()
