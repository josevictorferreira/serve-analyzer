# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch
import sys

from serve_analyzer.cli import run_analysis, InteractiveCalibrator

class TestCLIInteractiveDisplayFrame(unittest.TestCase):
    def test_default_display_frame_uses_start_frame(self):
        # Set start_frame to a non-zero value
        start_frame = 42
        captured = {}

        def mock_run_interactive(self, display_frame=0):
            captured['display_frame'] = display_frame
            # Return dummy calibration points and ball position
            return (10, 10), (20, 20), (30, 30)

        sys.argv = [
            'serve_analyzer.cli',
            'dummy.mp4',
            '--real-distance', '1.0',
            '--start-frame', str(start_frame)
        ]

        with patch.object(InteractiveCalibrator, 'run_interactive', new=mock_run_interactive), \
             patch('serve_analyzer.cli.track_ball_template', return_value=[(0, 0), (10, 0)]), \
             patch('serve_analyzer.cli.get_video_info', return_value={'fps': 30.0, 'width': 640, 'height': 480, 'frame_count': 10, 'duration_sec': 10/30.0}):
            # Import here to ensure argv is set before parsing
            from serve_analyzer import cli
            cli.main()

        # Verify that the display_frame passed to the interactive calibrator equals start_frame
        self.assertEqual(captured.get('display_frame'), start_frame)

    def test_display_frame_mismatch_raises(self):
        # Set start_frame and a different display_frame
        start_frame = 10
        display_frame = 20
        captured = {}

        def mock_run_interactive(self, display_frame=0):
            captured['display_frame'] = display_frame
            return (10, 10), (20, 20), (30, 30)

        sys.argv = [
            'serve_analyzer.cli',
            'dummy.mp4',
            '--real-distance', '1.0',
            '--start-frame', str(start_frame),
            '--display-frame', str(display_frame)
        ]

        with patch.object(InteractiveCalibrator, 'run_interactive', new=mock_run_interactive), \
             patch('serve_analyzer.cli.track_ball_template', return_value=[(0, 0), (10, 0)]), \
             patch('serve_analyzer.cli.get_video_info', return_value={'fps': 30.0, 'width': 640, 'height': 480, 'frame_count': 10, 'duration_sec': 10/30.0}):
            from serve_analyzer import cli
            # CLI should exit with non‑zero status due to mismatched display_frame
            exit_code = cli.main()
            self.assertEqual(exit_code, 1)

        # Ensure the interactive calibrator still receives the mismatched display_frame
        self.assertIsNone(captured.get('display_frame'))



if __name__ == '__main__':
    unittest.main()
