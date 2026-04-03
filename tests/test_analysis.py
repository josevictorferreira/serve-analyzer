"""
Unit tests for serve_analyzer.analysis module.

Tests focus on pure functions for:
- Scale calibration
- Velocity computation
- Coordinate transformations
"""

import unittest
import numpy as np
import tempfile
import os
import cv2

from serve_analyzer.analysis import (
    compute_scale_factor,
    compute_velocity_series,
    get_video_fps,
    get_video_info
)


class TestComputeScaleFactor(unittest.TestCase):
    """Tests for scale calibration function."""
    
    def test_horizontal_distance(self):
        """Test horizontal calibration."""
        scale = compute_scale_factor((100, 200), (400, 200), 1.0)
        # 300 pixels = 1 meter
        self.assertAlmostEqual(scale, 1.0 / 300, places=6)
    
    def test_vertical_distance(self):
        """Test vertical calibration."""
        scale = compute_scale_factor((100, 100), (100, 500), 2.0)
        # 400 pixels = 2 meters
        self.assertAlmostEqual(scale, 2.0 / 400, places=6)
    
    def test_diagonal_distance(self):
        """Test diagonal calibration."""
        # 3-4-5 triangle: points are (0,0) and (300,400)
        # Distance = sqrt(300^2 + 400^2) = 500 pixels
        scale = compute_scale_factor((0, 0), (300, 400), 5.0)
        self.assertAlmostEqual(scale, 5.0 / 500, places=6)
    
    def test_zero_real_distance_raises(self):
        """Zero real distance should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_scale_factor((0, 0), (100, 100), 0.0)
    
    def test_negative_real_distance_raises(self):
        """Negative real distance should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_scale_factor((0, 0), (100, 100), -1.0)
    
    def test_same_points_raises(self):
        """Same calibration points should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_scale_factor((100, 100), (100, 100), 1.0)


class TestComputeVelocitySeries(unittest.TestCase):
    """Tests for velocity computation function."""
    
    def test_constant_velocity(self):
        """Test constant velocity motion."""
        # 100 pixels/frame at 60 fps, scale 0.01 m/px
        # Speed = 100 * 0.01 * 60 = 60 m/s = 216 km/h
        centers = [(i * 100, 0) for i in range(10)]
        speeds_mps, speeds_kmh, stats = compute_velocity_series(
            centers, fps=60.0, scale_factor=0.01
        )
        
        # Most speeds should be ~60 m/s (allowing edge effects from smoothing)
        self.assertEqual(len(speeds_mps), 9)  # n-1 velocities
        # Check middle values (away from smoothing edges)
        self.assertTrue(np.all(speeds_mps[1:-1] > 55))
        self.assertAlmostEqual(stats['max_mps'], 60.0, places=0)
        # Mean is affected by smoothing, just check it's reasonable
        self.assertGreater(stats['mean_mps'], 50.0)
    
    def test_unit_conversion(self):
        """Test m/s to km/h conversion."""
        centers = [(0, 0), (100, 0)]
        speeds_mps, speeds_kmh, stats = compute_velocity_series(
            centers, fps=1.0, scale_factor=0.01
        )
        
        # 100 pixels = 1 meter, 1 frame = 1 second => 1 m/s
        self.assertAlmostEqual(speeds_mps[0], 1.0, places=5)
        self.assertAlmostEqual(speeds_kmh[0], 3.6, places=4)
    
    def test_two_points_minimum(self):
        """Test with minimum 2 points."""
        centers = [(0, 0), (10, 10)]
        speeds_mps, speeds_kmh, stats = compute_velocity_series(
            centers, fps=30.0, scale_factor=0.01
        )
        
        self.assertEqual(len(speeds_mps), 1)
        self.assertEqual(stats['frame_count'], 2)
    
    def test_insufficient_points_raises(self):
        """Less than 2 points should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_velocity_series([(0, 0)], fps=60.0, scale_factor=0.01)
    
    def test_invalid_fps_raises(self):
        """Zero/negative FPS should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_velocity_series([(0, 0), (10, 0)], fps=0.0, scale_factor=0.01)
        
        with self.assertRaises(ValueError):
            compute_velocity_series([(0, 0), (10, 0)], fps=-1.0, scale_factor=0.01)
    
    def test_invalid_scale_raises(self):
        """Zero/negative scale should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_velocity_series([(0, 0), (10, 0)], fps=60.0, scale_factor=0.0)
        
        with self.assertRaises(ValueError):
            compute_velocity_series([(0, 0), (10, 0)], fps=60.0, scale_factor=-0.01)
    
    def test_summary_statistics(self):
        """Test summary stats are computed correctly."""
        # Create varying velocity: 10, 20, 30, 40, 50 m/s
        centers = [(0, 0)]
        for i in range(1, 6):
            centers.append((centers[-1][0] + i * 100, 0))
        
        speeds_mps, speeds_kmh, stats = compute_velocity_series(
            centers, fps=100.0, scale_factor=0.01, smoothing_window=1
        )
        
        self.assertIn('max_mps', stats)
        self.assertIn('mean_mps', stats)
        self.assertIn('median_mps', stats)
        self.assertIn('max_kmh', stats)
        self.assertIn('mean_kmh', stats)
        self.assertIn('median_kmh', stats)
        self.assertIn('frame_count', stats)
        self.assertIn('duration_sec', stats)
        
        # Duration should be 6 frames / 100 fps = 0.06 seconds
        self.assertAlmostEqual(stats['duration_sec'], 0.05, places=2)

    def test_duration_sec(self):
        # Verify duration_sec uses frame intervals (n-1)/fps
        centers = [(0, 0), (100, 0), (200, 0)]
        speeds_mps, speeds_kmh, stats = compute_velocity_series(
            centers, fps=50.0, scale_factor=0.01, smoothing_window=1
        )
        expected = (len(centers) - 1) / 50.0
        self.assertAlmostEqual(stats['duration_sec'], expected, places=5)



class TestVideoInfo(unittest.TestCase):
    """Tests for video metadata functions."""
    
    def setUp(self):
        """Create a synthetic test video."""
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, 'test_video.mp4')
        
        # Create a simple test video (10 frames, 640x480, 30 fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 30.0, (640, 480))
        
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (i * 25, 0, 0)  # Varying blue
            out.write(frame)
        
        out.release()
    
    def tearDown(self):
        """Clean up test video."""
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        os.rmdir(self.temp_dir)
    
    def test_get_video_fps(self):
        """Test FPS extraction."""
        fps = get_video_fps(self.video_path)
        self.assertEqual(fps, 30.0)
    
    def test_get_video_info(self):
        """Test video info extraction."""
        info = get_video_info(self.video_path)
        
        self.assertEqual(info['fps'], 30.0)
        self.assertEqual(info['width'], 640)
        self.assertEqual(info['height'], 480)
        self.assertEqual(info['frame_count'], 10)
        self.assertAlmostEqual(info['duration_sec'], 10/30, places=2)
    
    def test_invalid_video_path(self):
        """Test with invalid video path."""
        with self.assertRaises(IOError):
            get_video_fps('/nonexistent/video.mp4')
        
        with self.assertRaises(IOError):
            get_video_info('/nonexistent/video.mp4')


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple functions."""
    
    def test_full_pipeline_math(self):
        """Test complete velocity computation pipeline (math only)."""
        # Simulate tracking results
        # Ball moves 200 pixels/frame, video is 60 fps
        # Scale: 100 pixels = 1 meter
        # Expected: 200 px/frame * 0.01 m/px * 60 fps = 120 m/s
        
        centers = [(i * 200, 300) for i in range(10)]
        
        # Calibration: points at (0,0) and (100,0) represent 1 meter
        scale = compute_scale_factor((0, 0), (100, 0), 1.0)
        
        speeds_mps, speeds_kmh, stats = compute_velocity_series(
            centers, fps=60.0, scale_factor=scale
        )
        
        # Should be approximately 120 m/s (432 km/h)
        # Allow some tolerance due to smoothing
        self.assertGreater(stats['max_mps'], 100)
        self.assertGreater(stats['max_kmh'], 360)


if __name__ == '__main__':
    unittest.main()
