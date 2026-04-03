"""
Core analysis logic for serve velocity estimation.

This module provides reusable functions for:
- Manual 2-point scale calibration
- Ball tracking (template matching)
- Velocity computation from tracked positions

All core functions are pure and testable without video I/O.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def compute_scale_factor(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    real_distance: float
) -> float:
    """
    Compute pixel-to-meter scale factor from two calibration points.
    
    Args:
        point1: First calibration point (x, y) in pixels
        point2: Second calibration point (x, y) in pixels
        real_distance: Known real-world distance between points in meters
    
    Returns:
        Scale factor (meters per pixel)
    
    Example:
        >>> scale = compute_scale_factor((100, 200), (400, 200), 1.0)
        >>> # 300 pixels = 1 meter
        >>> scale
        0.0033333333333333335
    """
    if real_distance <= 0:
        raise ValueError("Real distance must be positive")
    
    pixel_distance = np.sqrt(
        (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2
    )
    
    if pixel_distance == 0:
        raise ValueError("Calibration points must be different")
    
    return real_distance / pixel_distance


def compute_velocity_series(
    centers: List[Tuple[float, float]],
    fps: float,
    scale_factor: float,
    smoothing_window: int = 3
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute velocity series from tracked ball centers.
    
    Args:
        centers: List of ball center positions (x, y) in pixels
        fps: Video frame rate
        scale_factor: Meters per pixel from calibration
        smoothing_window: Window size for velocity smoothing (default 3)
    
    Returns:
        Tuple of (speeds_mps, speeds_kmh, summary_stats)
        - speeds_mps: Array of speeds in m/s
        - speeds_kmh: Array of speeds in km/h
        - summary_stats: Dict with max, mean, median speeds
    
    Example:
        >>> centers = [(0, 0), (100, 0), (200, 0)]
        >>> speeds_mps, speeds_kmh, stats = compute_velocity_series(
        ...     centers, fps=60.0, scale_factor=0.01
        ... )
    """
    if len(centers) < 2:
        raise ValueError("Need at least 2 center positions")
    
    if fps <= 0:
        raise ValueError("FPS must be positive")
    
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive")
    
    centers_array = np.array(centers)
    
    # Compute frame-to-frame displacements
    displacements = np.sqrt(
        np.sum(np.diff(centers_array, axis=0)**2, axis=1)
    )
    
    # Convert to real-world distances (meters)
    distances_m = displacements * scale_factor
    
    # Time interval between frames
    dt = 1.0 / fps
    
    # Compute speeds (m/s)
    speeds_mps = distances_m / dt
    
    # Apply simple moving average smoothing
    if smoothing_window > 1 and len(speeds_mps) >= smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        speeds_mps_smoothed = np.convolve(speeds_mps, kernel, mode='same')
    else:
        speeds_mps_smoothed = speeds_mps
    
    # Convert to km/h
    speeds_kmh = speeds_mps_smoothed * 3.6
    
    # Compute summary statistics
    summary_stats = {
        'max_mps': float(np.max(speeds_mps_smoothed)),
        'max_kmh': float(np.max(speeds_kmh)),
        'mean_mps': float(np.mean(speeds_mps_smoothed)),
        'mean_kmh': float(np.mean(speeds_kmh)),
        'median_mps': float(np.median(speeds_mps_smoothed)),
        'median_kmh': float(np.median(speeds_kmh)),
        'frame_count': len(centers),
        'duration_sec': (len(centers) - 1) / fps
    }
    
    return speeds_mps_smoothed, speeds_kmh, summary_stats


def track_ball_template(
    video_path: str,
    start_frame: int,
    initial_center: Tuple[int, int],
    template_size: int = 30,
    search_radius: int = 100,
    max_frames: Optional[int] = None
) -> List[Tuple[float, float]]:
    """
    Track ball using template matching from a given starting point.
    
    Args:
        video_path: Path to video file
        start_frame: Frame number to start tracking (post-impact)
        initial_center: Initial ball position (x, y) in pixels
        template_size: Size of template square in pixels (default 30)
        search_radius: Search radius around previous position (default 100)
        max_frames: Maximum frames to track (None = until end)
    
    Returns:
        List of tracked ball centers (x, y) in pixel coordinates
    
    Note:
        This is a simple MVP tracker. It uses template matching with
        a fixed-size window and assumes the ball moves smoothly.
        For more robust tracking, consider optical flow or ML methods.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate start frame
    if start_frame < 0 or start_frame >= total_frames:
        cap.release()
        raise ValueError(f"Start frame {start_frame} out of range [0, {total_frames})")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read first frame")
    
    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # Extract initial template
    half_size = template_size // 2
    x, y = initial_center
    
    # Ensure template is within frame bounds
    y_min = max(0, y - half_size)
    y_max = min(gray_first.shape[0], y + half_size)
    x_min = max(0, x - half_size)
    x_max = min(gray_first.shape[1], x + half_size)
    
    template = gray_first[y_min:y_max, x_min:x_max]
    
    centers = [initial_center]
    prev_center = initial_center
    
    frames_tracked = 1
    end_frame = total_frames if max_frames is None else min(total_frames, start_frame + max_frames)
    
    while frames_tracked < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Define search region
        search_x_min = max(0, prev_center[0] - search_radius)
        search_x_max = min(gray.shape[1], prev_center[0] + search_radius)
        search_y_min = max(0, prev_center[1] - search_radius)
        search_y_max = min(gray.shape[0], prev_center[1] + search_radius)
        
        search_region = gray[search_y_min:search_y_max, search_x_min:search_x_max]
        
        # Check if search region is large enough
        if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
            # Search region too small, use full frame
            search_region = gray
            search_x_min = 0
            search_y_min = 0
        
        # Template matching
        try:
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Confidence threshold
            if max_val > 0.5:
                # Convert back to full-frame coordinates
                match_x = max_loc[0] + template.shape[1] // 2 + search_x_min
                match_y = max_loc[1] + template.shape[0] // 2 + search_y_min
                
                centers.append((match_x, match_y))
                prev_center = (match_x, match_y)
                
                # Update template adaptively
                new_y_min = max(0, match_y - half_size)
                new_y_max = min(gray.shape[0], match_y + half_size)
                new_x_min = max(0, match_x - half_size)
                new_x_max = min(gray.shape[1], match_x + half_size)
                template = gray[new_y_min:new_y_max, new_x_min:new_x_max]
            else:
                # Low confidence - keep previous position
                centers.append(prev_center)
        except cv2.error:
            # Template matching failed - keep previous position
            centers.append(prev_center)
        
        frames_tracked += 1
    
    cap.release()
    return centers


def extract_ball_centers(
    video_path: str,
    frames: List[int],
    manual_positions: List[Tuple[int, int]]
) -> List[Tuple[float, float]]:
    """
    Use manually specified ball positions for velocity calculation.
    
    This is useful when automatic tracking fails or for validation.
    
    Args:
        video_path: Path to video file (for validation)
        frames: List of frame numbers
        manual_positions: List of manually marked ball positions
    
    Returns:
        List of ball centers (converted to float for consistency)
    """
    if len(frames) != len(manual_positions):
        raise ValueError("frames and manual_positions must have same length")
    
    # Validate video can be opened
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    cap.release()
    
    # Convert to float tuples
    return [(float(x), float(y)) for x, y in manual_positions]


def get_video_fps(video_path: str) -> float:
    """
    Extract frame rate from video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Frame rate (fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps <= 0:
        raise ValueError(f"Invalid FPS detected: {fps}")
    
    return fps


def get_video_info(video_path: str) -> dict:
    """
    Get basic video metadata.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dict with fps, width, height, frame_count, duration_sec
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration_sec': frame_count / fps if fps > 0 else 0
    }
