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
import tempfile
import os
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
    max_frames: Optional[int] = None,
    confidence_threshold: float = 0.5,
    debug_output: Optional[str] = None
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
        confidence_threshold: Min match confidence 0-1 (default 0.5, lower=more lenient)
        debug_output: If provided, save debug video showing tracking to this path
    
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
    
    # Debug video writer
    debug_writer = None
    if debug_output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        debug_writer = cv2.VideoWriter(debug_output.replace('.mp4', '_debug.avi'), fourcc, fps, (width, height))
        # Write first frame with initial position marked
        debug_frame = first_frame.copy()
        cv2.circle(debug_frame, initial_center, half_size, (0, 255, 0), 3)
        cv2.rectangle(debug_frame, (x - half_size, y - half_size), (x + half_size, y + half_size), (255, 0, 0), 2)
        cv2.putText(debug_frame, f'Frame {start_frame} - INIT', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        debug_writer.write(debug_frame)
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
        match_found = False
        try:
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Confidence threshold
            if max_val > confidence_threshold:
                # Convert back to full-frame coordinates
                match_x = max_loc[0] + template.shape[1] // 2 + search_x_min
                match_y = max_loc[1] + template.shape[0] // 2 + search_y_min
                
                centers.append((match_x, match_y))
                prev_center = (match_x, match_y)
                match_found = True
                
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
        
        # Debug visualization
        if debug_writer:
            debug_frame = frame.copy()
            # Draw search region (yellow rectangle)
            cv2.rectangle(debug_frame, (search_x_min, search_y_min), (search_x_max, search_y_max), (0, 255, 255), 2)
            # Draw tracked position (green if matched, red if stuck)
            color = (0, 255, 0) if match_found else (0, 0, 255)
            cv2.circle(debug_frame, prev_center, half_size, color, 3)
            cv2.rectangle(debug_frame, (prev_center[0] - half_size, prev_center[1] - half_size),
                          (prev_center[0] + half_size, prev_center[1] + half_size), (255, 0, 0), 2)
            # Status text
            status = f'Frame {start_frame + frames_tracked} - conf={max_val:.2f}' if 'max_val' in dir() else f'Frame {start_frame + frames_tracked}'
            cv2.putText(debug_frame, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            debug_writer.write(debug_frame)
        
        frames_tracked += 1
    
    if debug_writer:
        debug_writer.release()
        print(f"Debug video saved (AVI): {debug_output.replace('.mp4', '_debug.avi')}")
    
    cap.release()
    return centers


def track_ball_color(
    video_path: str,
    start_frame: int,
    initial_center: Tuple[int, int],
    search_radius: int = 250,
    max_frames: Optional[int] = None,
    hsv_lower: Tuple[int, int, int] = (20, 100, 100),
    hsv_upper: Tuple[int, int, int] = (35, 255, 255),
    min_area: int = 50,
    debug_output: Optional[str] = None
) -> List[Tuple[float, float]]:
    """
    Track ball using HSV color detection (for yellow tennis balls).
    
    Unlike template matching, this finds the yellow object directly,
    which works better when the ball is the only yellow thing in frame.
    
    Args:
        video_path: Path to video file
        start_frame: Frame number to start tracking
        initial_center: Initial ball position (x, y) - used for search region
        search_radius: Search radius around previous position (default 250)
        max_frames: Maximum frames to track (None = until end)
        hsv_lower: Lower HSV bounds for yellow (default: tennis ball yellow)
        hsv_upper: Upper HSV bounds for yellow
        min_area: Minimum contour area in pixels (filters noise)
        debug_output: If provided, save debug video showing tracking
    
    Returns:
        List of tracked ball centers (x, y) in pixel coordinates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame < 0 or start_frame >= total_frames:
        cap.release()
        raise ValueError(f"Start frame {start_frame} out of range [0, {total_frames})")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read first frame to initialize
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read first frame")
    
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    
    centers = [initial_center]
    prev_center = initial_center
    
    frames_tracked = 1
    end_frame = total_frames if max_frames is None else min(total_frames, start_frame + max_frames)
    
    # Debug video writer
    debug_writer = None
    if debug_output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        debug_path = debug_output.replace('.mp4', '_color_debug.avi')
        debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (width, height))
        # Write first frame
        debug_frame = first_frame.copy()
        cv2.circle(debug_frame, initial_center, 40, (0, 255, 0), 3)
        cv2.putText(debug_frame, f'Frame {start_frame} - INIT', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        debug_writer.write(debug_frame)
    
    while frames_tracked < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define search region around previous position
        search_x_min = max(0, prev_center[0] - search_radius)
        search_x_max = min(frame.shape[1], prev_center[0] + search_radius)
        search_y_min = max(0, prev_center[1] - search_radius)
        search_y_max = min(frame.shape[0], prev_center[1] + search_radius)
        
        # Create mask for yellow color in search region
        hsv_region = hsv[search_y_min:search_y_max, search_x_min:search_x_max]
        mask = cv2.inRange(hsv_region, lower, upper)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        match_found = False
        if contours:
            # Find largest contour above min_area
            valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    # Compute centroid
                    cx = int(M["m10"] / M["m00"]) + search_x_min
                    cy = int(M["m01"] / M["m00"]) + search_y_min
                    centers.append((cx, cy))
                    prev_center = (cx, cy)
                    match_found = True
        
        if not match_found:
            # Keep previous position if no yellow found
            centers.append(prev_center)
        
        # Debug visualization
        if debug_writer:
            debug_frame = frame.copy()
            # Draw search region (yellow rectangle)
            cv2.rectangle(debug_frame, (search_x_min, search_y_min), 
                          (search_x_max, search_y_max), (0, 255, 255), 2)
            # Draw tracked position
            color = (0, 255, 0) if match_found else (0, 0, 255)
            cv2.circle(debug_frame, prev_center, 40, color, 3)
            # Show mask overlay in corner (scaled down)
            mask_small = cv2.resize(mask, (200, 200))
            mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            debug_frame[10:210, 10:210] = mask_color
            # Status
            status = 'FOUND' if match_found else 'LOST'
            cv2.putText(debug_frame, f'Frame {start_frame + frames_tracked} - {status}', 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            debug_writer.write(debug_frame)
        
        frames_tracked += 1
    
    if debug_writer:
        debug_writer.release()
        print(f"Debug video saved: {debug_output.replace('.mp4', '_color_debug.avi')}")
    
    cap.release()
    return centers


def track_ball_csrt(
    video_path: str,
    start_frame: int,
    initial_center: Tuple[int, int],
    bbox_size: int = 50,
    max_frames: Optional[int] = None,
    debug_output: Optional[str] = None
) -> List[Tuple[float, float]]:
    """
    Track ball using OpenCV CSRT tracker (discriminative correlation filter).
    
    CSRT is more robust than template matching for fast-moving objects because:
    - It learns object appearance online
    - It uses spatial reliability map to handle deformation
    - It adapts to scale changes
    
    Args:
        video_path: Path to video file
        start_frame: Frame number to start tracking
        initial_center: Initial ball position (x, y) in pixels
        bbox_size: Size of bounding box around ball (default 50)
        max_frames: Maximum frames to track (None = until end)
        debug_output: If provided, save debug video showing tracking
    
    Returns:
        List of tracked ball centers (x, y) in pixel coordinates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame < 0 or start_frame >= total_frames:
        cap.release()
        raise ValueError(f"Start frame {start_frame} out of range [0, {total_frames})")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read first frame")
    
    # Create CSRT tracker
    tracker = cv2.TrackerCSRT_create()
    
    # Define initial bounding box (x, y, width, height)
    x, y = initial_center
    half = bbox_size // 2
    bbox = (x - half, y - half, bbox_size, bbox_size)
    
    # Ensure bbox is within frame bounds
    h, w = first_frame.shape[:2]
    bbox = (
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(bbox_size, w - max(0, bbox[0])),
        min(bbox_size, h - max(0, bbox[1]))
    )
    
    # Initialize tracker
    tracker.init(first_frame, bbox)
    
    centers = [initial_center]
    
    frames_tracked = 1
    end_frame = total_frames if max_frames is None else min(total_frames, start_frame + max_frames)
    
    # Debug video writer
    debug_writer = None
    if debug_output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        debug_writer = cv2.VideoWriter(
            debug_output.replace('.mp4', '_csrt_debug.avi'),
            fourcc, fps, (w, h)
        )
        # Draw first frame
        debug_frame = first_frame.copy()
        cv2.rectangle(debug_frame, 
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                      (0, 255, 0), 3)
        cv2.circle(debug_frame, initial_center, 10, (0, 0, 255), -1)
        cv2.putText(debug_frame, f'Frame {start_frame} - INIT',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        debug_writer.write(debug_frame)
    
    tracking_lost = False
    lost_count = 0
    max_lost_frames = 10  # Stop after this many consecutive lost frames
    
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update tracker
        success, bbox = tracker.update(frame)
        
        if success:
            # Extract center from bbox
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            centers.append((float(cx), float(cy)))
            lost_count = 0
            status = 'TRACKING'
            color = (0, 255, 0)  # Green
        else:
            # Lost tracking - use last known position
            centers.append(centers[-1])
            lost_count += 1
            status = f'LOST ({lost_count})'
            color = (0, 0, 255)  # Red
            
            if lost_count >= max_lost_frames:
                print(f"CSRT tracker lost ball for {max_lost_frames} frames, stopping at frame {start_frame + frames_tracked}")
                break
        
        if debug_writer:
            debug_frame = frame.copy()
            if success:
                cv2.rectangle(debug_frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              color, 3)
            cx, cy = centers[-1]
            cv2.circle(debug_frame, (int(cx), int(cy)), 10, color, -1)
            cv2.putText(debug_frame, f'Frame {start_frame + frames_tracked} - {status}',
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            debug_writer.write(debug_frame)
        
        frames_tracked += 1
    
    if debug_writer:
        debug_writer.release()
        print(f"CSRT debug video saved: {debug_output.replace('.mp4', '_csrt_debug.avi')}")
    
    cap.release()
    print(f"CSRT tracked {len(centers)} frames ({len([c for i, c in enumerate(centers[1:], 1) if c != centers[i-1]])} with movement)")
    return centers


def track_ball_optical_flow(
    video_path: str,
    start_frame: int,
    initial_center: Tuple[int, int],
    search_radius: int = 200,
    max_frames: Optional[int] = None,
    debug_output: Optional[str] = None
) -> List[Tuple[float, float]]:
    """
    Track ball using sparse optical flow (Lucas-Kanade pyramidal).
    
    Optimized for FAST-MOVING objects like tennis serves:
    - Large pyramid levels (4) to handle big displacements
    - Large window size (31x31) for robust tracking
    - Uses HSV color detection to re-acquire ball if flow fails
    
    Args:
        video_path: Path to video file
        start_frame: Frame number to start tracking (should be at/after contact)
        initial_center: Initial ball position (x, y) in pixels
        search_radius: Search radius for color re-acquisition (default 200)
        max_frames: Maximum frames to track (None = until end)
        debug_output: If provided, save debug video showing tracking
    
    Returns:
        List of tracked ball centers (x, y) in pixel coordinates
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if start_frame < 0 or start_frame >= total_frames:
        cap.release()
        raise ValueError(f"Start frame {start_frame} out of range [0, {total_frames})")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read first frame")
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Lucas-Kanade params optimized for FAST motion
    lk_params = dict(
        winSize=(31, 31),      # Large window for fast motion
        maxLevel=4,            # More pyramid levels for large displacements
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Initial point
    x, y = initial_center
    prev_pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
    
    centers = [initial_center]
    
    end_frame = total_frames if max_frames is None else min(total_frames, start_frame + max_frames)
    frames_tracked = 1
    lost_count = 0
    max_lost = 5
    
    # Debug writer
    debug_writer = None
    if debug_output:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        debug_writer = cv2.VideoWriter(
            debug_output.replace('.mp4', '_optflow_debug.avi'),
            fourcc, fps, (w, h)
        )
        # First frame
        debug_frame = prev_frame.copy()
        cv2.circle(debug_frame, initial_center, 15, (0, 255, 0), -1)
        cv2.putText(debug_frame, f'Frame {start_frame} - INIT', 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        debug_writer.write(debug_frame)
    
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **lk_params
        )
        
        found = False
        new_center = centers[-1]
        
        if status is not None and status[0][0] == 1:
            nx, ny = next_pts[0][0]
            
            # Validate: check if new position has yellow ball
            roi_x1 = max(0, int(nx) - 30)
            roi_y1 = max(0, int(ny) - 30)
            roi_x2 = min(w, int(nx) + 30)
            roi_y2 = min(h, int(ny) + 30)
            
            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, (18, 80, 80), (45, 255, 255))
                yellow_pixels = cv2.countNonZero(mask)
                
                if yellow_pixels > 20:  # Found yellow at predicted location
                    new_center = (float(nx), float(ny))
                    found = True
                    lost_count = 0
        
        if not found:
            # Try color-based re-acquisition in larger search region
            px, py = centers[-1]
            search_x1 = max(0, int(px) - search_radius)
            search_y1 = max(0, int(py) - search_radius)
            search_x2 = min(w, int(px) + search_radius)
            search_y2 = min(h, int(py) + search_radius)
            
            roi = frame[search_y1:search_y2, search_x1:search_x2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (18, 80, 80), (45, 255, 255))
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if 30 < cv2.contourArea(c) < 5000]
            
            if valid:
                # Find closest to predicted position
                best = None
                best_dist = float('inf')
                for c in valid:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00'] + search_x1
                        cy = M['m01'] / M['m00'] + search_y1
                        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                        if dist < best_dist:
                            best_dist = dist
                            best = (cx, cy)
                
                if best:
                    new_center = best
                    found = True
                    lost_count = 0
        
        if not found:
            lost_count += 1
            if lost_count >= max_lost:
                print(f"Optical flow lost ball for {max_lost} frames at frame {start_frame + frames_tracked}")
                break
        
        centers.append(new_center)
        
        # Update for next iteration
        prev_gray = gray
        prev_pts = np.array([[[new_center[0], new_center[1]]]], dtype=np.float32)
        
        if debug_writer:
            debug_frame = frame.copy()
            cx, cy = new_center
            color = (0, 255, 0) if found else (0, 0, 255)
            status_txt = 'TRACKING' if found else f'SEARCHING ({lost_count})'
            cv2.circle(debug_frame, (int(cx), int(cy)), 15, color, -1)
            # Draw motion vector
            if len(centers) > 1:
                px, py = centers[-2]
                cv2.arrowedLine(debug_frame, (int(px), int(py)), (int(cx), int(cy)), 
                               (255, 255, 0), 3, tipLength=0.3)
            cv2.putText(debug_frame, f'Frame {start_frame + frames_tracked} - {status_txt}',
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            debug_writer.write(debug_frame)
        
        frames_tracked += 1
    
    if debug_writer:
        debug_writer.release()
        print(f"Optical flow debug video saved: {debug_output.replace('.mp4', '_optflow_debug.avi')}")
    
    cap.release()
    print(f"Optical flow tracked {len(centers)} frames")
    return centers


def track_ball_yolo(
    video_path: str,
    start_frame: int,
    initial_center: Tuple[int, int],
    model_path: str = "rjtp",
    max_frames: Optional[int] = None,
    conf_threshold: float = 0.25,
    search_radius: int = 300,
    debug_output: Optional[str] = None
) -> List[Tuple[float, float]]:
    """
    Track ball using YOLO deep learning detector.
    
    Best for FAST-MOVING objects like tennis serves. YOLO can detect
    the ball regardless of motion blur because it uses learned features,
    not template matching.
    
    Args:
        video_path: Path to video file
        start_frame: Frame number to start tracking
        initial_center: Initial ball position (x, y) - used to identify which detection is the ball
        model_path: Path to YOLO model weights or 'rjtp' for RJTPP tennis-ball model (default: rjtp)
        max_frames: Maximum frames to track (None = until end)
        conf_threshold: Minimum confidence for detection (default 0.25)
        search_radius: Max distance from previous position to consider same ball
        debug_output: If provided, save debug video showing tracking
    
    Returns:
        List of tracked ball centers (x, y) in pixel coordinates
    
    Note:
        Requires ultralytics package: pip install ultralytics
        First run will download model weights (~6MB for yolov8n).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Run: pip install ultralytics\n"
            "Or use 'nix develop' which auto-installs it."
        )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    if start_frame < 0 or start_frame >= total_frames:
        cap.release()
        raise ValueError(f"Start frame {start_frame} out of range [0, {total_frames})")
    
    # Load YOLO model
    # Resolve model alias and download if needed
    _is_rjtp = model_path in ("rjtp", "RJTPP/tennis-ball-detection")
    if _is_rjtp:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="RJTPP/tennis-ball-detection", filename="best.pt"
        )
        print(f"Using RJTPP tennis-ball model: {model_path}")
    else:
        print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    centers = [initial_center]
    last_pos = initial_center
    
    end_frame = total_frames if max_frames is None else min(total_frames, start_frame + max_frames)
    frames_tracked = 0
    lost_count = 0
    max_lost = 10  # Allow more lost frames since YOLO may have false negatives
    
    # Debug writer
    debug_writer = None
    if debug_output:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        debug_writer = cv2.VideoWriter(
            debug_output.replace('.mp4', '_yolo_debug.avi'),
            fourcc, fps, (w, h)
        )
    
    # Track static false positives to filter them out
    static_positions = []  # positions that appear in multiple frames without moving
    
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False,
            device='cpu'
        )
        
        found = False
        new_center = last_pos
        best_score = -1  # Use score instead of just distance
        
        # Process detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            candidates = []
            
            for i in range(len(boxes)):
                # Get box center
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                conf = float(boxes.conf[i])
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Check if this is a known static false positive
                is_static = False
                for sx, sy in static_positions:
                    if abs(cx - sx) < 50 and abs(cy - sy) < 50:
                        is_static = True
                        break
                
                if is_static:
                    continue  # Skip static false positives
                
                # Check if this could be a ball:
                is_small = box_w < 200 and box_h < 200
                is_square = 0.5 < (box_w / max(box_h, 1)) < 2.0
                dist = np.sqrt((cx - last_pos[0])**2 + (cy - last_pos[1])**2)
                is_near = dist < search_radius
                
                if is_small and is_square and is_near:
                    # Score: prefer high confidence and reasonable distance
                    # Penalize detections that are too close (static) or too far (wrong object)
                    if frames_tracked > 0 and dist < 10:
                        # Suspiciously static - might be false positive
                        score = conf * 0.3  # Heavy penalty
                    else:
                        score = conf
                    candidates.append((cx, cy, conf, score, dist))
            
            # Choose best candidate
            if candidates:
                # Sort by score (highest first)
                candidates.sort(key=lambda x: x[3], reverse=True)
                cx, cy, conf, score, dist = candidates[0]
                new_center = (float(cx), float(cy))
                found = True
                
                # If this detection is very static, mark it as potential false positive
                if frames_tracked > 2 and dist < 10:
                    static_positions.append((cx, cy))
        
        if found:
            lost_count = 0
            last_pos = new_center
        else:
            lost_count += 1
            # Use color detection as fallback
            px, py = last_pos
            search_x1 = max(0, int(px) - search_radius)
            search_y1 = max(0, int(py) - search_radius)
            search_x2 = min(w, int(px) + search_radius)
            search_y2 = min(h, int(py) + search_radius)
            
            roi = frame[search_y1:search_y2, search_x1:search_x2]
            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # Yellow ball detection
                mask = cv2.inRange(hsv, (18, 80, 80), (45, 255, 255))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in contours if 30 < cv2.contourArea(c) < 5000]
                
                if valid:
                    # Find closest to last position
                    best = None
                    best_d = float('inf')
                    for c in valid:
                        M = cv2.moments(c)
                        if M['m00'] > 0:
                            cx = M['m10'] / M['m00'] + search_x1
                            cy = M['m01'] / M['m00'] + search_y1
                            d = np.sqrt((cx - px)**2 + (cy - py)**2)
                            if d < best_d:
                                best_d = d
                                best = (cx, cy)
                    
                    if best:
                        new_center = best
                        last_pos = new_center
                        found = True
                        lost_count = 0
            
            if lost_count >= max_lost:
                print(f"YOLO lost ball for {max_lost} frames at frame {start_frame + frames_tracked}")
                break
        
        centers.append(new_center)
        
        if debug_writer:
            debug_frame = frame.copy()
            cx, cy = new_center
            color = (0, 255, 0) if found else (0, 0, 255)
            status_txt = 'YOLO' if found else f'FALLBACK ({lost_count})'
            cv2.circle(debug_frame, (int(cx), int(cy)), 20, color, 3)
            # Draw search radius
            cv2.circle(debug_frame, (int(last_pos[0]), int(last_pos[1])), search_radius, (255, 255, 0), 1)
            # Draw all YOLO detections
            if len(results) > 0 and results[0].boxes is not None:
                for i in range(len(results[0].boxes)):
                    x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy()
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
            # Draw motion vector
            if len(centers) > 1:
                px, py = centers[-2]
                cv2.arrowedLine(debug_frame, (int(px), int(py)), (int(cx), int(cy)),
                               (255, 255, 0), 3, tipLength=0.3)
            cv2.putText(debug_frame, f'Frame {start_frame + frames_tracked} - {status_txt}',
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
            debug_writer.write(debug_frame)
        
        frames_tracked += 1
    
    if debug_writer:
        debug_writer.release()
        print(f"YOLO debug video saved: {debug_output.replace('.mp4', '_yolo_debug.avi')}")
    
    cap.release()
    print(f"YOLO tracked {len(centers)} frames")
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


def generate_annotated_video(
    video_path: str,
    output_path: str,
    centers: List[Tuple[float, float]],
    speeds_kmh: np.ndarray,
    start_frame: int,
    trail_length: int = 20,
    ball_color: Tuple[int, int, int] = (0, 255, 0),
    trail_color: Tuple[int, int, int] = (255, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    clip_to_tracking: bool = True
) -> None:
    """
    Generate annotated video with speed overlay and ball trajectory.

    Args:
        video_path: Path to input video file
        output_path: Path to output video file (MP4)
        centers: List of tracked ball centers (x, y) in pixels
        speeds_kmh: Array of speeds in km/h (len = len(centers) - 1)
        start_frame: Frame number where tracking started
        trail_length: Number of previous positions to show as trail (default 20)
        ball_color: BGR color for ball marker (default green)
        trail_color: BGR color for trajectory trail (default cyan)
        text_color: BGR color for speed text (default white)

    Note:
        Output is MP4 with H.264 codec. The video includes:
        - Ball position marker (circle)
        - Trajectory trail (line connecting last N positions)
        - Speed label near the ball
        - Max speed indicator in corner
        
        If clip_to_tracking=True (default), only outputs frames with tracking data.
        This is MUCH faster for long videos where tracking starts late.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use MJPG codec (reliable) to temp file, then ffmpeg to H.264
    temp_avi = None
    use_ffmpeg = True
    
    # Try direct H.264 first
    for codec in ['avc1', 'H264', 'X264']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            # Test if it actually works by writing a dummy frame
            test_frame = np.zeros((height, width, 3), dtype=np.uint8)
            out.write(test_frame)
            out.release()
            # Check if file was created with reasonable size
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                use_ffmpeg = False
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                break
            os.remove(output_path)
        out.release()
    
    if use_ffmpeg:
        # Fall back to MJPG -> ffmpeg pipeline
        temp_avi = tempfile.NamedTemporaryFile(suffix='.avi', delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_avi, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise IOError("Cannot create temporary video file")

    # Precompute max speed for display
    max_speed = float(np.max(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0

    # Scale font/marker sizes based on video resolution
    scale = min(width, height) / 1080.0
    font_scale = max(0.5, scale * 1.2)
    thickness = max(1, int(scale * 2))
    ball_radius = max(5, int(scale * 15))
    trail_thickness = max(1, int(scale * 3))

    frame_idx = 0
    tracking_idx = 0  # Index into centers/speeds arrays

    # Skip to start_frame if clipping (MUCH faster)
    if clip_to_tracking and start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only annotate frames from start_frame onwards where we have tracking data
        if frame_idx >= start_frame and tracking_idx < len(centers):
            cx, cy = centers[tracking_idx]
            cx, cy = int(cx), int(cy)

            # Draw trajectory trail
            trail_start = max(0, tracking_idx - trail_length)
            for i in range(trail_start, tracking_idx):
                p1 = (int(centers[i][0]), int(centers[i][1]))
                p2 = (int(centers[i + 1][0]), int(centers[i + 1][1]))
                # Fade trail: older = more transparent (thinner line as approximation)
                age = tracking_idx - i
                line_thick = max(1, trail_thickness - int(age * trail_thickness / trail_length))
                cv2.line(frame, p1, p2, trail_color, line_thick)

            # Draw ball marker
            cv2.circle(frame, (cx, cy), ball_radius, ball_color, -1)
            cv2.circle(frame, (cx, cy), ball_radius + 2, (0, 0, 0), 2)  # Black outline

            # Draw speed label near ball (offset to avoid occlusion)
            if tracking_idx < len(speeds_kmh):
                speed = speeds_kmh[tracking_idx]
                label = f"{speed:.1f} km/h"
                label_x = cx + ball_radius + 10
                label_y = cy - ball_radius
                # Ensure label stays in frame
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                if label_x + text_size[0] > width:
                    label_x = cx - ball_radius - text_size[0] - 10
                if label_y - text_size[1] < 0:
                    label_y = cy + ball_radius + text_size[1] + 10
                # Draw text with black outline for readability
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

            tracking_idx += 1

        # Draw max speed indicator in top-left corner (always visible after tracking starts)
        if frame_idx >= start_frame:
            max_label = f"Max: {max_speed:.1f} km/h"
            cv2.putText(frame, max_label, (20, int(50 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 0), thickness + 2)
            cv2.putText(frame, max_label, (20, int(50 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 0, 255), thickness)

        out.write(frame)
        frame_idx += 1

        # Stop early if clipping and we've processed all tracked frames
        if clip_to_tracking and tracking_idx >= len(centers):
            break

    cap.release()
    out.release()

    # Convert temp AVI to MP4 with ffmpeg if needed
    if use_ffmpeg and temp_avi:
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_avi,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Compatibility
                output_path
            ], check=True, capture_output=True)
        finally:
            # Clean up temp file
            if os.path.exists(temp_avi):
                os.remove(temp_avi)
