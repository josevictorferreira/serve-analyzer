#!/usr/bin/env python3
"""
Multi-serve velocity analyzer.

Detects ALL serve events in a video, distinguishes toss vs post-contact phases,
and computes velocity for each phase of each serve.

Usage:
    python -m serve_analyzer.multi_serve video.mov --expected-serves 8

Algorithm:
    1. Run YOLO ball detection on entire video
    2. Analyze ball trajectory for serve patterns:
       - Toss: ball moving UP slowly
       - Peak: ball at apex, direction reversal
       - Post-contact: ball moving DOWN/AWAY fast
    3. Detect velocity spikes to find racket contact moments
    4. Segment each serve and compute phase velocities
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# Import from existing codebase
from serve_analyzer.analysis import (
    compute_velocity_series,
)


@dataclass
class ServeEvent:
    """Represents a single serve with toss and post-contact phases."""

    serve_number: int

    # Frame ranges
    toss_start_frame: int
    toss_end_frame: int  # = contact frame
    contact_frame: int
    post_contact_end_frame: int

    # Ball positions
    toss_positions: List[Tuple[float, float]]
    post_contact_positions: List[Tuple[float, float]]

    # Velocities (km/h)
    toss_max_velocity: float
    toss_mean_velocity: float
    post_contact_max_velocity: float
    post_contact_mean_velocity: float

    # Peak detection
    peak_position: Tuple[float, float]
    peak_frame: int


def detect_ball_yolo(
    video_path: str,
    model_path: str = "rjtp",
    conf_threshold: float = 0.20,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    start_frame: int = 0,  # Skip frames before this
    progress_interval: int = 100,
) -> Tuple[List[Optional[Tuple[float, float]]], float, int, Optional[float]]:
    """
    Detect ball in ALL frames using YOLO.

    Returns:
        (detections, fps, total_frames, estimated_scale)
        - detections: List where each element is (x, y) or None if no ball found
        - fps: Video frame rate
        - total_frames: Number of frames processed
        - estimated_scale: Estimated m/px based on ball size (None if not enough data)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required: pip install ultralytics")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

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

    print(
        f"Processing {total_frames} frames at {fps:.1f} FPS ({total_frames / fps:.1f}s video)..."
    )

    detections: List[Optional[Tuple[float, float]]] = []
    ball_sizes: List[float] = []  # Collect ball diameters for scale estimation

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames before start_frame or if frame_skip applies
        if frame_idx < start_frame or (frame_skip > 1 and frame_idx % frame_skip != 0):
            # Insert None for skipped frames (will be interpolated later)
            detections.append(None)
            frame_idx += 1
            continue
        # Run YOLO
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False,
            device="cpu",
            classes=[32] if not _is_rjtp else None,  # sports ball class in COCO (not needed for RJTPP)
        )

        ball_pos = None

        # Find best ball detection (smallest bounding box = likely the ball)
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            best_area = float("inf")

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                area = w * h

                # Filter: ball should be small relative to frame
                if area < width * height * 0.01 and area < best_area:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    ball_pos = (float(cx), float(cy))
                    best_area = area
                    ball_sizes.append((w + h) / 2)  # Avg diameter in pixels

        # Fallback: HSV color detection for yellow tennis ball
        if ball_pos is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Yellow tennis ball range
            mask = cv2.inRange(hsv, (18, 80, 80), (45, 255, 255))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            valid = [c for c in contours if 100 < cv2.contourArea(c) < 10000]
            if valid:
                # Take largest valid contour
                largest = max(valid, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    ball_pos = (float(cx), float(cy))
                    # Estimate diameter from contour area (circle: area = pi*r^2)
                    area = cv2.contourArea(largest)
                    diameter_px = 2 * np.sqrt(area / np.pi)
                    ball_sizes.append(diameter_px)

        detections.append(ball_pos)

        frame_idx += 1
        if frame_idx % progress_interval == 0:
            print(
                f"  Processed {frame_idx}/{total_frames} frames ({100 * frame_idx / total_frames:.1f}%)"
            )

    cap.release()

    # Estimate scale factor from ball sizes
    # Tennis ball diameter = 6.7cm = 0.067m
    estimated_scale = None
    if len(ball_sizes) >= 10:
        median_diameter_px = np.median(ball_sizes)
        estimated_scale = 0.067 / median_diameter_px  # m/px
        print(
            f"Estimated scale: {estimated_scale:.6f} m/px (from median ball diameter {median_diameter_px:.1f}px)"
        )

    print(
        f"Detection complete. Found ball in {sum(1 for d in detections if d is not None)}/{len(detections)} frames"
    )

    return detections, fps, len(detections), estimated_scale


def interpolate_missing_detections(
    detections: List[Optional[Tuple[float, float]]], max_gap: int = 10
) -> List[Tuple[float, float]]:
    """
    Fill gaps in detections using linear interpolation.

    Args:
        detections: List with (x,y) or None
        max_gap: Maximum gap size to interpolate (frames)

    Returns:
        List of (x, y) with gaps filled
    """
    result = list(detections)
    n = len(result)

    # Find gaps and interpolate
    i = 0
    while i < n:
        if result[i] is None:
            # Find gap boundaries
            gap_start = i
            while i < n and result[i] is None:
                i += 1
            gap_end = i
            gap_size = gap_end - gap_start

            # Interpolate if gap is small enough and we have both boundaries
            if gap_size <= max_gap:
                if gap_start > 0 and gap_end < n:
                    start_pos = result[gap_start - 1]
                    end_pos = result[gap_end]
                    if start_pos and end_pos:
                        for j in range(gap_start, gap_end):
                            t = (j - gap_start + 1) / (gap_size + 1)
                            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
                            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
                            result[j] = (x, y)
        else:
            i += 1

    # Replace remaining None with last known position (forward fill)
    last_pos = None
    for i in range(n):
        if result[i] is not None:
            last_pos = result[i]
        elif last_pos is not None:
            result[i] = last_pos

    # Backward fill for initial Nones
    first_pos = None
    for i in range(n - 1, -1, -1):
        if result[i] is not None:
            first_pos = result[i]
        elif first_pos is not None:
            result[i] = first_pos

    return [(p[0], p[1]) if p else (0.0, 0.0) for p in result]


def compute_frame_velocities(
    positions: List[Tuple[float, float]], fps: float, smooth_sigma: float = 2.0
) -> np.ndarray:
    """
    Compute instantaneous velocity magnitude at each frame.

    Returns:
        Array of velocities in pixels/frame (length = len(positions))
    """
    positions = np.array(positions)
    n = len(positions)

    if n < 2:
        return np.zeros(n)

    # Compute frame-to-frame displacements
    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])
    velocities = np.sqrt(dx**2 + dy**2)

    # Pad to match input length
    velocities = np.concatenate([[velocities[0]], velocities])

    # Smooth to reduce noise
    if smooth_sigma > 0:
        velocities = gaussian_filter1d(velocities, sigma=smooth_sigma)

    return velocities


def compute_vertical_velocity(
    positions: List[Tuple[float, float]], smooth_sigma: float = 2.0
) -> np.ndarray:
    """
    Compute vertical velocity (positive = downward in image coords).

    Returns:
        Array of vertical velocities (length = len(positions))
    """
    positions = np.array(positions)
    n = len(positions)

    if n < 2:
        return np.zeros(n)

    # Vertical displacement (y increases downward in image)
    dy = np.diff(positions[:, 1])

    # Pad
    dy = np.concatenate([[dy[0]], dy])

    # Smooth
    if smooth_sigma > 0:
        dy = gaussian_filter1d(dy, sigma=smooth_sigma)

    return dy


def detect_serve_events(
    positions: List[Tuple[float, float]],
    velocities: np.ndarray,
    vert_velocities: np.ndarray,
    fps: float,
    expected_serves: int = 8,
    min_serve_gap_sec: float = 3.0,
    velocity_spike_percentile: float = 90,
    toss_lookback_sec: float = 1.5,
    post_contact_duration_sec: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Detect serve events by finding velocity spikes after upward ball motion.

    A serve is characterized by:
    1. Ball moving UP (toss) - negative vert velocity in image coords
    2. Peak/apex - direction reversal
    3. Sudden velocity spike (racket contact)
    4. Fast motion (post-contact)

    Returns:
        List of serve event dictionaries
    """
    n = len(positions)
    min_gap_frames = int(min_serve_gap_sec * fps)
    toss_lookback = int(toss_lookback_sec * fps)
    post_duration = int(post_contact_duration_sec * fps)

    # Find velocity spikes (potential contact points)
    velocity_threshold = np.percentile(velocities, velocity_spike_percentile)

    # Find peaks in velocity
    peaks, properties = signal.find_peaks(
        velocities,
        height=velocity_threshold,
        distance=min_gap_frames,
        prominence=velocity_threshold * 0.3,
    )

    print(f"Found {len(peaks)} velocity peaks above {velocity_threshold:.1f} px/frame")

    # Validate each peak as a serve contact
    serve_events = []

    for peak_frame in peaks:
        # Look back to find toss phase (ball moving upward = negative vert velocity)
        search_start = max(0, peak_frame - toss_lookback)
        toss_region = vert_velocities[search_start:peak_frame]

        upward_fraction = (
            float(np.mean(toss_region < -1)) if len(toss_region) > 0 else 0.0
        )
        recent_window = int(0.6 * fps)
        recent_region = vert_velocities[
            max(search_start, peak_frame - recent_window) : peak_frame
        ]
        recent_upward_fraction = (
            float(np.mean(recent_region < -1)) if len(recent_region) > 0 else 0.0
        )

        # Find apex (highest point) first - this is where ball direction reverses
        toss_positions = positions[search_start : peak_frame + 1]
        if len(toss_positions) == 0:
            continue

        y_coords = [p[1] for p in toss_positions]
        apex_idx = np.argmin(y_coords)  # Min y = highest in image
        apex_frame = search_start + apex_idx
        apex_position = toss_positions[apex_idx]
        frames_after_apex = peak_frame - apex_frame
        drop_after_apex = positions[peak_frame][1] - apex_position[1]

        upward_motion = upward_fraction > 0.3 or (
            recent_upward_fraction > 0.35
            and frames_after_apex <= int(0.4 * fps)
            and drop_after_apex > 120
        )

        if not upward_motion:
            continue  # Not a serve - no toss detected

        # Find toss start: scan back from apex to find where ball had low velocity
        # (indicating release from hand)
        toss_start = search_start
        velocities_region = velocities[search_start : apex_frame + 1]

        if len(velocities_region) > 5:
            # Velocity threshold: 10% of max velocity in toss region
            vel_threshold = np.max(velocities_region) * 0.15

            # Scan backward from apex to find where velocity was low (ball release)
            for i in range(apex_frame - 1, search_start - 1, -1):
                if velocities[i] < vel_threshold:
                    toss_start = i
                    break

        # Post-contact phase
        # Post-contact phase
        post_end = min(n - 1, peak_frame + post_duration)

        serve_events.append(
            {
                "contact_frame": peak_frame,
                "toss_start_frame": toss_start,
                "apex_frame": apex_frame,
                "apex_position": apex_position,
                "post_contact_end_frame": post_end,
                "contact_velocity": velocities[peak_frame],
                "upward_fraction": upward_fraction,
                "recent_upward_fraction": recent_upward_fraction,
                "drop_after_apex": drop_after_apex,
            }
        )

    # Score each candidate to find the most likely true serves
    for event in serve_events:
        contact_frame = event["contact_frame"]
        apex_frame = event["apex_frame"]
        post_end = event["post_contact_end_frame"]
        contact_vel = event["contact_velocity"]
        upward_fraction = event["upward_fraction"]
        recent_upward_fraction = event["recent_upward_fraction"]
        drop_after_apex = event["drop_after_apex"]

        score = float(contact_vel)

        # Acceleration/change at contact
        pre_start = max(0, contact_frame - 5)
        if pre_start < contact_frame:
            pre_vel = np.mean(velocities[pre_start:contact_frame])
            score += float(max(0, contact_vel - pre_vel))

        score += upward_fraction * 80.0
        score += recent_upward_fraction * 120.0

        score += float(max(0.0, drop_after_apex) * 0.35)

        # Post-contact sustained motion
        post_vels = velocities[contact_frame + 1 : post_end]
        if len(post_vels) > 0:
            score += float(np.mean(post_vels) * 2.0)

        # Proximity after apex penalty
        frames_after_apex = contact_frame - apex_frame
        if frames_after_apex <= fps * 0.45:
            score += 120.0
        elif frames_after_apex <= fps * 0.75:
            score += 60.0
        if frames_after_apex > fps * 0.5:
            score -= float((frames_after_apex - fps * 0.5) / fps * 140.0)

        if drop_after_apex < 120:
            score -= float((120.0 - drop_after_apex) * 2.5)

        event["score"] = score

    # Sort by score descending to get best candidates
    serve_events.sort(key=lambda x: x["score"], reverse=True)

    # Filter to expected serves, preferring non-overlapping
    selected_events = []
    for event in serve_events:
        # Check overlap with already selected
        overlap = False
        for sel in selected_events:
            if abs(event["contact_frame"] - sel["contact_frame"]) < min_gap_frames:
                overlap = True
                break
        if not overlap:
            selected_events.append(event)
        if len(selected_events) >= expected_serves:
            break

    # Sort selected by temporal order
    selected_events.sort(key=lambda x: x["contact_frame"])

    print(f"Identified {len(selected_events)} serve events")

    return selected_events


def analyze_serve(
    serve_idx: int,
    event: Dict[str, Any],
    positions: List[Tuple[float, float]],
    fps: float,
    scale_factor: float = 0.001,  # Default: 1 pixel = 1mm (approximate)
) -> ServeEvent:
    """
    Analyze a single serve event and compute velocities.

    Args:
        serve_idx: 0-based serve index
        event: Serve event dict from detect_serve_events
        positions: Full position list
        fps: Video FPS
        scale_factor: meters per pixel

    Returns:
        ServeEvent with computed velocities
    """
    toss_start = event["toss_start_frame"]
    contact = event["contact_frame"]
    post_end = event["post_contact_end_frame"]

    # Extract phase positions
    toss_positions = positions[toss_start : contact + 1]
    post_positions = positions[contact : post_end + 1]

    # Compute velocities for each phase
    def phase_stats(phase_positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        if len(phase_positions) < 2:
            return 0.0, 0.0
        _, speeds_kmh, stats = compute_velocity_series(
            phase_positions, fps, scale_factor, smoothing_window=3
        )
        return stats["max_kmh"], stats["mean_kmh"]

    toss_max, toss_mean = phase_stats(toss_positions)
    post_max, post_mean = phase_stats(post_positions)

    return ServeEvent(
        serve_number=serve_idx + 1,
        toss_start_frame=toss_start,
        toss_end_frame=contact,
        contact_frame=contact,
        post_contact_end_frame=post_end,
        toss_positions=toss_positions,
        post_contact_positions=post_positions,
        toss_max_velocity=toss_max,
        toss_mean_velocity=toss_mean,
        post_contact_max_velocity=post_max,
        post_contact_mean_velocity=post_mean,
        peak_position=event["apex_position"],
        peak_frame=event["apex_frame"],
    )


def generate_debug_video(
    video_path: str,
    output_path: str,
    positions: List[Tuple[float, float]],
    velocities: np.ndarray,
    serve_events: List[ServeEvent],
    fps: float,
) -> None:
    """Generate annotated video showing all serve detections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create frame->serve mapping
    frame_to_serve: Dict[int, Tuple[ServeEvent, str]] = {}
    for serve in serve_events:
        for f in range(serve.toss_start_frame, serve.contact_frame + 1):
            frame_to_serve[f] = (serve, "toss")
        for f in range(serve.contact_frame, serve.post_contact_end_frame + 1):
            frame_to_serve[f] = (serve, "post")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(positions):
            pos = positions[frame_idx]
            vel = velocities[frame_idx] if frame_idx < len(velocities) else 0

            # Draw ball position
            cx, cy = int(pos[0]), int(pos[1])

            # Color based on phase
            if frame_idx in frame_to_serve:
                serve, phase = frame_to_serve[frame_idx]
                if phase == "toss":
                    color = (255, 255, 0)  # Cyan - toss
                    label = f"Serve {serve.serve_number} TOSS"
                else:
                    color = (0, 255, 0)  # Green - post-contact
                    label = f"Serve {serve.serve_number} POST ({serve.post_contact_max_velocity:.0f} km/h)"

                # Mark contact frame specially
                if frame_idx == serve.contact_frame:
                    cv2.circle(frame, (cx, cy), 50, (0, 0, 255), 5)
                    label = f"Serve {serve.serve_number} CONTACT!"
            else:
                color = (128, 128, 128)  # Gray - between serves
                label = ""

            cv2.circle(frame, (cx, cy), 20, color, -1)

            # Velocity bar
            bar_height = int(vel * 2)
            cv2.rectangle(
                frame, (50, height - 50 - bar_height), (100, height - 50), color, -1
            )

            # Labels
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Vel: {vel:.1f} px/f",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )
            if label:
                cv2.putText(
                    frame, label, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Debug video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect multiple serves in a video and compute velocities for toss and post-contact phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze video with 8 expected serves
    python -m serve_analyzer.multi_serve video.mov --expected-serves 8
    
    # With debug video output
    python -m serve_analyzer.multi_serve video.mov -n 8 --debug-video
    
    # Custom scale factor (if known)
    python -m serve_analyzer.multi_serve video.mov -n 8 --scale-factor 0.002
        """,
    )

    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "-n",
        "--expected-serves",
        type=int,
        default=8,
        help="Expected number of serves (default: 8)",
    )
    parser.add_argument(
        "--model", default="rjtp", help="Model path or 'rjtp' for tennis-ball model (default: rjtp)"
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.001,
        help="Meters per pixel (default: 0.001 = 1mm/px)",
    )
    parser.add_argument(
        "--debug-video", action="store_true", help="Generate annotated debug video"
    )
    parser.add_argument("--output", "-o", type=str, help="Output JSON file for results")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="YOLO confidence threshold (default: 0.20)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames, use 2-4 for faster processing)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip frames before this frame number (default: 0)",
    )

    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Step 1: Detect ball in all frames
    print("\n=== Step 1: Ball Detection ===")
    raw_detections, fps, total_frames, estimated_scale = detect_ball_yolo(
        video_path,
        model_path=args.model,
        conf_threshold=args.conf,
        frame_skip=args.frame_skip,
        start_frame=args.start_frame,
    )

    # Use estimated scale if not provided by user
    effective_scale = args.scale_factor
    if estimated_scale is not None and args.scale_factor == 0.001:
        effective_scale = estimated_scale
        print(f"Using auto-estimated scale factor: {effective_scale:.6f} m/px")

    # Step 2: Interpolate missing detections
    print("\n=== Step 2: Interpolating Gaps ===")
    positions = interpolate_missing_detections(raw_detections, max_gap=15)
    print(
        f"Interpolated {sum(1 for r, p in zip(raw_detections, positions) if r is None and p != (0, 0))} missing frames"
    )

    # Step 3: Compute velocities
    print("\n=== Step 3: Computing Velocities ===")
    velocities = compute_frame_velocities(positions, fps)
    vert_velocities = compute_vertical_velocity(positions)

    # Step 4: Detect serve events
    print("\n=== Step 4: Detecting Serve Events ===")
    events = detect_serve_events(
        positions,
        velocities,
        vert_velocities,
        fps=fps,
        expected_serves=args.expected_serves,
    )

    if len(events) < args.expected_serves:
        print(
            f"Warning: Only found {len(events)} serves, expected {args.expected_serves}"
        )

    # Step 5: Analyze each serve
    print("\n=== Step 5: Analyzing Serves ===")
    serve_events: List[ServeEvent] = []

    for i, event in enumerate(events):
        serve = analyze_serve(i, event, positions, fps, effective_scale)
        serve_events.append(serve)

        print(f"\nServe {serve.serve_number}:")
        print(
            f"  Contact frame: {serve.contact_frame} ({serve.contact_frame / fps:.2f}s)"
        )
        print(f"  Toss: frames {serve.toss_start_frame}-{serve.toss_end_frame}")
        print(f"    Max velocity: {serve.toss_max_velocity:.1f} km/h")
        print(f"    Mean velocity: {serve.toss_mean_velocity:.1f} km/h")
        print(
            f"  Post-contact: frames {serve.contact_frame}-{serve.post_contact_end_frame}"
        )
        print(f"    Max velocity: {serve.post_contact_max_velocity:.1f} km/h")
        print(f"    Mean velocity: {serve.post_contact_mean_velocity:.1f} km/h")

    # Step 6: Generate outputs
    if args.output:
        results = {
            "video": str(Path(video_path).name),
            "fps": fps,
            "total_frames": total_frames,
            "scale_factor_m_per_px": effective_scale,
            "serves": [
                {
                    "serve_number": int(s.serve_number),
                    "contact_frame": int(s.contact_frame),
                    "contact_time_sec": float(s.contact_frame / fps),
                    "toss": {
                        "start_frame": int(s.toss_start_frame),
                        "end_frame": int(s.toss_end_frame),
                        "max_velocity_kmh": float(s.toss_max_velocity),
                        "mean_velocity_kmh": float(s.toss_mean_velocity),
                    },
                    "post_contact": {
                        "start_frame": int(s.contact_frame),
                        "end_frame": int(s.post_contact_end_frame),
                        "max_velocity_kmh": float(s.post_contact_max_velocity),
                        "mean_velocity_kmh": float(s.post_contact_mean_velocity),
                    },
                    "peak_frame": int(s.peak_frame),
                    "peak_position": [float(x) for x in s.peak_position],
                }
                for s in serve_events
            ],
            "positions": [[float(p[0]), float(p[1])] if p else None for p in positions],
        }

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    if args.debug_video:
        debug_path = str(Path(video_path).stem) + "_multi_serve_debug.mp4"
        generate_debug_video(
            video_path, debug_path, positions, velocities, serve_events, fps
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Serve':<8} {'Contact':<12} {'Toss Max':<12} {'Post Max':<12}")
    print(f"{'#':<8} {'Time (s)':<12} {'(km/h)':<12} {'(km/h)':<12}")
    print("-" * 60)
    for s in serve_events:
        print(
            f"{s.serve_number:<8} {s.contact_frame / fps:<12.2f} {s.toss_max_velocity:<12.1f} {s.post_contact_max_velocity:<12.1f}"
        )
    print("=" * 60)

    return serve_events


if __name__ == "__main__":
    main()
