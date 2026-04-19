#!/usr/bin/env python3
"""
Annotated video generator for tennis serve analysis.

Creates a video with visual overlays:
- Ball tracking circle with motion trail
- Real-time velocity display
- Serve counter and phase indicator (TOSS / HIT / FLIGHT)
- Peak velocity highlight
- Timestamp

Usage:
    python -m serve_analyzer.annotate_video video.mov -o annotated.mp4

Requires pre-computed analysis (serves_analysis.json) or runs detection inline.
"""

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Reuse detection functions from multi_serve
from serve_analyzer.multi_serve import (
    detect_ball_yolo,
    interpolate_missing_detections,
    compute_frame_velocities,
    detect_serve_events,
    compute_vertical_velocity,
)


# ============================================================================
# DRAWING UTILITIES
# ============================================================================


def draw_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    txt_color: Tuple[int, int, int] = (255, 255, 255),
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.7,
    thickness: int = 2,
    margin: int = 6,
) -> np.ndarray:
    """Draw text with a solid background box for readability."""
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)

    # Rectangle corners (org is bottom-left of text)
    tl = (org[0] - margin, org[1] - h - margin)
    br = (org[0] + w + margin, org[1] + margin)

    cv2.rectangle(img, tl, br, bg_color, cv2.FILLED)
    cv2.putText(img, text, org, font, scale, txt_color, thickness, cv2.LINE_AA)
    return img


def draw_trail(
    img: np.ndarray,
    trail: deque,
    color_start: Tuple[int, int, int] = (0, 255, 255),  # Yellow (newest)
    color_end: Tuple[int, int, int] = (0, 100, 100),  # Dim yellow (oldest)
    thickness: int = 2,
) -> np.ndarray:
    """Draw fading motion trail from position history."""
    trail_len = len(trail)
    if trail_len < 2:
        return img

    for i in range(1, trail_len):
        if trail[i - 1] is None or trail[i] is None:
            continue

        # Interpolate color based on age
        t = i / trail_len
        color = tuple(
            int(color_start[c] * (1 - t) + color_end[c] * t) for c in range(3)
        )

        pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        pt2 = (int(trail[i][0]), int(trail[i][1]))
        cv2.line(img, pt1, pt2, color, thickness)

    return img


def draw_ball_marker(
    img: np.ndarray,
    pos: Tuple[float, float],
    radius: int = 12,
    color: Tuple[int, int, int] = (0, 255, 0),  # Green
    thickness: int = 3,
) -> np.ndarray:
    """Draw circle marker at ball position."""
    center = (int(pos[0]), int(pos[1]))
    cv2.circle(img, center, radius, color, thickness)
    # Inner dot
    cv2.circle(img, center, 3, color, cv2.FILLED)
    return img


def draw_velocity_near_ball(
    img: np.ndarray,
    pos: Tuple[float, float],
    velocity_kmh: float,
    offset: Tuple[int, int] = (20, -20),
) -> np.ndarray:
    """Draw velocity text near ball position."""
    text_pos = (int(pos[0]) + offset[0], int(pos[1]) + offset[1])

    # Color based on velocity (green->yellow->red)
    if velocity_kmh < 30:
        color = (100, 255, 100)  # Light green
    elif velocity_kmh < 60:
        color = (0, 255, 255)  # Yellow
    elif velocity_kmh < 90:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 0, 255)  # Red

    return draw_label(
        img,
        f"{velocity_kmh:.0f} km/h",
        text_pos,
        bg_color=(30, 30, 30),
        txt_color=color,
        scale=0.6,
        thickness=2,
    )


# ============================================================================
# VIDEO ANNOTATION PIPELINE
# ============================================================================


@dataclass
class AnnotationState:
    """Mutable state for annotation pipeline."""

    trail: deque
    current_serve: int
    current_phase: str  # "IDLE", "TOSS", "HIT", "FLIGHT"
    phase_start_frame: int
    peak_velocity_frame: int
    peak_velocity_value: float
    show_peak_flash: bool


def determine_phase(
    frame_idx: int,
    serve_events: List[Dict[str, Any]],
) -> Tuple[int, str, Optional[Dict]]:
    """
    Determine current serve number and phase.

    Returns:
        (serve_number, phase_name, serve_event_dict or None)
    """
    for i, ev in enumerate(serve_events):
        toss_start = ev.get("toss_start_frame", 0)
        contact = ev.get("contact_frame", 0)
        post_end = ev.get("post_contact_end_frame", 0)

        if toss_start <= frame_idx < contact:
            return i + 1, "TOSS", ev
        elif contact <= frame_idx < post_end:
            return i + 1, "FLIGHT", ev

    # Between serves or before first
    for i, ev in enumerate(serve_events):
        toss_start = ev.get("toss_start_frame", 0)
        if frame_idx < toss_start:
            return i, "IDLE", None

    return len(serve_events), "IDLE", None


def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    positions: List[Tuple[float, float]],
    velocities_kmh: np.ndarray,
    serve_events: List[Dict[str, Any]],
    state: AnnotationState,
    fps: float,
    total_serves: int,
) -> np.ndarray:
    """Apply all annotations to a single frame."""
    h, w = frame.shape[:2]

    # Get current position and velocity
    pos = positions[frame_idx] if frame_idx < len(positions) else None
    vel = velocities_kmh[frame_idx] if frame_idx < len(velocities_kmh) else 0.0

    # Update trail
    if pos and pos != (0.0, 0.0):
        state.trail.appendleft(pos)
    else:
        state.trail.appendleft(None)

    # Determine serve phase
    serve_num, phase, serve_ev = determine_phase(frame_idx, serve_events)

    # Track peak velocity within current serve
    if phase in ("TOSS", "FLIGHT") and vel > state.peak_velocity_value:
        state.peak_velocity_value = vel
        state.peak_velocity_frame = frame_idx
        state.show_peak_flash = True

    # Reset peak tracking on serve change
    if serve_num != state.current_serve:
        state.current_serve = serve_num
        state.peak_velocity_value = 0.0
        state.show_peak_flash = False

    # ---- DRAW OVERLAYS ----

    # 1. Motion trail
    frame = draw_trail(frame, state.trail)

    # 2. Ball marker
    if pos and pos != (0.0, 0.0):
        # Flash effect for peak velocity
        if state.show_peak_flash and frame_idx - state.peak_velocity_frame < int(
            fps * 0.3
        ):
            marker_color = (0, 0, 255)  # Red flash
            radius = 18
        else:
            marker_color = (0, 255, 0)  # Green
            radius = 12
            state.show_peak_flash = False

        frame = draw_ball_marker(frame, pos, radius=radius, color=marker_color)

        # 3. Velocity near ball (only during active phases)
        if phase in ("TOSS", "FLIGHT") and vel > 5:
            frame = draw_velocity_near_ball(frame, pos, vel)

    # 4. Top-left: Serve counter
    serve_text = f"Serve {serve_num}/{total_serves}" if serve_num > 0 else "Waiting..."
    frame = draw_label(frame, serve_text, (20, 40), bg_color=(50, 50, 50))

    # 5. Top-left: Phase indicator (below serve counter)
    phase_colors = {
        "IDLE": (128, 128, 128),  # Gray
        "TOSS": (255, 200, 0),  # Cyan
        "HIT": (0, 0, 255),  # Red
        "FLIGHT": (0, 255, 0),  # Green
    }
    if phase != "IDLE":
        frame = draw_label(
            frame,
            phase,
            (20, 85),
            bg_color=(30, 30, 30),
            txt_color=phase_colors.get(phase, (255, 255, 255)),
            scale=0.9,
            thickness=2,
        )

    # 6. Top-right: Timestamp
    timestamp = frame_idx / fps
    time_text = f"{timestamp:.1f}s"
    frame = draw_label(frame, time_text, (w - 100, 40), bg_color=(50, 50, 50))

    # 7. Bottom: Peak velocity for current serve
    if serve_ev and state.peak_velocity_value > 10:
        peak_text = f"Peak: {state.peak_velocity_value:.0f} km/h"
        frame = draw_label(
            frame,
            peak_text,
            (20, h - 30),
            bg_color=(0, 50, 0),
            txt_color=(0, 255, 0),
            scale=0.7,
        )

    return frame


def create_annotated_video(
    video_path: str,
    output_path: str,
    positions: List[Tuple[float, float]],
    velocities_kmh: np.ndarray,
    serve_events: List[Dict[str, Any]],
    fps: float,
    total_serves: int,
    codec: str = "mp4v",
    trail_length: int = 25,
    progress_interval: int = 100,
) -> None:
    """
    Create annotated video with all overlays.

    Args:
        video_path: Input video
        output_path: Output video path
        positions: Ball positions per frame
        velocities_kmh: Velocity in km/h per frame
        serve_events: List of serve event dicts
        fps: Frame rate
        total_serves: Number of serves
        codec: FourCC codec (mp4v, avc1, XVID)
        trail_length: Number of frames for motion trail
        progress_interval: Print progress every N frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot create output video: {output_path}")

    print(f"Creating annotated video: {output_path}")
    print(
        f"  Resolution: {width}x{height}, FPS: {video_fps:.1f}, Frames: {total_frames}"
    )

    state = AnnotationState(
        trail=deque(maxlen=trail_length),
        current_serve=0,
        current_phase="IDLE",
        phase_start_frame=0,
        peak_velocity_frame=0,
        peak_velocity_value=0.0,
        show_peak_flash=False,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply annotations
        annotated = annotate_frame(
            frame,
            frame_idx,
            positions,
            velocities_kmh,
            serve_events,
            state,
            fps,
            total_serves,
        )

        out.write(annotated)
        frame_idx += 1

        if frame_idx % progress_interval == 0:
            pct = 100 * frame_idx / total_frames
            print(f"  Processed {frame_idx}/{total_frames} ({pct:.1f}%)")

    cap.release()
    out.release()
    print(f"Done! Output: {output_path}")


# ============================================================================
# CLI
# ============================================================================


def load_analysis(analysis_path: str) -> Dict[str, Any]:
    """
    Load pre-computed analysis from JSON and transform to expected format.
    
    The JSON from multi_serve.py uses nested structure:
        serves[].toss.start_frame, serves[].post_contact.end_frame
    
    This function flattens it to:
        serve_events[].toss_start_frame, serve_events[].post_contact_end_frame
    """
    with open(analysis_path, "r") as f:
        raw = json.load(f)
    
    # Transform nested serve structure to flat serve_events
    serve_events = []
    for s in raw.get("serves", []):
        ev = {
            "serve_number": s.get("serve_number", 0),
            "toss_start_frame": s.get("toss", {}).get("start_frame", 0),
            "contact_frame": s.get("contact_frame", 0),
            "post_contact_end_frame": s.get("post_contact", {}).get("end_frame", 0),
            "toss_max_velocity_kmh": s.get("toss", {}).get("max_velocity_kmh", 0),
            "post_contact_max_velocity_kmh": s.get("post_contact", {}).get("max_velocity_kmh", 0),
            "peak_frame": s.get("peak_frame", 0),
            "peak_position": s.get("peak_position", [0, 0]),
        }
        serve_events.append(ev)
    
    return {
        "fps": raw.get("fps", 60.0),
        "total_frames": raw.get("total_frames", 0),
        "scale_factor": raw.get("scale_factor_m_per_px", 0.0006),
        "serve_events": serve_events,
        "positions": raw.get("positions", []),  # May be empty in older JSONs
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create annotated tennis serve video with velocity overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use pre-computed analysis
    python -m serve_analyzer.annotate_video video.mov -a serves_analysis.json -o annotated.mp4
    
    # Run detection inline (slower)
    python -m serve_analyzer.annotate_video video.mov -n 8 --frame-skip 4 -o annotated.mp4
""",
    )

    parser.add_argument("video", help="Input video file")
    parser.add_argument(
        "-o", "--output", default="annotated.mp4", help="Output video path"
    )
    parser.add_argument(
        "-a", "--analysis", help="Pre-computed analysis JSON (from multi_serve.py)"
    )
    parser.add_argument(
        "-n", "--expected-serves", type=int, default=8, help="Expected number of serves"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame for detection",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        help="Scale factor (m/px), auto-estimated if not provided",
    )
    parser.add_argument(
        "--codec", default="mp4v", help="Video codec (mp4v, avc1, XVID)"
    )
    parser.add_argument(
        "--trail-length", type=int, default=25, help="Motion trail length in frames"
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Load or compute analysis
    if args.analysis:
        print(f"Loading analysis from: {args.analysis}")
        data = load_analysis(args.analysis)
        fps = data.get("fps", 60.0)
        scale_factor = args.scale_factor or data.get("scale_factor", 0.0006)
        serve_events = data.get("serve_events", [])
        loaded_positions = data.get("positions", [])
        
        if loaded_positions:
            print(f"  Loaded {len(loaded_positions)} positions from JSON")
            positions = [tuple(p) if p else (0.0, 0.0) for p in loaded_positions]
        else:
            # Positions not in JSON - must run detection
            print("  Positions not in JSON, running detection...")
            raw_detections, fps_detected, total_frames, _ = detect_ball_yolo(
                str(video_path),
                frame_skip=args.frame_skip,
            )
            positions = interpolate_missing_detections(raw_detections)
    else:
        # Full detection mode
        print("Running ball detection (this may take a while)...")
        
        raw_detections, fps, total_frames, estimated_scale = detect_ball_yolo(
            str(video_path),
            frame_skip=args.frame_skip,
        )
        
        positions = interpolate_missing_detections(raw_detections)
        scale_factor = args.scale_factor or estimated_scale or 0.0006
        print(f"Using scale factor: {scale_factor:.6f} m/px")
        
        velocities_px = compute_frame_velocities(positions, fps)
        vert_velocities = compute_vertical_velocity(positions)
        
        serve_events = detect_serve_events(
            positions,
            velocities_px,
            vert_velocities,
            fps,
            expected_serves=args.expected_serves,
        )

    # Compute velocities for annotation
    velocities_px = compute_frame_velocities(positions, fps)
    velocities_mps = velocities_px * scale_factor * fps
    velocities_kmh = velocities_mps * 3.6

    total_serves = len(serve_events)
    print(f"Found {total_serves} serves")

    # Create annotated video
    create_annotated_video(
        str(video_path),
        args.output,
        positions,
        velocities_kmh,
        serve_events,
        fps,
        total_serves,
        codec=args.codec,
        trail_length=args.trail_length,
    )


if __name__ == "__main__":
    main()
