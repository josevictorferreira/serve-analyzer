#!/usr/bin/env python3
"""
Test Roboflow tennis ball model on serve-analyzer video.

Supports two usage modes:
  1. Roboflow hosted API (requires free API key):
       python -m serve_analyzer.test_roboflow_model IMG_9259.MOV --api-key YOUR_KEY
  2. Local .pt file (if you have Roboflow paid plan and downloaded weights):
       python -m serve_analyzer.test_roboflow_model IMG_9259.MOV --model weights.pt

The model page for this project is:
  https://universe.roboflow.com/tennis-kq1fm/tennis-ball-o7so8

Usage:
    # Option A — Roboflow hosted API (recommended, no local download needed)
    nix develop --command bash -c "source .venv/bin/activate; python -m serve_analyzer.test_roboflow_model \\
        IMG_9259.MOV \\
        --api-key YOUR_ROBOFLOW_API_KEY \\
        --model-id tennis-kq1fm/tennis-ball-o7so8/2 \\
        --start-frame 550 \\
        --frame-skip 4 \\
        --conf 0.25 \\
        -o roboflow_test.json"

    # Option B — Local .pt file (Roboflow paid plan only)
    nix develop --command bash -c "source .venv/bin/activate; python -m serve_analyzer.test_roboflow_model \\
        IMG_9259.MOV \\
        --model weights.pt \\
        --start-frame 550 \\
        --frame-skip 4 \\
        --conf 0.25 \\
        -o roboflow_test.json"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Roboflow hosted API
# ---------------------------------------------------------------------------
def detect_ball_via_roboflow_api(
    video_path: str,
    api_key: str,
    model_id: str,
    start_frame: int = 0,
    frame_skip: int = 1,
    conf: float = 0.25,
    max_frames: Optional[int] = None,
    progress_interval: int = 100,
) -> Tuple[List[Optional[Tuple[float, float]]], float, int]:
    """
    Detect tennis ball using Roboflow hosted inference API.

    Returns:
        (detections, fps, total_frames)
        detections[i] = (x, y) center or None if no detection
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    # Parse workspace/project/version from model_id like "tennis-kq1fm/tennis-ball-o7so8/2"
    parts = model_id.split("/")
    if len(parts) != 3:
        raise ValueError(f"model-id must be workspace/project/version, got: {model_id}")
    workspace_id, project_id, version = parts[0], parts[1], parts[2]
    project = rf.workspace(workspace_id).project(project_id)
    model = project.version(version).model

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(
        f"Roboflow API mode — processing {total_frames} frames at {fps:.1f} FPS "
        f"({total_frames / fps:.1f}s video)"
    )
    print(f"  start_frame={start_frame}, frame_skip={frame_skip}, conf={conf}")

    detections: List[Optional[Tuple[float, float]]] = []
    frame_idx = 0
    frames_processed = 0

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_idx < start_frame or (frame_skip > 1 and frame_idx % frame_skip != 0):
            detections.append(None)
            frame_idx += 1
            continue

        # Roboflow predict() returns JSON with 'predictions' key
        try:
            resp = model.predict(frame, confidence=conf)
            preds = resp.json()["predictions"] if hasattr(resp, 'json') else resp
        except Exception as e:
            print(f"  [WARN] Inference error at frame {frame_idx}: {e}")
            preds = []

        ball_pos = None
        if preds:
            # Take highest-confidence prediction as the ball
            best = max(preds, key=lambda p: p.get("confidence", 0))
            cx = best["x"]
            cy = best["y"]
            ball_pos = (float(cx), float(cy))

        detections.append(ball_pos)
        frames_processed += 1

        frame_idx += 1
        if frame_idx % progress_interval == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")

    cap.release()
    found = sum(1 for d in detections if d is not None)
    print(f"Detection complete. Found ball in {found}/{len(detections)} frames "
          f"({100*found/len(detections):.1f}%)")

    return detections, fps, len(detections)


# ---------------------------------------------------------------------------
# Local Ultralytics YOLO model (.pt file)
# ---------------------------------------------------------------------------
def detect_ball_via_local_model(
    video_path: str,
    model_path: str,
    start_frame: int = 0,
    frame_skip: int = 1,
    conf: float = 0.25,
    max_frames: Optional[int] = None,
    progress_interval: int = 100,
) -> Tuple[List[Optional[Tuple[float, float]]], float, int, Optional[float]]:
    """
    Detect tennis ball using a local Ultralytics YOLO .pt file.
    Mirrors the detection logic in multi_serve.py.

    Returns:
        (detections, fps, total_frames, estimated_scale)
    """
    from ultralytics import YOLO

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(
        f"Local model mode — processing {total_frames} frames at {fps:.1f} FPS "
        f"({total_frames / fps:.1f}s video)"
    )
    print(f"  model={model_path}, start_frame={start_frame}, frame_skip={frame_skip}, conf={conf}")

    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    detections: List[Optional[Tuple[float, float]]] = []
    ball_sizes: List[float] = []

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_idx < start_frame or (frame_skip > 1 and frame_idx % frame_skip != 0):
            detections.append(None)
            frame_idx += 1
            continue

        results = model.predict(
            source=frame,
            conf=conf,
            verbose=False,
            device="cpu",
            classes=[0],  # Roboflow YOLO models typically use class 0 as ball
        )

        ball_pos = None
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
                    ball_sizes.append((w + h) / 2)

        # Fallback: HSV detection for yellow tennis ball (same as multi_serve.py)
        if ball_pos is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (18, 80, 80), (45, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in contours if 100 < cv2.contourArea(c) < 10000]
            if valid:
                largest = max(valid, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    ball_pos = (float(cx), float(cy))
                    area = cv2.contourArea(largest)
                    diameter_px = 2 * np.sqrt(area / np.pi)
                    ball_sizes.append(diameter_px)

        detections.append(ball_pos)

        frame_idx += 1
        if frame_idx % progress_interval == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")

    cap.release()

    # Estimate scale factor from ball sizes (tennis ball = 6.7cm)
    estimated_scale = None
    if len(ball_sizes) >= 10:
        median_diameter_px = np.median(ball_sizes)
        estimated_scale = 0.067 / median_diameter_px
        print(f"Estimated scale: {estimated_scale:.6f} m/px (ball diameter {median_diameter_px:.1f}px)")

    found = sum(1 for d in detections if d is not None)
    print(f"Detection complete. Found ball in {found}/{len(detections)} frames "
          f"({100*found/len(detections):.1f}%)")

    return detections, fps, len(detections), estimated_scale


# ---------------------------------------------------------------------------
# Interpolate missing detections
# ---------------------------------------------------------------------------
def interpolate_missing_detections(
    detections: List[Optional[Tuple[float, float]]], max_gap: int = 10
) -> List[Tuple[float, float]]:
    """Fill gaps in detections using linear interpolation (same as multi_serve.py)."""
    result = list(detections)
    n = len(result)
    i = 0
    while i < n:
        if result[i] is None:
            gap_start = i
            while i < n and result[i] is None:
                i += 1
            gap_end = i
            gap_size = gap_end - gap_start
            if gap_size <= max_gap and gap_start > 0 and gap_end < n:
                x0, y0 = result[gap_start - 1]
                x1, y1 = result[gap_end]
                for j in range(gap_start, gap_end):
                    t = (j - gap_start + 1) / (gap_size + 1)
                    result[j] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            else:
                # Extend last known position into gap
                for j in range(gap_start, gap_end):
                    if gap_start > 0:
                        result[j] = result[gap_start - 1]
                    elif gap_end < n:
                        result[j] = result[gap_end]
        else:
            i += 1
    return result


# ---------------------------------------------------------------------------
# Compute velocity series
# ---------------------------------------------------------------------------
def compute_velocity_series(
    positions: List[Tuple[float, float]], fps: float
) -> Tuple[List[float], List[float]]:
    """Compute per-frame velocity magnitude (px/frame) and direction."""
    from scipy.ndimage import gaussian_filter1d

    n = len(positions)
    dx = np.zeros(n)
    dy = np.zeros(n)
    for i in range(1, n):
        dx[i] = positions[i][0] - positions[i - 1][0]
        dy[i] = positions[i][1] - positions[i - 1][1]

    # Smooth
    sigma = 2.0
    dx_smooth = gaussian_filter1d(dx, sigma)
    dy_smooth = gaussian_filter1d(dy, sigma)

    velocity = np.sqrt(dx_smooth**2 + dy_smooth**2)
    return dx_smooth.tolist(), dy_smooth.tolist(), velocity.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test Roboflow or local YOLO tennis ball model on video"
    )
    parser.add_argument("video", help="Path to video file (e.g. IMG_9259.MOV)")
    parser.add_argument("-o", "--output", default="roboflow_test.json", help="Output JSON path")
    # Model selection — mutually exclusive
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", dest="local_model", help="Local .pt YOLO model file")
    model_group.add_argument("--api-key", dest="api_key",
                             help="Roboflow API key (get free key at https://app.roboflow.com)")
    parser.add_argument("--model-id", default="tennis-kq1fm/tennis-ball-o7so8/2",
                       help="Roboflow model ID (default: tennis-kq1fm/tennis-ball-o7so8/2)")
    # Processing options
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to process")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every N frames")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # Run detection
    if args.api_key:
        detections, fps, total = detect_ball_via_roboflow_api(
            video_path=video_path,
            api_key=args.api_key,
            model_id=args.model_id,
            start_frame=args.start_frame,
            frame_skip=args.frame_skip,
            conf=args.conf,
            max_frames=args.max_frames,
        )
        scale = None
    else:
        detections, fps, total, scale = detect_ball_via_local_model(
            video_path=video_path,
            model_path=args.local_model,
            start_frame=args.start_frame,
            frame_skip=args.frame_skip,
            conf=args.conf,
            max_frames=args.max_frames,
        )

    # Interpolate gaps
    positions_interp = interpolate_missing_detections(detections)
    positions_raw = []
    for d, pos in zip(detections, positions_interp):
        if d is not None and pos is not None:
            positions_raw.append((float(pos[0]), float(pos[1])))
        else:
            positions_raw.append(None)

    # Compute velocities
    dx, dy, velocity = compute_velocity_series(positions_interp, fps)

    # Build output
    output = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total,
        "start_frame": args.start_frame,
        "frame_skip": args.frame_skip,
        "conf_threshold": args.conf,
        "detector": "roboflow_api" if args.api_key else f"local_{args.local_model}",
        "model_id": args.model_id if args.api_key else args.local_model,
        "scale_m_px": float(scale) if scale else None,
        "n_detections": sum(1 for d in detections if d is not None),
        "positions": positions_raw,
        "velocity": [float(v) for v in velocity],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {args.output}")
    print(f"  Total frames:   {total}")
    print(f"  FPS:            {fps:.3f}")
    print(f"  Ball detected:  {output['n_detections']}/{total} "
          f"({100*output['n_detections']/total:.1f}%)")
    if scale:
        print(f"  Scale:          {scale:.6f} m/px")
    if output["n_detections"] > 0:
        max_vel = max(v for v in velocity if np.isfinite(v))
        max_vel_kmh = max_vel * fps * scale * 3.6 if scale else None
        print(f"  Max velocity:   {max_vel:.2f} px/frame")
        if max_vel_kmh:
            print(f"  Max speed:      {max_vel_kmh:.1f} km/h")


if __name__ == "__main__":
    main()
