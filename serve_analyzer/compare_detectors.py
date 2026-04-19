"""
Compare YOLOv8n sports ball detector vs RJTPP tennis-ball-detection model.

Usage:
    python -m serve_analyzer.compare_detectors video.mp4 [--max-frames 100]

Outputs:
    Detection rate comparison, per-frame agreement, and recommendation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

from serve_analyzer.multi_serve import detect_ball_yolo


def detect_ball_rjtp(
    video_path: str,
    conf_threshold: float = 0.20,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    start_frame: int = 0,
    progress_interval: int = 100,
) -> Tuple[List[Optional[Tuple[float, float]]], float, int, Optional[float]]:
    """
    Detect ball using RJTPP/tennis-ball-detection model from Hugging Face.

    Same interface as detect_ball_yolo() for drop-in replacement.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required: pip install ultralytics")

    # Download model from Hugging Face if not cached
    try:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="RJTPP/tennis-ball-detection", filename="best.pt"
        )
        print(f"Downloaded RJTPP model to: {model_path}")
    except ImportError:
        print("huggingface_hub not installed, trying direct path...")
        model_path = "RJTPP/tennis-ball-detection"
    except Exception as e:
        print(f"Could not download from HF: {e}")
        raise

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Loading RJTPP model: {model_path}")
    model = YOLO(model_path)

    print(
        f"Processing {total_frames} frames at {fps:.1f} FPS ({total_frames / fps:.1f}s video)..."
    )

    detections: List[Optional[Tuple[float, float]]] = []
    ball_sizes: List[float] = []

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < start_frame or (frame_skip > 1 and frame_idx % frame_skip != 0):
            detections.append(None)
            frame_idx += 1
            continue

        # Run RJTPP model - no class filter needed (only detects tennis balls)
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            verbose=False,
            device="cpu",
        )

        ball_pos = None

        # Find best detection (smallest bbox = likely the ball)
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

        # No HSV fallback needed - RJTPP is tennis-ball specific
        detections.append(ball_pos)

        frame_idx += 1
        if frame_idx % progress_interval == 0:
            print(
                f"  Processed {frame_idx}/{total_frames} frames ({100 * frame_idx / total_frames:.1f}%)"
            )

    cap.release()

    # Estimate scale factor from ball sizes
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


def compare_detectors(
    video_path: str,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    conf_threshold: float = 0.20,
) -> dict:
    """
    Run both detectors and compare results.

    Returns dict with comparison metrics.
    """
    print("=" * 70)
    print("DETECTOR COMPARISON: YOLOv8n (sports ball) vs RJTPP (tennis ball)")
    print("=" * 70)
    print(f"Video: {video_path}")
    if max_frames:
        print(f"Max frames: {max_frames}")
    print(f"Frame skip: {frame_skip}")
    print(f"Confidence threshold: {conf_threshold}")
    print()

    # Run YOLOv8n
    print("-" * 70)
    print("RUN 1: YOLOv8n (COCO sports ball class)")
    print("-" * 70)
    t0 = time.time()
    yolo_dets, yolo_fps, yolo_total, yolo_scale = detect_ball_yolo(
        video_path,
        model_path="yolov8n.pt",
        conf_threshold=conf_threshold,
        max_frames=max_frames,
        frame_skip=frame_skip,
        progress_interval=50,
    )
    yolo_time = time.time() - t0

    print()

    # Run RJTPP
    print("-" * 70)
    print("RUN 2: RJTPP/tennis-ball-detection (fine-tuned)")
    print("-" * 70)
    t0 = time.time()
    rjtp_dets, rjtp_fps, rjtp_total, rjtp_scale = detect_ball_rjtp(
        video_path,
        conf_threshold=conf_threshold,
        max_frames=max_frames,
        frame_skip=frame_skip,
        progress_interval=50,
    )
    rjtp_time = time.time() - t0

    # Compare results
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    # Detection rates
    yolo_found = sum(1 for d in yolo_dets if d is not None)
    rjtp_found = sum(1 for d in rjtp_dets if d is not None)

    yolo_rate = yolo_found / len(yolo_dets) * 100 if yolo_dets else 0
    rjtp_rate = rjtp_found / len(rjtp_dets) * 100 if rjtp_dets else 0

    print("\nDetection Rate:")
    print(
        f"  YOLOv8n:  {yolo_found:>6}/{len(yolo_dets):<6} frames ({yolo_rate:>5.1f}%)"
    )
    print(
        f"  RJTPP:    {rjtp_found:>6}/{len(rjtp_dets):<6} frames ({rjtp_rate:>5.1f}%)"
    )

    # Per-frame agreement (where both detected)
    both_detected = 0
    yolo_only = 0
    rjtp_only = 0
    neither = 0
    distances = []

    for y, r in zip(yolo_dets, rjtp_dets):
        if y is not None and r is not None:
            both_detected += 1
            dist = np.sqrt((y[0] - r[0]) ** 2 + (y[1] - r[1]) ** 2)
            distances.append(dist)
        elif y is not None and r is None:
            yolo_only += 1
        elif y is None and r is not None:
            rjtp_only += 1
        else:
            neither += 1

    print("\nPer-frame Agreement:")
    print(f"  Both detected:     {both_detected:>6} frames")
    print(f"  YOLO only:         {yolo_only:>6} frames")
    print(f"  RJTPP only:        {rjtp_only:>6} frames")
    print(f"  Neither detected:  {neither:>6} frames")

    if distances:
        print("\nPosition Agreement (where both detected):")
        print(f"  Mean distance:     {np.mean(distances):.1f} px")
        print(f"  Median distance:   {np.median(distances):.1f} px")
        print(f"  Max distance:      {np.max(distances):.1f} px")

    # Performance
    print("\nPerformance:")
    print(f"  YOLOv8n:  {yolo_time:.1f}s ({yolo_total / yolo_time:.1f} FPS)")
    print(f"  RJTPP:    {rjtp_time:.1f}s ({rjtp_total / rjtp_time:.1f} FPS)")

    # Scale estimation
    print("\nScale Estimation:")
    if yolo_scale:
        print(f"  YOLOv8n:  {yolo_scale:.6f} m/px")
    else:
        print("  YOLOv8n:  insufficient data")
    if rjtp_scale:
        print(f"  RJTPP:    {rjtp_scale:.6f} m/px")
    else:
        print("  RJTPP:    insufficient data")

    # Recommendation
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if rjtp_rate > yolo_rate * 1.1:
        print("RJTPP model shows significantly better detection rate.")
        print("Recommendation: Use RJTPP/tennis-ball-detection")
    elif yolo_rate > rjtp_rate * 1.1:
        print("YOLOv8n shows significantly better detection rate.")
        print("Recommendation: Stick with YOLOv8n")
    else:
        print("Detection rates are comparable.")
        if distances and np.median(distances) < 50:
            print("Position agreement is good (median < 50px).")
        if rjtp_time < yolo_time:
            print("RJTPP is faster.")
            print("Recommendation: Use RJTPP/tennis-ball-detection")
        else:
            print("YOLOv8n is faster.")
            print("Recommendation: Stick with YOLOv8n")

    return {
        "yolo": {
            "detections": yolo_dets,
            "detection_rate": yolo_rate,
            "frames_found": yolo_found,
            "total_frames": len(yolo_dets),
            "time_sec": yolo_time,
            "scale": yolo_scale,
        },
        "rjtp": {
            "detections": rjtp_dets,
            "detection_rate": rjtp_rate,
            "frames_found": rjtp_found,
            "total_frames": len(rjtp_dets),
            "time_sec": rjtp_time,
            "scale": rjtp_scale,
        },
        "agreement": {
            "both_detected": both_detected,
            "yolo_only": yolo_only,
            "rjtp_only": rjtp_only,
            "neither": neither,
            "mean_distance_px": float(np.mean(distances)) if distances else None,
            "median_distance_px": float(np.median(distances)) if distances else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLOv8n vs RJTPP tennis ball detection"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit processing to first N frames (for quick test)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold (default: 0.20)",
    )

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    compare_detectors(
        args.video,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
        conf_threshold=args.conf,
    )


if __name__ == "__main__":
    main()
