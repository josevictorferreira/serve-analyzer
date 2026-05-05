#!/usr/bin/env python3
"""Sixth-generation autonomous serve detector.

v6 = v5 timing refinement + two-stage tracking density + lightweight detector
voting.  It is intentionally detector-only: manual timestamps stay in
``serve_analyzer.serve_evaluation``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from serve_analyzer.multi_serve import (
    analyze_serve,
    compute_frame_velocities,
    compute_horizontal_velocity,
    compute_vertical_velocity,
    detect_serve_events,
    interpolate_missing_detections,
)
from serve_analyzer.serve_attempts import (
    _detect_broad_trajectory_events,
    _merge_candidate_events,
    detect_serve_candidates,
    infer_serve_count,
)
from serve_analyzer.serve_attempts_v5 import (
    _as_position,
    _quality_gate_candidates,
    _recompute_peak_velocities,
    _refine_contact_hybrid,
    _serialize_positions,
)


Position = Optional[Tuple[float, float]]


@dataclass(frozen=True)
class DetectionVote:
    """One detector's ball-center vote for a frame."""

    source: str
    x: float
    y: float
    confidence: float


def _load_seed(input_detections: str) -> Dict[str, Any]:
    with open(input_detections, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_video_metadata(
    video_path: str,
    fallback_fps: float = 30.0,
    fallback_total_frames: int = 0,
) -> Tuple[float, int, int, int]:
    """Return fps, total frames, width, and height with safe fallbacks."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback_fps, fallback_total_frames, 0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fallback_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fallback_total_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, total_frames, width, height


def _merge_windows(
    windows: Sequence[Dict[str, Any]],
    total_frames: int,
) -> List[Dict[str, Any]]:
    """Merge overlapping inclusive frame windows and preserve source labels."""
    normalized: List[Dict[str, Any]] = []
    for window in windows:
        start = max(0, int(window["start_frame"]))
        end = int(window["end_frame"])
        if total_frames > 0:
            end = min(total_frames - 1, end)
        if end < start:
            continue
        sources = set(window.get("sources", []))
        source = window.get("source")
        if source:
            sources.add(str(source))
        normalized.append(
            {"start_frame": start, "end_frame": end, "sources": sorted(sources)}
        )

    normalized.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    merged: List[Dict[str, Any]] = []
    for window in normalized:
        if not merged or window["start_frame"] > merged[-1]["end_frame"] + 1:
            merged.append(dict(window))
            continue
        merged[-1]["end_frame"] = max(merged[-1]["end_frame"], window["end_frame"])
        sources = set(merged[-1].get("sources", [])) | set(window.get("sources", []))
        merged[-1]["sources"] = sorted(sources)
    return merged


def _candidate_windows(
    candidates: Sequence[Dict[str, Any]],
    fps: float,
    total_frames: int,
    window_sec: float,
) -> List[Dict[str, Any]]:
    """Build fine-scan windows around coarse candidate contact frames."""
    radius = max(1, int(round(window_sec * fps)))
    windows = []
    for candidate in candidates:
        contact = int(candidate["contact_frame"])
        windows.append(
            {
                "start_frame": contact - radius,
                "end_frame": contact + radius,
                "source": "coarse_candidate",
            }
        )
    return _merge_windows(windows, total_frames)


def _detect_motion_hsv_vote(
    frame: np.ndarray,
    previous_gray: Optional[np.ndarray],
) -> Optional[DetectionVote]:
    """Detect a moving yellow tennis-ball blob with frame differencing."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, (18, 80, 80), (45, 255, 255))
    if previous_gray is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, previous_gray)
        _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(color_mask, motion_mask)
    else:
        mask = color_mask

    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [contour for contour in contours if 50 < cv2.contourArea(contour) < 10000]
    if not valid:
        return None

    contour = max(valid, key=cv2.contourArea)
    moment = cv2.moments(contour)
    if moment["m00"] <= 0:
        return None
    area = float(cv2.contourArea(contour))
    confidence = min(1.0, area / 800.0)
    return DetectionVote(
        source="motion_hsv",
        x=float(moment["m10"] / moment["m00"]),
        y=float(moment["m01"] / moment["m00"]),
        confidence=confidence,
    )


def _motion_hsv_rescue_windows(
    video_path: str,
    fps: float,
    total_frames: int,
    frame_skip: int,
    window_sec: float,
    max_windows: int = 24,
) -> List[Dict[str, Any]]:
    """Find low-cost fine-scan windows from moving yellow blobs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_idx = 0
    previous_gray: Optional[np.ndarray] = None
    hit_frames: List[int] = []
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_idx % frame_skip == 0:
            vote = _detect_motion_hsv_vote(frame, previous_gray)
            if vote is not None:
                hit_frames.append(frame_idx)
        previous_gray = gray
        frame_idx += 1
    cap.release()

    if not hit_frames:
        return []

    max_gap = max(1, int(round(fps * 1.25)))
    clusters: List[List[int]] = []
    current = [hit_frames[0]]
    for frame in hit_frames[1:]:
        if frame - current[-1] <= max_gap:
            current.append(frame)
        else:
            clusters.append(current)
            current = [frame]
    clusters.append(current)

    radius = max(1, int(round(window_sec * fps)))
    windows = []
    for cluster in sorted(clusters, key=len, reverse=True)[:max_windows]:
        if len(cluster) < 2:
            continue
        center = cluster[len(cluster) // 2]
        windows.append(
            {
                "start_frame": center - radius,
                "end_frame": center + radius,
                "source": "motion_hsv_rescue",
            }
        )
    return _merge_windows(windows, total_frames)


def _load_yolo_model(model_path: str) -> Tuple[Any, bool]:
    """Load the YOLO model and return whether it is tennis-ball specific."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics required for v6 YOLO voting") from exc

    is_rjtp = model_path in ("rjtp", "RJTPP/tennis-ball-detection")
    is_yolo26n = model_path in ("yolo26n", "yolo26n.pt")
    if is_rjtp:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="RJTPP/tennis-ball-detection", filename="best.pt"
        )
    elif is_yolo26n:
        model_path = str(Path(__file__).parent.parent / "yolo26n.pt")
    return YOLO(model_path), bool(is_rjtp or is_yolo26n)


def _detect_yolo_vote(
    frame: np.ndarray,
    model: Any,
    tennis_specific: bool,
    conf_threshold: float,
) -> Optional[DetectionVote]:
    """Return the smallest plausible YOLO ball detection as a vote."""
    height, width = frame.shape[:2]
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        verbose=False,
        device="cpu",
        classes=None if tennis_specific else [32],
    )
    if not results or results[0].boxes is None:
        return None

    boxes = results[0].boxes
    best: Optional[DetectionVote] = None
    best_area = float("inf")
    for index in range(len(boxes)):
        x1, y1, x2, y2 = boxes.xyxy[index].cpu().numpy()
        area = float((x2 - x1) * (y2 - y1))
        if area >= width * height * 0.01 or area >= best_area:
            continue
        confidence = (
            float(boxes.conf[index].cpu().numpy()) if boxes.conf is not None else 0.5
        )
        best = DetectionVote(
            source="yolo",
            x=float((x1 + x2) / 2.0),
            y=float((y1 + y2) / 2.0),
            confidence=confidence,
        )
        best_area = area
    return best


def _combine_votes(
    votes: Sequence[DetectionVote],
    radius_px: float,
    min_vote_count: int,
) -> Tuple[Position, Optional[Dict[str, Any]]]:
    """Combine agreeing detector votes into one weighted centroid."""
    if not votes:
        return None, None

    best_cluster: List[DetectionVote] = []
    for vote in votes:
        cluster = [
            other
            for other in votes
            if math.hypot(vote.x - other.x, vote.y - other.y) <= radius_px
        ]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < min_vote_count:
        strongest = max(votes, key=lambda item: item.confidence)
        if strongest.confidence >= 0.55:
            return None, {
                "pending_position": [float(strongest.x), float(strongest.y)],
                "vote_count": 1,
                "sources": [strongest.source],
                "confidence": float(strongest.confidence),
                "accepted": False,
            }
        return None, {
            "vote_count": int(len(votes)),
            "sources": sorted({vote.source for vote in votes}),
            "confidence": float(max(vote.confidence for vote in votes)),
            "accepted": False,
        }

    weights = [max(0.05, vote.confidence) for vote in best_cluster]
    weight_sum = sum(weights)
    x = sum(vote.x * weight for vote, weight in zip(best_cluster, weights)) / weight_sum
    y = sum(vote.y * weight for vote, weight in zip(best_cluster, weights)) / weight_sum
    return (float(x), float(y)), {
        "vote_count": int(len(best_cluster)),
        "sources": sorted({vote.source for vote in best_cluster}),
        "confidence": float(max(vote.confidence for vote in best_cluster)),
        "accepted": True,
    }


def _near_accepted_neighbor(
    positions: Sequence[Position],
    frame_idx: int,
    position: Tuple[float, float],
    lookaround: int,
    max_jump_px: float,
) -> bool:
    """Return True if a pending single-detector point is locally consistent."""
    start = max(0, frame_idx - lookaround)
    end = min(len(positions), frame_idx + lookaround + 1)
    for neighbor in positions[start:end]:
        if neighbor is None:
            continue
        if (
            math.hypot(position[0] - neighbor[0], position[1] - neighbor[1])
            <= max_jump_px
        ):
            return True
    return False


def _fine_ensemble_detections(
    video_path: str,
    windows: Sequence[Dict[str, Any]],
    total_frames: int,
    fps: float,
    model_path: str,
    conf_threshold: float,
    fine_frame_skip: int,
    vote_radius_px: float,
    min_vote_count: int,
    tracknet_weights: Optional[str] = None,
    tracknet_device: str = "cpu",
) -> Tuple[List[Position], List[Optional[Dict[str, Any]]]]:
    """Run YOLO + motion-HSV (+ optional TrackNetV2) voting inside windows."""
    detections: List[Position] = [None] * total_frames
    diagnostics: List[Optional[Dict[str, Any]]] = [None] * total_frames
    if not windows:
        return detections, diagnostics

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return detections, diagnostics

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 3840)
    scaled_radius = float(vote_radius_px) * max(0.25, width / 3840.0)
    yolo_model, tennis_specific = _load_yolo_model(model_path)

    tracknet_detections: List[Position] = []
    if tracknet_weights:
        from serve_analyzer.tracknetv2 import detect_ball_tracknetv2

        tracknet_detections, _, _, _ = detect_ball_tracknetv2(
            video_path,
            weights_path=tracknet_weights,
            conf_threshold=conf_threshold,
            frame_skip=fine_frame_skip,
            device=tracknet_device,
        )

    sorted_windows = sorted(windows, key=lambda item: item["start_frame"])
    window_index = 0
    frame_idx = 0
    previous_gray: Optional[np.ndarray] = None
    pending: Dict[int, Dict[str, Any]] = {}

    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while window_index < len(sorted_windows) and frame_idx > int(
            sorted_windows[window_index]["end_frame"]
        ):
            window_index += 1
        in_window = (
            window_index < len(sorted_windows)
            and int(sorted_windows[window_index]["start_frame"]) <= frame_idx
            and frame_idx <= int(sorted_windows[window_index]["end_frame"])
        )

        if in_window and frame_idx % fine_frame_skip == 0:
            votes: List[DetectionVote] = []
            yolo_vote = _detect_yolo_vote(
                frame, yolo_model, tennis_specific, conf_threshold
            )
            if yolo_vote is not None:
                votes.append(yolo_vote)
            hsv_vote = _detect_motion_hsv_vote(frame, previous_gray)
            if hsv_vote is not None:
                votes.append(hsv_vote)
            if frame_idx < len(tracknet_detections) and tracknet_detections[frame_idx]:
                tx, ty = tracknet_detections[frame_idx]  # type: ignore[misc]
                votes.append(
                    DetectionVote(
                        source="tracknetv2",
                        x=float(tx),
                        y=float(ty),
                        confidence=0.8,
                    )
                )

            combined, stats = _combine_votes(votes, scaled_radius, min_vote_count)
            if combined is not None:
                detections[frame_idx] = combined
                diagnostics[frame_idx] = stats
            elif stats and "pending_position" in stats:
                pending[frame_idx] = stats
            elif stats:
                diagnostics[frame_idx] = stats

        previous_gray = gray
        frame_idx += 1

    cap.release()

    lookaround = max(1, int(round(0.20 * fps)))
    for frame_idx, stats in pending.items():
        px, py = stats["pending_position"]
        position = (float(px), float(py))
        if _near_accepted_neighbor(
            detections,
            frame_idx,
            position,
            lookaround=lookaround,
            max_jump_px=scaled_radius * 3.0,
        ):
            enriched = dict(stats)
            enriched["accepted"] = True
            enriched["accepted_single"] = True
            enriched.pop("pending_position", None)
            detections[frame_idx] = position
            diagnostics[frame_idx] = enriched
    return detections, diagnostics


def _fuse_positions(
    coarse_raw: Sequence[Position], fine_raw: Sequence[Position]
) -> List[Position]:
    """Prefer fine ensemble detections while preserving coarse detections elsewhere."""
    length = max(len(coarse_raw), len(fine_raw))
    fused: List[Position] = []
    for index in range(length):
        fine = fine_raw[index] if index < len(fine_raw) else None
        coarse = coarse_raw[index] if index < len(coarse_raw) else None
        fused.append(fine if fine is not None else coarse)
    return fused


def _build_candidates_from_positions(
    positions: Sequence[Position],
    raw_positions: Sequence[Position],
    fps: float,
    scale_factor: float,
    frame_skip: int,
    detector_label: str,
    pool_hint: int = 12,
) -> Dict[str, Any]:
    """Run the v1 event generator on already-fused frame-indexed tracks."""
    velocities = compute_frame_velocities(positions, fps)
    vert_velocities = compute_vertical_velocity(positions)
    horiz_velocities = compute_horizontal_velocity(positions)
    expected = max(1, int(pool_hint))

    detector_profiles = [
        {
            "expected_serves": max(expected * 3, expected + 8),
            "min_serve_gap_sec": 2.0,
            "velocity_spike_percentile": 85,
            "toss_lookback_sec": 1.8,
            "post_contact_duration_sec": 0.8,
        },
        {
            "expected_serves": max(expected * 3, expected + 8),
            "min_serve_gap_sec": 2.5,
            "velocity_spike_percentile": 80,
            "toss_lookback_sec": 2.0,
            "post_contact_duration_sec": 1.0,
        },
        {
            "expected_serves": max(expected * 4, expected + 12),
            "min_serve_gap_sec": 1.5,
            "velocity_spike_percentile": 75,
            "toss_lookback_sec": 2.2,
            "post_contact_duration_sec": 1.0,
        },
    ]
    candidate_event_groups = [
        detect_serve_events(
            positions,
            velocities,
            vert_velocities,
            horiz_velocities=horiz_velocities,
            fps=fps,
            **profile,
        )
        for profile in detector_profiles
    ]
    candidate_event_groups.append(
        _detect_broad_trajectory_events(
            positions,
            velocities,
            vert_velocities,
            horiz_velocities,
            fps,
            expected,
        )
    )
    candidate_event_groups.append(
        _detect_broad_trajectory_events(
            positions,
            velocities,
            vert_velocities,
            horiz_velocities,
            fps,
            expected,
            min_serve_gap_sec=1.0,
        )
    )

    candidate_events = _merge_candidate_events(candidate_event_groups, fps)
    flattened_events = [event for group in candidate_event_groups for event in group]
    max_merge_gap_frames = max(1, int(0.75 * fps))
    analyzed_serves = [
        analyze_serve(index, event, positions, fps, scale_factor)
        for index, event in enumerate(candidate_events)
    ]

    candidates: List[Dict[str, Any]] = []
    for index, (event, serve) in enumerate(zip(candidate_events, analyzed_serves)):
        support_events = [
            source_event
            for source_event in flattened_events
            if abs(int(source_event["contact_frame"]) - int(event["contact_frame"]))
            <= max_merge_gap_frames
        ]
        apex_frame = int(event.get("apex_frame", event["contact_frame"]))
        frames_after_apex = int(event["contact_frame"] - apex_frame)
        candidates.append(
            {
                "candidate_index": int(index),
                "contact_frame": int(event["contact_frame"]),
                "contact_time_sec": float(event["contact_frame"] / fps),
                "post_contact_max_kmh": float(serve.post_contact_max_velocity),
                "post_contact_mean_kmh": float(serve.post_contact_mean_velocity),
                "post_contact_max_mps": float(serve.post_contact_max_velocity / 3.6),
                "post_contact_mean_mps": float(serve.post_contact_mean_velocity / 3.6),
                "score": float(event.get("score", 0.0)),
                "support_count": int(len(support_events)),
                "contact_velocity": float(event.get("contact_velocity", 0.0)),
                "upward_fraction": float(event.get("upward_fraction", 0.0)),
                "recent_upward_fraction": float(
                    event.get("recent_upward_fraction", 0.0)
                ),
                "drop_after_apex": float(event.get("drop_after_apex", 0.0)),
                "frames_after_apex": int(frames_after_apex),
                "rightward_fraction": float(event.get("rightward_fraction", 0.0)),
                "net_rightward_displacement": float(
                    event.get("net_rightward_displacement", 0.0)
                ),
                "direction_unreliable": bool(event.get("direction_unreliable", False)),
                "toss_rise_px": float(event.get("toss_rise_px", 0.0)),
                "toss_duration_frames": int(event.get("toss_duration_frames", 0)),
                "early_post_downward_fraction": float(
                    event.get("early_post_downward_fraction", 0.0)
                ),
                "early_post_net_dy": float(event.get("early_post_net_dy", 0.0)),
            }
        )

    return {
        "candidates": candidates,
        "positions": list(positions),
        "raw_positions": list(raw_positions),
        "frame_skip": int(frame_skip),
        "detector": detector_label,
    }


def _attach_fine_support(
    candidates: List[Dict[str, Any]],
    diagnostics: Sequence[Optional[Dict[str, Any]]],
    radius_frames: int = 3,
) -> None:
    """Add local ensemble-vote evidence to each candidate."""
    for candidate in candidates:
        contact = int(candidate["contact_frame"])
        start = max(0, contact - radius_frames)
        end = min(len(diagnostics), contact + radius_frames + 1)
        nearby = [item for item in diagnostics[start:end] if item]
        if not nearby:
            candidate["v6_fine_vote_count"] = 0
            candidate["v6_fine_sources"] = []
            candidate["v6_fine_confirmed"] = False
            continue
        best = max(nearby, key=lambda item: int(item.get("vote_count", 0)))
        candidate["v6_fine_vote_count"] = int(best.get("vote_count", 0))
        candidate["v6_fine_sources"] = list(best.get("sources", []))
        candidate["v6_fine_confirmed"] = bool(best.get("accepted", False))


def _build_speed_rescue_candidates(
    existing_candidates: Sequence[Dict[str, Any]],
    positions: Sequence[Tuple[float, float]],
    fps: float,
    scale_factor: float,
    windows: Sequence[Dict[str, Any]],
    diagnostics: Sequence[Optional[Dict[str, Any]]],
    post_contact_sec: float = 1.0,
) -> List[Dict[str, Any]]:
    """Add one speed-based candidate for uncovered motion-HSV rescue windows."""
    rescue_windows = [
        window
        for window in windows
        if "motion_hsv_rescue" in set(window.get("sources", []))
    ]
    if not rescue_windows or not positions:
        return []

    velocities = compute_frame_velocities(list(positions), fps)
    vert_velocities = compute_vertical_velocity(list(positions))
    horiz_velocities = compute_horizontal_velocity(list(positions))
    global_threshold = float(np.percentile(velocities, 75))
    existing_frames = [
        int(candidate["contact_frame"]) for candidate in existing_candidates
    ]
    rescue_candidates: List[Dict[str, Any]] = []

    for window in rescue_windows:
        pad_before = int(round(0.5 * fps))
        pad_after = int(round(0.25 * fps))
        start = max(0, int(window["start_frame"]) - pad_before)
        end = min(len(positions) - 1, int(window["end_frame"]) + pad_after)
        if end <= start:
            continue
        if any(start <= frame <= end for frame in existing_frames):
            continue

        region = velocities[start : end + 1]
        if len(region) < 3:
            continue
        contact_frame = start + int(np.argmax(region))
        contact_velocity = float(velocities[contact_frame])
        if contact_velocity < global_threshold:
            continue

        post_end = min(
            len(positions) - 1, contact_frame + int(round(post_contact_sec * fps))
        )
        early_post_end = contact_frame + max(1, (post_end - contact_frame) // 2)
        post_horiz = horiz_velocities[contact_frame + 1 : early_post_end + 1]
        rightward_fraction = (
            float(np.mean(post_horiz > 0.5)) if len(post_horiz) else 0.0
        )
        post_positions = positions[contact_frame + 1 : post_end + 1]
        net_dx = (
            float(post_positions[-1][0] - post_positions[0][0])
            if len(post_positions) >= 2
            else 0.0
        )
        if rightward_fraction < 0.2 and net_dx <= 50.0:
            continue

        search_start = max(0, contact_frame - int(round(2.2 * fps)))
        toss_region = vert_velocities[search_start:contact_frame]
        upward_fraction = (
            float(np.mean(toss_region < -1.0)) if len(toss_region) > 0 else 0.0
        )
        recent_start = max(search_start, contact_frame - int(round(0.6 * fps)))
        recent_region = vert_velocities[recent_start:contact_frame]
        recent_upward_fraction = (
            float(np.mean(recent_region < -1.0)) if len(recent_region) > 0 else 0.0
        )

        y_coords = [
            position[1] for position in positions[search_start : contact_frame + 1]
        ]
        apex_frame = search_start + int(np.argmin(y_coords))
        apex_position = positions[apex_frame]
        drop_after_apex = float(positions[contact_frame][1] - apex_position[1])

        velocities_region = velocities[search_start : apex_frame + 1]
        toss_start = search_start
        if len(velocities_region) > 5:
            low_velocity = float(np.max(velocities_region) * 0.15)
            for frame in range(apex_frame - 1, search_start - 1, -1):
                if velocities[frame] < low_velocity:
                    toss_start = frame
                    break
        toss_rise_px = max(0.0, float(positions[toss_start][1] - apex_position[1]))
        toss_duration_frames = max(0, apex_frame - toss_start)

        early_post_vert = vert_velocities[
            contact_frame + 1 : min(post_end + 1, contact_frame + 9)
        ]
        early_post_downward_fraction = (
            float(np.mean(early_post_vert > 1.0)) if len(early_post_vert) > 0 else 0.0
        )
        early_post_positions = positions[
            contact_frame + 1 : min(post_end + 1, contact_frame + 9)
        ]
        early_post_net_dy = (
            float(early_post_positions[-1][1] - early_post_positions[0][1])
            if len(early_post_positions) >= 2
            else 0.0
        )

        event = {
            "contact_frame": int(contact_frame),
            "toss_start_frame": int(toss_start),
            "apex_frame": int(apex_frame),
            "apex_position": apex_position,
            "post_contact_end_frame": int(post_end),
        }
        serve = analyze_serve(
            len(existing_candidates) + len(rescue_candidates),
            event,
            list(positions),
            fps,
            scale_factor,
        )

        nearby = [
            item
            for item in diagnostics[
                max(0, contact_frame - int(round(0.15 * fps))) : min(
                    len(diagnostics), contact_frame + int(round(0.15 * fps)) + 1
                )
            ]
            if item
        ]
        fine_vote_count = max(
            (int(item.get("vote_count", 0)) for item in nearby), default=0
        )
        score = (
            contact_velocity
            + recent_upward_fraction * 120.0
            + max(0.0, drop_after_apex) * 0.25
            + max(0.0, net_dx) * 0.08
            + fine_vote_count * 120.0
        )

        rescue_candidates.append(
            {
                "candidate_index": int(
                    len(existing_candidates) + len(rescue_candidates)
                ),
                "contact_frame": int(contact_frame),
                "contact_time_sec": float(contact_frame / fps),
                "post_contact_max_kmh": float(serve.post_contact_max_velocity),
                "post_contact_mean_kmh": float(serve.post_contact_mean_velocity),
                "post_contact_max_mps": float(serve.post_contact_max_velocity / 3.6),
                "post_contact_mean_mps": float(serve.post_contact_mean_velocity / 3.6),
                "score": float(score),
                "support_count": int(max(1, fine_vote_count)),
                "contact_velocity": contact_velocity,
                "upward_fraction": upward_fraction,
                "recent_upward_fraction": recent_upward_fraction,
                "drop_after_apex": drop_after_apex,
                "frames_after_apex": int(contact_frame - apex_frame),
                "rightward_fraction": rightward_fraction,
                "net_rightward_displacement": net_dx,
                "direction_unreliable": bool(
                    rightward_fraction < 0.35 and net_dx <= 0.0
                ),
                "toss_rise_px": float(toss_rise_px),
                "toss_duration_frames": int(toss_duration_frames),
                "early_post_downward_fraction": early_post_downward_fraction,
                "early_post_net_dy": early_post_net_dy,
                "v6_rescue_source": "motion_hsv_speed",
                "v6_rescue_window": [
                    int(window["start_frame"]),
                    int(window["end_frame"]),
                ],
            }
        )
        existing_frames.append(contact_frame)

    return rescue_candidates


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _rank_v6_candidate(candidate: Dict[str, Any]) -> float:
    """Rank a refined candidate using ensemble support and serve geometry."""
    vote_bonus = _clip(float(candidate.get("v6_fine_vote_count", 0)) / 3.0)
    support_bonus = _clip(float(candidate.get("support_count", 0)) / 3.0)
    rightward_bonus = _clip(float(candidate.get("rightward_fraction", 0.0)))
    rightward_disp_bonus = _clip(
        float(candidate.get("net_rightward_displacement", 0.0)) / 160.0
    )
    toss_bonus = _clip(float(candidate.get("recent_upward_fraction", 0.0)))
    post_bonus = _clip(float(candidate.get("post_contact_mean_kmh", 0.0)) / 160.0)
    score_bonus = _clip(
        float(candidate.get("v5_contact_score", candidate.get("score", 0.0))) / 1800.0
    )

    leftward_penalty = _clip(
        -float(candidate.get("net_rightward_displacement", 0.0)) / 120.0
    )
    weak_motion_penalty = (
        1.0
        if (
            abs(float(candidate.get("drop_after_apex", 0.0))) < 10.0
            and abs(float(candidate.get("net_rightward_displacement", 0.0))) < 20.0
        )
        else 0.0
    )

    return float(
        0.24 * vote_bonus
        + 0.18 * support_bonus
        + 0.18 * rightward_bonus
        + 0.12 * rightward_disp_bonus
        + 0.12 * toss_bonus
        + 0.08 * post_bonus
        + 0.08 * score_bonus
        - 0.45 * leftward_penalty
        - 0.35 * weak_motion_penalty
    )


def _is_v6_selection_artifact(candidate: Dict[str, Any]) -> bool:
    """Return True for candidate shapes that are usually tracking artifacts."""
    frames_after_apex = float(candidate.get("frames_after_apex", 0.0))
    drop_after_apex = float(candidate.get("drop_after_apex", 0.0))
    if frames_after_apex <= 2.0 and abs(drop_after_apex) < 10.0:
        return True

    toss_duration = int(candidate.get("toss_duration_frames", 0))
    early_post_net_dy = float(candidate.get("early_post_net_dy", 0.0))
    fine_confirmed = bool(candidate.get("v6_fine_confirmed", False))
    return toss_duration <= 5 and early_post_net_dy > 300.0 and not fine_confirmed


def select_serves_v6(
    candidates: Sequence[Dict[str, Any]],
    min_gap_sec: float = 3.0,
    min_post_contact_kmh: float = 15.0,
    min_rightward_fraction: float = 0.2,
) -> List[Dict[str, Any]]:
    """Select v6 serves autonomously from the refined candidate pool."""
    qualified = _quality_gate_candidates(
        list(candidates),
        min_post_contact_kmh=min_post_contact_kmh,
        min_rightward_fraction=min_rightward_fraction,
    )
    qualified = [
        candidate for candidate in qualified if not _is_v6_selection_artifact(candidate)
    ]
    ranked: List[Dict[str, Any]] = []
    for candidate in qualified:
        enriched = dict(candidate)
        enriched["selector_rank"] = _rank_v6_candidate(enriched)
        ranked.append(enriched)
    ranked.sort(key=lambda item: float(item["selector_rank"]), reverse=True)

    target_count = infer_serve_count(
        ranked,
        min_rank_floor=0.08,
        relative_floor=0.50,
    )
    selected: List[Dict[str, Any]] = []
    for candidate in ranked:
        if float(candidate["selector_rank"]) < 0.0:
            continue
        candidate_time = float(candidate["contact_time_sec"])
        if all(
            abs(candidate_time - float(existing["contact_time_sec"])) >= min_gap_sec
            for existing in selected
        ):
            selected.append(dict(candidate))
        if len(selected) >= target_count:
            break

    selected.sort(key=lambda item: float(item["contact_time_sec"]))
    return selected


def detect_serve_candidates_v6(
    video_path: str,
    detector: str = "yolo",
    model: str = "rjtp",
    tracknet_weights: Optional[str] = None,
    tracknet_device: str = "cpu",
    scale_factor: float = 0.001,
    conf_threshold: float = 0.20,
    frame_skip: int = 4,
    fine_frame_skip: int = 1,
    fine_window_sec: float = 1.75,
    vote_radius_px: float = 35.0,
    min_vote_count: int = 2,
    input_detections: Optional[str] = None,
    quality_gate: bool = True,
    min_post_contact_kmh: float = 15.0,
    min_rightward_fraction: float = 0.2,
    apex_search_backward_sec: float = 1.5,
    max_backward_shift: int = 10,
    sg_window: int = 7,
    sg_polyorder: int = 3,
    peak_top_k: int = 5,
    post_contact_sec: float = 1.0,
) -> Dict[str, Any]:
    """Run autonomous v6 serve detection and return JSON-safe records."""
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if fine_frame_skip < 1:
        raise ValueError("fine_frame_skip must be at least 1")

    if input_detections:
        seed = _load_seed(input_detections)
    else:
        seed = detect_serve_candidates(
            video_path,
            expected_serves=None,
            detector=detector,
            model=model,
            tracknet_weights=tracknet_weights,
            tracknet_device=tracknet_device,
            scale_factor=scale_factor,
            conf_threshold=conf_threshold,
            frame_skip=frame_skip,
        )

    seed_candidates = list(seed.get("candidates", []))
    coarse_raw = [_as_position(value) for value in seed.get("raw_positions", [])]
    if not coarse_raw:
        coarse_raw = [_as_position(value) for value in seed.get("positions", [])]

    fallback_total = len(coarse_raw)
    fps, total_frames, _width, _height = _extract_video_metadata(
        video_path,
        fallback_total_frames=fallback_total,
    )
    total_frames = max(total_frames, fallback_total)
    if len(coarse_raw) < total_frames:
        coarse_raw.extend([None] * (total_frames - len(coarse_raw)))

    windows: List[Dict[str, Any]] = []
    if fine_window_sec > 0:
        windows.extend(
            _candidate_windows(seed_candidates, fps, total_frames, fine_window_sec)
        )
        windows.extend(
            _motion_hsv_rescue_windows(
                video_path,
                fps=fps,
                total_frames=total_frames,
                frame_skip=max(1, frame_skip),
                window_sec=fine_window_sec,
            )
        )
    windows = _merge_windows(windows, total_frames)

    fine_raw: List[Position] = [None] * total_frames
    fine_diagnostics: List[Optional[Dict[str, Any]]] = [None] * total_frames
    if windows:
        fine_raw, fine_diagnostics = _fine_ensemble_detections(
            video_path,
            windows=windows,
            total_frames=total_frames,
            fps=fps,
            model_path=model,
            conf_threshold=conf_threshold,
            fine_frame_skip=fine_frame_skip,
            vote_radius_px=vote_radius_px,
            min_vote_count=min_vote_count,
            tracknet_weights=tracknet_weights,
            tracknet_device=tracknet_device,
        )

    fused_raw = _fuse_positions(coarse_raw, fine_raw)
    positions = interpolate_missing_detections(fused_raw, max_gap=15)
    built = _build_candidates_from_positions(
        positions=positions,
        raw_positions=fused_raw,
        fps=fps,
        scale_factor=scale_factor,
        frame_skip=frame_skip,
        detector_label="v6",
    )

    search_backward_frames = max(1, int(apex_search_backward_sec * fps))
    built_candidates = list(built["candidates"])
    built_candidates.extend(
        _build_speed_rescue_candidates(
            built_candidates,
            positions,
            fps,
            scale_factor,
            windows,
            fine_diagnostics,
            post_contact_sec=post_contact_sec,
        )
    )
    built_candidates.sort(key=lambda candidate: float(candidate["contact_time_sec"]))
    for index, candidate in enumerate(built_candidates):
        candidate["candidate_index"] = int(index)

    refined_all = _refine_contact_hybrid(
        built_candidates,
        positions,
        fps,
        search_backward_frames=search_backward_frames,
        max_backward_shift=max_backward_shift,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )
    refined_all.sort(key=lambda candidate: float(candidate["contact_time_sec"]))
    for index, candidate in enumerate(refined_all):
        candidate["candidate_index"] = int(index)

    _attach_fine_support(
        refined_all, fine_diagnostics, radius_frames=max(3, int(0.15 * fps))
    )

    _recompute_peak_velocities(
        refined_all,
        positions,
        fps,
        scale_factor=scale_factor,
        post_contact_sec=post_contact_sec,
        top_k=peak_top_k,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )

    selected = select_serves_v6(
        refined_all,
        min_post_contact_kmh=min_post_contact_kmh if quality_gate else 0.0,
        min_rightward_fraction=min_rightward_fraction if quality_gate else 0.0,
    )

    return {
        "video_path": str(video_path),
        "detector": "v6",
        "seed_detector": detector,
        "expected_serves": None,
        "count_inferred": True,
        "inferred_count": int(len(selected)),
        "frame_skip": int(frame_skip),
        "fine_frame_skip": int(fine_frame_skip),
        "selected_serves": selected,
        "candidates": refined_all,
        "positions": _serialize_positions(positions),
        "raw_positions": _serialize_positions(fused_raw),
        "v6_windows": windows,
        "v6_parameters": {
            "coarse_frame_skip": int(frame_skip),
            "fine_frame_skip": int(fine_frame_skip),
            "fine_window_sec": float(fine_window_sec),
            "vote_radius_px": float(vote_radius_px),
            "min_vote_count": int(min_vote_count),
            "quality_gate": bool(quality_gate),
            "min_post_contact_kmh": float(min_post_contact_kmh),
            "min_rightward_fraction": float(min_rightward_fraction),
            "apex_search_backward_sec": float(apex_search_backward_sec),
            "max_backward_shift": int(max_backward_shift),
            "sg_window": int(sg_window),
            "sg_polyorder": int(sg_polyorder),
            "peak_top_k": int(peak_top_k),
            "post_contact_sec": float(post_contact_sec),
        },
        "v6_fine_detection_count": int(sum(1 for item in fine_raw if item is not None)),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for autonomous v6 detection."""
    parser = argparse.ArgumentParser(
        description="Autonomous v6 serve detector with two-stage ensemble tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m serve_analyzer.serve_attempts_v6 video.mov \
        --frame-skip 4 \
        --output run_v6_auto.json
        """,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--detector",
        choices=("yolo", "tracknetv2"),
        default="yolo",
        help="Coarse ball detector backend (default: yolo)",
    )
    parser.add_argument(
        "--model",
        default="rjtp",
        help="YOLO model path or 'rjtp' for tennis-ball model (default: rjtp)",
    )
    parser.add_argument(
        "--tracknet-weights",
        help="Path to TrackNetV2 weights for optional ensemble voting",
    )
    parser.add_argument(
        "--tracknet-device",
        default="cpu",
        help="Torch device for TrackNetV2 inference (default: cpu)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.001,
        help="Meters per pixel (default: 0.001)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Detector confidence threshold (default: 0.20)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Coarse pass frame skip (default: 4)",
    )
    parser.add_argument(
        "--fine-frame-skip",
        type=int,
        default=1,
        help="Fine pass frame skip inside windows (default: 1)",
    )
    parser.add_argument(
        "--fine-window-sec",
        type=float,
        default=1.75,
        help="Seconds before/after coarse events for fine pass (default: 1.75)",
    )
    parser.add_argument(
        "--vote-radius-px",
        type=float,
        default=35.0,
        help="Agreement radius for detector votes at 4K width (default: 35)",
    )
    parser.add_argument(
        "--min-vote-count",
        type=int,
        default=2,
        help="Votes required for immediate acceptance (default: 2)",
    )
    parser.add_argument(
        "--input-detections",
        help="Cached coarse detector JSON to reuse before v6 fine scanning",
    )
    parser.add_argument(
        "--no-quality-gate",
        action="store_true",
        help="Disable final v6 quality gate",
    )
    parser.add_argument("--output", "-o", help="Output JSON file for detector results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    result = detect_serve_candidates_v6(
        args.video,
        detector=args.detector,
        model=args.model,
        tracknet_weights=args.tracknet_weights,
        tracknet_device=args.tracknet_device,
        scale_factor=args.scale_factor,
        conf_threshold=args.conf,
        frame_skip=args.frame_skip,
        fine_frame_skip=args.fine_frame_skip,
        fine_window_sec=args.fine_window_sec,
        vote_radius_px=args.vote_radius_px,
        min_vote_count=args.min_vote_count,
        input_detections=args.input_detections,
        quality_gate=not args.no_quality_gate,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
