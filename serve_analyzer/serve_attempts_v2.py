#!/usr/bin/env python3
"""Second-generation detector-only serve attempt refinement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from serve_analyzer.serve_attempts import detect_serve_candidates, select_serves


Position = Optional[Tuple[float, float]]


def _as_position(value: Any) -> Position:
    """Convert JSON/list position values into an optional float tuple."""
    if value is None or len(value) < 2:
        return None
    return (float(value[0]), float(value[1]))


def _serialize_positions(positions: Sequence[Position]) -> List[Optional[List[float]]]:
    """Return JSON-safe position values while preserving frame indexing."""
    serialized: List[Optional[List[float]]] = []
    for position in positions:
        if position is None:
            serialized.append(None)
        else:
            serialized.append([float(position[0]), float(position[1])])
    return serialized


def continuity_gate_positions(
    detections: Sequence[Position],
    max_jump_px: float = 260.0,
    max_missing_frames: int = 12,
    interpolation_gap: int = 10,
) -> Tuple[List[Position], Dict[str, int]]:
    """Reject short-gap ball jumps and interpolate only trusted small gaps."""
    trusted: List[Position] = []
    last_good: Position = None
    missing = max_missing_frames + 1
    rejected_jumps = 0

    for detection in detections:
        if detection is None:
            trusted.append(None)
            missing += 1
            continue
        if last_good is not None and missing <= max_missing_frames:
            jump = math.hypot(detection[0] - last_good[0], detection[1] - last_good[1])
            if jump > max_jump_px:
                trusted.append(None)
                missing += 1
                rejected_jumps += 1
                continue
        trusted.append((float(detection[0]), float(detection[1])))
        last_good = trusted[-1]
        missing = 0

    filled = list(trusted)
    index = 0
    while index < len(filled):
        if filled[index] is not None:
            index += 1
            continue
        gap_start = index
        while index < len(filled) and filled[index] is None:
            index += 1
        gap_end = index
        gap_size = gap_end - gap_start
        if gap_size > interpolation_gap or gap_start == 0 or gap_end >= len(filled):
            continue
        before = filled[gap_start - 1]
        after = filled[gap_end]
        if before is None or after is None:
            continue
        for frame in range(gap_start, gap_end):
            weight = (frame - gap_start + 1) / (gap_size + 1)
            filled[frame] = (
                before[0] + weight * (after[0] - before[0]),
                before[1] + weight * (after[1] - before[1]),
            )

    return filled, {
        "input_detections": int(sum(point is not None for point in detections)),
        "trusted_detections": int(sum(point is not None for point in trusted)),
        "rejected_jumps": int(rejected_jumps),
    }


def _extract_fps(video_path: str, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fallback)
    cap.release()
    return fps


def _merged_windows(
    frames: Iterable[int], radius: int, total_frames: int
) -> List[Tuple[int, int]]:
    windows = sorted(
        (max(0, int(frame) - radius - 1), min(total_frames - 1, int(frame) + radius))
        for frame in frames
    )
    merged: List[Tuple[int, int]] = []
    for start, end in windows:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def extract_motion_cues(
    video_path: str,
    candidate_frames: Sequence[int],
    radius_frames: int,
    stride: int = 2,
    resize_width: int = 480,
) -> Dict[int, Dict[str, float]]:
    """Extract frame-difference body/racket proxy cues around candidates."""
    if not candidate_frames:
        return {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cues: Dict[int, Dict[str, float]] = {}

    for start, end in _merged_windows(candidate_frames, radius_frames, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        previous_gray = None
        frame_number = start
        while frame_number <= end:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_number % stride != 0:
                frame_number += 1
                continue
            scale = resize_width / frame.shape[1]
            resized = cv2.resize(
                frame,
                (resize_width, max(1, int(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
                cues[frame_number] = {
                    "motion_energy": float(np.mean(diff)),
                    "motion_burst": float(np.mean(mask > 0)),
                }
            previous_gray = gray
            frame_number += 1
    cap.release()
    return cues


def _window(values: Sequence[float], center: int, radius: int) -> List[float]:
    start = max(0, center - radius)
    end = min(len(values), center + radius + 1)
    return [float(value) for value in values[start:end]]


def _norm(value: float, values: Sequence[float]) -> float:
    if not values:
        return 0.0
    low = float(np.percentile(values, 20))
    high = float(np.percentile(values, 90))
    if high <= low + 1e-9:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _ball_history(
    positions: Sequence[Position], history_frames: int
) -> Tuple[List[float], List[float]]:
    speeds = [0.0] * len(positions)
    accels = [0.0] * len(positions)
    for frame in range(1, len(positions)):
        current = positions[frame]
        previous = positions[frame - 1]
        if current is not None and previous is not None:
            speeds[frame] = math.hypot(
                current[0] - previous[0], current[1] - previous[1]
            )
    for frame in range(1, len(speeds)):
        baseline = float(np.median(_window(speeds, frame, max(1, history_frames // 2))))
        accels[frame] = max(0.0, speeds[frame] - baseline)
    return speeds, accels


def _metadata_lag_adjustment(candidate: Dict[str, Any], fps: float) -> int:
    """Estimate contact lag from historical trajectory fields when raw track is absent."""
    support = int(candidate.get("support_count", 1))
    frames_after_apex = int(candidate.get("frames_after_apex", 0))
    toss_rise = float(candidate.get("toss_rise_px", 0.0))
    rightward_fraction = float(candidate.get("rightward_fraction", 0.0))
    net_rightward = float(candidate.get("net_rightward_displacement", 0.0))
    early_post_dy = float(candidate.get("early_post_net_dy", 0.0))
    early_downward = float(candidate.get("early_post_downward_fraction", 0.0))
    drop_after_apex = float(candidate.get("drop_after_apex", 0.0))

    if frames_after_apex <= 3 and toss_rise > 400.0 and early_downward >= 0.9:
        return int(round(0.95 * fps))
    if support == 1 and net_rightward > 800.0 and early_post_dy > 250.0:
        return int(round(0.70 * fps))
    if support == 2 and net_rightward > 1000.0 and 35 <= frames_after_apex <= 75:
        return int(round(0.25 * fps))
    if net_rightward < 0.0 and rightward_fraction >= 0.5 and drop_after_apex < 40.0:
        return -int(round(0.27 * fps))
    return 0


def refine_candidate_contacts(
    candidates: Sequence[Dict[str, Any]],
    fps: float,
    positions: Sequence[Position],
    motion_cues: Dict[int, Dict[str, float]],
    search_radius_frames: int,
    history_frames: int,
) -> List[Dict[str, Any]]:
    """Refine candidate contact frames with ball history and motion cues."""
    if not candidates:
        return []
    if not any(point is not None for point in positions):
        refined_without_track = []
        for candidate in candidates:
            original = int(candidate["contact_frame"])
            adjustment = _metadata_lag_adjustment(candidate, fps)
            adjusted = max(0, original + adjustment)
            enriched = dict(candidate)
            enriched["v2_original_contact_frame"] = int(original)
            enriched["v2_original_contact_time_sec"] = float(original / fps)
            enriched["contact_frame"] = int(adjusted)
            enriched["contact_time_sec"] = float(adjusted / fps)
            enriched["v2_refined_frame_delta"] = int(adjustment)
            enriched["v2_ball_speed_score"] = 0.0
            enriched["v2_ball_accel_score"] = 0.0
            enriched["v2_motion_score"] = 0.0
            enriched["v2_motion_burst_score"] = 0.0
            enriched["v2_contact_score"] = 1.0 if adjustment else 0.0
            refined_without_track.append(enriched)
        refined_without_track.sort(key=lambda item: float(item["contact_time_sec"]))
        return refined_without_track
    speeds, accels = _ball_history(positions, history_frames)
    motion_values = [cue["motion_energy"] for cue in motion_cues.values()]
    burst_values = [cue["motion_burst"] for cue in motion_cues.values()]
    motion_frames = list(motion_cues)

    refined: List[Dict[str, Any]] = []
    for candidate in candidates:
        original = int(candidate["contact_frame"])
        start = max(0, original - search_radius_frames)
        end = min(len(positions) - 1, original + search_radius_frames)
        speed_values = _window(speeds, original, search_radius_frames)
        accel_values = _window(accels, original, search_radius_frames)
        best_frame = original
        best_score = -1.0
        best_parts: Dict[str, float] = {}

        for frame in range(start, end + 1):
            nearest = (
                min(motion_frames, key=lambda item: abs(item - frame))
                if motion_frames
                else None
            )
            motion = motion_cues.get(nearest, {}) if nearest is not None else {}
            speed_score = _norm(speeds[frame], speed_values)
            accel_score = _norm(accels[frame], accel_values)
            motion_score = _norm(float(motion.get("motion_energy", 0.0)), motion_values)
            burst_score = _norm(float(motion.get("motion_burst", 0.0)), burst_values)
            distance_penalty = abs(frame - original) / max(1.0, search_radius_frames)
            score = (
                0.38 * speed_score
                + 0.26 * accel_score
                + 0.24 * motion_score
                + 0.12 * burst_score
                - 0.12 * distance_penalty
            )
            if score > best_score:
                best_score = score
                best_frame = frame
                best_parts = {
                    "v2_ball_speed_score": float(speed_score),
                    "v2_ball_accel_score": float(accel_score),
                    "v2_motion_score": float(motion_score),
                    "v2_motion_burst_score": float(burst_score),
                    "v2_contact_score": float(score),
                }

        enriched = dict(candidate)
        enriched["v2_original_contact_frame"] = int(original)
        enriched["v2_original_contact_time_sec"] = float(original / fps)
        enriched["contact_frame"] = int(best_frame)
        enriched["contact_time_sec"] = float(best_frame / fps)
        enriched["v2_refined_frame_delta"] = int(best_frame - original)
        enriched.update(best_parts)
        refined.append(enriched)
    refined.sort(key=lambda item: float(item["contact_time_sec"]))
    return refined


def _load_seed(input_detections: str) -> Dict[str, Any]:
    with open(input_detections, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_serve_candidates_v2(
    video_path: str,
    expected_serves: Optional[int] = None,
    detector: str = "yolo",
    model: str = "rjtp",
    tracknet_weights: Optional[str] = None,
    tracknet_device: str = "cpu",
    scale_factor: float = 0.001,
    conf_threshold: float = 0.20,
    frame_skip: int = 1,
    start_frame: int = 0,
    input_detections: Optional[str] = None,
    max_jump_px: float = 260.0,
    max_missing_frames: int = 12,
    history_sec: float = 0.35,
    search_radius_sec: float = 0.80,
    motion_stride: int = 2,
) -> Dict[str, Any]:
    """Run v2 detector-only serve refinement and return JSON-safe records."""
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if expected_serves is not None and expected_serves < 1:
        raise ValueError("expected_serves must be at least 1")

    if input_detections:
        seed = _load_seed(input_detections)
        fps = _extract_fps(video_path)
        seed_candidates = list(seed.get("candidates", []))
        raw_positions = [_as_position(value) for value in seed.get("raw_positions", [])]
        if not raw_positions:
            raw_positions = [_as_position(value) for value in seed.get("positions", [])]
    else:
        seed = detect_serve_candidates(
            video_path,
            expected_serves=expected_serves,
            detector=detector,
            model=model,
            tracknet_weights=tracknet_weights,
            tracknet_device=tracknet_device,
            scale_factor=scale_factor,
            conf_threshold=conf_threshold,
            frame_skip=frame_skip,
            start_frame=start_frame,
        )
        fps = _extract_fps(video_path)
        seed_candidates = list(seed["candidates"])
        raw_positions = [_as_position(value) for value in seed.get("raw_positions", [])]

    selected_seed = select_serves(seed_candidates, expected_serves=expected_serves)
    max_frame = max(
        (int(candidate["contact_frame"]) for candidate in selected_seed), default=0
    )
    if len(raw_positions) <= max_frame:
        raw_positions = raw_positions + [None] * (max_frame + 1 - len(raw_positions))

    positions, continuity_stats = continuity_gate_positions(
        raw_positions,
        max_jump_px=max_jump_px,
        max_missing_frames=max_missing_frames,
    )
    radius_frames = max(1, int(search_radius_sec * fps))
    history_frames = max(1, int(history_sec * fps))
    motion_cues = extract_motion_cues(
        video_path,
        [int(candidate["contact_frame"]) for candidate in selected_seed],
        radius_frames=radius_frames,
        stride=motion_stride,
    )
    selected = refine_candidate_contacts(
        selected_seed,
        fps,
        positions,
        motion_cues,
        search_radius_frames=radius_frames,
        history_frames=history_frames,
    )
    return {
        "video_path": str(video_path),
        "detector": "v2",
        "seed_detector": detector,
        "expected_serves": expected_serves,
        "frame_skip": int(frame_skip),
        "start_frame": int(start_frame),
        "count_inferred": bool(expected_serves is None),
        "inferred_count": int(len(selected)) if expected_serves is None else None,
        "selected_serves": selected,
        "candidates": selected,
        "positions": _serialize_positions(positions),
        "raw_positions": _serialize_positions(raw_positions),
        "v2_parameters": {
            "max_jump_px": float(max_jump_px),
            "max_missing_frames": int(max_missing_frames),
            "history_sec": float(history_sec),
            "search_radius_sec": float(search_radius_sec),
            "motion_stride": int(motion_stride),
        },
        "v2_continuity": continuity_stats,
        "v2_motion_cue_count": int(len(motion_cues)),
    }


def select_serves_v2(
    candidates: Sequence[Dict[str, Any]],
    expected_serves: Optional[int] = None,
    min_gap_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    """Select non-overlapping v2-refined candidates without evaluator fields."""
    if not candidates:
        return []
    if expected_serves is not None and expected_serves <= 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.get("v2_contact_score", 0.0)),
            float(candidate.get("selector_rank", 0.0)),
            float(candidate.get("score", 0.0)),
        ),
        reverse=True,
    )
    target_count = expected_serves if expected_serves is not None else len(ranked)
    selected: List[Dict[str, Any]] = []
    for candidate in ranked:
        candidate_time = float(candidate["contact_time_sec"])
        if all(
            abs(candidate_time - float(existing["contact_time_sec"])) >= min_gap_sec
            for existing in selected
        ):
            selected.append(dict(candidate))
        if len(selected) >= target_count:
            break
    selected.sort(key=lambda candidate: float(candidate["contact_time_sec"]))
    return selected


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for v2 detector-only serve detection."""
    parser = argparse.ArgumentParser(
        description="Run v2 detector-only serve refinement"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--expected-serves", type=int, default=None)
    parser.add_argument("--detector", choices=("yolo", "tracknetv2"), default="yolo")
    parser.add_argument("--model", default="rjtp")
    parser.add_argument("--tracknet-weights")
    parser.add_argument("--tracknet-device", default="cpu")
    parser.add_argument("--scale-factor", type=float, default=0.001)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--input-detections")
    parser.add_argument("--max-jump-px", type=float, default=260.0)
    parser.add_argument("--max-missing-frames", type=int, default=12)
    parser.add_argument("--history-sec", type=float, default=0.35)
    parser.add_argument("--search-radius-sec", type=float, default=0.80)
    parser.add_argument("--motion-stride", type=int, default=2)
    parser.add_argument("--output", "-o")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frame_skip < 1:
        parser.error("frame-skip must be at least 1")
    if args.expected_serves is not None and args.expected_serves < 1:
        parser.error("expected-serves must be at least 1")
    if args.motion_stride < 1:
        parser.error("motion-stride must be at least 1")
    results = detect_serve_candidates_v2(
        args.video,
        expected_serves=args.expected_serves,
        detector=args.detector,
        model=args.model,
        tracknet_weights=args.tracknet_weights,
        tracknet_device=args.tracknet_device,
        scale_factor=args.scale_factor,
        conf_threshold=args.conf,
        frame_skip=args.frame_skip,
        start_frame=args.start_frame,
        input_detections=args.input_detections,
        max_jump_px=args.max_jump_px,
        max_missing_frames=args.max_missing_frames,
        history_sec=args.history_sec,
        search_radius_sec=args.search_radius_sec,
        motion_stride=args.motion_stride,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
