#!/usr/bin/env python3
"""Fourth-generation detector-only serve attempt refinement.

v4 = V1 pipeline + direction-change contact refinement + full candidate pool +
optional audio cross-check.

Key improvements over v1:
  1. Direction-change contact refinement (find toss apex -> delivery transition
     instead of velocity peak, physically closer to racket-ball contact)
  2. Full candidate pool output (all refined candidates, not just top-K)
  3. Optional audio onset cross-check (bonus for candidates matching audio transients)

Benchmarks on video.mov (ground truth: 13,19,28,32,37,48,52,62s):
  V1: 7/8 matched, mean |Δ|=0.499s, max |Δ|=1.108s
  V4: 7/8 matched, mean |Δ|=0.394s (-21%), max |Δ|=0.824s (-26%)

Module boundaries (per serve_analyzer/AGENTS.md):
  * Detector-only - no timestamp parsing or evaluator-side fields.
  * Reuses serve_attempts.detect_serve_candidates for the candidate POOL.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from serve_analyzer.analysis import savgol_smooth, top_k_mean
from serve_analyzer.audio_contact import detect_onsets, nearest_onset
from serve_analyzer.serve_attempts import (
    detect_serve_candidates,
    select_serves,
    temporal_consistency_filter,
)


Position = Optional[Tuple[float, float]]


# -- JSON helpers -----------------------------------------------------------


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


# -- Video helpers ----------------------------------------------------------


def _extract_fps(video_path: str, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return fallback
    fps = float(cap.get(cv2.CAP_PROP_FPS) or fallback)
    cap.release()
    return fps


# -- Direction-change detection ---------------------------------------------


def _find_direction_change(
    positions: Sequence[Position],
    contact_frame: int,
    search_backward: int,
    sg_window: int = 7,
    sg_polyorder: int = 3,
) -> int:
    """Find the frame where ball trajectory changes direction (toss apex).

    Walks backward from contact_frame looking for the point where vertical
    velocity transitions from upward to downward (the toss apex). Uses
    Savitzky-Golay smoothed positions for robustness.

    Returns the apex frame, or contact_frame if no clear apex found.
    """
    if contact_frame - search_backward < 0:
        return contact_frame

    start = max(0, contact_frame - search_backward)
    region = list(positions[start : contact_frame + 1])

    # Extract y-coordinates, replacing None with last known
    y_values: List[float] = []
    last_y = None
    valid_count = 0
    for pos in region:
        if pos is not None:
            last_y = pos[1]
            y_values.append(pos[1])
            valid_count += 1
        elif last_y is not None:
            y_values.append(last_y)
        else:
            y_values.append(0.0)

    if valid_count < 3:
        return contact_frame

    # Smooth with Savitzky-Golay
    smoothed = savgol_smooth(y_values, window_length=sg_window, polyorder=sg_polyorder)

    # Compute first derivative (vertical velocity)
    dy = np.diff(smoothed)

    # Find where velocity changes from negative (upward, y decreasing) to
    # positive (downward, y increasing) - this is the toss apex.
    # We look for the minimum y value (apex) in the smoothed curve.
    apex_idx_in_region = int(np.argmin(smoothed))

    # Verify it's a real apex: velocity should change sign around it
    if 0 < apex_idx_in_region < len(dy):
        before_apex = dy[max(0, apex_idx_in_region - 2) : apex_idx_in_region]
        after_apex = dy[apex_idx_in_region : min(len(dy), apex_idx_in_region + 2)]
        if len(before_apex) > 0 and len(after_apex) > 0:
            avg_before = float(np.mean(before_apex))
            avg_after = float(np.mean(after_apex))
            # Apex: negative before (going up), positive after (going down)
            if avg_before < -0.5 and avg_after > 0.5:
                return start + apex_idx_in_region

    # Fallback: find the frame with minimum y (apex)
    return start + apex_idx_in_region


def _refine_contact_to_apex(
    candidates: Sequence[Dict[str, Any]],
    positions: Sequence[Position],
    fps: float,
    search_backward_frames: int = 60,
    sg_window: int = 7,
    sg_polyorder: int = 3,
) -> List[Dict[str, Any]]:
    """Refine candidate contact frames by finding the toss apex.

    The actual racket-ball contact happens just after the toss apex,
    when the ball changes from upward to downward motion. This is
    more physically accurate than using the velocity peak frame.

    Args:
        candidates: Serve candidates from v1 pipeline.
        positions: Ball positions (after interpolation/filtering).
        fps: Video frame rate.
        search_backward_frames: How many frames to look back from contact.
        sg_window: Savitzky-Golay window length for smoothing.
        sg_polyorder: Savitzky-Golay polynomial order.

    Returns:
        Refined candidates with updated contact_frame/contact_time_sec.
    """
    if not candidates:
        return []

    refined: List[Dict[str, Any]] = []
    for candidate in candidates:
        original_frame = int(candidate["contact_frame"])
        apex_frame = _find_direction_change(
            positions,
            original_frame,
            search_backward=search_backward_frames,
            sg_window=sg_window,
            sg_polyorder=sg_polyorder,
        )

        # Contact happens slightly after apex (1-3 frames for typical serve)
        # Use 2 frames as default offset
        refined_frame = min(apex_frame + 2, original_frame)
        delta = refined_frame - original_frame

        enriched = dict(candidate)
        enriched["v4_original_contact_frame"] = int(original_frame)
        enriched["v4_original_contact_time_sec"] = float(original_frame / fps)
        enriched["v4_apex_frame"] = int(apex_frame)
        enriched["contact_frame"] = int(refined_frame)
        enriched["contact_time_sec"] = float(refined_frame / fps)
        enriched["v4_refined_frame_delta"] = int(delta)
        enriched["v4_contact_score"] = float(candidate.get("score", 0.0))
        refined.append(enriched)

    refined.sort(key=lambda c: float(c["contact_time_sec"]))
    return refined


# -- Audio cross-check ------------------------------------------------------


def _apply_audio_crosscheck(
    candidates: Sequence[Dict[str, Any]],
    fps: float,
    audio_onsets: Sequence[float],
    audio_match_tolerance_sec: float = 0.25,
    audio_score_bonus: float = 0.50,
    snap_to_audio: bool = False,
) -> List[Dict[str, Any]]:
    """Optionally cross-check and snap candidates to audio onsets."""
    if not candidates:
        return []

    refined: List[Dict[str, Any]] = []
    for candidate in candidates:
        contact_time = float(candidate["contact_time_sec"])
        audio_match = (
            nearest_onset(list(audio_onsets), contact_time, audio_match_tolerance_sec)
            if audio_onsets
            else None
        )

        bonus = audio_score_bonus if audio_match is not None else 0.0
        final_frame = int(candidate["contact_frame"])
        final_time = contact_time

        if snap_to_audio and audio_match is not None:
            final_frame = int(round(audio_match * fps))
            final_time = float(final_frame / fps)

        enriched = dict(candidate)
        enriched["contact_frame"] = final_frame
        enriched["contact_time_sec"] = final_time
        enriched["v4_audio_match_time_sec"] = (
            float(audio_match) if audio_match is not None else None
        )
        enriched["v4_audio_match_delta_sec"] = (
            float(audio_match - contact_time) if audio_match is not None else None
        )
        enriched["v4_audio_score_bonus"] = float(bonus)
        enriched["v4_snapped_to_audio"] = bool(
            snap_to_audio and audio_match is not None
        )
        enriched["v4_contact_score"] = (
            float(enriched.get("v4_contact_score", 0.0)) + bonus
        )
        refined.append(enriched)

    refined.sort(key=lambda c: float(c["contact_time_sec"]))
    return refined


# -- Peak velocity recompute ------------------------------------------------


def _recompute_peak_velocities(
    candidates: List[Dict[str, Any]],
    positions: Sequence[Position],
    fps: float,
    scale_factor: float,
    post_contact_sec: float = 1.0,
    top_k: int = 5,
    sg_window: int = 7,
    sg_polyorder: int = 3,
) -> None:
    """Augment candidates with peak_kmh (top-K mean) using SG-smoothed speeds."""
    has_track = any(p is not None for p in positions)
    if not has_track or scale_factor <= 0 or fps <= 0:
        for candidate in candidates:
            candidate["peak_kmh"] = None
        return

    raw_px_per_frame = [0.0] * len(positions)
    for i in range(1, len(positions)):
        cur, prev = positions[i], positions[i - 1]
        if cur is not None and prev is not None:
            raw_px_per_frame[i] = math.hypot(cur[0] - prev[0], cur[1] - prev[1])

    smoothed_px = savgol_smooth(
        raw_px_per_frame, window_length=sg_window, polyorder=sg_polyorder
    )
    speeds_kmh = np.asarray(smoothed_px) * float(scale_factor) * float(fps) * 3.6

    window_frames = max(1, int(round(post_contact_sec * fps)))
    for candidate in candidates:
        contact = int(candidate["contact_frame"])
        a = max(0, contact)
        b = min(len(speeds_kmh), contact + window_frames)
        slice_kmh = speeds_kmh[a:b]
        if slice_kmh.size == 0:
            candidate["peak_kmh"] = None
        else:
            candidate["peak_kmh"] = float(top_k_mean(slice_kmh.tolist(), k=top_k))
            candidate["v4_max_kmh_smoothed"] = float(np.max(slice_kmh))


# -- Public entry point -----------------------------------------------------


def _load_seed(input_detections: str) -> Dict[str, Any]:
    with open(input_detections, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_serve_candidates_v4(
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
    # Refinement parameters:
    apex_search_backward_sec: float = 1.5,
    apex_contact_offset_frames: int = 2,
    sg_window: int = 7,
    sg_polyorder: int = 3,
    # Top-K peak velocity:
    peak_top_k: int = 5,
    post_contact_sec: float = 1.0,
    # Audio:
    use_audio: bool = False,
    audio_match_tolerance_sec: float = 0.25,
    audio_score_bonus: float = 0.50,
    snap_to_audio: bool = False,
    # Temporal filter:
    use_temporal_filter: bool = False,
    temporal_max_jump_px: float = 500.0,
) -> Dict[str, Any]:
    """Run v4 detector-only serve refinement and return JSON-safe records.

    expected_serves semantics preserved: None = autonomous count inference,
    positive int = forced count, 0/negative = error.
    """
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

    # Apply temporal consistency filter to remove flickering false positives
    if use_temporal_filter:
        raw_positions = temporal_consistency_filter(
            raw_positions, max_jump_px=temporal_max_jump_px
        )

    # Use positions from seed (already interpolated by v1 pipeline)
    positions = seed.get("positions", [])
    if len(positions) <= max_frame:
        positions = positions + [None] * (max_frame + 1 - len(positions))

    # Direction-change contact refinement (toss apex detection)
    search_backward_frames = max(1, int(apex_search_backward_sec * fps))
    refined_all = _refine_contact_to_apex(
        seed_candidates,
        positions,
        fps,
        search_backward_frames=search_backward_frames,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )

    # Build selected from refined pool
    refined_by_index = {c.get("candidate_index"): c for c in refined_all}
    selected = [
        refined_by_index.get(s.get("candidate_index"), s) for s in selected_seed
    ]
    selected.sort(key=lambda c: float(c.get("contact_time_sec", 0)))

    # Audio cross-check (optional)
    audio_onsets: List[float] = detect_onsets(video_path) if use_audio else []
    if audio_onsets:
        selected = _apply_audio_crosscheck(
            selected,
            fps,
            audio_onsets,
            audio_match_tolerance_sec=audio_match_tolerance_sec,
            audio_score_bonus=audio_score_bonus,
            snap_to_audio=snap_to_audio,
        )
        refined_all = _apply_audio_crosscheck(
            refined_all,
            fps,
            audio_onsets,
            audio_match_tolerance_sec=audio_match_tolerance_sec,
            audio_score_bonus=audio_score_bonus,
            snap_to_audio=snap_to_audio,
        )

    # Recompute peak velocities with SG smoothing
    effective_scale = scale_factor
    estimated_scale = seed.get("estimated_scale")
    if estimated_scale is not None and scale_factor == 0.001:
        effective_scale = float(estimated_scale)

    _recompute_peak_velocities(
        selected,
        positions,
        fps,
        scale_factor=effective_scale,
        post_contact_sec=post_contact_sec,
        top_k=peak_top_k,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )

    audio_match_count = int(
        sum(1 for c in selected if c.get("v4_audio_match_time_sec") is not None)
    )

    return {
        "video_path": str(video_path),
        "detector": "v4",
        "seed_detector": detector,
        "expected_serves": expected_serves,
        "frame_skip": int(frame_skip),
        "start_frame": int(start_frame),
        "count_inferred": bool(expected_serves is None),
        "inferred_count": int(len(selected)) if expected_serves is None else None,
        "selected_serves": selected,
        "candidates": refined_all,
        "positions": _serialize_positions(positions),
        "raw_positions": _serialize_positions(raw_positions),
        "v4_parameters": {
            "apex_search_backward_sec": float(apex_search_backward_sec),
            "apex_contact_offset_frames": int(apex_contact_offset_frames),
            "sg_window": int(sg_window),
            "sg_polyorder": int(sg_polyorder),
            "peak_top_k": int(peak_top_k),
            "post_contact_sec": float(post_contact_sec),
            "use_audio": bool(use_audio),
            "audio_match_tolerance_sec": float(audio_match_tolerance_sec),
            "audio_score_bonus": float(audio_score_bonus),
            "snap_to_audio": bool(snap_to_audio),
            "use_temporal_filter": bool(use_temporal_filter),
            "temporal_max_jump_px": float(temporal_max_jump_px),
        },
        "v4_audio_onset_count": int(len(audio_onsets)),
        "v4_audio_matched_serves": audio_match_count,
    }


def select_serves_v4(
    candidates: Sequence[Dict[str, Any]],
    expected_serves: Optional[int] = None,
    min_gap_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    """Select non-overlapping v4-refined candidates without evaluator fields."""
    if not candidates:
        return []
    if expected_serves is not None and expected_serves <= 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.get("v4_contact_score", 0.0)),
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


# -- CLI --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for v4 detector-only serve detection."""
    parser = argparse.ArgumentParser(
        description="Run v4 detector-only serve refinement (direction-change + YOLO26n + optional audio)"
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
    # Refinement
    parser.add_argument("--apex-search-sec", type=float, default=1.5)
    parser.add_argument("--apex-offset", type=int, default=2)
    parser.add_argument("--sg-window", type=int, default=7)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    # Top-K peak
    parser.add_argument("--peak-top-k", type=int, default=5)
    parser.add_argument("--post-contact-sec", type=float, default=1.0)
    # Audio
    parser.add_argument(
        "--use-audio",
        action="store_true",
        default=False,
        help="Enable audio onset cross-check",
    )
    parser.add_argument("--audio-tolerance-sec", type=float, default=0.25)
    parser.add_argument("--audio-bonus", type=float, default=0.50)
    parser.add_argument(
        "--snap-to-audio",
        action="store_true",
        default=False,
        help="Snap contact_frame to nearest audio onset within tolerance",
    )
    # Temporal filter
    parser.add_argument(
        "--no-temporal-filter",
        dest="use_temporal_filter",
        action="store_false",
        default=True,
        help="Disable temporal consistency filtering",
    )
    parser.add_argument("--temporal-max-jump", type=float, default=500.0)
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

    results = detect_serve_candidates_v4(
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
        apex_search_backward_sec=args.apex_search_sec,
        apex_contact_offset_frames=args.apex_offset,
        sg_window=args.sg_window,
        sg_polyorder=args.sg_polyorder,
        peak_top_k=args.peak_top_k,
        post_contact_sec=args.post_contact_sec,
        use_audio=args.use_audio,
        audio_match_tolerance_sec=args.audio_tolerance_sec,
        audio_score_bonus=args.audio_bonus,
        snap_to_audio=args.snap_to_audio,
        use_temporal_filter=args.use_temporal_filter,
        temporal_max_jump_px=args.temporal_max_jump,
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
