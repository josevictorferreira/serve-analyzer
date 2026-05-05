#!/usr/bin/env python3
"""Fifth-generation detector-only serve attempt refinement.

v5 = V1 pipeline + hybrid contact timing + post-selection quality gating +
full candidate pool output.

Key improvements over v1:
  1. Hybrid contact timing - keeps V1's peak frame as primary, uses apex
     detection to cap excessive backward shifts (max 10 frames)
  2. Post-selection quality gating - rejects candidates with insufficient
     post-contact speed (< 30 km/h) or direction evidence
  3. Adaptive apex-to-contact offset based on toss height

Module boundaries (per serve_analyzer/AGENTS.md):
  * Detector-only - no timestamp parsing or evaluator-side fields.
  * Reuses serve_attempts.detect_serve_candidates for the candidate POOL.

Benchmarks on video.mov (ground truth: 13,19,28,32,37,48,52,62s):
  V1: 7/8 matched, mean |Δ|=0.499s, max |Δ|=1.108s
  V5: TBD
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
from serve_analyzer.serve_attempts import detect_serve_candidates, select_serves


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


# -- Hybrid contact timing -------------------------------------------------


def _find_apex_frame(
    positions: Sequence[Position],
    contact_frame: int,
    search_backward: int,
    sg_window: int = 7,
    sg_polyorder: int = 3,
) -> int:
    """Find the toss apex frame (minimum y) near the contact frame."""
    if contact_frame - search_backward < 0:
        return contact_frame

    start = max(0, contact_frame - search_backward)
    region = list(positions[start : contact_frame + 1])

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

    smoothed = savgol_smooth(y_values, window_length=sg_window, polyorder=sg_polyorder)
    apex_idx = int(np.argmin(smoothed))
    return start + apex_idx


def _adaptive_apex_offset(toss_rise_px: float, max_offset: int = 15) -> int:
    """Compute adaptive offset from apex to contact based on toss height.

    Higher toss = longer time between apex and racket contact.
    Uses linear scaling: offset = toss_rise_px / 20, capped at max_offset.
    """
    return max(2, min(max_offset, int(toss_rise_px / 20)))


def _refine_contact_hybrid(
    candidates: Sequence[Dict[str, Any]],
    positions: Sequence[Position],
    fps: float,
    search_backward_frames: int = 60,
    max_backward_shift: int = 10,
    sg_window: int = 7,
    sg_polyorder: int = 3,
) -> List[Dict[str, Any]]:
    """Refine contact frames using hybrid timing.

    Keeps V1's peak frame as primary. Uses apex detection to identify
    excessive backward shifts (V1 is too late) and caps them at max_backward_shift.

    This preserves V1's accuracy when it's good, but corrects gross overestimates.

    Args:
        candidates: Serve candidates from V1 pipeline.
        positions: Ball positions (after interpolation).
        fps: Video frame rate.
        search_backward_frames: How many frames to look back from contact.
        max_backward_shift: Maximum frames to shift backward (default 10).
        sg_window: Savitzky-Golay window length.
        sg_polyorder: Savitzky-Golay polynomial order.

    Returns:
        Refined candidates with updated contact_frame/contact_time_sec.
    """
    if not candidates:
        return []

    refined: List[Dict[str, Any]] = []
    for candidate in candidates:
        original_frame = int(candidate["contact_frame"])
        toss_rise = float(candidate.get("toss_rise_px", 0.0))

        # Find toss apex
        apex_frame = _find_apex_frame(
            positions,
            original_frame,
            search_backward=search_backward_frames,
            sg_window=sg_window,
            sg_polyorder=sg_polyorder,
        )

        frames_before_apex = original_frame - apex_frame

        # Only shift backward if V1 is significantly late (> max_backward_shift frames after apex)
        if frames_before_apex > max_backward_shift:
            shift = frames_before_apex - max_backward_shift
            refined_frame = original_frame - shift
        else:
            # V1 is close enough to apex; keep it unchanged
            refined_frame = original_frame
            shift = 0

        refined_frame = max(0, refined_frame)

        enriched = dict(candidate)
        enriched["v5_original_contact_frame"] = int(original_frame)
        enriched["v5_original_contact_time_sec"] = float(original_frame / fps)
        enriched["v5_apex_frame"] = int(apex_frame)
        enriched["v5_frames_after_apex"] = int(refined_frame - apex_frame)
        enriched["contact_frame"] = int(refined_frame)
        enriched["contact_time_sec"] = float(refined_frame / fps)
        enriched["v5_refined_frame_delta"] = int(refined_frame - original_frame)
        enriched["v5_contact_score"] = float(candidate.get("score", 0.0))
        enriched["v5_adaptive_offset"] = int(_adaptive_apex_offset(toss_rise))
        refined.append(enriched)

    refined.sort(key=lambda c: float(c["contact_time_sec"]))
    return refined


# -- Post-selection quality gating -----------------------------------------


def _quality_gate_candidates(
    candidates: List[Dict[str, Any]],
    min_post_contact_kmh: float = 15.0,
    min_rightward_fraction: float = 0.2,
) -> List[Dict[str, Any]]:
    """Filter candidates that don't meet minimum serve quality thresholds.

    Removes candidates with:
    - post_contact_max_kmh < min_post_contact_kmh (no measurable ball speed)
    - rightward_fraction < min_rightward_fraction (ball doesn't move right)

    Args:
        candidates: Refined serve candidates.
        min_post_contact_kmh: Minimum post-contact speed in km/h.
        min_rightward_fraction: Minimum fraction of rightward motion.

    Returns:
        Filtered list of candidates.
    """
    passed = []
    for c in candidates:
        max_kmh = float(c.get("post_contact_max_kmh", 0.0))
        right_frac = float(c.get("rightward_fraction", 0.0))

        if max_kmh < min_post_contact_kmh:
            continue
        if right_frac < min_rightward_fraction:
            continue

        passed.append(c)

    return passed


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
            candidate["v5_max_kmh_smoothed"] = float(np.max(slice_kmh))


# -- Public entry point -----------------------------------------------------


def _load_seed(input_detections: str) -> Dict[str, Any]:
    with open(input_detections, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_serve_candidates_v5(
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
    # Hybrid timing parameters:
    apex_search_backward_sec: float = 1.5,
    max_backward_shift: int = 10,
    sg_window: int = 7,
    sg_polyorder: int = 3,
    # Quality gating:
    quality_gate: bool = True,
    min_post_contact_kmh: float = 15.0,
    min_rightward_fraction: float = 0.2,
    # Top-K peak velocity:
    peak_top_k: int = 5,
    post_contact_sec: float = 1.0,
) -> Dict[str, Any]:
    """Run v5 detector-only serve refinement and return JSON-safe records.

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

    # Get selected serves from V1 pipeline
    selected_seed = select_serves(seed_candidates, expected_serves=expected_serves)
    max_frame = max(
        (int(candidate["contact_frame"]) for candidate in selected_seed), default=0
    )
    if len(raw_positions) <= max_frame:
        raw_positions = raw_positions + [None] * (max_frame + 1 - len(raw_positions))

    # Use positions from seed (already interpolated by V1 pipeline)
    positions = seed.get("positions", [])
    if len(positions) <= max_frame:
        positions = positions + [None] * (max_frame + 1 - len(positions))

    # Hybrid contact refinement: keep V1's timing, cap excessive backward shifts
    search_backward_frames = max(1, int(apex_search_backward_sec * fps))
    refined_all = _refine_contact_hybrid(
        seed_candidates,
        positions,
        fps,
        search_backward_frames=search_backward_frames,
        max_backward_shift=max_backward_shift,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )

    # Build selected from refined pool
    refined_by_index = {c.get("candidate_index"): c for c in refined_all}
    selected = [
        refined_by_index.get(s.get("candidate_index"), s) for s in selected_seed
    ]
    selected.sort(key=lambda c: float(c.get("contact_time_sec", 0)))

    # Post-selection quality gating
    if quality_gate:
        selected = _quality_gate_candidates(
            selected,
            min_post_contact_kmh=min_post_contact_kmh,
            min_rightward_fraction=min_rightward_fraction,
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

    return {
        "video_path": str(video_path),
        "detector": "v5",
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
        "v5_parameters": {
            "apex_search_backward_sec": float(apex_search_backward_sec),
            "max_backward_shift": int(max_backward_shift),
            "quality_gate": bool(quality_gate),
            "min_post_contact_kmh": float(min_post_contact_kmh),
            "min_rightward_fraction": float(min_rightward_fraction),
            "sg_window": int(sg_window),
            "sg_polyorder": int(sg_polyorder),
            "peak_top_k": int(peak_top_k),
            "post_contact_sec": float(post_contact_sec),
        },
    }


def select_serves_v5(
    candidates: Sequence[Dict[str, Any]],
    expected_serves: Optional[int] = None,
    min_gap_sec: float = 2.0,
    min_post_contact_kmh: float = 15.0,
    min_rightward_fraction: float = 0.2,
) -> List[Dict[str, Any]]:
    """Select non-overlapping v5-refined candidates with quality gating."""
    if not candidates:
        return []
    if expected_serves is not None and expected_serves <= 0:
        return []

    # Apply quality gate first
    qualified = _quality_gate_candidates(
        list(candidates),
        min_post_contact_kmh=min_post_contact_kmh,
        min_rightward_fraction=min_rightward_fraction,
    )
    if not qualified:
        return []

    ranked = sorted(
        qualified,
        key=lambda candidate: (
            float(candidate.get("v5_contact_score", 0.0)),
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
    """Create CLI parser for v5 detector-only serve detection."""
    parser = argparse.ArgumentParser(
        description="Run v5 detector-only serve refinement (hybrid timing + quality gating)"
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
    # Hybrid timing
    parser.add_argument("--apex-search-sec", type=float, default=1.5)
    parser.add_argument("--max-backward-shift", type=int, default=10)
    parser.add_argument("--sg-window", type=int, default=7)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    # Quality gating
    parser.add_argument(
        "--no-quality-gate",
        dest="quality_gate",
        action="store_false",
        default=True,
        help="Disable post-selection quality gating",
    )
    parser.add_argument("--min-post-contact-kmh", type=float, default=15.0)
    parser.add_argument("--min-rightward-fraction", type=float, default=0.2)
    # Top-K peak
    parser.add_argument("--peak-top-k", type=int, default=5)
    parser.add_argument("--post-contact-sec", type=float, default=1.0)
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

    results = detect_serve_candidates_v5(
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
        max_backward_shift=args.max_backward_shift,
        sg_window=args.sg_window,
        sg_polyorder=args.sg_polyorder,
        quality_gate=args.quality_gate,
        min_post_contact_kmh=args.min_post_contact_kmh,
        min_rightward_fraction=args.min_rightward_fraction,
        peak_top_k=args.peak_top_k,
        post_contact_sec=args.post_contact_sec,
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
