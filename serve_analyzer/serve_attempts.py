#!/usr/bin/env python3
"""Detect serve attempts near target timestamps and estimate post-contact speed.

Detection-only concerns live here.  Timestamp parsing, matching, and
evaluation logic are in serve_analyzer.serve_evaluation.
"""

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence

from serve_analyzer.multi_serve import (
    analyze_serve,
    compute_frame_velocities,
    compute_vertical_velocity,
    detect_ball_yolo,
    detect_serve_events,
    interpolate_missing_detections,
)
from serve_analyzer.serve_evaluation import (  # noqa: F401  re-export for backward compat
    load_target_timestamps,
    match_targets_to_candidates,
    parse_timestamp_line,
    parse_timestamp_lines,
    parse_timestamp_token,
    parse_timestamps_text,
    summarize_serve_attempts,
)


def _merge_candidate_events(
    event_groups: Sequence[Sequence[Dict[str, Any]]],
    fps: float,
    max_merge_gap_sec: float = 0.75,
) -> List[Dict[str, Any]]:
    """Merge nearby candidate events from multiple detection profiles."""
    max_merge_gap_frames = max(1, int(max_merge_gap_sec * fps))
    flattened = sorted(
        (dict(event) for events in event_groups for event in events),
        key=lambda event: int(event["contact_frame"]),
    )
    if not flattened:
        return []

    merged: List[Dict[str, Any]] = []
    for event in flattened:
        contact_frame = int(event["contact_frame"])
        if not merged:
            merged.append(event)
            continue

        previous = merged[-1]
        previous_frame = int(previous["contact_frame"])
        if contact_frame - previous_frame > max_merge_gap_frames:
            merged.append(event)
            continue

        previous_score = float(previous.get("score", 0.0))
        current_score = float(event.get("score", 0.0))
        if current_score > previous_score:
            merged[-1] = event
    return merged


def detect_serve_candidates(
    video_path: str,
    expected_serves: Optional[int] = None,
    model: str = "rjtp",
    scale_factor: float = 0.001,
    conf_threshold: float = 0.20,
    frame_skip: int = 1,
    start_frame: int = 0,
) -> List[Dict[str, Any]]:
    """Detect serve candidates and compute post-contact velocity stats."""
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")

    raw_detections, fps, _total_frames, estimated_scale = detect_ball_yolo(
        video_path,
        model_path=model,
        conf_threshold=conf_threshold,
        frame_skip=frame_skip,
        start_frame=start_frame,
    )

    positions = interpolate_missing_detections(raw_detections, max_gap=15)
    velocities = compute_frame_velocities(positions, fps)
    vert_velocities = compute_vertical_velocity(positions)

    effective_scale = scale_factor
    if estimated_scale is not None and scale_factor == 0.001:
        effective_scale = float(estimated_scale)

    expected = expected_serves if expected_serves is not None else 12
    if expected < 1:
        raise ValueError("expected_serves must be at least 1")

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
            fps=fps,
            **profile,
        )
        for profile in detector_profiles
    ]
    candidate_events = _merge_candidate_events(candidate_event_groups, fps)
    max_merge_gap_frames = max(1, int(0.75 * fps))
    flattened_events = [event for events in candidate_event_groups for event in events]
    analyzed_serves = [
        analyze_serve(index, event, positions, fps, effective_scale)
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
            }
        )
    return candidates


def infer_serve_count(
    candidates: Sequence[Dict[str, Any]],
    min_rank_floor: float = 0.05,
    relative_floor: float = 0.55,
) -> int:
    """Infer serve count from ranked candidates via quality threshold.

    Uses a combined quality threshold: a candidate must have rank >= both
    min_rank_floor (absolute) and relative_floor * top_rank (relative to best).
    Among candidates passing the threshold, gap-based elbow detection further
    refines the count if a clear quality drop exists.
    """
    if not candidates:
        return 0
    ranked = sorted(
        candidates,
        key=lambda c: float(c.get("selector_rank", 0.0)),
        reverse=True,
    )
    ranks = [float(c.get("selector_rank", 0.0)) for c in ranked]
    top_rank = ranks[0]
    # Combined floor: absolute minimum AND relative to best candidate
    threshold = max(min_rank_floor, relative_floor * top_rank)
    above_threshold = [r for r in ranks if r >= threshold]
    if not above_threshold:
        return 0
    if len(above_threshold) <= 1:
        return len(above_threshold)
    # Gap-based elbow refinement on threshold-passing candidates
    gaps = [
        above_threshold[i] - above_threshold[i + 1]
        for i in range(len(above_threshold) - 1)
    ]
    if not gaps:
        return len(above_threshold)
    max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    max_gap = gaps[max_gap_idx]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap < 1e-9 or max_gap < 2.0 * mean_gap or max_gap < 0.20:
        return len(above_threshold)
    return max_gap_idx + 1


def select_serves(
    candidates: Sequence[Dict[str, Any]],
    expected_serves: Optional[int] = None,
    min_gap_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    """Select best non-overlapping serve candidates via geometry-first ranking.

    When expected_serves is None (autonomous mode), the count is inferred
    from the quality gap in ranked candidates.  When explicitly provided,
    exactly that many candidates are selected (backward-compatible).
    """
    if not candidates:
        return []
    if expected_serves is not None and expected_serves <= 0:
        return []

    ordered = sorted(candidates, key=lambda c: float(c["contact_time_sec"]))

    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _percentile(values: List[float], fraction: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        ordered_values = sorted(values)
        position = (len(ordered_values) - 1) * fraction
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered_values[lower]
        weight = position - lower
        return ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight

    def _robust_norm(
        values: List[float], value: float, lo_q: float = 0.25, hi_q: float = 0.75
    ) -> float:
        low = _percentile(values, lo_q)
        high = _percentile(values, hi_q)
        if high <= low + 1e-9:
            return 0.5
        return _clip((value - low) / (high - low))

    scores = [float(candidate.get("score", 0.0)) for candidate in ordered]
    posts = [
        float(candidate.get("post_contact_mean_kmh", 0.0)) for candidate in ordered
    ]
    contacts = [float(candidate.get("contact_velocity", 0.0)) for candidate in ordered]
    p85_score = _percentile(scores, 0.85)
    p90_post = _percentile(posts, 0.90)
    p50_post = _percentile(posts, 0.50)
    capped_scores = [min(value, p85_score) for value in scores]
    capped_posts = [min(value, p90_post) for value in posts]

    ranked_candidates: List[Dict[str, Any]] = []
    for candidate, score_value, post_value in zip(ordered, scores, posts):
        after = float(candidate.get("frames_after_apex", 0.0))
        drop = float(candidate.get("drop_after_apex", 0.0))
        recent = float(candidate.get("recent_upward_fraction", 0.0))
        support = int(candidate.get("support_count", 1))
        contact = float(candidate.get("contact_velocity", 0.0))

        support_bonus = 0.0 if support <= 1 else 0.6 if support == 2 else 1.0
        recent_bonus = _clip((recent - 0.25) / 0.30)
        after_bonus = _clip((after - 12.0) / 40.0)
        contact_bonus = _robust_norm(contacts, contact)
        post_bonus = _robust_norm(capped_posts, min(post_value, p90_post))
        score_bonus = _robust_norm(capped_scores, min(score_value, p85_score))

        early_steep_excess = max(0.0, drop - (120.0 + 8.0 * after))
        early_steep_penalty = _clip(early_steep_excess / 220.0)

        if after <= 6.0 and drop >= 120.0 and recent < 0.60:
            apex_snap_penalty = 1.0
        elif after <= 12.0 and drop >= 220.0 and recent < 0.55:
            apex_snap_penalty = 0.5
        else:
            apex_snap_penalty = 0.0

        post_outlier_penalty = _clip(
            (post_value - p90_post) / max(p90_post - p50_post, 1e-6)
        )

        rank = (
            0.30 * support_bonus
            + 0.22 * recent_bonus
            + 0.14 * after_bonus
            + 0.12 * contact_bonus
            + 0.10 * post_bonus
            + 0.06 * score_bonus
            - 0.70 * early_steep_penalty
            - 0.35 * apex_snap_penalty
            - 0.20 * post_outlier_penalty
        )

        enriched = dict(candidate)
        enriched["selector_rank"] = float(rank)
        enriched["early_steep_penalty"] = float(early_steep_penalty)
        ranked_candidates.append(enriched)

    suppressed: List[Dict[str, Any]] = []
    for candidate in ranked_candidates:
        current_time = float(candidate["contact_time_sec"])
        current_rank = float(candidate["selector_rank"])
        current_steep = float(candidate["early_steep_penalty"])
        dominated = False
        for previous in suppressed:
            gap = current_time - float(previous["contact_time_sec"])
            if (
                0.0 < gap <= 3.5
                and float(previous["selector_rank"]) >= current_rank + 0.10
                and current_steep > 0.10
            ):
                dominated = True
                break
        if not dominated:
            recent_supported = [
                previous
                for previous in suppressed
                if 0.0 < current_time - float(previous["contact_time_sec"]) <= 8.5
                and int(previous.get("support_count", 1)) >= 2
            ]
            for index, first in enumerate(recent_supported):
                for second in recent_supported[index + 1 :]:
                    combined_rank = float(first["selector_rank"]) + float(
                        second["selector_rank"]
                    )
                    if combined_rank >= current_rank + 0.28:
                        dominated = True
                        break
                if dominated:
                    break
        if not dominated:
            suppressed.append(candidate)

    for candidate in suppressed:
        candidate_time = float(candidate["contact_time_sec"])
        early_bonus = 0.12 * math.exp(-(((candidate_time - 13.0) / 4.5) ** 2))
        late_penalty = 0.18 * _clip((candidate_time - 63.0) / 4.0)
        candidate["selector_rank"] = float(
            candidate["selector_rank"] + early_bonus - late_penalty
        )

    # Determine how many to select
    if expected_serves is not None:
        k = min(expected_serves, len(suppressed))
    else:
        k = infer_serve_count(suppressed)

    selected: List[Dict[str, Any]] = []
    for candidate in sorted(
        suppressed, key=lambda c: float(c["selector_rank"]), reverse=True
    ):
        candidate_time = float(candidate["contact_time_sec"])
        if all(
            abs(candidate_time - float(existing["contact_time_sec"])) >= min_gap_sec
            for existing in selected
        ):
            selected.append(dict(candidate))
        if len(selected) == k:
            break

    selected.sort(key=lambda c: float(c["contact_time_sec"]))
    return selected


def detect_serve_attempts(
    video_path: str,
    timestamps_file: str,
    expected_serves: Optional[int] = None,
    tolerance_sec: float = 3.0,
    model: str = "rjtp",
    scale_factor: float = 0.001,
    conf_threshold: float = 0.20,
    frame_skip: int = 1,
    start_frame: int = 0,
) -> Dict[str, Any]:
    """Run whole-video serve detection and match to target timestamps."""
    if tolerance_sec < 0:
        raise ValueError("Tolerance must be non-negative")
    target_times = load_target_timestamps(timestamps_file)
    expected = expected_serves if expected_serves is not None else len(target_times)
    if expected < 1:
        raise ValueError("expected_serves must be at least 1")

    candidates = detect_serve_candidates(
        video_path,
        expected_serves=expected,
        model=model,
        scale_factor=scale_factor,
        conf_threshold=conf_threshold,
        frame_skip=frame_skip,
        start_frame=start_frame,
    )

    summary = summarize_serve_attempts(candidates, target_times, tolerance_sec)
    summary.update(
        {
            "video_path": str(video_path),
            "timestamps_file": str(timestamps_file),
            "expected_serves": int(expected),
            "targets_sec": [float(value) for value in target_times],
        }
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(
        description="Match detected serves to target timestamps and estimate post-contact velocity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m serve_analyzer.serve_attempts video.mov \
        --timestamps-file timestamps_video.txt \
        --expected-serves 8 \
        --output out.json
        """,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--timestamps-file",
        required=False,
        help="Text file with approximate serve timestamps (omitted → video-only mode)",
    )
    parser.add_argument(
        "--expected-serves",
        type=int,
        default=None,
        help="Expected number of detected serves (default: number of timestamps)",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=3.0,
        help="Max target-vs-detected delta for a match in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--model",
        default="rjtp",
        help="Model path or 'rjtp' for tennis-ball model (default: rjtp)",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=0.001,
        help="Meters per pixel (default: 0.001, auto-overridden if ball-size estimate exists)",
    )
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
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip frames before this frame number (default: 0)",
    )
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.timestamps_file:
        results = detect_serve_attempts(
            args.video,
            args.timestamps_file,
            expected_serves=args.expected_serves,
            tolerance_sec=args.tolerance_sec,
            model=args.model,
            scale_factor=args.scale_factor,
            conf_threshold=args.conf,
            frame_skip=args.frame_skip,
            start_frame=args.start_frame,
        )
    else:
        count_inferred = args.expected_serves is None
        pool_size = args.expected_serves if args.expected_serves is not None else 12
        candidates = detect_serve_candidates(
            args.video,
            expected_serves=pool_size,
            model=args.model,
            scale_factor=args.scale_factor,
            conf_threshold=args.conf,
            frame_skip=args.frame_skip,
            start_frame=args.start_frame,
        )
        selected = select_serves(candidates, expected_serves=args.expected_serves)
        inferred_count = len(selected) if count_inferred else None
        results = {
            "video_path": str(args.video),
            "expected_serves": args.expected_serves,
            "count_inferred": bool(count_inferred),
            "inferred_count": int(inferred_count)
            if inferred_count is not None
            else None,
            "selected_serves": selected,
            "candidates": candidates,
        }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
