#!/usr/bin/env python3
"""Third-generation detector-only serve attempt refinement.

v3 = v2 (gating + motion + history) with four upgrades:

  1. Savitzky-Golay smoothing replaces moving-average for ball-history speeds
     (preserves single-frame velocity peaks).
  2. 4-state Kalman filter replaces the v2 fixed-jump continuity gate; handles
     gaps via prediction with chi^2 outlier gating.
  3. Top-K mean peak velocity reported alongside max as `peak_kmh` (robust
     to single-frame tracking spikes).
  4. Optional audio-onset cross-check: candidates whose contact_time matches
     a 2-5 kHz audio transient within `audio_match_tolerance_sec` get a score
     bonus and (if --snap-to-audio) snap to the matched audio time.

This is a STANDALONE reimplementation (not importing from serve_attempts_v2)
per the v3 design decision in .docs/plans/. Public surface mirrors v2 with
`detect_serve_candidates_v3` and `select_serves_v3`. Output JSON is a
superset of v2 with extra `v3_*` fields.

Module boundaries (per serve_analyzer/AGENTS.md):
  * Detector-only - no timestamp parsing or evaluator-side fields.
  * Reuses serve_attempts.detect_serve_candidates / select_serves for the
    candidate POOL only (we trust the v2 finding that the pool already
    contains the real serves; v3 improves selection/refinement quality).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from serve_analyzer.analysis import savgol_smooth, top_k_mean
from serve_analyzer.audio_contact import detect_onsets, nearest_onset
from serve_analyzer.kalman import smooth_track
from serve_analyzer.serve_attempts import detect_serve_candidates, select_serves


Position = Optional[Tuple[float, float]]


# -- JSON helpers ---------------------------------------------------------


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


# -- Video helpers --------------------------------------------------------


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
    """Frame-difference body/racket proxy cues around candidates."""
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


# -- Scoring helpers ------------------------------------------------------


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


def _ball_history_v3(
    positions: Sequence[Position],
    history_frames: int,
    sg_window: int = 7,
    sg_polyorder: int = 3,
) -> Tuple[List[float], List[float]]:
    """Compute per-frame ball speed and excess-acceleration with SG smoothing.

    v2 used a raw frame-to-frame hypot speed and a median-baseline accel
    proxy (which acts like a moving-average and attenuates peaks). v3 keeps
    the same shape but smooths the speed series with Savitzky-Golay before
    computing the accel proxy, so a real impact spike survives smoothing.
    """
    raw_speeds = [0.0] * len(positions)
    for frame in range(1, len(positions)):
        current = positions[frame]
        previous = positions[frame - 1]
        if current is not None and previous is not None:
            raw_speeds[frame] = math.hypot(
                current[0] - previous[0], current[1] - previous[1]
            )
    smoothed = savgol_smooth(
        raw_speeds, window_length=sg_window, polyorder=sg_polyorder
    )
    speeds: List[float] = [float(v) for v in smoothed]

    accels: List[float] = [0.0] * len(speeds)
    for frame in range(1, len(speeds)):
        baseline = float(np.median(_window(speeds, frame, max(1, history_frames // 2))))
        accels[frame] = max(0.0, speeds[frame] - baseline)
    return speeds, accels


def _metadata_lag_adjustment(candidate: Dict[str, Any], fps: float) -> int:
    """Estimate contact lag from historical trajectory fields when raw track is absent.

    Identical to v2's heuristic: kept here so v3 stays self-contained.
    """
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


# -- Refinement core -----------------------------------------------------


def refine_candidate_contacts_v3(
    candidates: Sequence[Dict[str, Any]],
    fps: float,
    positions: Sequence[Position],
    motion_cues: Dict[int, Dict[str, float]],
    search_radius_frames: int,
    history_frames: int,
    audio_onsets: Sequence[float] = (),
    audio_match_tolerance_sec: float = 0.25,
    audio_score_bonus: float = 0.50,
    snap_to_audio: bool = False,
    sg_window: int = 7,
    sg_polyorder: int = 3,
) -> List[Dict[str, Any]]:
    """Refine candidate contact frames using SG-smoothed ball history,
    motion cues, and optional audio onsets.

    The score is the v2-style weighted blend of normalized speed/accel/motion
    minus a distance penalty, plus an additional audio_score_bonus when an
    audio onset lies within `audio_match_tolerance_sec` of the candidate frame.
    """
    if not candidates:
        return []

    has_track = any(point is not None for point in positions)

    if not has_track:
        # No raw track - fall back to v2's metadata-lag heuristic.
        refined_without_track: List[Dict[str, Any]] = []
        for candidate in candidates:
            original = int(candidate["contact_frame"])
            adjustment = _metadata_lag_adjustment(candidate, fps)
            adjusted = max(0, original + adjustment)
            adjusted_time = float(adjusted / fps)

            audio_match = (
                nearest_onset(
                    list(audio_onsets), adjusted_time, audio_match_tolerance_sec
                )
                if audio_onsets
                else None
            )

            if snap_to_audio and audio_match is not None:
                snapped_frame = int(round(audio_match * fps))
                final_frame = snapped_frame
                final_time = float(snapped_frame / fps)
            else:
                final_frame = adjusted
                final_time = adjusted_time

            enriched = dict(candidate)
            enriched["v3_original_contact_frame"] = int(original)
            enriched["v3_original_contact_time_sec"] = float(original / fps)
            enriched["contact_frame"] = int(final_frame)
            enriched["contact_time_sec"] = float(final_time)
            enriched["v3_refined_frame_delta"] = int(final_frame - original)
            enriched["v3_ball_speed_score"] = 0.0
            enriched["v3_ball_accel_score"] = 0.0
            enriched["v3_motion_score"] = 0.0
            enriched["v3_motion_burst_score"] = 0.0
            enriched["v3_audio_match_time_sec"] = (
                float(audio_match) if audio_match is not None else None
            )
            enriched["v3_audio_match_delta_sec"] = (
                float(audio_match - adjusted_time) if audio_match is not None else None
            )
            enriched["v3_audio_score_bonus"] = (
                float(audio_score_bonus) if audio_match is not None else 0.0
            )
            enriched["v3_contact_score"] = (1.0 if adjustment else 0.0) + (
                audio_score_bonus if audio_match is not None else 0.0
            )
            enriched["v3_snapped_to_audio"] = bool(
                snap_to_audio and audio_match is not None
            )
            refined_without_track.append(enriched)
        refined_without_track.sort(key=lambda item: float(item["contact_time_sec"]))
        return refined_without_track

    speeds, accels = _ball_history_v3(
        positions, history_frames, sg_window=sg_window, sg_polyorder=sg_polyorder
    )
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
            nearest_motion = (
                min(motion_frames, key=lambda item: abs(item - frame))
                if motion_frames
                else None
            )
            motion = (
                motion_cues.get(nearest_motion, {})
                if nearest_motion is not None
                else {}
            )
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
                    "v3_ball_speed_score": float(speed_score),
                    "v3_ball_accel_score": float(accel_score),
                    "v3_motion_score": float(motion_score),
                    "v3_motion_burst_score": float(burst_score),
                }

        best_time = float(best_frame / fps)
        audio_match = (
            nearest_onset(list(audio_onsets), best_time, audio_match_tolerance_sec)
            if audio_onsets
            else None
        )

        bonus = audio_score_bonus if audio_match is not None else 0.0
        if snap_to_audio and audio_match is not None:
            snapped_frame = int(round(audio_match * fps))
            final_frame = snapped_frame
            final_time = float(snapped_frame / fps)
        else:
            final_frame = best_frame
            final_time = best_time

        enriched = dict(candidate)
        enriched["v3_original_contact_frame"] = int(original)
        enriched["v3_original_contact_time_sec"] = float(original / fps)
        enriched["contact_frame"] = int(final_frame)
        enriched["contact_time_sec"] = float(final_time)
        enriched["v3_refined_frame_delta"] = int(final_frame - original)
        enriched.update(best_parts)
        enriched["v3_audio_match_time_sec"] = (
            float(audio_match) if audio_match is not None else None
        )
        enriched["v3_audio_match_delta_sec"] = (
            float(audio_match - best_time) if audio_match is not None else None
        )
        enriched["v3_audio_score_bonus"] = float(bonus)
        enriched["v3_snapped_to_audio"] = bool(
            snap_to_audio and audio_match is not None
        )
        enriched["v3_contact_score"] = float(best_score + bonus)
        refined.append(enriched)
    refined.sort(key=lambda item: float(item["contact_time_sec"]))
    return refined


# -- Peak-velocity recompute ---------------------------------------------


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
    """Augment candidates with `peak_kmh` (top-K mean) using SG-smoothed speeds.

    Mutates each candidate in-place. Requires a non-empty raw track; otherwise
    leaves existing post_contact_* fields untouched and writes peak_kmh=None.
    """
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
            candidate["v3_max_kmh_smoothed"] = float(np.max(slice_kmh))



# -- Public entry point ---------------------------------------------------


def _load_seed(input_detections: str) -> Dict[str, Any]:
    with open(input_detections, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_serve_candidates_v3(
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
    # Kalman parameters (replace v2's max_jump_px / max_missing_frames):
    kalman_sigma_a: float = 50.0,
    kalman_sigma_z: float = 4.0,
    kalman_gate_chi2: float = 9.21,
    max_imputed_run: int = 12,
    # Refinement parameters (carry-over from v2):
    history_sec: float = 0.35,
    search_radius_sec: float = 0.80,
    motion_stride: int = 2,
    # SG parameters:
    sg_window: int = 7,
    sg_polyorder: int = 3,
    # Top-K peak velocity:
    peak_top_k: int = 5,
    post_contact_sec: float = 1.0,
    # Audio:
    use_audio: bool = True,
    audio_match_tolerance_sec: float = 0.25,
    audio_score_bonus: float = 0.50,
    snap_to_audio: bool = False,
) -> Dict[str, Any]:
    """Run v3 detector-only serve refinement and return JSON-safe records.

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

    # Kalman smoothing replaces v2's continuity_gate_positions.
    kalman_result = smooth_track(
        raw_positions,
        sigma_a=kalman_sigma_a,
        sigma_z=kalman_sigma_z,
        gate_chi2=kalman_gate_chi2,
        max_imputed_run=max_imputed_run,
    )
    positions = kalman_result.positions

    radius_frames = max(1, int(search_radius_sec * fps))
    history_frames = max(1, int(history_sec * fps))
    motion_cues = extract_motion_cues(
        video_path,
        [int(candidate["contact_frame"]) for candidate in selected_seed],
        radius_frames=radius_frames,
        stride=motion_stride,
    )

    audio_onsets: List[float] = detect_onsets(video_path) if use_audio else []

    selected = refine_candidate_contacts_v3(
        selected_seed,
        fps,
        positions,
        motion_cues,
        search_radius_frames=radius_frames,
        history_frames=history_frames,
        audio_onsets=audio_onsets,
        audio_match_tolerance_sec=audio_match_tolerance_sec,
        audio_score_bonus=audio_score_bonus,
        snap_to_audio=snap_to_audio,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )

    _recompute_peak_velocities(
        selected,
        positions,
        fps,
        scale_factor=scale_factor,
        post_contact_sec=post_contact_sec,
        top_k=peak_top_k,
        sg_window=sg_window,
        sg_polyorder=sg_polyorder,
    )

    audio_match_count = int(
        sum(1 for c in selected if c.get("v3_audio_match_time_sec") is not None)
    )

    return {
        "video_path": str(video_path),
        "detector": "v3",
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
        "v3_parameters": {
            "kalman_sigma_a": float(kalman_sigma_a),
            "kalman_sigma_z": float(kalman_sigma_z),
            "kalman_gate_chi2": float(kalman_gate_chi2),
            "max_imputed_run": int(max_imputed_run),
            "history_sec": float(history_sec),
            "search_radius_sec": float(search_radius_sec),
            "motion_stride": int(motion_stride),
            "sg_window": int(sg_window),
            "sg_polyorder": int(sg_polyorder),
            "peak_top_k": int(peak_top_k),
            "post_contact_sec": float(post_contact_sec),
            "use_audio": bool(use_audio),
            "audio_match_tolerance_sec": float(audio_match_tolerance_sec),
            "audio_score_bonus": float(audio_score_bonus),
            "snap_to_audio": bool(snap_to_audio),
        },
        "v3_kalman_stats": {
            "input_detections": int(sum(p is not None for p in raw_positions)),
            "kalman_accepted": int(kalman_result.accepted),
            "kalman_rejected_jumps": int(kalman_result.rejected_jumps),
            "kalman_imputed": int(kalman_result.imputed_count),
        },
        "v3_motion_cue_count": int(len(motion_cues)),
        "v3_audio_onset_count": int(len(audio_onsets)),
        "v3_audio_matched_serves": audio_match_count,
    }


def select_serves_v3(
    candidates: Sequence[Dict[str, Any]],
    expected_serves: Optional[int] = None,
    min_gap_sec: float = 2.0,
) -> List[Dict[str, Any]]:
    """Select non-overlapping v3-refined candidates without evaluator fields."""
    if not candidates:
        return []
    if expected_serves is not None and expected_serves <= 0:
        return []

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.get("v3_contact_score", 0.0)),
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


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for v3 detector-only serve detection."""
    parser = argparse.ArgumentParser(
        description="Run v3 detector-only serve refinement (SG + Kalman + Top-K + Audio)"
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
    # Kalman
    parser.add_argument("--kalman-sigma-a", type=float, default=50.0)
    parser.add_argument("--kalman-sigma-z", type=float, default=4.0)
    parser.add_argument("--kalman-gate-chi2", type=float, default=9.21)
    parser.add_argument("--max-imputed-run", type=int, default=12)
    # Refinement
    parser.add_argument("--history-sec", type=float, default=0.35)
    parser.add_argument("--search-radius-sec", type=float, default=0.80)
    parser.add_argument("--motion-stride", type=int, default=2)
    # SG
    parser.add_argument("--sg-window", type=int, default=7)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    # Top-K peak
    parser.add_argument("--peak-top-k", type=int, default=5)
    parser.add_argument("--post-contact-sec", type=float, default=1.0)
    # Audio
    parser.add_argument(
        "--no-audio",
        dest="use_audio",
        action="store_false",
        default=True,
        help="Disable audio onset cross-check",
    )
    parser.add_argument("--audio-tolerance-sec", type=float, default=0.25)
    parser.add_argument("--audio-bonus", type=float, default=0.50)
    parser.add_argument(
        "--snap-to-audio",
        action="store_true",
        default=False,
        help="Snap contact_frame to nearest audio onset within tolerance",
    )
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
    if args.peak_top_k < 1:
        parser.error("peak-top-k must be at least 1")
    results = detect_serve_candidates_v3(
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
        kalman_sigma_a=args.kalman_sigma_a,
        kalman_sigma_z=args.kalman_sigma_z,
        kalman_gate_chi2=args.kalman_gate_chi2,
        max_imputed_run=args.max_imputed_run,
        history_sec=args.history_sec,
        search_radius_sec=args.search_radius_sec,
        motion_stride=args.motion_stride,
        sg_window=args.sg_window,
        sg_polyorder=args.sg_polyorder,
        peak_top_k=args.peak_top_k,
        post_contact_sec=args.post_contact_sec,
        use_audio=args.use_audio,
        audio_match_tolerance_sec=args.audio_tolerance_sec,
        audio_score_bonus=args.audio_bonus,
        snap_to_audio=args.snap_to_audio,
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
