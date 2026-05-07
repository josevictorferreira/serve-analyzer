"""Autonomous wall-impact detection with manual-correction overlay.

Detects the frame and pixel location where a ball hits a calibrated wall
plane from lateral video.  Uses frame-difference + brightness peak analysis
near the calibrated wall x-coordinate.  A manual-correction override can
replace the final impact values while preserving the autonomous candidate
for comparison.

Public symbols
--------------
- :class:`WallImpactResult` — frozen result dataclass.
- :func:`detect_wall_impact` — main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from serve_analyzer.wall_calibration import (
    WallCalibration,
    compute_wall_homography,
    pixel_to_wall,
    WallCalibrationError,
)
from serve_analyzer.wall_outputs import (
    assemble_wall_analysis_result,
    to_json,
    serve_to_csv_row,
    write_csv,
)


@dataclass(frozen=True)
class WallImpactResult:
    """Result of wall-impact detection for one serve window.

    Attributes
    ----------
    impact_frame
        Final impact frame (equals *manual_correction* value when provided,
        otherwise the autonomous candidate).  ``None`` when insufficient
        track data was collected.
    impact_pixel
        Final impact pixel ``(x, y)`` — same override semantics as
        *impact_frame*.
    autonomous_frame
        Impact frame chosen by the autonomous detector.  ``None`` when
        the track was insufficient or detection failed.
    autonomous_pixel
        Impact pixel ``(x, y)`` chosen by the autonomous detector.
    candidate_track
        Raw ball-position candidates as ``(frame, x_px, y_px)`` tuples.
    warnings
        List of warning-code strings drawn from
        :data:`WARNING_CODES`.
    confidence
        Dict with detection-confidence metadata.
    """

    impact_frame: Optional[int]
    impact_pixel: Optional[Tuple[float, float]]
    autonomous_frame: Optional[int]
    autonomous_pixel: Optional[Tuple[float, float]]
    candidate_track: List[Tuple[int, float, float]]
    warnings: List[str]
    confidence: Dict


@dataclass(frozen=True)
class PreWallSpeedResult:
    """Result of pre-wall speed estimation for one serve.

    Attributes
    ----------
    speed_m_s
        Estimated pre-impact speed magnitude in m/s, or ``None`` when
        refused (e.g. insufficient track).
    speed_km_h
        Estimated speed in km/h, or ``None`` when refused.
    speed_mph
        Estimated speed in mph, or ``None`` when refused.
    uncertainty_m_s
        Combined uncertainty in m/s (±1 frame, homography residuals,
        degraded intrinsics).  ``0.0`` when refused.
    samples_used
        Number of clean pre-impact positions used in the estimate.
    warnings
        List of warning-code strings.
    metadata
        Extra data for downstream consumers (e.g. velocity vector).
    """

    speed_m_s: Optional[float]
    speed_km_h: Optional[float]
    speed_mph: Optional[float]
    uncertainty_m_s: float
    samples_used: int
    warnings: List[str]
    metadata: Dict


@dataclass(frozen=True)
class CourtProjectionResult:
    """Regulation-court landing projection from a pre-wall speed estimate.

    Interprets the projected landing as the equivalent no-wall ball landing
    on a real tennis court (not wall rebound).  Gravity-only; no spin or drag.

    Coordinate conventions
    --------------------
    Court frame:
        - Origin at the net centerline.
        - ``landing_x_m``: lateral offset from centerline.  Positive = ad side
          (camera-left if camera faces the wall from behind the net).
        - ``landing_z_m``: distance from the net along the court length.
          Positive toward the server's baseline (camera side).
          Server baseline at z = COURT_LENGTH_M / 2 = 11.885 m.
        - Net at z = 0.

    Assumptions
    -----------
    - Wall is aligned with the net (court z = 0).
    - The ball at serve contact is at height ``serve_contact_height_m`` and
      horizontal distance ``serve_contact_distance_m`` from the net along z.
    - Monocular assumption: horizontal velocity is projected onto the z-axis
      (wall-perpendicular); lateral velocity vx is taken from the wall-frame
      measurement.

    Attributes
    ----------
    landing_x_m
        Lateral offset from court centerline (m).  Positive = ad side.
        ``None`` when projection refused.
    landing_z_m
        Distance from net along court length (m).  Positive toward baseline.
        ``None`` when projection refused.
    in_service_box
        ``True`` when landing falls inside the regulation service box.
        ``None`` when projection refused.
    service_box_side
        ``"deuce"`` | ``"ad"`` | ``None``.
    assumptions
        Dict of modelling assumptions used.
    uncertainty
        Dict with ``landing_z_sensitivity_m`` and ``landing_x_sensitivity_m``.
    warnings
        List of warning-code strings.
    """

    landing_x_m: Optional[float]
    landing_z_m: Optional[float]
    in_service_box: Optional[bool]
    service_box_side: Optional[str]
    assumptions: Dict
    uncertainty: Dict
    warnings: List[str]


def _find_ball_in_frame(
    gray: np.ndarray,
    wall_x_px: float,
    search_half_width: float = 40.0,
    y_min: int = 0,
    y_max: Optional[int] = None,
    brightness_threshold: float = 200.0,
) -> Optional[Tuple[float, float]]:
    """Locate the brightest blob near *wall_x_px* in a grayscale frame.

    Returns the centroid ``(x, y)`` or ``None`` if no bright-enough region
    is found.
    """
    if y_max is None:
        y_max = gray.shape[0]

    x_lo = max(0, int(wall_x_px - search_half_width))
    x_hi = min(gray.shape[1], int(wall_x_px + search_half_width))

    roi = gray[y_min:y_max, x_lo:x_hi]
    if roi.size == 0:
        return None

    _, thresh = cv2.threshold(roi, int(brightness_threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    moment = cv2.moments(largest)
    if moment["m00"] <= 0:
        return None

    cx = moment["m10"] / moment["m00"] + x_lo
    cy = moment["m01"] / moment["m00"] + y_min
    return (float(cx), float(cy))


@dataclass(frozen=True)
class ImpactWindow:
    """One candidate impact window found by multi-impact scanning.

    Attributes
    ----------
    start_frame
        First frame of the impact episode.
    end_frame
        Last frame of the impact episode.
    candidate_track
        Ball-position candidates as ``(frame, x_px, y_px)`` within this episode.
    brightness_changes
        Brightness deltas as ``(frame, delta)`` within this episode.
    impact_frame
        Best-guess impact frame within this episode.
    impact_x_px
        Impact pixel x coordinate.
    impact_y_px
        Impact pixel y coordinate.
    confidence
        Composite confidence score ``[0, 1]`` for this window.
    """

    start_frame: int
    end_frame: int
    candidate_track: List[Tuple[int, float, float]]
    brightness_changes: List[Tuple[int, float]]
    impact_frame: int
    impact_x_px: float
    impact_y_px: float
    confidence: float


def scan_wall_candidates(
    video_path: Union[str, Path],
    wall_x_px: float,
    *,
    frame_skip: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Dict:
    """Scan a video for ball candidates and wall-band brightness changes.

    Parameters
    ----------
    video_path
        Path to the video file.
    wall_x_px
        Calibrated wall x-coordinate in pixels.
    frame_skip
        Process one frame every *frame_skip* frames.  Skipped frames are
        advanced with ``VideoCapture.grab()`` to avoid unnecessary decoding.

    Returns
    -------
    Dict
        Metadata and scan data with keys ``candidate_track``,
        ``brightness_changes``, ``fps``, ``frame_count``, ``width``, and
        ``height``.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_skip = max(1, int(frame_skip))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    search_half_width = max(60.0, width * 0.15)
    candidate_track: List[Tuple[int, float, float]] = []
    brightness_changes: List[Tuple[int, float]] = []
    prev_gray: Optional[np.ndarray] = None
    effective_end = end_frame + 1 if end_frame is not None else frame_count
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < effective_end:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pos = _find_ball_in_frame(
            gray,
            wall_x_px=wall_x_px,
            search_half_width=search_half_width,
            brightness_threshold=200.0,
        )
        if pos is not None:
            candidate_track.append((frame_idx, pos[0], pos[1]))

        x_lo = max(0, int(wall_x_px - 15))
        x_hi = min(width, int(wall_x_px + 15))
        wall_band = gray[:, x_lo:x_hi]
        mean_brightness = float(np.mean(wall_band)) if wall_band.size > 0 else 0.0
        if prev_gray is not None:
            prev_band = prev_gray[:, x_lo:x_hi]
            prev_brightness = float(np.mean(prev_band)) if prev_band.size > 0 else 0.0
            delta = abs(mean_brightness - prev_brightness)
            brightness_changes.append((frame_idx, delta))

        prev_gray = gray
        for _ in range(frame_skip - 1):
            if not cap.grab():
                break
            frame_idx += 1
        frame_idx += 1

    cap.release()
    return {
        "candidate_track": candidate_track,
        "brightness_changes": brightness_changes,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }


def estimate_local_bounce_score(
    candidate_track: List[Tuple[int, float, float]],
    frame_number: int,
    *,
    frame_radius: int = 5,
) -> float:
    """Estimate whether local x-motion shows a wall bounce.

    Parameters
    ----------
    candidate_track
        Episode track points as ``(frame, x_px, y_px)`` tuples.
    frame_number
        Candidate impact frame around which to compare x-velocity.
    frame_radius
        Number of frames before and after *frame_number* used for the local
        velocity estimate.

    Returns
    -------
    float
        ``1.0`` when x increases before impact and decreases after impact,
        otherwise ``0.0``.
    """
    before = [
        p for p in candidate_track if frame_number - frame_radius <= p[0] < frame_number
    ]
    after = [
        p for p in candidate_track if frame_number < p[0] <= frame_number + frame_radius
    ]
    if len(before) < 2 or len(after) < 2:
        return 0.0

    before_velocity = before[-1][1] - before[0][1]
    after_velocity = after[-1][1] - after[0][1]
    return 1.0 if before_velocity > 0.0 and after_velocity < 0.0 else 0.0


def segment_wall_impacts(
    candidate_track: List[Tuple[int, float, float]],
    brightness_changes: List[Tuple[int, float]],
    fps: float,
    wall_x_px: float,
) -> List[ImpactWindow]:
    """Segment candidate tracks into scored wall-impact windows.

    Parameters
    ----------
    candidate_track
        Full-video ball-position candidates as ``(frame, x_px, y_px)``.
    brightness_changes
        Full-video wall-band brightness deltas as ``(frame, delta)``.
    fps
        Video frame rate used to derive temporal thresholds.
    wall_x_px
        Calibrated wall x-coordinate in pixels.

    Returns
    -------
    List[ImpactWindow]
        Scored candidate impact windows sorted by impact frame.
    """
    fps = fps if fps > 0.0 else 30.0
    max_gap_frames = max(1, round(fps * 0.25))
    min_window_frames = max(2, round(fps * 0.08))
    max_brightness = max((delta for _, delta in brightness_changes), default=0.0)

    def make_window(track: List[Tuple[int, float, float]]) -> Optional[ImpactWindow]:
        if len(track) < min_window_frames:
            return None

        impact_entry = min(track, key=lambda p: abs(p[1] - wall_x_px))
        impact_frame, impact_x_px, impact_y_px = impact_entry
        start_frame = track[0][0]
        end_frame = track[-1][0]
        window_brightness = [
            (frame, delta)
            for frame, delta in brightness_changes
            if start_frame <= frame <= end_frame
        ]
        min_wall_distance = abs(impact_x_px - wall_x_px)
        wall_proximity = max(
            0.0, 1.0 - (min_wall_distance / max(1.0, abs(wall_x_px) * 0.15))
        )
        peak_delta = max((delta for _, delta in window_brightness), default=0.0)
        brightness = peak_delta / max_brightness if max_brightness > 0.0 else 0.0
        bounce = estimate_local_bounce_score(track, impact_frame)
        score = (0.60 * wall_proximity) + (0.30 * brightness) + (0.10 * bounce)
        if score < 0.35:
            return None
        return ImpactWindow(
            start_frame=start_frame,
            end_frame=end_frame,
            candidate_track=track,
            brightness_changes=window_brightness,
            impact_frame=impact_frame,
            impact_x_px=impact_x_px,
            impact_y_px=impact_y_px,
            confidence=float(score),
        )

    windows: List[ImpactWindow] = []
    if candidate_track:
        current: List[Tuple[int, float, float]] = [candidate_track[0]]
        for point in candidate_track[1:]:
            if point[0] - current[-1][0] <= max_gap_frames:
                current.append(point)
            else:
                window = make_window(current)
                if window is not None:
                    windows.append(window)
                current = [point]
        window = make_window(current)
        if window is not None:
            windows.append(window)

    if not windows and brightness_changes:
        from scipy.signal import find_peaks

        frames = [frame for frame, _ in brightness_changes]
        deltas = np.array([delta for _, delta in brightness_changes], dtype=float)
        peak_indices, _ = find_peaks(deltas, distance=max(1, round(fps * 0.4)))
        for peak_idx in peak_indices:
            peak_frame = frames[int(peak_idx)]
            track = [
                point
                for point in candidate_track
                if abs(point[0] - peak_frame) <= max_gap_frames
            ]
            if not track and candidate_track:
                track = [min(candidate_track, key=lambda p: abs(p[0] - peak_frame))]
            window = make_window(track)
            if window is not None:
                windows.append(window)

    return sorted(windows, key=lambda window: window.impact_frame)


def detect_wall_impacts(
    video_path: Union[str, Path],
    calibration: WallCalibration,
    *,
    frame_skip: int = 1,
    trim_start_frame: Optional[int] = None,
    trim_end_frame: Optional[int] = None,
) -> List[WallImpactResult]:
    """Detect multiple wall-impact candidates from a full video.

    Parameters
    ----------
    video_path
        Path to the video file.
    calibration
        A validated :class:`WallCalibration` instance.  Wall x pixel
        coordinate is inferred from the right-most reference point.
    frame_skip
        Process one frame every *frame_skip* frames during scanning.

    Returns
    -------
    List[WallImpactResult]
        One result for each accepted impact window.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    if calibration.wall_reference_points:
        wall_x_px = max(p.pixel[0] for p in calibration.wall_reference_points)
    else:
        wall_x_px = width * 0.75

    scan = scan_wall_candidates(
        video_path, wall_x_px, frame_skip=frame_skip,
        start_frame=trim_start_frame or 0, end_frame=trim_end_frame,
    )
    windows = segment_wall_impacts(
        scan["candidate_track"],
        scan["brightness_changes"],
        scan["fps"],
        wall_x_px,
    )

    results: List[WallImpactResult] = []
    for window in windows:
        impact_pixel = (window.impact_x_px, window.impact_y_px)
        results.append(
            WallImpactResult(
                impact_frame=window.impact_frame,
                impact_pixel=impact_pixel,
                autonomous_frame=window.impact_frame,
                autonomous_pixel=impact_pixel,
                candidate_track=window.candidate_track,
                warnings=[],
                confidence={
                    "track_length": len(window.candidate_track),
                    "method": "multi_impact_segment",
                    "score": window.confidence,
                    "start_frame": window.start_frame,
                    "end_frame": window.end_frame,
                    "brightness_changes": window.brightness_changes,
                },
            )
        )
    return results


def detect_wall_impact(
    video_path: Union[str, Path],
    calibration: WallCalibration,
    *,
    serve_window: Optional[Tuple[int, int]] = None,
    manual_correction: Optional[Dict] = None,
    trim_start_frame: Optional[int] = None,
    trim_end_frame: Optional[int] = None,
) -> WallImpactResult:
    """Detect ball-against-wall impact frame and pixel from video.

    Algorithm (MVP):
      1. Open video; iterate frames inside *serve_window*.
      2. For each frame, locate the brightest blob within a horizontal
         band around the calibrated wall x.
      3. Collect candidates into a track; apply plausibility gates
         (monotonic approach toward wall, minimum motion).
      4. Pick impact frame: the last frame whose blob centre is at or
         past the wall plane, or the frame with the largest brightness
         change (discontinuity).
      5. If *manual_correction* is provided, override the final
        ``impact_frame``/``impact_pixel`` while preserving the
        autonomous candidate.

    Parameters
    ----------
    video_path
        Path to the video file.
    calibration
        A validated :class:`WallCalibration` instance.  Wall x pixel
        coordinate is inferred from the right-most reference point.
    serve_window
        ``(start_frame, end_frame)`` inclusive range to search.  When
        ``None`` the entire video is scanned.
    manual_correction
        Optional dict with keys ``"impact_frame"`` (int) and
        ``"impact_pixel"`` (``(x, y)`` tuple).

    Returns
    -------
    WallImpactResult
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Derive wall_x_px from calibration reference points (right-most pixel x).
    if calibration.wall_reference_points:
        wall_x_px = max(p.pixel[0] for p in calibration.wall_reference_points)
    else:
        # Fallback: 75 % of frame width as heuristic.
        wall_x_px = width * 0.75

    # Frame range.
    start_frame = trim_start_frame if trim_start_frame is not None else 0
    end_frame = trim_end_frame if trim_end_frame is not None else total_frames - 1
    if serve_window is not None:
        start_frame = max(start_frame, serve_window[0])
        end_frame = min(end_frame, serve_window[1])

    # --- Pass 1: collect candidate ball positions near the wall ---
    search_half_width = max(60.0, width * 0.15)
    candidate_track: List[Tuple[int, float, float]] = []
    prev_gray: Optional[np.ndarray] = None
    brightness_changes: List[Tuple[int, float]] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pos = _find_ball_in_frame(
            gray,
            wall_x_px=wall_x_px,
            search_half_width=search_half_width,
            brightness_threshold=200.0,
        )

        if pos is not None:
            candidate_track.append((frame_idx, pos[0], pos[1]))

        # Track brightness near the wall for discontinuity detection.
        x_lo = max(0, int(wall_x_px - 15))
        x_hi = min(width, int(wall_x_px + 15))
        wall_band = gray[:, x_lo:x_hi]
        mean_brightness = float(np.mean(wall_band)) if wall_band.size > 0 else 0.0

        if prev_gray is not None:
            prev_band = prev_gray[:, x_lo:x_hi]
            prev_brightness = float(np.mean(prev_band)) if prev_band.size > 0 else 0.0
            delta = abs(mean_brightness - prev_brightness)
            brightness_changes.append((frame_idx, delta))

        prev_gray = gray

    cap.release()

    # --- Plausibility gating ---
    # Keep only candidates that show monotonic approach toward wall (x increasing).
    gated_track: List[Tuple[int, float, float]] = []
    for entry in candidate_track:
        gated_track.append(entry)
    # Relaxed monotonicity: just require the track has enough entries.

    # --- Insufficient-track check ---
    if len(gated_track) < 3:
        return WallImpactResult(
            impact_frame=None,
            impact_pixel=None,
            autonomous_frame=None,
            autonomous_pixel=None,
            candidate_track=gated_track,
            warnings=["insufficient_track"],
            confidence={"track_length": len(gated_track), "method": "brightness_peak"},
        )

    # --- Pick autonomous impact ---
    # Strategy: find the frame where ball x is closest to wall_x_px.
    # The ball approaches from the left; the impact is where x ≈ wall_x_px.
    autonomous_frame: Optional[int] = None
    autonomous_pixel: Optional[Tuple[float, float]] = None

    best_idx = -1
    best_dist = float("inf")
    for i, (fidx, x_px, y_px) in enumerate(gated_track):
        dist = abs(x_px - wall_x_px)
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx >= 0:
        autonomous_frame = gated_track[best_idx][0]
        autonomous_pixel = (gated_track[best_idx][1], gated_track[best_idx][2])

    # Refine using brightness discontinuity if available.
    # The ball disappearing causes a brightness drop at the wall.
    if brightness_changes:
        # Find the peak brightness change near the autonomous candidate.
        search_radius = 5
        candidates_near = [
            (fidx, delta)
            for fidx, delta in brightness_changes
            if autonomous_frame is not None
            and abs(fidx - autonomous_frame) <= search_radius
        ]
        if candidates_near:
            peak_frame, _ = max(candidates_near, key=lambda t: t[1])
            # Use the peak brightness change frame if it's close.
            # Find the track entry closest to this peak frame.
            for i, (fidx, x_px, y_px) in enumerate(gated_track):
                if fidx == peak_frame:
                    autonomous_frame = fidx
                    autonomous_pixel = (x_px, y_px)
                    break

    # --- Manual correction overlay ---
    warnings: List[str] = []
    confidence: Dict = {
        "track_length": len(gated_track),
        "method": "brightness_peak",
    }

    final_frame = autonomous_frame
    final_pixel = autonomous_pixel

    if manual_correction is not None:
        warnings.append("manual_correction_used")
        if "impact_frame" in manual_correction:
            final_frame = int(manual_correction["impact_frame"])
        if "impact_pixel" in manual_correction:
            px = manual_correction["impact_pixel"]
            final_pixel = (float(px[0]), float(px[1]))

    return WallImpactResult(
        impact_frame=final_frame,
        impact_pixel=final_pixel,
        autonomous_frame=autonomous_frame,
        autonomous_pixel=autonomous_pixel,
        candidate_track=gated_track,
        warnings=warnings,
        confidence=confidence,
    )


def estimate_pre_wall_speed(
    impact_result: WallImpactResult,
    calibration: WallCalibration,
    *,
    fps: float,
    min_samples: int = 4,
) -> PreWallSpeedResult:
    """Estimate pre-impact speed magnitude from the final clean track segment.

    Trims *impact_result.candidate_track* to frames strictly before
    *impact_result.impact_frame*, converts pixel positions to wall-plane
    meters via ``pixel_to_wall``, and computes a finite-difference velocity
    over the last *min_samples* clean positions.  The speed is anchored on
    the last pre-impact point (the point closest to impact).

    Parameters
    ----------
    impact_result
        A :class:`WallImpactResult` with populated ``impact_frame`` and
        ``candidate_track``.
    calibration
        A validated :class:`WallCalibration` with at least 4 wall reference
        points.
    fps
        Video frame rate (frames per second).
    min_samples
        Minimum number of clean pre-impact positions required to report a
        speed (default 4).

    Returns
    -------
    PreWallSpeedResult
        Speed fields are ``None`` when refused.  ``uncertainty_m_s``
        combines ±1 frame impact ambiguity, homography reprojection RMS,
        and a degraded-intrinsics multiplicative factor.
    """
    warnings: List[str] = []
    metadata: Dict = {}

    # --- 1. Trim track to strictly pre-impact frames ---
    impact_frame = impact_result.impact_frame
    if impact_frame is None:
        warnings.append("insufficient_track")
        return PreWallSpeedResult(
            speed_m_s=None,
            speed_km_h=None,
            speed_mph=None,
            uncertainty_m_s=0.0,
            samples_used=0,
            warnings=warnings,
            metadata=metadata,
        )

    pre_track = [
        (f, x, y) for f, x, y in impact_result.candidate_track if f < impact_frame
    ]

    if len(pre_track) < min_samples:
        warnings.append("insufficient_track")
        return PreWallSpeedResult(
            speed_m_s=None,
            speed_km_h=None,
            speed_mph=None,
            uncertainty_m_s=0.0,
            samples_used=len(pre_track),
            warnings=warnings,
            metadata=metadata,
        )

    # --- 2. Compute wall-plane homography ---
    image_points = np.array(
        [p.pixel for p in calibration.wall_reference_points], dtype=np.float64
    )
    world_points = np.array(
        [p.wall_m for p in calibration.wall_reference_points], dtype=np.float64
    )

    try:
        H, residuals = compute_wall_homography(
            image_points, world_points, intrinsics=calibration.intrinsics
        )
    except WallCalibrationError:
        warnings.append("insufficient_track")
        return PreWallSpeedResult(
            speed_m_s=None,
            speed_km_h=None,
            speed_mph=None,
            uncertainty_m_s=0.0,
            samples_used=len(pre_track),
            warnings=warnings,
            metadata=metadata,
        )

    H_inv = np.linalg.inv(H)

    # --- 3. Convert pre-impact pixels to wall meters ---
    pixels = np.array([[x, y] for _, x, y in pre_track], dtype=np.float64)
    wall_m = pixel_to_wall(H_inv, pixels)

    # --- 4. Finite-difference velocity (central differences, fallback forward) ---
    # Use the final *min_samples* points for the clean segment.
    segment = wall_m[-min_samples:]
    frames = np.array([f for f, _, _ in pre_track[-min_samples:]], dtype=np.float64)

    if len(segment) < 2:
        warnings.append("insufficient_track")
        return PreWallSpeedResult(
            speed_m_s=None,
            speed_km_h=None,
            speed_mph=None,
            uncertainty_m_s=0.0,
            samples_used=len(pre_track),
            warnings=warnings,
            metadata=metadata,
        )

    dt = 1.0 / fps

    # Central differences for interior points, forward/backward for edges.
    velocities = np.zeros_like(segment)
    for i in range(len(segment)):
        if i == 0:
            velocities[i] = (segment[i + 1] - segment[i]) / dt
        elif i == len(segment) - 1:
            velocities[i] = (segment[i] - segment[i - 1]) / dt
        else:
            velocities[i] = (segment[i + 1] - segment[i - 1]) / (2.0 * dt)

    # Speed magnitude at the last pre-impact point (anchor).
    vx, vy = velocities[-1]
    speed_m_s = float(np.sqrt(vx**2 + vy**2))

    # --- 5. Unit conversions ---
    speed_km_h = speed_m_s * 3.6
    speed_mph = speed_m_s * 2.2369362920544

    # --- 6. Uncertainty budget ---
    # a) ±1 frame impact ambiguity: recompute speed shifting anchor by ±1 frame.
    speeds_shifted: List[float] = []
    for shift in (-1, 1):
        shifted_idx = len(wall_m) + shift
        if 1 <= shifted_idx <= len(wall_m):
            shifted_segment = wall_m[max(0, shifted_idx - min_samples) : shifted_idx]
            if len(shifted_segment) >= 2:
                # Recompute velocity at the new anchor.
                if len(shifted_segment) >= 3:
                    v_shifted = (shifted_segment[-1] - shifted_segment[-3]) / (2.0 * dt)
                else:
                    v_shifted = (shifted_segment[-1] - shifted_segment[-2]) / dt
                speeds_shifted.append(
                    float(np.sqrt(v_shifted[0] ** 2 + v_shifted[1] ** 2))
                )

    if len(speeds_shifted) >= 2:
        ambiguity = (max(speeds_shifted) - min(speeds_shifted)) / 2.0
    elif speeds_shifted:
        ambiguity = abs(speeds_shifted[0] - speed_m_s)
    else:
        ambiguity = 0.0

    # b) Homography residual scaling.
    rms_px = residuals.get("reprojection_rms_px", 0.0)
    # Convert pixel RMS to speed uncertainty: RMS_px * scale_factor / dt.
    # Approximate scale factor from the homography (meters per pixel near the wall).
    # Use the mean of diagonal elements of the linear part as a rough scale.
    scale_factor = float(np.mean([H[0, 0], H[1, 1]]))
    if scale_factor <= 0:
        scale_factor = 1.0
    residual_speed_uncertainty = (rms_px * scale_factor) / dt

    # c) Degraded intrinsics factor.
    degraded_factor = 1.0
    if (
        calibration.intrinsics is not None
        and calibration.intrinsics.source == "approx_exif"
    ):
        degraded_factor = 1.5
        warnings.append("degraded_intrinsics")

    # Combine in quadrature, then apply degraded factor multiplicatively.
    uncertainty_m_s = (
        np.sqrt(ambiguity**2 + residual_speed_uncertainty**2) * degraded_factor
    )

    # --- 7. Metadata for downstream consumers ---
    metadata["velocity_vector_wall_m_s"] = (float(vx), float(vy))
    metadata["homography_residuals"] = residuals
    metadata["scale_factor_approx"] = scale_factor

    return PreWallSpeedResult(
        speed_m_s=speed_m_s,
        speed_km_h=speed_km_h,
        speed_mph=speed_mph,
        uncertainty_m_s=float(uncertainty_m_s),
        samples_used=len(pre_track),
        warnings=warnings,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Court projection
# ---------------------------------------------------------------------------


def _compute_landing(
    h0: float,
    vz0: float,
    vx0: float,
    vy0: float,
    gravity_m_s2: float,
) -> Tuple[float, float]:
    """Solve gravity-only projectile for time-of-flight and return (x, z).

    y(t) = h0 + vy0*t - 0.5*g*t^2 = 0
    t* = (vy0 + sqrt(vy0^2 + 2*g*h0)) / g
    x(t*) = vx0 * t*
    z(t*) = vz0 * t*
    """
    discriminant = vy0**2 + 2.0 * gravity_m_s2 * h0
    t_flight = (vy0 + float(np.sqrt(max(0.0, discriminant)))) / gravity_m_s2
    return vx0 * t_flight, vz0 * t_flight


def _classify_service_box(
    landing_x_m: float, landing_z_m: float
) -> Tuple[bool, Optional[str]]:
    """Classify landing relative to the regulation service box.

    Service box bounds (server's side):
        z ∈ [0, SERVICE_BOX_DEPTH_M]  (net → service line)
        |x| ≤ SERVICE_BOX_WIDTH_M
    """
    from serve_analyzer.wall_calibration import (
        SERVICE_BOX_DEPTH_M,
        SERVICE_BOX_WIDTH_M,
    )

    in_box = (
        0.0 <= landing_z_m <= SERVICE_BOX_DEPTH_M
        and abs(landing_x_m) <= SERVICE_BOX_WIDTH_M
    )
    side: Optional[str] = None
    if in_box:
        side = "ad" if landing_x_m > 0 else "deuce"
    return in_box, side


def project_to_court(
    speed_result: PreWallSpeedResult,
    calibration: WallCalibration,
    *,
    gravity_m_s2: float = 9.81,
) -> CourtProjectionResult:
    """Project a serve onto a regulation tennis court (gravity-only, no wall).

    Uses the inferred pre-wall speed and velocity vector to compute the
    equivalent no-wall landing position on a regulation court.  Refuses
    projection (returns all-``None`` geometry + ``"projection_refused"`` warning)
    when speed is unavailable or calibration is incomplete.

    Parameters
    ----------
    speed_result
        A :class:`PreWallSpeedResult` from :func:`estimate_pre_wall_speed`.
    calibration
        A :class:`WallCalibration` with ``serve_contact_height_m`` set.
    gravity_m_s2
        Gravitational acceleration (default 9.81 m/s^2).

    Returns
    -------
    CourtProjectionResult
        Landing coordinates in the court frame, service-box classification,
        modelling assumptions, sensitivity analysis, and warnings.

    Notes
    -----
    **Monocular vz assumption**: With a single lateral camera we cannot
    observe depth velocity directly.  We assume the ball travels primarily
    along the wall-perpendicular (z) axis.  Given the measured speed
    magnitude and the wall-frame (vx, vy), we compute:

    .. code-block:: python

        vz = sqrt(speed^2 - vx^2 - vy^2)

    This vz is the speed toward the wall.  For the no-wall continuation
    the ball would have been *traveling toward* the wall from the server,
    so in the court frame (z positive toward server), the ball at contact
    is moving *in the negative-z direction* (toward net/wall).  We set
    ``vz0 = -vz`` so that ``z(t) = z_contact + vz0 * t`` decreases from
    the contact point toward (and past) the net.

    **Wall = net assumption**: We assume the wall is aligned with the net
    (court z = 0).  The serve contact is at
    ``z_contact = calibration.serve_contact_distance_m`` from the net.
    """
    from serve_analyzer.wall_calibration import (
        COURT_LENGTH_M,
        SERVICE_BOX_DEPTH_M,
        SERVICE_BOX_WIDTH_M,
    )

    warnings: List[str] = []

    # --- Refusal checks ---
    if speed_result.speed_m_s is None:
        return CourtProjectionResult(
            landing_x_m=None,
            landing_z_m=None,
            in_service_box=None,
            service_box_side=None,
            assumptions={"model": "gravity_only", "refused": True},
            uncertainty={
                "landing_z_sensitivity_m": 0.0,
                "landing_x_sensitivity_m": 0.0,
            },
            warnings=["projection_refused"],
        )

    if "insufficient_track" in speed_result.warnings:
        return CourtProjectionResult(
            landing_x_m=None,
            landing_z_m=None,
            in_service_box=None,
            service_box_side=None,
            assumptions={"model": "gravity_only", "refused": True},
            uncertainty={
                "landing_z_sensitivity_m": 0.0,
                "landing_x_sensitivity_m": 0.0,
            },
            warnings=["projection_refused"],
        )

    if calibration.serve_contact_height_m is None:
        return CourtProjectionResult(
            landing_x_m=None,
            landing_z_m=None,
            in_service_box=None,
            service_box_side=None,
            assumptions={"model": "gravity_only", "refused": True},
            uncertainty={
                "landing_z_sensitivity_m": 0.0,
                "landing_x_sensitivity_m": 0.0,
            },
            warnings=["projection_refused"],
        )

    # --- Extract velocity components ---
    vel = speed_result.metadata.get("velocity_vector_wall_m_s", (0.0, 0.0))
    vx_wall = float(vel[0])  # wall-frame lateral velocity
    vy_wall = float(vel[1])  # wall-frame vertical velocity
    speed = float(speed_result.speed_m_s)

    # Monocular vz assumption: project remaining speed onto z-axis
    vz_sq = max(0.0, speed**2 - vx_wall**2 - vy_wall**2)
    vz_toward_wall = float(np.sqrt(vz_sq))

    # Court frame setup
    h0 = float(calibration.serve_contact_height_m)
    z_contact = float(calibration.serve_contact_distance_m)

    # In court frame, ball at contact moves toward net (z decreasing)
    # vz0 is negative (toward wall/net direction)
    vz0 = -vz_toward_wall
    vx0 = vx_wall  # lateral velocity carries over directly
    vy0 = vy_wall  # vertical velocity carries over directly

    # --- Solve trajectory ---
    landing_x, landing_z_rel = _compute_landing(h0, vz0, vx0, vy0, gravity_m_s2)
    landing_z = z_contact + landing_z_rel  # shift by contact position

    # --- Classify service box ---
    in_service_box, service_box_side = _classify_service_box(landing_x, landing_z)

    # --- Sensitivity analysis ---
    # ±10% speed
    z_plus, x_plus = [], []
    for speed_factor in (0.9, 1.1):
        s = speed * speed_factor
        vz_sq_s = max(0.0, s**2 - vx_wall**2 - vy_wall**2)
        vz_s = float(np.sqrt(vz_sq_s))
        lx_s, lz_rel_s = _compute_landing(h0, -vz_s, vx0, vy0, gravity_m_s2)
        z_plus.append(z_contact + lz_rel_s)
        x_plus.append(lx_s)

    # ±0.1 m contact height
    for h_delta in (-0.1, 0.1):
        lx_h, lz_rel_h = _compute_landing(h0 + h_delta, vz0, vx0, vy0, gravity_m_s2)
        z_plus.append(z_contact + lz_rel_h)
        x_plus.append(lx_h)

    landing_z_sensitivity_m = (max(z_plus) - min(z_plus)) / 2.0
    landing_x_sensitivity_m = (max(x_plus) - min(x_plus)) / 2.0

    # --- Build assumptions dict ---
    assumptions: Dict = {
        "model": "gravity_only",
        "contact_height_m": h0,
        "serve_contact_distance_m": z_contact,
        "no_wall_continuation": True,
        "wall_aligned_with_net": True,
        "court_length_m": COURT_LENGTH_M,
        "service_box_depth_m": SERVICE_BOX_DEPTH_M,
        "service_box_width_m": SERVICE_BOX_WIDTH_M,
        "monocular_vz_assumption": (
            "vz = sqrt(speed^2 - vx^2 - vy^2); "
            "horizontal velocity projected onto wall-perpendicular z-axis"
        ),
    }

    uncertainty: Dict = {
        "landing_z_sensitivity_m": landing_z_sensitivity_m,
        "landing_x_sensitivity_m": landing_x_sensitivity_m,
    }

    return CourtProjectionResult(
        landing_x_m=landing_x,
        landing_z_m=landing_z,
        in_service_box=in_service_box,
        service_box_side=service_box_side,
        assumptions=assumptions,
        uncertainty=uncertainty,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse
import glob
import json
import sys
from typing import Sequence


def _error_json(message: str, code: str = "validation_error") -> int:
    """Write structured error to stderr and return exit code 2."""
    sys.stderr.write(json.dumps({"error": message, "code": code}) + "\n")
    return 2


def _build_parser():
    """Build argparse for wall analysis orchestration CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m serve_analyzer.wall_serve",
        description=(
            "Analyze wall-serve videos: detect impact, estimate speed, project to court. "
            "Supports single video or batch glob mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  %(prog)s --video serve_01.MOV --metadata setup.json --output-dir results/

  # Batch mode
  %(prog)s --batch "videos/wall/*.MOV" --metadata setup.json --output-dir results/

  # With per-video override and manual corrections
  %(prog)s --video serve_01.MOV --metadata setup.json --output-dir results/ \\
      --override override.json --manual-corrections corrections.json
        """,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video",
        help="Path to a single video file.",
    )
    source.add_argument(
        "--batch",
        help='Glob pattern for batch processing (e.g. "videos/wall/*.MOV").',
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to JSON setup metadata file (from wall_calibration CLI).",
    )
    parser.add_argument(
        "--override",
        help="Optional per-video override JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-video results and aggregate CSV.",
    )
    parser.add_argument(
        "--manual-corrections",
        help=(
            "Optional JSON mapping serve_index -> "
            '{"pixel_x": int, "pixel_y": int, "impact_frame": int (optional)}'
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        default=False,
        help="Suppress annotated MP4 generation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Suppress plot PNG generation.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video frame rate (frames per second).",
    )

    return parser


def _load_json(path: str) -> dict:
    """Load and parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _get_fps(video_path: str, fps_override: float | None = None) -> float:
    """Return video fps, using override if provided."""
    if fps_override is not None:
        return fps_override
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS ({fps}) for video: {video_path}")
    return float(fps)


def _apply_override(
    calibration: WallCalibration, override: dict | None
) -> WallCalibration:
    """Apply per-video override dict onto a calibration instance."""
    if override is None:
        return calibration
    data = calibration.to_dict()
    vo = override.get("video_override", override)
    if isinstance(vo, dict):
        for key, value in vo.items():
            if key == "wall_reference_points" and isinstance(value, list):
                data["setup"][key] = value
            elif key == "hook_reference" and isinstance(value, dict):
                data["setup"][key] = value
            elif key == "chair_references" and isinstance(value, list):
                data["setup"][key] = value
            else:
                data["setup"][key] = value
    return WallCalibration.from_dict(data)


def _load_manual_corrections(path: str | None) -> dict | None:
    """Load manual corrections JSON mapping serve_index -> correction dict."""
    if path is None:
        return None
    raw = _load_json(path)
    # Normalize to dict keyed by string serve_index
    corrections: dict = {}
    for key, value in raw.items():
        corrections[str(key)] = value
    return corrections


def _process_video(
    video_path: str,
    calibration: WallCalibration,
    output_dir: Path,
    *,
    no_video: bool = False,
    no_plots: bool = False,
    manual_corrections: dict | None = None,
    fps_override: float | None = None,
    trim_start_frame: Optional[int] = None,
    trim_end_frame: Optional[int] = None,
) -> list[dict]:
    """Run full analysis pipeline on one video and write artifacts.

    Returns a list of dicts with per-serve results for aggregation.
    Multiple dicts are returned when multiple wall impacts are detected.
    """
    video_path_obj = Path(video_path)
    video_stem = video_path_obj.stem
    fps = _get_fps(video_path, fps_override)

    # Determine manual correction for serve_index 0 (MVP: one serve per video)
    correction = None
    if manual_corrections is not None:
        correction = manual_corrections.get("0") or manual_corrections.get(0)
        if correction is not None:
            px = correction.get("pixel_x")
            py = correction.get("pixel_y")
            if_f = correction.get("impact_frame")
            if px is not None and py is not None:
                correction = {"impact_pixel": (float(px), float(py))}
                if if_f is not None:
                    correction["impact_frame"] = int(if_f)

    # --- Detection (multi-impact) ---
    impact_results = detect_wall_impacts(
        video_path, calibration,
        trim_start_frame=trim_start_frame, trim_end_frame=trim_end_frame,
    )

    # Fallback: if multi-impact found nothing, try single-impact
    if not impact_results:
        impact_results = [detect_wall_impact(
            video_path, calibration, manual_correction=correction,
            trim_start_frame=trim_start_frame, trim_end_frame=trim_end_frame,
        )]

    # Apply manual correction to primary impact if provided
    if correction is not None and impact_results:
        primary = impact_results[0]
        if "manual_correction_used" not in primary.warnings:
            new_frame = correction.get("impact_frame", primary.impact_frame)
            new_pixel = correction.get("impact_pixel", primary.impact_pixel)
            impact_results[0] = WallImpactResult(
                impact_frame=new_frame,
                impact_pixel=new_pixel,
                autonomous_frame=primary.autonomous_frame,
                autonomous_pixel=primary.autonomous_pixel,
                candidate_track=primary.candidate_track,
                warnings=list(primary.warnings) + ["manual_correction_used"],
                confidence=primary.confidence,
            )

    # --- Process each impact ---
    per_impact: list[dict] = []
    for idx, ir in enumerate(impact_results):
        sr = estimate_pre_wall_speed(ir, calibration, fps=fps, min_samples=4)
        sr.metadata["fps"] = fps
        pr = project_to_court(sr, calibration)
        per_impact.append({
            "impact_index": idx,
            "impact_result": ir,
            "speed_result": sr,
            "projection_result": pr,
        })

    # --- Artifacts (primary impact only for MVP) ---
    primary_data = per_impact[0]
    primary_ir = primary_data["impact_result"]
    primary_sr = primary_data["speed_result"]
    primary_pr = primary_data["projection_result"]

    artifact_paths: dict[str, Any] = {"annotated_video": None, "plots": {}}
    try:
        from serve_analyzer.wall_artifacts import render_annotated_video, render_plots
    except ImportError:
        render_annotated_video = None  # type: ignore[assignment]
        render_plots = None  # type: ignore[assignment]

    if not no_video and render_annotated_video is not None:
        annotated_path = output_dir / f"{video_stem}_annotated.mp4"
        try:
            render_annotated_video(
                video_path, primary_ir, primary_sr,
                primary_pr, calibration, str(annotated_path),
            )
            artifact_paths["annotated_video"] = str(annotated_path)
        except Exception as exc:
            sys.stderr.write(f"Warning: annotated video generation failed: {exc}\n")

    if not no_plots and render_plots is not None:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_paths = render_plots(
                primary_ir, primary_sr, primary_pr,
                calibration, str(plots_dir), video_stem=video_stem,
                per_impact_results=per_impact if len(per_impact) > 1 else None,
            )
            artifact_paths["plots"] = {k: str(v) for k, v in plot_paths.items()}
        except Exception as exc:
            sys.stderr.write(f"Warning: plot generation failed: {exc}\n")

    # --- Assemble WallAnalysisResult ---
    result = assemble_wall_analysis_result(
        video_path, calibration, primary_ir, primary_sr,
        primary_pr, artifact_paths=artifact_paths,
        per_impact_results=per_impact if len(per_impact) > 1 else None,
    )

    # --- Write JSON ---
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(to_json(result), indent=2), encoding="utf-8")

    # --- Write CSV (one row per impact) ---
    serve_rows: list[dict] = []
    for pi in per_impact:
        ir = pi["impact_result"]
        sr = pi["speed_result"]
        pr = pi["projection_result"]
        per_result = assemble_wall_analysis_result(
            video_path, calibration, ir, sr, pr,
        )
        serve_row = {
            "impact_index": pi["impact_index"],
            "video": video_stem,
            "serve_index": result.measured.get("serve_index", 0),
            "impact_time_sec": per_result.measured.get("impact_time_sec"),
            "impact_frame": per_result.measured.get("impact_frame"),
            "wall_x_m": per_result.measured.get("wall_x_m"),
            "wall_y_m": per_result.measured.get("wall_y_m"),
            "speed_m_s": per_result.inferred.get("speed_m_s"),
            "speed_km_h": per_result.inferred.get("speed_km_h"),
            "speed_mph": per_result.inferred.get("speed_mph"),
            "landing_x_m": per_result.inferred.get("landing_x_m"),
            "landing_z_m": per_result.inferred.get("landing_z_m"),
            "in_service_box": per_result.inferred.get("in_service_box"),
            "confidence_score": per_result.confidence.get("aggregate_score"),
            "warnings": per_result.warnings,
        }
        serve_rows.append(serve_row)

    csv_path = output_dir / "result.csv"
    write_csv(csv_path, [serve_to_csv_row(r) for r in serve_rows])

    return serve_rows


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for wall analysis orchestration.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 on success, 2 on validation error, 1 on unexpected error).
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        raise

    # Load metadata
    try:
        metadata = _load_json(args.metadata)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return _error_json(
            f"Failed to load metadata: {exc}", code="metadata_load_error"
        )

    # Load optional override
    override = None
    if args.override:
        try:
            override = _load_json(args.override)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            return _error_json(
                f"Failed to load override: {exc}", code="override_load_error"
            )

    # Load optional manual corrections
    manual_corrections = None
    if args.manual_corrections:
        try:
            manual_corrections = _load_manual_corrections(args.manual_corrections)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            return _error_json(
                f"Failed to load manual corrections: {exc}",
                code="manual_corrections_load_error",
            )

    # Resolve video list
    video_paths: list[str] = []
    if args.video:
        video_paths = [args.video]
    elif args.batch:
        video_paths = glob.glob(args.batch)
        if not video_paths:
            return _error_json(
                f"No videos matched glob pattern: {args.batch}",
                code="batch_no_match",
            )
        video_paths.sort()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for video_path in video_paths:
        try:
            calibration = WallCalibration.from_dict(metadata)
        except Exception as exc:
            return _error_json(
                f"Calibration validation failed for {video_path}: {exc}",
                code="calibration_error",
            )

        if override is not None:
            try:
                calibration = _apply_override(calibration, override)
            except Exception as exc:
                return _error_json(
                    f"Override application failed for {video_path}: {exc}",
                    code="override_error",
                )

        video_output_dir = output_dir / Path(video_path).stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            rows = _process_video(
                video_path,
                calibration,
                video_output_dir,
                no_video=args.no_video,
                no_plots=args.no_plots,
                manual_corrections=manual_corrections,
                fps_override=args.fps,
            )
            all_rows.extend(rows)
        except Exception as exc:
            sys.stderr.write(f"Error processing {video_path}: {exc}\n")
            return 1

    # Aggregate CSV
    if len(all_rows) > 1 or (len(all_rows) == 1 and args.batch):
        aggregate_path = output_dir / "all_serves.csv"
        csv_rows = [serve_to_csv_row(row) for row in all_rows]
        write_csv(aggregate_path, csv_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
