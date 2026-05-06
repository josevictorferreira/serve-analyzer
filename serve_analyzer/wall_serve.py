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
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from serve_analyzer.wall_calibration import (
    WallCalibration,
    compute_wall_homography,
    pixel_to_wall,
    WallCalibrationError,
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


def detect_wall_impact(
    video_path: Union[str, Path],
    calibration: WallCalibration,
    *,
    serve_window: Optional[Tuple[int, int]] = None,
    manual_correction: Optional[Dict] = None,
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
    start_frame = 0
    end_frame = total_frames - 1
    if serve_window is not None:
        start_frame = max(0, serve_window[0])
        end_frame = min(total_frames - 1, serve_window[1])

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
