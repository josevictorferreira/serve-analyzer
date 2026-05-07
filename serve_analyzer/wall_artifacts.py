"""Wall-analysis artifact generation: annotated MP4 + plots.

Produces two kinds of output artifacts for a single wall-serve analysis:

1. Annotated MP4 -- ball track, impact points, speed, landing projection,
   and warning codes overlaid on the original video frames.

2. Matplotlib plots -- speed-vs-frame, wall-impact scatter, and court-landing
   projection with service-box boundaries.

Public functions
----------------
- :func:`render_annotated_video` -- generate an annotated MP4.
- :func:`render_plots` -- generate three PNG plots and return their paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from serve_analyzer.wall_calibration import (
    COURT_LENGTH_M,
    SERVICE_BOX_DEPTH_M,
    SERVICE_BOX_WIDTH_M,
    WallCalibration,
    compute_wall_homography,
    pixel_to_wall,
)
from serve_analyzer.wall_outputs import PLOT_FILENAMES
from serve_analyzer.wall_serve import (
    CourtProjectionResult,
    PreWallSpeedResult,
    WallImpactResult,
)

# ---------------------------------------------------------------------------
# Drawing helpers (cv2 overlays)
# ---------------------------------------------------------------------------

_COLOUR_BG = (40, 40, 40)
_COLOUR_TEXT = (255, 255, 255)
_COLOUR_AUTONOMOUS = (0, 165, 255)  # orange (BGR)
_COLOUR_CORRECTED = (255, 0, 0)  # red (BGR)
_COLOUR_TRACK = (0, 255, 255)  # yellow (BGR)


def _draw_info_panel(
    frame: np.ndarray,
    lines: list[str],
    *,
    x: int = 10,
    y: int = 30,
    font_scale: float = 0.5,
    thickness: int = 1,
    line_height: int = 22,
) -> np.ndarray:
    """Draw a semi-transparent text panel on *frame*."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(max(lines, key=len), font, font_scale, thickness)
    panel_h = line_height * len(lines) + 10

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 4, y - 18),
        (x + tw + 8, y + panel_h),
        _COLOUR_BG,
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            font,
            font_scale,
            _COLOUR_TEXT,
            thickness,
            cv2.LINE_AA,
        )
    return frame


def _compute_wall_coords(
    impact_result: WallImpactResult,
    calibration: WallCalibration,
    pixel: Tuple[float, float],
) -> Tuple[float, float] | None:
    """Convert a pixel position to wall meters using calibration homography."""
    try:
        image_pts = np.array(
            [p.pixel for p in calibration.wall_reference_points], dtype=np.float64
        )
        world_pts = np.array(
            [p.wall_m for p in calibration.wall_reference_points], dtype=np.float64
        )
        H, _ = compute_wall_homography(image_pts, world_pts)
        H_inv = np.linalg.inv(H)
        wall_xy = pixel_to_wall(H_inv, np.array([pixel], dtype=np.float64))
        return float(wall_xy[0, 0]), float(wall_xy[0, 1])
    except Exception:
        return None


def _build_info_lines(
    impact_result: WallImpactResult,
    speed_result: PreWallSpeedResult,
    projection_result: CourtProjectionResult,
    calibration: WallCalibration,
) -> list[str]:
    """Build the static text lines for the info panel."""
    lines: list[str] = []

    # Wall coordinates
    if impact_result.impact_pixel is not None:
        wc = _compute_wall_coords(
            impact_result, calibration, impact_result.impact_pixel
        )
        if wc is not None:
            lines.append(f"Wall: ({wc[0]:.2f}, {wc[1]:.2f}) m")
        else:
            lines.append("Wall: N/A")
    else:
        lines.append("Wall: N/A")

    # Speed
    if speed_result.speed_m_s is not None:
        lines.append(
            f"Speed: {speed_result.speed_m_s:.1f} m/s  "
            f"({speed_result.speed_km_h:.1f} km/h)"
        )
    else:
        lines.append("Speed: N/A")

    # Projected landing
    if (
        projection_result.landing_x_m is not None
        and projection_result.landing_z_m is not None
    ):
        box_str = "IN" if projection_result.in_service_box else "OUT"
        lines.append(
            f"Land: ({projection_result.landing_x_m:.2f}, "
            f"{projection_result.landing_z_m:.2f}) m  [{box_str}]"
        )
    else:
        lines.append("Land: Projection refused")

    # Warnings
    all_warnings = sorted(
        set(impact_result.warnings + speed_result.warnings + projection_result.warnings)
    )
    if all_warnings:
        lines.append(f"Warn: {';'.join(all_warnings)}")

    return lines


# ---------------------------------------------------------------------------
# Annotated video
# ---------------------------------------------------------------------------


def render_annotated_video(
    video_path: str | Path,
    impact_result: WallImpactResult,
    speed_result: PreWallSpeedResult,
    projection_result: CourtProjectionResult,
    calibration: WallCalibration,
    output_path: str | Path,
    *,
    fps: float | None = None,
    overwrite: bool = True,
) -> Path:
    """Generate an annotated MP4 with ball track, impact points, and info panel.

    Args:
        video_path: Source video.
        impact_result: Wall impact detection result.
        speed_result: Pre-wall speed estimation result.
        projection_result: Court projection result.
        calibration: Wall calibration data.
        output_path: Destination MP4 path.
        fps: Override FPS (default: read from source video).
        overwrite: If False and *output_path* exists, raise FileExistsError.

    Returns:
        The output Path.

    Raises:
        FileExistsError: If output exists and overwrite=False.
        IOError: If video cannot be read or written.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = fps if fps is not None else float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0:
        video_fps = 60.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"Cannot create output: {output_path}")

    info_lines = _build_info_lines(
        impact_result, speed_result, projection_result, calibration
    )

    auto_frame = impact_result.autonomous_frame
    impact_frame = impact_result.impact_frame
    auto_pixel = impact_result.autonomous_pixel
    impact_pixel = impact_result.impact_pixel

    track_map: Dict[int, Tuple[float, float]] = {
        f: (x, y) for f, x, y in impact_result.candidate_track
    }
    track_frames_sorted = sorted(track_map.keys())

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Track trail up to current frame
        current_trail = [
            (int(x), int(y))
            for f in track_frames_sorted
            if f <= idx
            for x, y in [track_map[f]]
        ]
        if len(current_trail) >= 2:
            pts = np.array(current_trail, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, _COLOUR_TRACK, 1, cv2.LINE_AA)
        for pt in current_trail:
            cv2.circle(frame, pt, 2, _COLOUR_TRACK, cv2.FILLED)

        # Autonomous impact marker
        if auto_frame is not None and auto_pixel is not None and idx == auto_frame:
            cx, cy = int(auto_pixel[0]), int(auto_pixel[1])
            cv2.circle(frame, (cx, cy), 8, _COLOUR_AUTONOMOUS, 2)

        # Corrected/final impact marker (only if differs from autonomous)
        if (
            impact_frame is not None
            and impact_pixel is not None
            and idx == impact_frame
            and (impact_frame != auto_frame or impact_pixel != auto_pixel)
        ):
            ix, iy = int(impact_pixel[0]), int(impact_pixel[1])
            cv2.circle(frame, (ix, iy), 8, _COLOUR_CORRECTED, 2)

        _draw_info_panel(frame, info_lines)
        out.write(frame)

    cap.release()
    out.release()

    # Re-encode to H.264 so Chrome/other browsers can play the MP4.
    # cv2.VideoWriter with 'mp4v' produces MPEG-4 Part 2 which browsers
    # cannot decode. Fall back to the raw file if ffmpeg is unavailable.
    try:
        import os
        import subprocess

        temp_path = str(output_path) + ".raw.mp4"
        os.replace(str(output_path), temp_path)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", temp_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        os.remove(temp_path)
    except Exception:
        # ffmpeg not available or re-encode failed; keep the raw mp4v file
        if os.path.exists(temp_path) and not os.path.exists(str(output_path)):
            os.replace(temp_path, str(output_path))

    return output_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_filename(template: str, video_stem: str, idx: int = 0) -> str:
    """Render a PLOT_FILENAMES template to a concrete filename."""
    return template.format(video_stem=video_stem, idx=idx)


def render_plots(
    impact_result: WallImpactResult,
    speed_result: PreWallSpeedResult,
    projection_result: CourtProjectionResult,
    calibration: WallCalibration,
    output_dir: str | Path,
    *,
    video_stem: str,
    overwrite: bool = True,
    per_impact_results: list[dict] | None = None,
) -> dict[str, Path]:
    """Generate three analysis plots and return their paths.

    Args:
        impact_result: Wall impact detection result.
        speed_result: Pre-wall speed estimation result.
        projection_result: Court projection result.
        calibration: Wall calibration data.
        output_dir: Directory to write PNG files.
        video_stem: Video filename stem (without extension) for naming.
        overwrite: If False and any output exists, raise FileExistsError.
        per_impact_results: Optional list of per-impact dicts (impact_result,
            speed_result, projection_result) for multi-impact plotting.

    Returns:
        Dict with keys ``"speed"``, ``"wall_impact"``, ``"court_landing"``
        mapping to the respective PNG Paths.

    Raises:
        FileExistsError: If any output exists and overwrite=False.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speed_path = output_dir / _plot_filename(PLOT_FILENAMES["speed"], video_stem)
    wall_path = output_dir / _plot_filename(PLOT_FILENAMES["wall_impact"], video_stem)
    court_landing_tpl = PLOT_FILENAMES.get(
        "court_landing", "{video_stem}_serve{idx:02d}_court_landing.png"
    )
    court_path = output_dir / _plot_filename(court_landing_tpl, video_stem)

    if not overwrite:
        for p in (speed_path, wall_path, court_path):
            if p.exists():
                raise FileExistsError(f"Output already exists: {p}")

    _plot_speed(impact_result, speed_result, calibration, str(speed_path))
    _plot_wall_impact(impact_result, calibration, str(wall_path), per_impact_results=per_impact_results)
    _plot_court_landing(projection_result, str(court_path))

    return {
        "speed": speed_path,
        "wall_impact": wall_path,
        "court_landing": court_path,
    }


def _compute_speeds_from_track(
    impact_result: WallImpactResult,
    calibration: WallCalibration,
    fps: float,
) -> tuple[list[int], list[float]]:
    """Compute per-frame speed (m/s) from the candidate track via finite diffs.

    Returns (frames, speeds_m_s) aligned to the track entries.
    """
    track = impact_result.candidate_track
    if len(track) < 2:
        return [t[0] for t in track], [float("nan")] * len(track)

    image_pts = np.array(
        [p.pixel for p in calibration.wall_reference_points], dtype=np.float64
    )
    world_pts = np.array(
        [p.wall_m for p in calibration.wall_reference_points], dtype=np.float64
    )
    H, _ = compute_wall_homography(image_pts, world_pts)
    H_inv = np.linalg.inv(H)

    pixels = np.array([[x, y] for _, x, y in track], dtype=np.float64)
    wall_m = pixel_to_wall(H_inv, pixels)

    frames = [t[0] for t in track]
    speeds: list[float] = []
    dt = 1.0 / fps

    for i in range(len(wall_m)):
        if i == 0:
            dx = wall_m[i + 1] - wall_m[i]
        elif i == len(wall_m) - 1:
            dx = wall_m[i] - wall_m[i - 1]
        else:
            dx = (wall_m[i + 1] - wall_m[i - 1]) / 2.0
        speeds.append(float(np.sqrt(dx[0] ** 2 + dx[1] ** 2)) / dt)

    return frames, speeds


def _plot_speed(
    impact_result: WallImpactResult,
    speed_result: PreWallSpeedResult,
    calibration: WallCalibration,
    output_path: str,
) -> None:
    """Speed vs frame plot with impact frame marked."""
    fps = speed_result.metadata.get("fps", 60.0)

    fig, ax = plt.subplots(figsize=(10, 5))

    track = impact_result.candidate_track
    if track:
        frames, speeds = _compute_speeds_from_track(impact_result, calibration, fps)
        speeds_kmh = [s * 3.6 for s in speeds]
        ax.plot(
            frames, speeds_kmh, marker="o", markersize=3, linewidth=1, color="#2E86AB"
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Speed (km/h)")
        ax.set_title("Pre-Impact Ball Speed Profile")
        ax.grid(True, alpha=0.3)

    if impact_result.impact_frame is not None:
        ax.axvline(
            x=impact_result.impact_frame,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Impact frame",
        )
        ax.legend()

    if speed_result.speed_m_s is not None:
        ax.axhline(
            y=speed_result.speed_km_h,
            color="green",
            linestyle=":",
            alpha=0.6,
            label=f"Est. {speed_result.speed_km_h:.1f} km/h",
        )
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_wall_impact(
    impact_result: WallImpactResult,
    calibration: WallCalibration,
    output_path: str,
    per_impact_results: list[dict] | None = None,
) -> None:
    """Wall-impact scatter with calibration reference points.

    When *per_impact_results* contains more than one entry, all impact
    points are plotted with numbered markers and speed annotations.
    """
    COLORS = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Calibration reference points
    for pt in calibration.wall_reference_points:
        wx, wy = pt.wall_m
        ax.plot(
            wx,
            wy,
            "s",
            color="gray",
            markersize=8,
            label="Ref" if not ax.get_lines() else "",
        )
        ax.annotate(
            pt.name,
            (wx, wy),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="gray",
        )

    # Wall extent rectangle
    all_wx = [p.wall_m[0] for p in calibration.wall_reference_points]
    all_wy = [p.wall_m[1] for p in calibration.wall_reference_points]
    if all_wx and all_wy:
        rect = mpatches.Rectangle(
            (min(all_wx), min(all_wy)),
            max(all_wx) - min(all_wx),
            max(all_wy) - min(all_wy),
            fill=False,
            edgecolor="lightgray",
            linestyle="--",
            linewidth=1,
        )
        ax.add_patch(rect)

    multi = per_impact_results is not None and len(per_impact_results) > 1

    if multi:
        # Plot all impacts
        for i, pi in enumerate(per_impact_results):
            ir = pi["impact_result"]
            sr = pi["speed_result"]
            if ir.impact_pixel is None or ir.impact_frame is None:
                continue
            wc = _compute_wall_coords(ir, calibration, ir.impact_pixel)
            if wc is None:
                continue
            speed_kmh = sr.speed_km_h if sr.speed_km_h is not None else float("nan")
            if i == 0:
                ax.plot(wc[0], wc[1], "r*", markersize=20, label=f"#{1} Impact", zorder=5)
            else:
                color = COLORS[(i - 1) % len(COLORS)]
                ax.plot(wc[0], wc[1], "o", color=color, markersize=12, label=f"#{i + 1} Impact", zorder=5)
            ax.annotate(
                f"#{i + 1}: {speed_kmh:.1f} km/h",
                (wc[0], wc[1]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                fontweight="bold",
                zorder=6,
            )
        title = f"Wall Impact Locations ({len(per_impact_results)} impacts)"
    else:
        # Single impact (backward compat)
        if (
            impact_result.impact_pixel is not None
            and impact_result.impact_frame is not None
        ):
            wc = _compute_wall_coords(
                impact_result, calibration, impact_result.impact_pixel
            )
            if wc is not None:
                ax.plot(wc[0], wc[1], "r*", markersize=20, label="Impact", zorder=5)

        # Autonomous point (if different from final)
        if (
            impact_result.autonomous_pixel is not None
            and impact_result.autonomous_frame is not None
            and (
                impact_result.autonomous_frame != impact_result.impact_frame
                or impact_result.autonomous_pixel != impact_result.impact_pixel
            )
        ):
            wc = _compute_wall_coords(
                impact_result, calibration, impact_result.autonomous_pixel
            )
            if wc is not None:
                ax.plot(
                    wc[0],
                    wc[1],
                    "o",
                    color="orange",
                    markersize=10,
                    label="Autonomous",
                    zorder=4,
                )
        title = "Wall Impact Location"

    ax.set_xlabel("Wall X (m)")
    ax.set_ylabel("Wall Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_court_landing(
    projection_result: CourtProjectionResult,
    output_path: str,
) -> None:
    """Court landing scatter with service-box boundaries.

    If landing coordinates are None (projection refused), renders empty axes
    with an annotation.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if projection_result.landing_x_m is None or projection_result.landing_z_m is None:
        ax.set_xlim(-SERVICE_BOX_WIDTH_M * 2, SERVICE_BOX_WIDTH_M * 2)
        ax.set_ylim(-1, COURT_LENGTH_M / 2 + 1)
        ax.text(
            0.5,
            0.5,
            "Projection refused",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
            color="red",
            fontweight="bold",
        )
        ax.set_title("Court Landing Projection -- Refused")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Net line (z=0)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=2, label="Net (z=0)")
    # Centerline (x=0)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=1, label="Centerline")

    # Service box rectangle
    box = mpatches.Rectangle(
        (-SERVICE_BOX_WIDTH_M, 0),
        SERVICE_BOX_WIDTH_M * 2,
        SERVICE_BOX_DEPTH_M,
        fill=False,
        edgecolor="green",
        linewidth=2,
        label="Service box",
    )
    ax.add_patch(box)

    # Deuce/ad sub-rectangles
    ax.add_patch(
        mpatches.Rectangle(
            (-SERVICE_BOX_WIDTH_M, 0),
            SERVICE_BOX_WIDTH_M,
            SERVICE_BOX_DEPTH_M,
            fill=True,
            facecolor="lightgreen",
            alpha=0.2,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0),
            SERVICE_BOX_WIDTH_M,
            SERVICE_BOX_DEPTH_M,
            fill=True,
            facecolor="lightcoral",
            alpha=0.2,
        )
    )

    # Landing point
    lx, lz = projection_result.landing_x_m, projection_result.landing_z_m
    in_box = projection_result.in_service_box
    color = "green" if in_box else "red"
    ax.plot(
        lx,
        lz,
        "o",
        color=color,
        markersize=12,
        label=f"Landing ({'IN' if in_box else 'OUT'})",
    )

    # Side labels
    ax.text(
        -SERVICE_BOX_WIDTH_M + 0.3,
        SERVICE_BOX_DEPTH_M / 2,
        "Deuce",
        fontsize=9,
        color="darkgreen",
        alpha=0.6,
    )
    ax.text(
        SERVICE_BOX_WIDTH_M - 0.8,
        SERVICE_BOX_DEPTH_M / 2,
        "Ad",
        fontsize=9,
        color="darkred",
        alpha=0.6,
    )

    ax.set_xlabel("Lateral offset from centerline (m)")
    ax.set_ylabel("Distance from net along court (m)")
    ax.set_title("Court Landing Projection")
    ax.set_xlim(-SERVICE_BOX_WIDTH_M * 1.5, SERVICE_BOX_WIDTH_M * 1.5)
    ax.set_ylim(-1, COURT_LENGTH_M / 2 + 1)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
