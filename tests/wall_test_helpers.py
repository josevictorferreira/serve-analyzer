"""
Pure helper functions for generating synthetic wall-impact test videos.

No TestCase here — just deterministic OpenCV video generation and ground-truth
computation for use by tests/test_wall_synthetic.py and later wall-analysis
test modules.
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Tuple

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class WallFixtureGroundTruth:
    """Ground-truth metadata for a synthetic wall-impact video fixture."""

    video_path: str
    fps: int
    total_frames: int
    impact_frame: int
    impact_pixel: Tuple[int, int]
    ball_positions: List[Tuple[int, int, int]]  # (frame_idx, x, y)
    ball_speed_px_per_frame: float
    wall_x_px: int
    expected_uncertainty_px: float


def generate_wall_impact_video(
    path: str,
    *,
    width: int = 320,
    height: int = 240,
    fps: int = 60,
    total_frames: int = 90,
    impact_frame: int = 60,
    ball_radius: int = 6,
    ball_speed_px_per_frame: float = 8.0,
    wall_x_px: int = 240,
    blur_sigma: float = 0.0,
) -> WallFixtureGroundTruth:
    """Generate a synthetic MP4 video of a ball hitting a vertical wall.

    The ball moves left-to-right horizontally at constant speed.  It reaches
    ``wall_x_px`` exactly on ``impact_frame``, then disappears for all
    subsequent frames.  The background is solid gray; the ball is a filled
    white circle.  Optional Gaussian blur simulates motion blur.

    Args:
        path: Destination file path (should end in ``.mp4``).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames to write.
        impact_frame: Frame index (0-based) at which the ball centre reaches
            ``wall_x_px``.
        ball_radius: Radius of the filled white circle.
        ball_speed_px_per_frame: Horizontal speed in pixels per frame.
        wall_x_px: X coordinate of the vertical wall plane.
        blur_sigma: If > 0, apply ``cv2.GaussianBlur`` to every frame with
            kernel size derived from sigma.

    Returns:
        A :class:`WallFixtureGroundTruth` dataclass with deterministic
        metadata derived from the input arguments.

    Raises:
        ValueError: If ``impact_frame`` is not in ``[0, total_frames)`` or
            if the computed start position would place the ball outside the
            frame before impact.
    """
    if not (0 <= impact_frame < total_frames):
        raise ValueError(
            f"impact_frame ({impact_frame}) must be in [0, {total_frames})"
        )

    # Compute start_x so that x[impact_frame] == wall_x_px exactly.
    start_x = float(wall_x_px) - ball_speed_px_per_frame * impact_frame
    if start_x < ball_radius:
        raise ValueError(
            f"Computed start_x ({start_x:.1f}) is too close to the left edge; "
            "increase width, decrease impact_frame, or decrease ball_speed."
        )

    y = height // 2

    # Pre-compute ground-truth ball positions (only visible frames).
    ball_positions: List[Tuple[int, int, int]] = []
    for t in range(total_frames):
        x = start_x + ball_speed_px_per_frame * t
        if x > wall_x_px + ball_radius:
            # Ball has passed the wall; stop recording positions.
            break
        ball_positions.append((t, int(round(x)), y))

    # Video writer setup.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")

    # Background colour (medium gray).
    bg_color = (128, 128, 128)

    # Blur kernel size (must be odd).
    blur_ksize = 0
    if blur_sigma > 0.0:
        blur_ksize = int(math.ceil(blur_sigma * 3) * 2 + 1)

    for t in range(total_frames):
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        x = start_x + ball_speed_px_per_frame * t
        if x <= wall_x_px + ball_radius:
            # Ball is still visible (pre-impact or exactly at impact).
            cx = int(round(x))
            cy = y
            cv2.circle(frame, (cx, cy), ball_radius, (255, 255, 255), thickness=-1)

        if blur_ksize > 0:
            frame = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), blur_sigma)

        out.write(frame)

    out.release()

    # Deterministic uncertainty scales linearly with blur_sigma.
    expected_uncertainty_px = 1.0 + blur_sigma * 0.5

    return WallFixtureGroundTruth(
        video_path=path,
        fps=fps,
        total_frames=total_frames,
        impact_frame=impact_frame,
        impact_pixel=(wall_x_px, y),
        ball_positions=ball_positions,
        ball_speed_px_per_frame=ball_speed_px_per_frame,
        wall_x_px=wall_x_px,
        expected_uncertainty_px=expected_uncertainty_px,
    )
