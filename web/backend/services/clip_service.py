"""Generate short MP4 clips for selected serves with ball overlay.

Each clip is a temporal slice around the serve contact time, with a
green circle + crosshair drawn on the detected ball position for every
frame. Positions are interpolated for frames between sampled detections.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from web.backend.paths import get_clips_dir


def _interpolate_position(
    frame_num: int,
    positions: List[Optional[Tuple[float, float]]],
    frame_skip: int,
) -> Optional[Tuple[float, float]]:
    """Get ball position for any frame via linear interpolation.

    Positions are at frame_skip intervals. For in-between frames,
    linearly interpolate between the two nearest sampled positions.

    Parameters
    ----------
    frame_num:
        Original video frame number.
    positions:
        List of (x, y) tuples, one per sampled frame. May contain None.
    frame_skip:
        Sampling interval used during detection.

    Returns
    -------
    tuple or None
        Interpolated (x, y) position, or None if out of range.
    """
    float_idx = frame_num / frame_skip
    lo = int(float_idx)
    hi = lo + 1

    if lo < 0 or lo >= len(positions):
        return None

    pos_lo = positions[lo]
    if pos_lo is None:
        return None

    if hi >= len(positions):
        return pos_lo

    pos_hi = positions[hi]
    if pos_hi is None:
        return pos_lo

    frac = float_idx - lo
    x = pos_lo[0] * (1 - frac) + pos_hi[0] * frac
    y = pos_lo[1] * (1 - frac) + pos_hi[1] * frac
    return (x, y)


def _draw_ball_marker(frame: np.ndarray, x: float, y: float) -> None:
    """Draw green circle + crosshair on ball position (in-place)."""
    cx, cy = int(round(x)), int(round(y))
    h, w = frame.shape[:2]
    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return
    # Green circle (radius 12, thickness 2)
    cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
    # Crosshair lines (20px arms)
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)


def generate_clips(
    video_path: str,
    selected_serves: List[Dict[str, Any]],
    positions: List[Optional[Tuple[float, float]]],
    detection_frame_skip: int,
) -> List[Dict[str, Any]]:
    """Create one MP4 clip per selected serve with ball overlay.

    The clip window is deterministic: ``max(contact_frame - 2.25*fps, 0)``
    to ``contact_frame + 1.75*fps``. Each frame has a green circle + crosshair
    drawn at the detected/interpolated ball position. Output is scaled to
    480px width.

    Parameters
    ----------
    video_path:
        Absolute path to the source video.
    selected_serves:
        List of serve dicts with ``contact_frame`` and ``contact_time_sec``.
    positions:
        Full positions array from detection (one entry per sampled frame).
    detection_frame_skip:
        Frame skip used during detection (for position interpolation).

    Returns
    -------
    list[dict]
        Metadata for each generated clip.
    """
    clips_dir = get_clips_dir()
    clip_metadata: List[Dict[str, Any]] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output dimensions: 480px wide, preserve aspect ratio
    out_width = 480
    scale = out_width / orig_width
    out_height = int(orig_height * scale)
    # Ensure even height for codec compatibility
    out_height = out_height + (out_height % 2)

    for idx, serve in enumerate(selected_serves, start=1):
        contact_frame = int(serve["contact_frame"])
        contact_time_sec = float(serve["contact_time_sec"])

        start_frame = max(0, contact_frame - int(2.25 * fps))
        end_frame = min(total_frames, contact_frame + int(1.75 * fps))
        duration = round((end_frame - start_frame) / fps, 3)

        filename = f"serve-{idx:02d}.mp4"
        output_path = os.path.join(clips_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to output dimensions
            if scale != 1.0:
                frame = cv2.resize(frame, (out_width, out_height))

            # Draw ball marker if position available
            ball_pos = _interpolate_position(frame_num, positions, detection_frame_skip)
            if ball_pos is not None:
                # Scale position to match resized frame
                scaled_x = ball_pos[0] * scale
                scaled_y = ball_pos[1] * scale
                _draw_ball_marker(frame, scaled_x, scaled_y)

            writer.write(frame)

        writer.release()

        # Re-encode to H.264 for browser compatibility
        temp_path = output_path + ".tmp.mp4"
        os.rename(output_path, temp_path)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        os.remove(temp_path)

        clip_metadata.append(
            {
                "filename": filename,
                "url_path": f"/clips/{filename}",
                "serve_index": idx,
                "contact_time_sec": contact_time_sec,
                "duration": duration,
            }
        )

    cap.release()
    return clip_metadata


def cleanup_clips() -> None:
    """Remove all files from the clips temp directory."""
    clips_dir = get_clips_dir()
    if not os.path.isdir(clips_dir):
        return
    for entry in os.listdir(clips_dir):
        path = os.path.join(clips_dir, entry)
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                import shutil

                shutil.rmtree(path)
        except OSError:
            pass
