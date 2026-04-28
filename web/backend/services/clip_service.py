"""Generate short clean MP4 clips for selected serves.

Each clip is a temporal slice around the serve contact time. Ball overlay
metadata is returned separately so the browser can draw accurate markers only
when a detector produced a position for the current video frame.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
from web.backend.paths import get_clips_dir


def _position_at_frame(
    frame_num: int,
    positions: List[Optional[Tuple[float, float]]],
) -> Optional[Tuple[float, float]]:
    """Return the detector position for one source frame, if present.

    Parameters
    ----------
    frame_num:
        Original video frame number.
    positions:
        List of (x, y) tuples keyed by original frame number. May contain None.

    Returns
    -------
    tuple or None
        Detector (x, y) position, or None if the ball was not detected.
    """
    if frame_num < 0 or frame_num >= len(positions):
        return None
    return positions[frame_num]


def generate_clips(
    video_path: str,
    selected_serves: List[Dict[str, Any]],
    positions: List[Optional[Tuple[float, float]]],
    detection_frame_skip: int,
    overlay_positions: Optional[List[Optional[Tuple[float, float]]]] = None,
) -> List[Dict[str, Any]]:
    """Create one clean MP4 clip per selected serve with overlay metadata.

    The clip window is deterministic: ``max(contact_frame - 2.25*fps, 0)``
    to ``contact_frame + 1.75*fps``. Output is scaled to at most 960px width.
    Ball positions are returned in clip pixel coordinates instead of being
    burned into the video.

    Parameters
    ----------
    video_path:
        Absolute path to the source video.
    selected_serves:
        List of serve dicts with ``contact_frame`` and ``contact_time_sec``.
    positions:
        Full interpolated positions array from detection.
    detection_frame_skip:
        Frame skip used during detection.
    overlay_positions:
        Raw detector positions keyed by source frame number. When omitted,
        ``positions`` is used as a fallback.

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

    # Output dimensions: large enough for review, preserve aspect ratio.
    out_width = min(orig_width, 960)
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
        overlay_source = (
            overlay_positions if overlay_positions is not None else positions
        )
        overlay_points: List[Dict[str, float | int]] = []

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

            ball_pos = _position_at_frame(frame_num, overlay_source)
            if ball_pos is not None:
                scaled_x = ball_pos[0] * scale
                scaled_y = ball_pos[1] * scale
                if 0 <= scaled_x < out_width and 0 <= scaled_y < out_height:
                    overlay_points.append(
                        {
                            "frame_number": int(frame_num),
                            "clip_time_sec": round((frame_num - start_frame) / fps, 4),
                            "x": round(float(scaled_x), 2),
                            "y": round(float(scaled_y), 2),
                        }
                    )

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
                "fps": float(fps),
                "width": int(out_width),
                "height": int(out_height),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "contact_frame": int(contact_frame),
                "contact_clip_time_sec": round((contact_frame - start_frame) / fps, 4),
                "velocity_kmh": (
                    float(serve["post_contact_max_kmh"])
                    if serve.get("post_contact_max_kmh") is not None
                    else None
                ),
                "mean_velocity_kmh": (
                    float(serve["post_contact_mean_kmh"])
                    if serve.get("post_contact_mean_kmh") is not None
                    else None
                ),
                "ball_positions": overlay_points,
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
