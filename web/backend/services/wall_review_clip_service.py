"""Generate impact-centered review clips for wall analysis results.

Each clip is a short temporal window around the detected impact time,
generated via ffmpeg for browser-playable H.264 output.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _compute_clip_window(
    impact_time_sec: float,
    video_duration_sec: float,
    pre_impact_sec: float = 1.5,
    post_impact_sec: float = 1.0,
) -> Tuple[float, float]:
    """Compute review clip start/end times clamped to video bounds.

    Parameters
    ----------
    impact_time_sec:
        Detected impact time in seconds.
    video_duration_sec:
        Total video duration in seconds.
    pre_impact_sec:
        Seconds before impact to include (default 1.5).
    post_impact_sec:
        Seconds after impact to include (default 1.0).

    Returns
    -------
    tuple
        ``(start_time_sec, end_time_sec)``.
    """
    start_time_sec = max(impact_time_sec - pre_impact_sec, 0.0)
    end_time_sec = min(impact_time_sec + post_impact_sec, video_duration_sec)
    return start_time_sec, end_time_sec


def generate_impact_review_clip(
    video_path: str,
    output_dir: str,
    impact_time_sec: float,
    video_duration_sec: float,
    video_stem: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Generate a silent H.264 MP4 review clip centered on the impact time.

    Uses ffmpeg with ``-ss`` / ``-to`` for fast clipping. Output is
    browser-playable H.264 with audio removed.

    Parameters
    ----------
    video_path:
        Absolute path to the source video.
    output_dir:
        Directory where the clip will be written.
    impact_time_sec:
        Detected impact time in seconds.
    video_duration_sec:
        Total video duration in seconds.
    video_stem:
        Video stem used for naming (e.g. ``IMG_9340``).

    Returns
    -------
    tuple or None
        ``(output_path, metadata_dict)`` on success, or ``None`` on failure.
        ``metadata_dict`` contains ``impact_time_sec``, ``impact_frame``,
        ``start_time_sec``, ``end_time_sec``, ``duration_sec``.
    """
    start_time_sec, end_time_sec = _compute_clip_window(
        impact_time_sec, video_duration_sec
    )

    if end_time_sec <= start_time_sec:
        logger.warning(
            "Review clip window is empty (start=%.3f, end=%.3f); skipping.",
            start_time_sec,
            end_time_sec,
        )
        return None

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{video_stem}_impact_review.mp4"
    output_path = os.path.join(output_dir, filename)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_time_sec),
        "-to",
        str(end_time_sec),
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-an",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "ffmpeg failed for review clip (returncode=%d): %s",
            exc.returncode,
            exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "",
        )
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out after 120s for review clip.")
        return None
    except FileNotFoundError:
        logger.warning("ffmpeg not found; cannot generate review clip.")
        return None

    duration_sec = round(end_time_sec - start_time_sec, 3)

    metadata: Dict[str, Any] = {
        "impact_time_sec": impact_time_sec,
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "duration_sec": duration_sec,
    }

    return output_path, metadata
