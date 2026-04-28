"""Adapter service that wraps the detector stack for the web backend.

Imports serve_analyzer.serve_attempts directly instead of shelling out,
and normalizes the result into a stable job payload.
"""

import os
from typing import Any, Callable, Dict, List, Optional

from serve_analyzer.analysis import get_video_info
from serve_analyzer.serve_attempts import (
    detect_serve_candidates,
    select_serves,
)


def _recommended_frame_skip(video_path: str) -> int:
    """Choose frame_skip based on video resolution and length.

    Rules (web adapter only):
        - 4K / very long videos -> frame_skip=4
        - 1080p / long videos   -> frame_skip=2
        - smaller / shorter     -> frame_skip=1
    """
    info = get_video_info(video_path)
    width = info.get("width", 0)
    height = info.get("height", 0)
    frame_count = info.get("frame_count", 0)

    if width >= 3000 or height >= 1700 or frame_count >= 3500:
        return 4
    if width >= 1900 or height >= 1000 or frame_count >= 1800:
        return 2
    return 1


Phases = ["analyzing", "clipping", "done"]


def _detector_config() -> Dict[str, Optional[str]]:
    """Read detector backend settings from environment variables."""
    detector = os.environ.get("SERVE_ANALYZER_DETECTOR", "yolo").lower()
    if detector not in {"yolo", "tracknetv2"}:
        raise ValueError(f"Unsupported detector: {detector}")
    return {
        "detector": detector,
        "model": os.environ.get("SERVE_ANALYZER_MODEL", "rjtp"),
        "tracknet_weights": os.environ.get("SERVE_ANALYZER_TRACKNET_WEIGHTS"),
        "tracknet_device": os.environ.get("SERVE_ANALYZER_TRACKNET_DEVICE", "cpu"),
    }


def estimate_analysis_duration(video_path: str) -> float:
    """Estimate total analysis time in seconds based on video metadata.

    Uses frame count, resolution, and recommended frame_skip to predict
    how long YOLO detection + select_serves will take. Conservative estimate
    at ~0.3 seconds per sampled frame (includes YOLO inference + HSV tracking).

    Returns
    -------
    float
        Estimated duration in seconds.
    """
    info = get_video_info(video_path)
    frame_count = info.get("frame_count", 0)
    width = info.get("width", 0)
    height = info.get("height", 0)

    # Replicate _recommended_frame_skip logic inline (avoids double get_video_info call)
    if width >= 3000 or height >= 1700 or frame_count >= 3500:
        frame_skip = 4
    elif width >= 1900 or height >= 1000 or frame_count >= 1800:
        frame_skip = 2
    else:
        frame_skip = 1

    sampled_frames = max(1, frame_count // frame_skip)
    detector = os.environ.get("SERVE_ANALYZER_DETECTOR", "yolo").lower()
    seconds_per_sample = 0.7 if detector == "tracknetv2" else 0.3
    return float(sampled_frames * seconds_per_sample)


def run_analysis(
    video_path: str,
    expected_serves: Optional[int] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run detector + selector on *video_path* and return a normalized payload.

    Parameters
    ----------
    video_path:
        Absolute path to the uploaded video file.
    expected_serves:
        Force exactly this many serves, or ``None`` to let the selector infer
        the count autonomously.
    on_progress:
        Optional callback that receives coarse phase strings
        (``"analyzing"``, ``"clipping"``, ``"done"``).

    Returns
    -------
    dict
        Normalized result with keys:
        ``video_path``, ``expected_serves``, ``count_inferred``,
        ``inferred_count``, ``selected_serves``, ``candidates``.
    """
    if on_progress:
        on_progress("analyzing")

    frame_skip = _recommended_frame_skip(video_path)

    # Pool size follows CLI default at serve_attempts.py:507-508
    # In autonomous mode (expected_serves=None) pass None through
    # so detector uses its own default pool expansion logic.
    pool_size = expected_serves
    detector_config = _detector_config()

    detection_result = detect_serve_candidates(
        video_path,
        expected_serves=pool_size,
        detector=detector_config["detector"] or "yolo",
        model=detector_config["model"] or "rjtp",
        tracknet_weights=detector_config["tracknet_weights"],
        tracknet_device=detector_config["tracknet_device"] or "cpu",
        frame_skip=frame_skip,
    )
    candidates: List[Dict[str, Any]] = detection_result["candidates"]
    positions = detection_result["positions"]
    detection_frame_skip = detection_result["frame_skip"]

    selected: List[Dict[str, Any]] = select_serves(
        candidates,
        expected_serves=expected_serves,
    )

    count_inferred = expected_serves is None
    inferred_count: Optional[int] = len(selected) if count_inferred else None

    # Normalize numpy scalars to plain Python types for JSON safety
    if inferred_count is not None:
        inferred_count = int(inferred_count)

    result: Dict[str, Any] = {
        "video_path": str(video_path),
        "expected_serves": expected_serves,
        "count_inferred": bool(count_inferred),
        "inferred_count": inferred_count,
        "detector": detector_config["detector"],
        "selected_serves": selected,
        "candidates": candidates,
        "positions": positions,
        "raw_positions": detection_result.get("raw_positions", positions),
        "detection_frame_skip": detection_frame_skip,
    }

    if on_progress:
        on_progress("clipping")
        on_progress("done")

    return result
