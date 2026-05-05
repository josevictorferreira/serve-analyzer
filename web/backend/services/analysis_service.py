"""Adapter service that wraps the detector stack for the web backend.

Selects a detector-version service instead of shelling out, and normalizes
the result into a stable job payload.
"""

import os
from typing import Any, Callable, Dict, Optional

from serve_analyzer.analysis import get_video_info
from web.backend.services.detection_services import (
    get_detector_service,
    resolve_detector_version,
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


def estimate_analysis_duration(
    video_path: str, detector_version: Optional[str] = None
) -> float:
    """Estimate total analysis time in seconds based on video metadata.

    Uses frame count, resolution, recommended frame_skip, and selected detector
    version to predict how long the detector stack will take.

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
    tracking_detector = os.environ.get("SERVE_ANALYZER_DETECTOR", "yolo").lower()
    service = get_detector_service(detector_version)
    seconds_per_sample = service.estimate_seconds_per_sample(tracking_detector)
    return float(sampled_frames * seconds_per_sample)


def run_analysis(
    video_path: str,
    expected_serves: Optional[int] = None,
    detector_version: Optional[str] = None,
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

    selected_detector_version = resolve_detector_version(detector_version)
    detector_service = get_detector_service(selected_detector_version)
    frame_skip = _recommended_frame_skip(video_path)

    # Pool size follows CLI default at serve_attempts.py:507-508
    # In autonomous mode (expected_serves=None) pass None through
    # so detector uses its own default pool expansion logic.
    pool_size = expected_serves
    detector_config = _detector_config()

    detection_result = detector_service.run(
        video_path,
        expected_serves=pool_size,
        frame_skip=frame_skip,
        tracking_config=detector_config,
    )
    candidates = detection_result["candidates"]
    positions = detection_result["positions"]
    detection_frame_skip = detection_result["detection_frame_skip"]
    selected = detection_result["selected_serves"]

    result_expected_serves = detection_result.get("expected_serves", expected_serves)
    count_inferred = detection_result.get("count_inferred", expected_serves is None)
    inferred_count: Optional[int] = detection_result.get(
        "inferred_count", len(selected) if count_inferred else None
    )

    # Normalize numpy scalars to plain Python types for JSON safety
    if inferred_count is not None:
        inferred_count = int(inferred_count)

    result: Dict[str, Any] = {
        "video_path": str(video_path),
        "expected_serves": result_expected_serves,
        "count_inferred": bool(count_inferred),
        "inferred_count": inferred_count,
        "detector": detection_result["detector"],
        "detector_version": detection_result["detector_version"],
        "detector_label": detection_result["detector_label"],
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
