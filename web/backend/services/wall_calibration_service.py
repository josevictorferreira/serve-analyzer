"""Wall calibration persistence service — in-memory storage with thread lock."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from serve_analyzer.wall_calibration import WallCalibration

_wall_calibration: Dict[str, Any] = {}
_wall_calibration_lock = threading.Lock()


def store_calibration(
    video_id: str,
    calibration_frame: int,
    calibration_time_sec: float,
    calibration: WallCalibration,
    trim_start_frame: Optional[int] = None,
    trim_end_frame: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist a validated WallCalibration alongside video/frame metadata.

    Args:
        video_id: The staged video identifier.
        calibration_frame: Frame number used for calibration.
        calibration_time_sec: Timestamp (seconds) of the calibration frame.
        calibration: A validated :class:`WallCalibration` instance.

    Returns:
        A dict with ``video_id``, ``point_count``, and ``rms_m`` (if available).
    """
    global _wall_calibration
    result: Dict[str, Any] = {
        "video_id": video_id,
        "calibration_frame": calibration_frame,
        "calibration_time_sec": calibration_time_sec,
        "point_count": len(calibration.wall_reference_points),
        "rms_m": None,
    }
    with _wall_calibration_lock:
        _wall_calibration["current"] = {
            "video_id": video_id,
            "calibration_frame": calibration_frame,
            "calibration_time_sec": calibration_time_sec,
            "calibration": calibration,
            "result": result,
            "trim_start_frame": trim_start_frame,
            "trim_end_frame": trim_end_frame,
        }
    return result


def get_calibration() -> Optional[Dict[str, Any]]:
    """Return the persisted calibration payload, or None if none exists."""
    with _wall_calibration_lock:
        entry = _wall_calibration.get("current")
        if entry is None:
            return None
        return {
            "video_id": entry["video_id"],
            "calibration_frame": entry["calibration_frame"],
            "calibration_time_sec": entry["calibration_time_sec"],
            "calibration": entry["calibration"].to_dict(),
            "point_count": entry["result"]["point_count"],
            "rms_m": entry["result"]["rms_m"],
            "trim_start_frame": entry.get("trim_start_frame"),
            "trim_end_frame": entry.get("trim_end_frame"),
        }


def clear_calibration() -> None:
    """Clear the persisted calibration state."""
    global _wall_calibration
    with _wall_calibration_lock:
        _wall_calibration.clear()


def _strip_none_values(obj: Any) -> Any:
    """Recursively remove dict keys whose value is None."""
    if isinstance(obj, dict):
        return {k: _strip_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none_values(v) for v in obj]
    return obj


def validate_and_store(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate payload via :meth:`WallCalibration.from_dict` and persist it.

    Args:
        payload: Dictionary expected by :class:`WallCalibration` plus
            ``video_id``, ``calibration_frame``, and ``calibration_time_sec``.

    Returns:
        The stored result dict with ``video_id`` and ``point_count``.

    Raises:
        WallCalibrationError: If validation fails.
    """
    video_id = str(payload["video_id"])
    calibration_frame = int(payload["calibration_frame"])
    calibration_time_sec = float(payload["calibration_time_sec"])
    cleaned = _strip_none_values(payload)
    calibration = WallCalibration.from_dict(cleaned)
    trim_start_frame = payload.get("trim_start_frame")
    trim_end_frame = payload.get("trim_end_frame")
    return store_calibration(
        video_id, calibration_frame, calibration_time_sec, calibration,
        trim_start_frame=trim_start_frame, trim_end_frame=trim_end_frame,
    )
