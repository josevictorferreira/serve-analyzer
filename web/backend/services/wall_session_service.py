"""Wall video session service — staged upload, metadata probing, and session state."""

from __future__ import annotations

import dataclasses
import os
import shutil
import threading
import uuid
from typing import Optional

import cv2
from fastapi import HTTPException

from web.backend.paths import get_wall_temp_dir as _get_wall_temp_dir
from web.backend.paths import make_wall_video_path


@dataclasses.dataclass(frozen=True)
class WallVideoMetadata:
    """Metadata extracted from a staged wall video."""

    video_id: str
    filename: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class WallSessionState:
    """In-memory state for the currently staged wall video."""

    video_id: str
    video_path: str
    video_url: str
    metadata: WallVideoMetadata


_wall_session: dict[str, WallSessionState] = {}
_wall_lock = threading.Lock()


def _probe_video_metadata(path: str, video_id: str, filename: str) -> WallVideoMetadata:
    """Probe width, height, fps, and frame count via OpenCV."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            # Fallback: count frames manually
            fps = fps if fps > 0 else 30.0
            manual_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                manual_count += 1
            frame_count = manual_count
        duration_sec = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()
    return WallVideoMetadata(
        video_id=video_id,
        filename=filename,
        duration_sec=duration_sec,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def stage_video(upload_file_path: str, filename: str) -> WallSessionState:
    """Move an uploaded file into the wall temp directory and probe metadata.

    Args:
        upload_file_path: Absolute path to the temporarily saved upload.
        filename: Original filename (for display/metadata).

    Returns:
        A :class:`WallSessionState` with the staged video details.
    """
    global _wall_session
    video_id = uuid.uuid4().hex
    dest_path = make_wall_video_path(video_id)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.move(upload_file_path, dest_path)

    try:
        metadata = _probe_video_metadata(dest_path, video_id, filename)
    except Exception:
        try:
            os.remove(dest_path)
        except OSError:
            pass
        raise
    state = WallSessionState(
        video_id=video_id,
        video_path=dest_path,
        video_url=f"/api/wall/video/{video_id}",
        metadata=metadata,
    )
    with _wall_lock:
        previous = _wall_session.get("current")
        _wall_session.clear()
        _wall_session["current"] = state
    if previous is not None and os.path.isfile(previous.video_path):
        try:
            os.remove(previous.video_path)
        except OSError:
            pass
    return state


def get_session() -> Optional[WallSessionState]:
    """Return the current wall session, or None if none is staged."""
    with _wall_lock:
        return _wall_session.get("current")


def get_session_or_raise() -> WallSessionState:
    """Return the current wall session, raising 404 if none is staged."""
    session = get_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No wall video is currently staged")
    return session


def clear_session() -> None:
    """Delete the staged video file and reset in-memory session state."""
    with _wall_lock:
        session = _wall_session.get("current")
        _wall_session.clear()
    if session is not None and os.path.isfile(session.video_path):
        try:
            os.remove(session.video_path)
        except OSError:
            pass


def get_wall_temp_dir() -> str:
    """Return the wall-specific temp subdirectory (with makedirs)."""
    return _get_wall_temp_dir()
