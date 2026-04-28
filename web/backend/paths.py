"""Path helpers for temp directories and clip storage."""

import os
import shutil
import tempfile


def get_session_temp_dir() -> str:
    """Return a session-scoped temp directory for uploads and clips."""
    base = os.environ.get("SERVE_ANALYZER_TEMP")
    if base is None:
        base = os.path.join(tempfile.gettempdir(), "serve_analyzer")
        os.makedirs(base, exist_ok=True)
    return base


def get_clips_dir() -> str:
    """Return the clips subdirectory inside the session temp dir."""
    clips = os.path.join(get_session_temp_dir(), "clips")
    os.makedirs(clips, exist_ok=True)
    return clips


def get_annotation_root() -> str:
    """Return the local root directory for annotation sessions."""
    root = os.environ.get("SERVE_ANALYZER_ANNOTATION_DIR")
    if root is None:
        root = os.path.join(tempfile.gettempdir(), "serve_analyzer_annotations")
    os.makedirs(root, exist_ok=True)
    return root


def get_annotation_session_dir(session_id: str) -> str:
    """Return the directory for one annotation session."""
    session_dir = os.path.join(get_annotation_root(), session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def clean_temp_clips() -> None:
    """Remove all files from the clips directory on startup."""
    clips = get_clips_dir()
    if os.path.isdir(clips):
        for entry in os.listdir(clips):
            path = os.path.join(clips, entry)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except OSError:
                pass


def make_temp_video_path() -> str:
    """Return a unique temp path for an uploaded video."""
    return os.path.join(get_session_temp_dir(), f"upload_{os.urandom(4).hex()}.mp4")
