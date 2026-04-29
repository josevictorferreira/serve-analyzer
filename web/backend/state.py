"""Shared in-memory state for a single job."""

from enum import Enum
from typing import Any, Dict
import threading


class JobPhase(str, Enum):
    """Phases a job can be in."""

    IDLE = "idle"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    CLIPPING = "clipping"
    DONE = "done"
    ERROR = "error"


# Single global job state.
_job_state: Dict[str, Any] = {
    "status": JobPhase.IDLE,
    "phase": None,
    "error": None,
    "clips": [],
    "selected_serves": [],
    "candidates": [],
    "count_inferred": None,
    "inferred_count": None,
    "detector": None,
    "detector_version": None,
    "detector_label": None,
    "estimated_duration_sec": None,
}

_state_lock = threading.Lock()


def get_state() -> Dict[str, Any]:
    """Return a shallow copy of the current job state."""
    with _state_lock:
        return dict(_job_state)


def set_state(updates: Dict[str, Any]) -> None:
    """Apply key/value updates to the global job state."""
    global _job_state
    with _state_lock:
        _job_state.update(updates)


def reset_state() -> None:
    """Reset job state to idle with empty collections."""
    global _job_state
    with _state_lock:
        _job_state = {
            "status": JobPhase.IDLE,
            "phase": None,
            "error": None,
            "clips": [],
            "selected_serves": [],
            "candidates": [],
            "count_inferred": None,
            "inferred_count": None,
            "detector": None,
            "detector_version": None,
            "detector_label": None,
            "estimated_duration_sec": None,
        }


def is_job_active() -> bool:
    """Return True if a job is currently running (not idle, done, or error)."""
    with _state_lock:
        return _job_state["status"] in (
            JobPhase.UPLOADING,
            JobPhase.ANALYZING,
            JobPhase.CLIPPING,
        )
