"""Pydantic schemas for wall serve analysis API endpoints."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class WallVideoUploadResponse(BaseModel):
    """Response from POST /api/wall/video after staging a wall video."""

    video_id: str
    video_url: str
    filename: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


class WallVideoMetadataResponse(BaseModel):
    """Response from GET /api/wall/video/{video_id}/metadata."""

    video_id: str
    filename: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


class WallJobResetResponse(BaseModel):
    """Response from POST /api/wall/job/reset."""

    status: str
    message: str


class WallJobStatus(BaseModel):
    """Current wall job state returned by GET /api/wall/job."""

    status: str
    phase: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
