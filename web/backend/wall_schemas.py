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


class WallReviewClipArtifact(BaseModel):
    """Browser URL for the impact-centered wall review clip."""

    url: str


class WallReviewMetadata(BaseModel):
    """Timing metadata for an impact-centered wall review clip."""

    impact_time_sec: float
    impact_frame: Optional[int] = None
    start_time_sec: float
    end_time_sec: float
    duration_sec: float


class WallCalibrationPoint(BaseModel):
    """One wall reference point with pixel and wall-frame coordinates."""

    name: str
    pixel: list[float]
    wall_m: list[float]


class WallCalibrationSetup(BaseModel):
    """Setup section of a wall calibration payload."""

    serve_contact_distance_m: float = 6.11
    camera_wall_distance_m: float = 1.57
    serve_contact_height_m: float
    wall_reference_points: list[WallCalibrationPoint]
    hook_reference: Optional[Dict[str, Any]] = None
    chair_references: Optional[list[Dict[str, Any]]] = None


class WallCalibrationRequest(BaseModel):
    """Payload for POST /api/wall/calibration."""

    video_id: str
    calibration_frame: int
    calibration_time_sec: float
    setup: WallCalibrationSetup
    video_override: Optional[Dict[str, Any]] = None
    intrinsics: Optional[Dict[str, Any]] = None
    manual_corrections: Optional[Dict[str, Any]] = None


class WallCalibrationResponse(BaseModel):
    """Response from POST /api/wall/calibration."""

    video_id: str
    point_count: int
    rms_m: Optional[float] = None


class WallCalibrationGetResponse(BaseModel):
    """Response from GET /api/wall/calibration."""

    video_id: str
    calibration_frame: int
    calibration_time_sec: float
    calibration: Dict[str, Any]
    point_count: int
    rms_m: Optional[float] = None


class WallCalibrationDeleteResponse(BaseModel):
    """Response from DELETE /api/wall/calibration."""

    status: str
    message: str


class WallAnalyzeResponse(BaseModel):
    """Response from POST /api/wall/analyze."""

    status: str
    message: str
