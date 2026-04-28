"""Pydantic schemas for API request/response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class JobStatus(BaseModel):
    """Current job state returned by GET /api/job."""

    status: str
    phase: Optional[str] = None
    error: Optional[str] = None
    clips: List[Dict[str, Any]] = []
    selected_serves: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    count_inferred: Optional[bool] = None
    inferred_count: Optional[int] = None
    detector: Optional[str] = None
    estimated_duration_sec: Optional[float] = None


class AnalyzeResponse(BaseModel):
    """Response from POST /api/analyze."""

    status: str
    message: str


class AnnotationReviewRequest(BaseModel):
    """Request body for reviewing one annotation frame."""

    action: str
    bbox: Optional[Dict[str, float]] = None


class AnnotationSessionResponse(BaseModel):
    """Response containing a full annotation session manifest."""

    session: Dict[str, Any]


class AnnotationSessionsResponse(BaseModel):
    """Response containing annotation session summaries."""

    sessions: List[Dict[str, Any]]


class AnnotationExportResponse(BaseModel):
    """Response from exporting a YOLO dataset."""

    export: Dict[str, Any]


class AnnotationEvaluationResponse(BaseModel):
    """Response from baseline evaluation."""

    evaluation: Dict[str, Any]


class TrainingEnvironmentResponse(BaseModel):
    """Response from the local training environment check."""

    environment: Dict[str, Any]
