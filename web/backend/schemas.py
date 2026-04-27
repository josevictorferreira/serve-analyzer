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
    estimated_duration_sec: Optional[float] = None



class AnalyzeResponse(BaseModel):
    """Response from POST /api/analyze."""

    status: str
    message: str
