"""FastAPI application for serve analyzer web backend."""

import os
import shutil
import threading
from typing import Any
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .paths import (
    clean_temp_clips,
    get_clips_dir,
    get_session_temp_dir,
    get_wall_temp_dir,
    make_temp_video_path,
)
from .paths import get_wall_output_dir
from .schemas import (
    AnalyzeResponse,
    AnnotationEvaluationResponse,
    AnnotationExportResponse,
    AnnotationReviewRequest,
    AnnotationSessionResponse,
    AnnotationSessionsResponse,
    DetectorVersionsResponse,
    JobStatus,
    TrainingEnvironmentResponse,
)
from .state import (
    get_wall_state,
    is_any_job_active,
    reset_state,
    set_wall_state,
    set_state,
    JobPhase,
    WallJobPhase,
    reset_wall_state,
)
from .services.analysis_service import run_analysis
from .services import annotation_service
from .services.clip_service import generate_clips
from .services.detection_services import (
    default_detector_version,
    list_detector_versions,
    resolve_detector_version,
)
from .wall_schemas import (
WallJobStatus,
WallJobResetResponse,
WallVideoMetadataResponse,
WallVideoUploadResponse,
WallCalibrationRequest,
WallCalibrationResponse,
WallCalibrationGetResponse,
    WallCalibrationDeleteResponse,
    WallAnalyzeResponse,
)
from .services.wall_calibration_service import (
    clear_calibration,
    get_calibration,
    validate_and_store,
)
from serve_analyzer.wall_calibration import WallCalibrationError
from serve_analyzer.wall_calibration import WallCalibration
from .services.wall_analysis_service import run_wall_analysis
from .services.wall_session_service import (
    clear_session,
    get_session,
    stage_video,
)

app = FastAPI(title="Serve Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    """Clean leftover temp clips on startup."""
    clean_temp_clips()


@app.get("/api/job", response_model=JobStatus)
async def get_job() -> dict[str, Any]:
    """Return current job state."""
    from .state import get_state

    return get_state()


@app.get("/api/detectors", response_model=DetectorVersionsResponse)
async def get_detectors() -> dict[str, Any]:
    """Return serve detector versions available to the web frontend."""
    return {
        "detectors": list_detector_versions(),
        "default_version": default_detector_version(),
    }


ALLOWED_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/x-matroska",
}
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


@app.post("/api/analyze", status_code=202, response_model=AnalyzeResponse)
async def analyze(
    video: UploadFile = File(...), detector_version: str | None = Form(None)
) -> dict[str, str]:
    """Accept a video upload and start analysis in the background."""
    if is_any_job_active():
        raise HTTPException(status_code=409, detail="A job is already active")
    if video.content_type and video.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {video.content_type}"
        )
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file extension: {ext}"
        )
    try:
        selected_detector_version = resolve_detector_version(detector_version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    reset_state()
    set_state({"status": JobPhase.UPLOADING})

    temp_video = make_temp_video_path()
    try:
        with open(temp_video, "wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as exc:
        set_state({"status": JobPhase.ERROR, "error": str(exc)})
        raise HTTPException(
            status_code=500, detail=f"Failed to save upload: {exc}"
        ) from exc
    finally:
        video.file.close()

    from .services.analysis_service import estimate_analysis_duration

    estimated = estimate_analysis_duration(temp_video, selected_detector_version)
    set_state(
        {
            "status": JobPhase.ANALYZING,
            "estimated_duration_sec": estimated,
            "detector_version": selected_detector_version,
        }
    )

    threading.Thread(
        target=_run_analysis_thread,
        args=(temp_video, selected_detector_version),
        daemon=True,
    ).start()

    return {"status": "accepted", "message": "Analysis started"}


def _run_analysis_thread(video_path: str, detector_version: str) -> None:
    """Run blocking analysis, generate clips, and update shared state."""
    try:
        result = run_analysis(
            video_path,
            expected_serves=None,
            detector_version=detector_version,
            on_progress=lambda phase: set_state({"phase": phase}),
        )
        selected_serves = result["selected_serves"]
        set_state(
            {
                "status": JobPhase.CLIPPING,
                "phase": None,
                "selected_serves": selected_serves,
                "candidates": result["candidates"],
                "count_inferred": result["count_inferred"],
                "inferred_count": result["inferred_count"],
                "detector": result["detector"],
                "detector_version": result["detector_version"],
                "detector_label": result["detector_label"],
            }
        )
        clip_metadata = generate_clips(
            video_path,
            selected_serves,
            positions=result["positions"],
            overlay_positions=result.get("raw_positions"),
            detection_frame_skip=result["detection_frame_skip"],
        )
        set_state(
            {
                "status": JobPhase.DONE,
                "phase": None,
                "estimated_duration_sec": None,
                "clips": clip_metadata,
            }
        )
    except Exception as exc:
        set_state(
            {
                "status": JobPhase.ERROR,
                "phase": None,
                "estimated_duration_sec": None,
                "error": str(exc),
            }
        )

    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


@app.get(
    "/api/annotation/sessions",
    response_model=AnnotationSessionsResponse,
)
async def list_annotation_sessions() -> dict[str, Any]:
    """Return saved tennis-ball annotation sessions."""
    return {"sessions": annotation_service.list_annotation_sessions()}


@app.post(
    "/api/annotation/sessions",
    status_code=201,
    response_model=AnnotationSessionResponse,
)
async def create_annotation_session(
    video: UploadFile = File(...),
    max_frames: int = Query(240, ge=1, le=2000),
    frame_step: int | None = Query(None, ge=1),
    prelabel: bool = Query(True),
    confidence: float = Query(0.20, ge=0.01, le=1.0),
) -> dict[str, Any]:
    """Create an annotation session by extracting frames from an uploaded video."""
    if video.content_type and video.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {video.content_type}"
        )
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file extension: {ext}"
        )

    temp_video = make_temp_video_path()
    try:
        with open(temp_video, "wb") as f:
            shutil.copyfileobj(video.file, f)
        session = annotation_service.create_annotation_session(
            temp_video,
            video.filename or f"upload{ext}",
            max_frames=max_frames,
            frame_step=frame_step,
            prelabel=prelabel,
            confidence=confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to create annotation session: {exc}"
        ) from exc
    finally:
        video.file.close()
        try:
            os.unlink(temp_video)
        except OSError:
            pass

    return {"session": session}


@app.get(
    "/api/annotation/sessions/{session_id}",
    response_model=AnnotationSessionResponse,
)
async def get_annotation_session(session_id: str) -> dict[str, Any]:
    """Return one annotation session manifest."""
    try:
        session = annotation_service.get_annotation_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session}


@app.get("/api/annotation/sessions/{session_id}/frames/{frame_id}/image")
async def serve_annotation_frame(session_id: str, frame_id: str) -> FileResponse:
    """Serve an extracted annotation frame image."""
    try:
        image_path = annotation_service.get_frame_image_path(session_id, frame_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(image_path)


@app.post(
    "/api/annotation/sessions/{session_id}/frames/{frame_id}/review",
    response_model=AnnotationSessionResponse,
)
async def review_annotation_frame(
    session_id: str, frame_id: str, review: AnnotationReviewRequest
) -> dict[str, Any]:
    """Review one annotation frame as accepted, corrected, absent, or skipped."""
    try:
        session = annotation_service.review_frame(
            session_id, frame_id, review.action, review.bbox
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session}


@app.post(
    "/api/annotation/sessions/{session_id}/frames/{frame_id}/undo",
    response_model=AnnotationSessionResponse,
)
async def undo_annotation_frame(session_id: str, frame_id: str) -> dict[str, Any]:
    """Undo the review decision for one annotation frame."""
    try:
        session = annotation_service.undo_frame_review(session_id, frame_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session": session}


@app.post(
    "/api/annotation/sessions/{session_id}/export",
    response_model=AnnotationExportResponse,
)
async def export_annotation_dataset(session_id: str) -> dict[str, Any]:
    """Export reviewed annotation frames as a YOLO dataset."""
    try:
        exported = annotation_service.export_yolo_dataset(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"export": exported}


@app.get(
    "/api/annotation/sessions/{session_id}/baseline",
    response_model=AnnotationEvaluationResponse,
)
async def evaluate_annotation_baseline(
    session_id: str, iou_threshold: float = Query(0.30, ge=0.01, le=1.0)
) -> dict[str, Any]:
    """Evaluate stored RJTPP predictions against reviewed labels."""
    try:
        evaluation = annotation_service.evaluate_rjtpp_baseline(
            session_id, iou_threshold=iou_threshold
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"evaluation": evaluation}


@app.get(
    "/api/annotation/training-environment",
    response_model=TrainingEnvironmentResponse,
)
async def get_training_environment() -> dict[str, Any]:
    """Return local PyTorch/Ultralytics training environment readiness."""
    return {"environment": annotation_service.check_training_environment()}


@app.post("/api/job/reset")
async def reset_job() -> dict[str, str]:
    """Reset the job state and clean temp artifacts."""
    session_dir = get_session_temp_dir()
    if os.path.isdir(session_dir):
        for entry in os.listdir(session_dir):
            path = os.path.join(session_dir, entry)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except OSError:
                pass
    reset_state()
    return {"status": "reset"}


@app.get("/clips/{filename}")
async def serve_clip(filename: str) -> FileResponse:
    """Serve a clip file from the temp clips directory."""
    clips_dir = get_clips_dir()
    safe_name = os.path.basename(filename)
    if safe_name != filename or ".." in filename or "%2e%2e" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(clips_dir, safe_name)
    if not os.path.realpath(file_path).startswith(os.path.realpath(clips_dir)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(file_path)


# Wall analysis routes (Task 1 — session staging)
# ============================================================


@app.get("/api/wall/job", response_model=WallJobStatus)
async def get_wall_job() -> dict[str, Any]:
    """Return current wall job state."""
    return get_wall_state()


@app.post("/api/wall/video", response_model=WallVideoUploadResponse)
async def upload_wall_video(video: UploadFile = File(...)) -> dict[str, Any]:
    """Stage a wall-practice video upload and return metadata."""
    if is_any_job_active():
        raise HTTPException(status_code=409, detail="A job is already active")
    if video.content_type and video.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {video.content_type}"
        )
    ext = os.path.splitext(video.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file extension: {ext}"
        )

    reset_wall_state()
    set_wall_state({"status": WallJobPhase.UPLOADING})
    temp_video = make_temp_video_path()
    try:
        with open(temp_video, "wb") as f:
            shutil.copyfileobj(video.file, f)
    except Exception as exc:
        set_wall_state({"status": WallJobPhase.ERROR, "error": str(exc)})
        raise HTTPException(
            status_code=500, detail=f"Failed to save upload: {exc}"
        ) from exc
    finally:
        video.file.close()

    try:
        state = stage_video(temp_video, video.filename or f"upload{ext}")
    except Exception as exc:
        set_wall_state({"status": WallJobPhase.ERROR, "error": str(exc)})
        try:
            os.unlink(temp_video)
        except OSError:
            pass
        raise HTTPException(
            status_code=500, detail=f"Failed to stage video: {exc}"
        ) from exc

    set_wall_state({"status": WallJobPhase.DONE, "phase": None, "error": None})
    meta = state.metadata
    return {
        "video_id": state.video_id,
        "video_url": state.video_url,
        "filename": meta.filename,
        "duration_sec": meta.duration_sec,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "width": meta.width,
        "height": meta.height,
    }


@app.get("/api/wall/video/{video_id}")
async def get_wall_video(video_id: str) -> FileResponse:
    """Serve the staged wall video file by video_id."""
    session = get_session()
    if session is None or session.video_id != video_id:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(session.video_path)


@app.get(
    "/api/wall/video/{video_id}/metadata", response_model=WallVideoMetadataResponse
)
async def get_wall_video_metadata(video_id: str) -> dict[str, Any]:
    """Return metadata for the staged wall video."""
    session = get_session()
    if session is None or session.video_id != video_id:
        raise HTTPException(status_code=404, detail="Video not found")
    meta = session.metadata
    return {
        "video_id": session.video_id,
        "filename": meta.filename,
        "duration_sec": meta.duration_sec,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "width": meta.width,
        "height": meta.height,
    }


@app.post("/api/wall/job/reset", response_model=WallJobResetResponse)
async def reset_wall_job() -> dict[str, str]:
    """Reset wall job state, delete staged video, and clean wall artifacts.

    NOTE: This intentionally does NOT clear calibration state. Calibration
    is persisted independently and survives a job reset so the user can
    re-analyze the same video without recalibrating.
    """
    clear_session()
    wall_temp = get_wall_temp_dir()
    if os.path.isdir(wall_temp):
        for entry in os.listdir(wall_temp):
            path = os.path.join(wall_temp, entry)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except OSError:
                pass

    reset_wall_state()
    return {"status": "reset", "message": "Wall job reset successfully"}


# Wall calibration routes (Task 2 — calibration persistence)
# ============================================================


@app.post("/api/wall/calibration", response_model=WallCalibrationResponse)
async def create_wall_calibration(payload: WallCalibrationRequest) -> dict[str, Any]:
    """Persist wall calibration after validating via WallCalibration.from_dict()."""
    session = get_session()
    if session is None or session.video_id != payload.video_id:
        raise HTTPException(
            status_code=400,
            detail="No staged video matches the provided video_id",
        )
    try:
        result = validate_and_store(payload.model_dump())
    except WallCalibrationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@app.get("/api/wall/calibration", response_model=WallCalibrationGetResponse)
async def get_wall_calibration() -> dict[str, Any]:
    """Return the persisted wall calibration, or 404 if none exists."""
    calibration = get_calibration()
    if calibration is None:
        raise HTTPException(status_code=404, detail="No calibration found")
    return calibration


@app.delete("/api/wall/calibration", response_model=WallCalibrationDeleteResponse)
async def delete_wall_calibration() -> dict[str, str]:
    """Clear the persisted wall calibration state."""
    clear_calibration()
    return {"status": "deleted", "message": "Wall calibration cleared"}


# Wall analysis routes (Task 3 — analysis + artifact serving)
# ============================================================


@app.post("/api/wall/analyze", response_model=WallAnalyzeResponse)
async def analyze_wall() -> dict[str, str]:
    """Start wall analysis in the background. Requires staged video + calibration."""
    if is_any_job_active():
        raise HTTPException(status_code=409, detail="A job is already active")

    session = get_session()
    if session is None:
        raise HTTPException(status_code=400, detail="No wall video is currently staged")

    cal_entry = get_calibration()
    if cal_entry is None:
        raise HTTPException(status_code=400, detail="No calibration saved")

    video_id = session.video_id
    if cal_entry.get("video_id") != video_id:
        raise HTTPException(
            status_code=400, detail="Saved calibration does not match staged video"
        )

    # Reconstruct WallCalibration from stored dict
    calibration = WallCalibration.from_dict(cal_entry["calibration"])

    reset_wall_state()
    set_wall_state({"status": WallJobPhase.ANALYZING})

    def _on_progress(phase: str) -> None:
        if phase == "artifacting":
            set_wall_state({"status": WallJobPhase.ARTIFACTING})

    def _run() -> None:
        try:
            result = run_wall_analysis(
                session.video_path,
                calibration,
                video_id,
                session.metadata.duration_sec,
                on_progress=_on_progress,
            )
            set_wall_state({"status": WallJobPhase.DONE, "result": result})
        except Exception as exc:
            set_wall_state({"status": WallJobPhase.ERROR, "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "accepted", "message": "Wall analysis started"}


@app.get("/api/wall/artifacts/{artifact_path:path}")
async def serve_wall_artifact(artifact_path: str) -> FileResponse:
    """Serve a wall analysis artifact file with path-traversal protection."""
    # Reject literal or encoded traversal sequences
    if ".." in artifact_path or "%2e%2e" in artifact_path:
        raise HTTPException(status_code=400, detail="Invalid artifact path")

    session = get_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No staged video")

    output_dir = Path(get_wall_output_dir(session.video_id))
    target = (output_dir / artifact_path).resolve()
    output_dir_resolved = output_dir.resolve()

    if not str(target).startswith(str(output_dir_resolved) + os.sep) and target != output_dir_resolved:
        raise HTTPException(status_code=403, detail="Access denied")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(str(target))
