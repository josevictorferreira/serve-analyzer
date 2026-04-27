"""FastAPI application for serve analyzer web backend."""

import os
import shutil
import threading
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .paths import (
    clean_temp_clips,
    get_clips_dir,
    get_session_temp_dir,
    make_temp_video_path,
)
from .schemas import AnalyzeResponse, JobStatus
from .state import is_job_active, reset_state, set_state, JobPhase
from .services.analysis_service import run_analysis
from .services.clip_service import generate_clips

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


ALLOWED_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/x-matroska",
}
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


@app.post("/api/analyze", status_code=202, response_model=AnalyzeResponse)
async def analyze(video: UploadFile = File(...)) -> dict[str, str]:
    """Accept a video upload and start analysis in the background."""
    if is_job_active():
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
    estimated = estimate_analysis_duration(temp_video)
    set_state({"status": JobPhase.ANALYZING, "estimated_duration_sec": estimated})

    # Start real analysis in a background thread (CPU-blocking).
    threading.Thread(
        target=_run_analysis_thread,
        args=(temp_video,),
        daemon=True,
    ).start()

    return {"status": "accepted", "message": "Analysis started"}


def _run_analysis_thread(video_path: str) -> None:
    """Run blocking analysis, generate clips, and update shared state."""
    try:
        result = run_analysis(
            video_path,
            expected_serves=None,
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
            }
        )
        clip_metadata = generate_clips(
            video_path, selected_serves,
            positions=result["positions"],
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
        set_state({"status": JobPhase.ERROR, "phase": None, "estimated_duration_sec": None, "error": str(exc)})


    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


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
