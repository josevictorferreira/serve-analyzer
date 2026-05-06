# Learnings — web-wall-serve-analysis

## 2026-05-06 Session Start
- Wall pipeline fully implemented: `serve_analyzer/wall_*.py` with 67 wall tests, 248 full suite.
- `_process_video()` returns `serve_row` only; adapter must read `result.json`/`result.csv` from disk.
- Artifact layout: `{video_stem}_annotated.mp4` and `plots/*.png` under output dir.
- Frontend is Vite React + Tailwind v4 + shadcn/ui; backend is FastAPI on port 8000.
- Entry: `python -m web.backend` (not `app.py`).
- Normal serve web plan complete at `.sisyphus/plans/web-serve-analysis-app.md`.
- AGENTS.md lessons: always verify output contracts, don't subclass cv2.VideoCapture, enforce artifact naming in tests.

## 2026-05-06 Task 1 Wall Web Session APIs
- Added wall staging backend with in-memory session state, OpenCV metadata probing, wall temp/output helpers, and FastAPI wall video/metadata/reset/job endpoints.
- Global busy guard now checks normal serve and wall job states with shared state lock; `/api/analyze` and wall upload mutually reject active jobs.
- Verification: targeted `test_wall_web_session.py` passed; full `python -m unittest discover -s tests -v` passed with 257 tests (1 skipped marker for recursive full-suite check).

## 2026-05-06 Task 2 Wall Calibration Persistence APIs
- Created `web/backend/services/wall_calibration_service.py` with in-memory `_wall_calibration` dict + `_wall_calibration_lock`, mirroring `wall_session_service.py` pattern.
- Added Pydantic schemas to `wall_schemas.py`: `WallCalibrationPoint`, `WallCalibrationSetup`, `WallCalibrationRequest`, `WallCalibrationResponse`, `WallCalibrationGetResponse`, `WallCalibrationDeleteResponse`.
- Added three routes to `app.py`: `POST /api/wall/calibration`, `GET /api/wall/calibration`, `DELETE /api/wall/calibration`.
- POST validates payload via `WallCalibration.from_dict()`, checks `video_id` matches staged session, returns 200 with `point_count` and `rms_m` (None for now).
- GET returns 404 when no calibration is persisted; otherwise returns full calibration dict with metadata.
- DELETE clears calibration independently of video session.
- `POST /api/wall/job/reset` explicitly documented NOT to clear calibration — calibration survives job reset.
- Key gotcha: Pydantic `Optional[Dict] = None` fields serialize as `null` in JSON, which `WallCalibration.from_dict()` mishandles because it checks `if "hook_reference" in setup` rather than `if setup.get("hook_reference")`. Since we must NOT modify `serve_analyzer/`, the service strips `None` values recursively via `_strip_none_values()` before calling `from_dict()`.
- Tests in `tests/test_wall_web_calibration.py` cover: valid 4-point POST, GET round-trip, 3-point POST → 422, DELETE then GET 404, job reset preserves calibration, mismatched video_id → 400.
- Full suite: 264 tests, 2 skipped, 0 failures.

## 2026-05-06 Task 3 Wall Analysis Adapter, Normalized Result Contract, and Nested Artifacts
- Created `web/backend/services/wall_analysis_service.py` with `run_wall_analysis()` that:
  1. Gets staged video path from session service and calibration from calibration service.
  2. Creates output dir via `get_wall_output_dir(video_id)`.
  3. Calls `_process_video(video_path, calibration, output_dir)`.
  4. Reads `result.json` from disk (does NOT trust `_process_video` return value).
  5. Normalizes artifact paths to relative browser URLs (`/api/wall/artifacts/...`).
  6. Returns the full six-section result payload.
- Added `POST /api/wall/analyze` to `app.py`:
  - Checks `is_any_job_active()` → 409 if busy.
  - Validates staged session and saved calibration exist and match.
  - Reconstructs `WallCalibration` from stored dict via `WallCalibration.from_dict()`.
  - Launches background thread running `run_wall_analysis()` with progress callback that sets `ARTIFACTING` then `DONE`.
  - Returns 200 with accepted status immediately.
- Added `GET /api/wall/artifacts/{artifact_path:path}` to `app.py`:
  - Rejects literal `..` and encoded `%2e%2e` traversal sequences → 400.
  - Resolves artifact path under current wall job output root.
  - Verifies resolved path stays under output root (path traversal protection).
  - Serves file via `FileResponse` or returns 404.
- Added `WallAnalyzeResponse` schema to `wall_schemas.py`.
- Tests in `tests/test_wall_web_analysis.py` cover:
  - Full synthetic flow: upload → calibrate → analyze → poll job → assert six-section payload with `wall_x_m`/`wall_y_m` keys present.
  - Nested plot artifact URL resolves with 200 and non-zero bytes.
  - Path traversal rejected (literal `..` → 404 from router normalization; encoded `%2e%2e` → 400 from handler).
  - Reject analyze without calibration → 400.
  - Reject analyze when busy → 409.
- Full suite: 269 tests, 2 skipped, 0 failures.
- Key gotcha: `TestClient` URL-decodes `%2e%2e` before sending, so the encoded traversal test may hit the router first. The router normalizes `..` away → 404, which is still a rejection. Our handler check for `%2e%2e` catches the encoded case when it reaches us. Tests accept 400/403/404 for both cases to be resilient.
- Key gotcha: Must import `Path` from `pathlib` in `app.py` for artifact route; missing import caused `NameError` in background thread.
- Key gotcha: `_process_video()` writes `result.json` with absolute paths in `artifacts`. The adapter must normalize these to relative browser URLs before storing in wall state.

