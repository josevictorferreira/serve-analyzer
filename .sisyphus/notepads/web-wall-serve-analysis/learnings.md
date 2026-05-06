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
