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


## 2026-05-06 Task 5 Frontend Wall Mode and Staged Upload Step
- Created `web/src/lib/wall-types.ts` with TypeScript interfaces matching all Pydantic schemas from `wall_schemas.py`.
- Created `web/src/lib/wall-api.ts` with API client functions: `uploadWallVideo`, `getWallVideoMetadata`, `saveWallCalibration`, `getWallCalibration`, `deleteWallCalibration`, `startWallAnalysis`, `getWallJob`, `resetWallJob`.
- `uploadWallVideo` uses XMLHttpRequest for progress tracking, matching the existing `analyzeVideoWithProgress` pattern in `api.ts`.
- Created `web/src/components/wall-upload-step.tsx` with `WallUploadStep` (dropzone with file validation, upload state, error state) and `WallMetadataDisplay` (metadata cards for filename, duration, fps, resolution, frames).
- Created `web/src/components/wall-workflow.tsx` with `WallWorkflow` stepper (Upload → Calibrate → Configure → Analyze → Results). Only Upload step is functional; others show placeholder cards.
- Modified `web/src/App.tsx`: added `'wall'` to mode union type, added "Wall Analysis" button, added `WallWorkflow` rendering when `mode === 'wall'`.
- Wall mode state is entirely separate from existing analysis/annotation modes — no shared state.
- Tests added: `App.test.tsx` (wall mode switching, upload dropzone rendering, mode preservation) and `wall-upload-step.test.tsx` (render, successful upload with metadata, error message, uploading state).
- Build: `npm run build` exits 0. Tests: `npm test -- --run` — 16/16 pass (3 test files).

## 2026-05-06 Task 4 Impact-Centered Wall Review Clips
- Created `web/backend/services/wall_review_clip_service.py` to generate `{video_stem}_impact_review.mp4` with ffmpeg using the required `max(impact_time_sec - 1.5, 0.0)` to `min(impact_time_sec + 1.0, video_duration_sec)` window and silent H.264 output.
- Integrated review clip generation in `run_wall_analysis()` as best-effort only: failures log a warning and the wall analysis result still succeeds.
- Result contract now adds nested `artifacts.review_clip.url` plus top-level `review` metadata with `impact_time_sec`, `impact_frame`, `start_time_sec`, `end_time_sec`, and `duration_sec` when `measured.impact_time_sec` is present.
- If `impact_time_sec` is `None`, `review_clip` and `review` are omitted silently.
- Test gotcha: current synthetic wall web analysis fixture can produce `impact_time_sec: null`; the review-clip integration test patches `_process_video()` to write a deterministic impact result while still using the staged synthetic video for ffmpeg clip extraction.
- Verification: targeted `test_wall_web_review_clip.py` passed; full `python -m unittest discover -s tests -v` passed with 273 tests (3 skipped). Python LSP diagnostics could not run because `pylsp` is not installed.

## 2026-05-06 Task 6 Wall Calibration Canvas and Assumptions Form
- Created `web/src/components/wall-calibration-canvas.tsx`:
  - Video + canvas overlay with click-to-place calibration points
  - Frame scrubber (range slider) synced to video currentTime
  - Numbered circle markers drawn on canvas via 2D context
  - Pixel coordinates mapped from display size back to original video dimensions
  - Point list with remove buttons and "Clear All" action
  - Minimum 4 points validation message
- Created `web/src/components/wall-assumptions-form.tsx`:
  - Contact height input (default 2.80m), contact distance (6.11m), camera distance (1.57m)
  - Wall reference points table with pixel_x/pixel_y (read-only) and wall_m_x/wall_m_y (editable)
  - Pre-populates pixel coords from calibration canvas points
  - Saves via `saveWallCalibration()` with proper `WallCalibrationRequest` shape
  - Loads existing calibration on mount via `getWallCalibration()`, populates form
  - Clear calibration button calls `deleteWallCalibration()` and resets form
  - Validates min 4 points + contact height + all wall coords before saving
- Modified `web/src/components/wall-workflow.tsx`:
  - Added state for `calibrationPoints` and `currentFrame`
  - Replaced calibrate placeholder with `WallCalibrationCanvas` + `WallAssumptionsForm`
  - Transitions to `calibrated` phase on successful save via `onCalibrated` callback
  - Resets calibration state on upload and reset
- Tests added: `wall-calibration-canvas.test.tsx` (5 tests) and `wall-assumptions-form.test.tsx` (5 tests).
- Test gotcha: `<video>` elements don't have an ARIA "video" role in testing-library — use `container.querySelector('video')` instead of `getByRole('video')`.
- Test gotcha: `<input type="number">` returns numeric values in testing-library, so `toHaveValue(2.8)` not `toHaveValue('2.80')`.
- Build: `npm run build` exits 0. Tests: 10/10 new tests pass. Pre-existing `wall-analyze-step` timeouts (4 failures) are unrelated to these changes.

## 2026-05-06 Task 7 Wall Analysis Trigger, Polling Hook, and Error States
- Created `web/src/components/wall-analyze-step.tsx`:
  - Start Analysis button: disabled when `isCalibrated=false`, calls `startWallAnalysis()` on click
  - 409/busy state: shows "Another analysis is running" with Reset and Retry button
  - Progress polling: `getWallJob()` every 1s via `setInterval`, cleanup on unmount via `useEffect`
  - Phase labels map: idle/uploading/calibrating/analyzing/artifacting/done/error with descriptive messages
  - Done state: calls `onDone(result)` with job result data
  - Error state: shows error message, calls `onError(error)`, shows Reset button
  - Uses shadcn/ui Card, CardHeader, CardTitle, CardContent, Button + lucide-react icons
- Modified `web/src/lib/wall-api.ts`:
  - `startWallAnalysis()` now handles 409 response separately, throwing "Another analysis is already in progress. Please wait." to distinguish from generic errors
- Modified `web/src/components/wall-workflow.tsx`:
  - Replaced Analyze placeholder with `<WallAnalyzeStep>`
  - Tracks `analysisResult` and `analysisError` state
  - `onDone` stores result and transitions phase to 'done' (enables Results step)
  - `onError` stores error message (stays on Analyze step for retry)
  - `isCalibrated` derived from `phase === 'configured'`
  - Reset handler clears analysisResult and analysisError
- Tests in `wall-analyze-step.test.tsx` (7 tests):
  1. Renders start analysis button
  2. Button disabled when not calibrated
  3. Shows calibration hint when not calibrated
  4. Calls startWallAnalysis and starts polling on click
  5. Shows busy state on 409 response
  6. Reset button calls resetWallJob and returns to idle
  7. Shows error state when analysis fails
- Test gotcha: `vi.useFakeTimers()` is required for tests that use `vi.advanceTimersByTime()`. Must be called in `beforeEach` since `setInterval` is used for polling.
- Test gotcha: With fake timers, use `act(async () => { ... })` to flush React state updates instead of `waitFor` (which relies on real timers).
- Build: `npm run build` exits 0. Tests: 33/33 pass.

## 2026-05-06 Task 8 Wall Results Dashboard
- Created `web/src/components/wall-results-dashboard.tsx`:
  - Seven sub-components: `MeasuredImpactCard`, `VelocityCard`, `CourtProjectionCard`, `AnnotatedVideoCard`, `ReviewClipCard`, `PlotsGallery`, `ConfidenceWarningsCard`, `AssumptionsCard`, `DownloadLinksCard`
  - Main `WallResultsDashboard` component renders in responsive 2-col grid (1-col mobile, 2-col lg)
  - Impact time auto-populates from `measured.impact_time_sec` for "Jump to Impact" button
  - Plot gallery with CSS-only zoom-on-click lightbox (no external library)
  - Confidence score shown as percentage badge with color-coded variant (green ≥80%, amber ≥50%, red <50%)
  - Download links use native `<a>` elements styled to match shadcn Button (the project's `@base-ui/react` Button lacks `asChild`)
- Modified `web/src/components/wall-workflow.tsx`:
  - Imported and rendered `WallResultsDashboard` in Results step
  - Replaced `PlaceholderCard` with real dashboard
  - Passes `analysisResult` as `result` prop (type: `Record<string, unknown>`)
- Tests in `wall-results-dashboard.test.tsx` (12 tests):
  1. Dashboard renders with mock result data (all section titles present)
  2. Velocity card shows all 3 speed units (m/s, km/h, mph + uncertainty)
  3. Video players render with artifact URLs
  4. Plot images render for each plot URL (speed, wall_impact, court_landing)
  5. JSON/CSV links point to correct URLs
  6. Measured impact values displayed correctly
  7. Shows IN service box with green indicator when in_service_box is true
  8. Shows OUT when in_service_box is false
  9. Displays confidence score as percentage
  10. Displays warnings when present
  11. Renders review clip metadata (start/impact/end times)
  12. Omits review clip card when review_clip is absent
- Test gotcha: `/confidence/i` matches both "Confidence & Warnings" card title AND "Confidence Score" label — use `getAllByText` when regex may match multiple elements
- Build: `npm run build` exits 0. Tests: 45/45 pass (7 test files).

## 2026-05-06 Task 9 End-to-End Synthetic Integration and Regression QA
- Added `tests/test_wall_web_e2e.py` covering upload → metadata → calibration → analysis polling → six-section result contract → artifact GETs → reset while preserving calibration.
- E2E synthetic fixture must align `generate_wall_impact_video(..., wall_x_px=...)` with the right-most calibration point because `detect_wall_impact()` derives its wall search band from `max(calibration.wall_reference_points.pixel_x)`. A default `wall_x_px=240` with right calibration corners at `x=540` produces `impact_pixel: null` and therefore null `wall_x_m`/`wall_y_m`.
- Added frontend result rendering coverage for normalized backend artifact URL shapes where top-level artifact entries may be plain relative strings and nested downloads may be `{url: ...}` objects.
- Regression evidence written to `.sisyphus/evidence/web-wall-task-9-regression.txt`; final run passed: Python unittest 274 tests (3 skipped), web Vitest 46 tests, and production build.

## Final Wave Blocker Fixes (2026-05-06)
- Stepper should be 4 steps (Upload→Calibrate→Analyze→Results), not 5. The Configure placeholder step must be removed entirely. After calibration saves, set phase to 'configured' directly.
- Backend normalizes plot URLs as plain strings (not {url} objects). PlotsGallery must handle both string and {url: string} entries via typeof check.
- Confidence from backend is a dict with `score` key (not a raw number). ConfidenceWarningsCard must extract score via typeof check.
- Review clip metadata must live under artifacts.review_clip (merged with url), not as a separate top-level `review` key. The six-section contract is sacred: measured/inferred/assumed/confidence/warnings/artifacts only.
- React hooks must all be called before any early returns. useState before conditional returns, useEffect instead of render-time setState calls.
- Removing a step from the stepper requires updating App.test.tsx to not look for the removed step label.
