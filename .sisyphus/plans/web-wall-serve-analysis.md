# Complete Wall Serve Web UI Flow

## TL;DR
> **Summary**: Extend the existing local FastAPI + Vite app with a complete wall-serve workflow: staged wall-video upload, interactive calibration on the uploaded video, analysis execution, browser-safe artifact serving, and a results dashboard with graphs, impact review crop, annotated video, velocity, wall impact, equivalent court landing, JSON/CSV, warnings, and confidence.
> **Deliverables**:
> - Backend wall session APIs: `POST /api/wall/video`, `GET /api/wall/video/{video_id}`, `GET /api/wall/video/{video_id}/metadata`, `POST /api/wall/analyze`, `GET /api/wall/job`, `POST /api/wall/job/reset`, `POST|GET|DELETE /api/wall/calibration`, `GET /api/wall/artifacts/{artifact_path:path}`.
> - Frontend wall mode with stepper: Upload → Calibrate → Configure → Analyze → Review Results.
> - Result UI covering the exact six-section wall contract: `measured`, `inferred`, `assumed`, `confidence`, `warnings`, `artifacts`.
> - Impact-centered H.264 review clip plus full annotated MP4, graph gallery, JSON viewer/link, CSV preview/link.
> - Synthetic-only backend, frontend, and Playwright/manual-QA verification.
> **Effort**: Large
> **Parallel**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4 → Tasks 5-8 → Task 9

## Context

### Original Request
"Ok, then do a complete plan for the whole web ui flow, uploading a wall video, calibrating, receiving results with graphs, video crops, ball velocity information, court position information and so on"

### Existing System Facts
- Existing normal serve web app is in `web/` and already supports upload, `/api/analyze`, `/api/job`, detector selection, clips, annotation mode, and `/clips/{filename}`.
- Existing wall pipeline is implemented under `serve_analyzer/wall_*.py` and produces JSON/CSV, annotated MP4, and plots.
- `serve_analyzer/wall_outputs.py` defines the exact `WallAnalysisResult` contract with top-level keys: `measured`, `inferred`, `assumed`, `confidence`, `warnings`, `artifacts`.
- `serve_analyzer/wall_serve._process_video()` returns a `serve_row`, not the full six-section web result. The web adapter must read `result.json` and `result.csv` from the output directory after `_process_video()` completes.
- `serve_analyzer/wall_artifacts.py` generates full annotated MP4 and plot PNGs; impact review crops are not part of wall artifacts today and must be produced by a web-side ffmpeg helper following `web/backend/services/clip_service.py` style.

### Oracle + Metis Decisions Incorporated
- Upload and analysis are separate: users upload/stage a video before calibration, then analysis reuses that staged backend video. The browser must never calibrate one file and analyze another.
- Calibration preview uses direct video serving plus a frontend canvas overlay. No backend still-frame extraction endpoint in MVP.
- Artifact URLs are browser-safe relative URLs. The frontend must never consume local absolute filesystem paths from `result.json`.
- Artifact serving supports nested plot paths via `GET /api/wall/artifacts/{artifact_path:path}` with path traversal protection.
- Single global CPU/video job across regular serve and wall analysis. If either `/api/analyze` or `/api/wall/analyze` is active, the other returns HTTP 409.
- `POST /api/wall/job/reset` clears current wall job/video/artifacts only and preserves calibration. `DELETE /api/wall/calibration` clears calibration.
- Manual correction UI is out of MVP. The UI must show autonomous vs final impact information and a future correction note.
- Because the user explicitly asked for video crops, MVP includes an impact-centered review clip generated around `impact_time_sec`.

### Metis Review (gaps addressed)
- Added staged upload APIs and metadata before calibration.
- Added exact adapter rule: call `_process_video()`, then read `result.json` and `result.csv`.
- Added nested artifact URL normalization and traversal tests.
- Resolved reset semantics and global concurrency policy.
- Added JSON/CSV viewer/link requirements.
- Added impact-centered review clip and annotated-video seek behavior.
- Added synthetic-only QA assertions for six-section payload, numeric wall meters, artifacts, URLs, resets, and UI rendering.

## Work Objectives

### Core Objective
Provide a complete local web UI flow for wall serve analysis so a user can upload a wall-practice video, calibrate the wall in-browser, run analysis, and inspect all outputs without using the CLI.

### Deliverables
- Backend staged-video session and metadata APIs.
- Backend wall calibration persistence APIs.
- Backend wall analysis adapter and normalized result/artifact schema.
- Backend impact review clip generation.
- Frontend wall workflow mode with upload, calibration, analysis, polling, and results.
- Result dashboard with graphs, annotated full video, impact crop, velocity, wall impact, court projection, confidence/warnings, raw JSON, and CSV.
- Backend unittest/FastAPI tests, frontend Vitest tests, and Playwright/manual QA evidence.

### Definition of Done (agent-verifiable)
- `nix develop --command python -m unittest discover -s tests -p 'test_wall_web*.py' -v` passes.
- `nix develop --command python -m unittest discover -s tests -v` passes.
- `cd web && npm test -- --run` passes.
- `cd web && npm run build` passes.
- Playwright/manual QA evidence shows upload → calibration → analyze → results with synthetic wall video and no real-video dependency.
- A synthetic web analysis returns exactly `measured`, `inferred`, `assumed`, `confidence`, `warnings`, `artifacts` and non-null numeric `measured.wall_x_m` / `measured.wall_y_m`.
- Browser artifact URLs resolve for `result.json`, `result.csv`, annotated MP4, impact review clip, and nested `plots/*.png`.

### Must Have
- Backend-staged upload before calibration.
- Direct video playback for calibration frame selection.
- Minimum 4 wall calibration points with image pixel and wall meter coordinates.
- Metadata/contact assumptions required by wall analysis.
- Full six-section wall output rendered in UI.
- Impact-centered review clip around `impact_time_sec`.
- Annotated MP4 player with jump-to-impact control.
- Graph gallery for speed, trajectory/court landing, and wall impact plots.
- Raw JSON viewer/link and CSV preview/link.
- Relative URL normalization for artifacts.
- Path traversal protection for artifact serving.

### Must NOT Have
- No DB, auth, deployment, cloud storage, queue service, WebSocket implementation, or remote multi-user session.
- No manual correction UI in this plan.
- No real wall videos in tests.
- No frontend consumption of absolute filesystem paths.
- No changes to the existing normal serve UX except adding a third wall mode and shared global-busy handling.
- No pytest migration; project uses `unittest` for Python.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after, using Python `unittest`, FastAPI `TestClient`, Vitest, and Playwright/manual QA.
- QA policy: every task includes happy-path and failure/edge-case scenarios.
- Evidence: `.sisyphus/evidence/web-wall-task-{N}-{slug}.{txt|png|json}`.

## Execution Strategy

### Parallel Execution Waves
- **Wave 1**: Tasks 1-4 backend/session foundations.
- **Wave 2**: Tasks 5-8 frontend workflow and result UI, using Wave 1 contracts.
- **Wave 3**: Task 9 full integration/QA hardening.

### Dependency Matrix
- Task 1 blocks Tasks 2, 3, 5, 7.
- Task 2 blocks Tasks 3, 6, 7.
- Task 3 blocks Tasks 4, 7, 8, 9.
- Task 4 blocks Tasks 8, 9.
- Tasks 5 and 6 block Task 7.
- Tasks 7 and 8 block Task 9.
- Task 9 blocks final verification wave.

### Agent Dispatch Summary
- Wave 1: 4 tasks — business-logic/backend, quick/service, security-aware route review.
- Wave 2: 4 tasks — visual-engineering/frontend + business-logic hooks.
- Wave 3: 1 task — unspecified-high QA/integration.

## TODOs

- [x] 1. Backend wall video session APIs and global busy guard

  **What to do**: Add wall video session state under `web/backend/`, with routes `POST /api/wall/video`, `GET /api/wall/video/{video_id}`, `GET /api/wall/video/{video_id}/metadata`, and `POST /api/wall/job/reset`. Reuse existing upload validation from `web/backend/app.py` (`ALLOWED_TYPES`, temp output patterns) and store staged video under a wall-specific temp directory. Return `video_id`, `video_url`, filename, duration, fps, frame_count, width, height. Add shared global-busy helper so `/api/analyze` and `/api/wall/analyze` mutually reject active CPU/video jobs with HTTP 409. Job reset deletes staged video, job state, and artifacts but preserves calibration.
  **Must NOT do**: Do not run wall analysis here. Do not clear calibration on job reset. Do not introduce DB/auth/queue.

  **Recommended Agent Profile**:
  - Category: `business-logic` - backend state/routes and concurrency contract.
  - Skills: []
  - Omitted: [`developing-containers`] - no deployment/container work.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2,3,5,7 | Blocked By: none

  **References**:
  - Pattern: `web/backend/app.py` - existing upload validation, job route style, background job patterns.
  - Pattern: `web/backend/state.py` - existing job state conventions.
  - Pattern: `web/backend/schemas.py` - Pydantic response model style.

  **Acceptance Criteria**:
  - [ ] `POST /api/wall/video` rejects unsupported MIME/extensions with existing error style.
  - [ ] Successful synthetic MP4 upload returns `video_id`, relative `video_url`, and numeric metadata.
  - [ ] `GET /api/wall/video/{video_id}` streams the staged file and rejects unknown IDs.
  - [ ] `POST /api/wall/job/reset` deletes video/artifacts/job state but leaves calibration state untouched.
  - [ ] Starting regular serve analysis while wall analysis is active, or wall analysis while regular serve analysis is active, returns HTTP 409.

  **QA Scenarios**:
  ```
  Scenario: Stage synthetic wall video for calibration
    Tool: Bash
    Steps: Run FastAPI TestClient test uploading tests synthetic MP4 to POST /api/wall/video, then GET returned video_url and metadata endpoint.
    Expected: 200 responses, metadata dimensions/fps/frame_count > 0, video bytes non-empty.
    Evidence: .sisyphus/evidence/web-wall-task-1-video-session.txt

  Scenario: Reject path/state misuse
    Tool: Bash
    Steps: Upload .txt, request unknown video_id, call job reset, then GET old video_id.
    Expected: 400/404 responses as appropriate; calibration fixture remains present after job reset.
    Evidence: .sisyphus/evidence/web-wall-task-1-errors.txt
  ```

  **Commit**: YES | Message: `feat(web): add wall video session endpoints` | Files: `web/backend/*`, `tests/test_wall_web_session.py`

- [x] 2. Wall calibration persistence APIs

  **What to do**: Add `POST /api/wall/calibration`, `GET /api/wall/calibration`, and `DELETE /api/wall/calibration`. Payload must include `video_id`, `calibration_frame`, `calibration_time_sec`, at least 4 points each with image pixel `{x,y}` and wall meters `{x_m,y_m}`, plus required metadata/contact assumptions. Persist as process-session JSON in a wall-specific backend temp/state location. Validate via `WallCalibration.from_dict()` and return RMS/point count where available.
  **Must NOT do**: Do not require calibration to be tied to a real file path exposed to the browser. Do not delete calibration during job reset.

  **Recommended Agent Profile**:
  - Category: `business-logic` - schema validation and persistence.
  - Skills: []
  - Omitted: [`auditing-security`] - route security is limited and covered by task acceptance.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3,6,7 | Blocked By: 1

  **References**:
  - API/Type: `serve_analyzer/wall_calibration.py` - calibration schema and validation.
  - Pattern: `tests/test_wall_calibration_cli.py` - valid/invalid calibration fixture shapes.

  **Acceptance Criteria**:
  - [ ] POST with 4 valid points persists calibration and GET returns same values.
  - [ ] POST with fewer than 4 points returns 422/400 with actionable message.
  - [ ] DELETE clears calibration; job reset does not.
  - [ ] Calibration includes frame/time used by frontend overlay.

  **QA Scenarios**:
  ```
  Scenario: Save and reuse calibration
    Tool: Bash
    Steps: Upload video, POST 4-point calibration, GET calibration, reset job, GET calibration again.
    Expected: Calibration survives job reset with same frame/time and point count.
    Evidence: .sisyphus/evidence/web-wall-task-2-calibration.txt

  Scenario: Clear calibration explicitly
    Tool: Bash
    Steps: DELETE /api/wall/calibration then GET /api/wall/calibration.
    Expected: DELETE 200/204; GET returns empty/not configured state.
    Evidence: .sisyphus/evidence/web-wall-task-2-reset.txt
  ```

  **Commit**: YES | Message: `feat(web): persist wall calibration state` | Files: `web/backend/*`, `tests/test_wall_web_calibration.py`

- [x] 3. Wall analysis adapter, normalized result contract, and nested artifacts

  **What to do**: Add `POST /api/wall/analyze` and `GET /api/wall/job`. The analyze route must require a staged `video_id` and saved calibration. It must call `serve_analyzer.wall_serve._process_video()` using the staged video, calibration JSON, and wall output dir; then read `result.json` and `result.csv` from disk to build the API response. Normalize artifact paths into relative browser URLs under `GET /api/wall/artifacts/{artifact_path:path}`. Preserve the exact top-level result keys: `measured`, `inferred`, `assumed`, `confidence`, `warnings`, `artifacts`. Add path traversal protection: resolved artifact path must stay under the current wall job output root.
  **Must NOT do**: Do not trust `_process_video()` return value for full JSON. Do not expose absolute filesystem paths. Do not flatten plots if nested serving is chosen.

  **Recommended Agent Profile**:
  - Category: `business-logic` - adapter and API contract.
  - Skills: []
  - Omitted: [`frontend-ui-ux`] - backend only.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4,7,8,9 | Blocked By: 1,2

  **References**:
  - API/Type: `serve_analyzer/wall_outputs.py` - exact six-section contract and wall-meter fields.
  - Pattern: `serve_analyzer/wall_serve.py` - `_process_video()` writes artifacts and returns only `serve_row`.
  - Pattern: `serve_analyzer/wall_artifacts.py` - annotated video and plot output layout.
  - Pattern: `web/backend/app.py` - route style and background-thread job handling.

  **Acceptance Criteria**:
  - [ ] Synthetic analysis response has exactly `measured`, `inferred`, `assumed`, `confidence`, `warnings`, `artifacts` at top level.
  - [ ] `measured.wall_x_m` and `measured.wall_y_m` are numbers for calibrated synthetic impact.
  - [ ] `result.json`, `result.csv`, annotated MP4, and all plot PNG artifact URLs are relative browser URLs.
  - [ ] Nested `plots/*.png` artifact URLs resolve.
  - [ ] `../` and encoded traversal artifact paths are rejected with 400/404.

  **QA Scenarios**:
  ```
  Scenario: Analyze staged calibrated synthetic video
    Tool: Bash
    Steps: TestClient uploads video, saves calibration, POSTs /api/wall/analyze, polls /api/wall/job until done, reads payload.
    Expected: Done payload has six exact sections, numeric wall meters, speed fields, landing fields, and relative artifact URLs.
    Evidence: .sisyphus/evidence/web-wall-task-3-analysis-contract.json

  Scenario: Serve nested plot safely
    Tool: Bash
    Steps: GET returned speed plot URL, then GET /api/wall/artifacts/../result.json and encoded traversal variant.
    Expected: Plot 200 image/png and non-zero bytes; traversal rejected.
    Evidence: .sisyphus/evidence/web-wall-task-3-artifacts.txt
  ```

  **Commit**: YES | Message: `feat(web): expose wall analysis API` | Files: `web/backend/*`, `tests/test_wall_web_analysis.py`

- [x] 4. Impact-centered review clip service

  **What to do**: Add a wall impact review clip generator under `web/backend/services/` using ffmpeg patterns from `clip_service.py`. After successful wall analysis, generate `impact_review.mp4` or `{video_stem}_impact_review.mp4` from the original staged video or annotated video. Use window `start_time_sec = max(impact_time_sec - 1.5, 0.0)` and `end_time_sec = min(impact_time_sec + 1.0, video_duration_sec)`. Encode browser-playable H.264/AAC or silent H.264 MP4 consistent with existing clip service. Add `artifacts.review_clip.url` and `artifacts.review` metadata: `impact_time_sec`, `impact_frame`, `start_time_sec`, `end_time_sec`, `duration_sec`.
  **Must NOT do**: Do not replace the full annotated MP4. Do not require manual editing/correction.

  **Recommended Agent Profile**:
  - Category: `business-logic` - media artifact generation.
  - Skills: []
  - Omitted: [`vision-tools`] - implementation uses ffmpeg/file checks, not visual analysis.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 8,9 | Blocked By: 3

  **References**:
  - Pattern: `web/backend/services/clip_service.py` - ffmpeg invocation and browser-compatible clips.
  - API/Type: `serve_analyzer/wall_outputs.py` - `measured.impact_time_sec` and `measured.impact_frame`.

  **Acceptance Criteria**:
  - [ ] Completed wall job includes review clip URL and review metadata.
  - [ ] Review clip exists, is non-zero, and OpenCV can read first frame.
  - [ ] Clip window clamps at video start/end.
  - [ ] If `impact_time_sec` is absent, API omits review clip and emits a non-fatal warning.

  **QA Scenarios**:
  ```
  Scenario: Generate impact review crop
    Tool: Bash
    Steps: Run synthetic wall analysis through API and inspect returned review clip with OpenCV.
    Expected: MP4 exists, first frame readable, metadata start <= impact <= end.
    Evidence: .sisyphus/evidence/web-wall-task-4-review-clip.txt

  Scenario: Clamp early impact window
    Tool: Bash
    Steps: Unit-test clip window calculation with impact_time_sec=0.4 and duration=2.0.
    Expected: start_time_sec=0.0, end_time_sec=1.4.
    Evidence: .sisyphus/evidence/web-wall-task-4-clamp.txt
  ```

  **Commit**: YES | Message: `feat(web): generate wall impact review clips` | Files: `web/backend/services/*`, `tests/test_wall_web_review_clip.py`

- [x] 5. Frontend wall mode and staged upload step

  **What to do**: Add a third mode to `web/src/App.tsx` (or equivalent mode state) named `Wall Analysis`. Build a wall workflow shell with steps: Upload, Calibrate, Configure, Analyze, Results. Upload step calls `POST /api/wall/video`, stores `video_id`, `video_url`, metadata, and displays filename/duration/fps/dimensions. Existing normal serve analysis and annotation modes must continue to work.
  **Must NOT do**: Do not mix wall staged upload with regular serve upload state. Do not start analysis on upload.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - UI flow and state.
  - Skills: [`frontend-ui-ux`] - polish workflow clarity.
  - Omitted: [`browser-debug-tools`] - only needed during QA.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 6,7 | Blocked By: 1

  **References**:
  - Pattern: `web/src/App.tsx` - current analysis/annotation mode state.
  - Pattern: `web/src/components/upload-dropzone.tsx` - existing upload UX.

  **Acceptance Criteria**:
  - [ ] User can switch between normal Analysis, Annotation, and Wall Analysis.
  - [ ] Wall upload shows metadata before calibration.
  - [ ] Upload errors render actionable messages.
  - [ ] Existing normal serve flow remains visually and functionally unchanged except for mode navigation.

  **QA Scenarios**:
  ```
  Scenario: Upload wall video in UI
    Tool: Playwright
    Steps: Open app, select Wall Analysis, upload synthetic video, wait for metadata cards.
    Expected: Upload step completes, video metadata visible, Calibrate step enabled.
    Evidence: .sisyphus/evidence/web-wall-task-5-upload.png

  Scenario: Reject invalid upload in UI
    Tool: Playwright
    Steps: Select Wall Analysis and upload invalid text file.
    Expected: Error message shown; Calibrate step remains disabled.
    Evidence: .sisyphus/evidence/web-wall-task-5-invalid.png
  ```

  **Commit**: YES | Message: `feat(web): add wall analysis upload flow` | Files: `web/src/*`, `web/src/components/*`

- [x] 6. Interactive calibration canvas and assumptions form

  **What to do**: Build `WallCalibrationCanvas` and wall assumptions form. Use the staged `video_url` in an HTML video element with scrub controls; overlay a canvas for point selection on the visible frame. Require at least 4 point rows, each pairing clicked image coordinates with wall coordinates in meters. Capture `calibration_frame`/`calibration_time_sec`. Include required wall metadata/contact assumptions needed by the wall pipeline, with clear labels and units. POST to `/api/wall/calibration`; show saved status and calibration RMS/point count.
  **Must NOT do**: Do not implement manual impact correction. Do not use only browser object URLs without backend `video_id`.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - interactive canvas UX.
  - Skills: [`frontend-ui-ux`] - clear calibration instructions and validation.
  - Omitted: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 2,5

  **References**:
  - API/Type: `serve_analyzer/wall_calibration.py` - required calibration point shape.
  - Pattern: `web/src/components/annotation-workspace.tsx` - video/canvas interaction concepts if useful.

  **Acceptance Criteria**:
  - [ ] Canvas click records image pixel coordinates in video coordinate space, not CSS-scaled coordinates.
  - [ ] User can edit wall meter coordinates for each point.
  - [ ] Save disabled until 4 valid points and required assumptions are present.
  - [ ] Saved calibration persists across job reset and reload through GET endpoint.
  - [ ] Delete calibration clears UI and backend state.

  **QA Scenarios**:
  ```
  Scenario: Save four-point calibration
    Tool: Playwright
    Steps: Upload synthetic video, scrub to frame, click four known corners, enter wall meters/assumptions, save.
    Expected: Saved badge appears; GET calibration shows 4 points and selected frame/time.
    Evidence: .sisyphus/evidence/web-wall-task-6-calibration.png

  Scenario: Validation blocks incomplete calibration
    Tool: Playwright
    Steps: Try saving with 3 points or missing contact height.
    Expected: Save disabled or validation message names missing input.
    Evidence: .sisyphus/evidence/web-wall-task-6-validation.png
  ```

  **Commit**: YES | Message: `feat(web): add wall calibration UI` | Files: `web/src/components/*`, `web/src/*`, `web/src/**/*.test.*`

- [x] 7. Wall analyze trigger, polling hook, and error states

  **What to do**: Add frontend API client/hook for `POST /api/wall/analyze` and `GET /api/wall/job`. Analysis starts only when `video_id` and saved calibration exist. Show progress phases: `idle`, `uploaded`, `calibrated`, `queued`, `analyzing`, `artifacting`, `done`, `error`. Handle HTTP 409 global-busy responses with a clear message that another analysis is active. Preserve final payload in state for results dashboard.
  **Must NOT do**: Do not re-upload the video during analysis. Do not poll normal `/api/job` for wall state.

  **Recommended Agent Profile**:
  - Category: `business-logic` - frontend state and API contract.
  - Skills: []
  - Omitted: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8,9 | Blocked By: 3,5,6

  **References**:
  - Pattern: `web/src/App.tsx` - existing polling for normal analysis.
  - API/Type: backend schemas from Tasks 1-3.

  **Acceptance Criteria**:
  - [ ] Analyze button disabled until upload and calibration are complete.
  - [ ] Polling stops on `done` or `error`.
  - [ ] 409 global-busy response is displayed and does not clear current calibration/video state.
  - [ ] Error states expose backend error message and allow retry after reset.

  **QA Scenarios**:
  ```
  Scenario: Run wall analysis from UI
    Tool: Playwright
    Steps: Complete upload/calibration, click Analyze, wait for done.
    Expected: Progress reaches Results and payload stored.
    Evidence: .sisyphus/evidence/web-wall-task-7-polling.png

  Scenario: Global busy handling
    Tool: Bash + Playwright
    Steps: Force backend active job state, click Wall Analyze.
    Expected: UI shows 409 busy message and keeps staged video/calibration.
    Evidence: .sisyphus/evidence/web-wall-task-7-busy.txt
  ```

  **Commit**: YES | Message: `feat(web): run wall analysis from UI` | Files: `web/src/*`, `web/src/lib/*`, `web/src/**/*.test.*`

- [ ] 8. Wall results dashboard with graphs, video crop, velocity, and court position

  **What to do**: Build `WallResultsDashboard` that renders the full six-section contract. Required panels: measured impact (`impact_time_sec`, `impact_frame`, `impact_pixel`, `wall_x_m`, `wall_y_m`, RMS, raw sample summary), velocity (`speed_m_s`, `speed_km_h`, `speed_mph`, uncertainty), court projection (`landing_x_m`, `landing_z_m`, `in_service_box`, `service_box_side`, sensitivities), assumptions, confidence, warnings, artifact links, raw JSON viewer/link, CSV preview/link. Add full annotated MP4 player with “Jump to impact” seeking to `impact_time_sec`. Add impact review crop player for `review_clip.url`. Add graph gallery for speed, trajectory/court landing, and wall impact plots; every image must have alt text and loading/error states. Show autonomous vs final impact fields and an out-of-MVP note for future manual correction if detection is wrong.
  **Must NOT do**: Do not hide warnings/confidence behind raw JSON only. Do not require a user to inspect local files outside browser.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - rich results UI.
  - Skills: [`frontend-ui-ux`] - dashboard hierarchy and media presentation.
  - Omitted: []

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 9 | Blocked By: 3,4,7

  **References**:
  - API/Type: `serve_analyzer/wall_outputs.py` - exact result fields.
  - Pattern: `web/src/components/video-player.tsx` - native video controls/metadata card style.
  - Pattern: existing normal serve result/timeline components in `web/src/`.

  **Acceptance Criteria**:
  - [ ] UI renders all six top-level result sections.
  - [ ] Velocity shown in m/s, km/h, and mph with uncertainty.
  - [ ] Wall position and equivalent court landing shown with units and service-box status.
  - [ ] Annotated video and impact review clip both play in browser.
  - [ ] “Jump to impact” seeks annotated video to `impact_time_sec` within ±0.25s.
  - [ ] JSON link/viewer and CSV link/preview are visible and load from relative artifact URLs.
  - [ ] Plot gallery renders nested plot artifact URLs.

  **QA Scenarios**:
  ```
  Scenario: Review complete wall results
    Tool: Playwright
    Steps: Load completed synthetic result, inspect dashboard fields, click Jump to impact, open JSON/CSV links, verify plots visible.
    Expected: All required panels present; video currentTime near impact; artifact links return 200.
    Evidence: .sisyphus/evidence/web-wall-task-8-dashboard.png

  Scenario: Warnings and missing optional artifacts
    Tool: Vitest
    Steps: Render dashboard with warning payload and missing optional review_clip.
    Expected: Warnings visible; optional artifact missing state is graceful; required fields still render.
    Evidence: .sisyphus/evidence/web-wall-task-8-edge.txt
  ```

  **Commit**: YES | Message: `feat(web): display wall analysis results` | Files: `web/src/components/*`, `web/src/*`, `web/src/**/*.test.*`

- [ ] 9. End-to-end synthetic integration and regression QA

  **What to do**: Add complete backend and frontend integration coverage for the whole wall web flow. Use synthetic wall fixtures/helpers only. Verify full commands: Python wall web tests, full Python suite, Vitest, web build, and Playwright/manual QA. Include evidence files under `.sisyphus/evidence/`. Update no project docs unless needed by implementation; if docs are changed, keep them scoped to web wall usage.
  **Must NOT do**: Do not introduce real video test dependencies. Do not mark final verification tasks complete before reviewers approve.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - cross-stack QA and regression hardening.
  - Skills: [`browser-debug-tools`] - Playwright/manual UI verification.
  - Omitted: []

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: F1-F4 | Blocked By: 1,2,3,4,5,6,7,8

  **References**:
  - Test: `tests/wall_test_helpers.py` - synthetic wall video/helper patterns.
  - Test: `tests/test_wall_cli.py` - output artifact contract expectations.
  - Command: `python -m unittest discover -s tests -v` - Python suite.
  - Command: `cd web && npm test -- --run`; `cd web && npm run build` - frontend checks.

  **Acceptance Criteria**:
  - [ ] Backend synthetic flow covers upload, calibration, analyze, job polling, artifact URLs, reset semantics.
  - [ ] Frontend tests cover mode switching, upload, calibration validation, analyze disabled/enabled, result rendering.
  - [ ] Playwright/manual QA completes Upload → Calibrate → Analyze → Results and captures screenshots/evidence.
  - [ ] Every required artifact exists on disk and resolves through browser URL.
  - [ ] No absolute filesystem paths appear in frontend-visible artifact URLs.
  - [ ] Full Python and web checks pass.

  **QA Scenarios**:
  ```
  Scenario: Full wall web flow
    Tool: Playwright
    Steps: Start backend/frontend, upload synthetic wall video, place four calibration points, enter assumptions, analyze, inspect dashboard.
    Expected: Results show numeric wall meters, speed, court landing, plots, annotated video, impact crop, JSON/CSV links, warnings/confidence.
    Evidence: .sisyphus/evidence/web-wall-task-9-e2e.png

  Scenario: Full regression commands
    Tool: Bash
    Steps: Run Python wall web tests, full Python unittest suite, Vitest, and web build.
    Expected: All commands exit 0.
    Evidence: .sisyphus/evidence/web-wall-task-9-regression.txt
  ```

  **Commit**: YES | Message: `test(web): cover wall analysis flow` | Files: `tests/*`, `web/src/**/*.test.*`, `.sisyphus/evidence/*`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ Playwright full wall flow and media/artifact inspection)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit after each task with the message specified in the task.
- Do not combine backend session/calibration/analysis/result UI commits unless implementation is inseparable.
- Do not push unless explicitly requested.

## Success Criteria
- A local user can open the web app, choose Wall Analysis, upload a wall video, calibrate on that exact uploaded video, run analysis, and review all wall analysis outputs in-browser.
- The web payload matches the CLI wall result contract and avoids the prior output-contract failure mode: wall-meter fields and artifact paths are verified by tests and manual QA.
- Existing normal serve analysis and annotation modes still pass their tests/build and remain usable.
