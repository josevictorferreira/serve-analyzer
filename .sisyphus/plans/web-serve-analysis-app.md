# Web Serve Analysis App

## TL;DR
> **Summary**: Build a local-only web stack under `web/` with a React/TypeScript/Vite/Tailwind/shadcn frontend and a FastAPI-backed Python runner that uploads one video, runs the existing serve detector, generates short per-serve MP4 clips, and renders them in a bottom timeline player.
> **Deliverables**:
> - `web/` frontend app and dev/build tooling
> - `web/backend/` local API runner that wraps existing Python analysis
> - single-job upload/progress/results flow
> - H.264 serve clip generation and static serving
> - automated tests + agent-executed QA evidence
> **Effort**: Large
> **Parallel**: YES - 2 waves
> **Critical Path**: 1 → 4 → 5 → 7

## Context
### Original Request
Create a React/TypeScript/Vite/Tailwind/shadcn web app in a new `web` directory. MVP: drag-and-drop a video, process it with the latest working serve analysis, then show all cropped serves as their own small videos in a bottom timeline where clicking plays the selected clip.

### Interview Summary
- Runtime: local web UI + local Python API/process runner
- Scope: single-job only; no queue UI; no remote deployment
- Review UI: serve clips only; no full original-video player in MVP
- Exports: none; in-app review only
- Artifact lifecycle: temporary per session
- Test strategy: TDD where practical + mandatory agent QA
- Default applied: interpret “cropped serves” as short derived serve clips from the original video timeline, scaled down for lightweight playback; no spatial player/ball crop in MVP

### Metis Review (gaps addressed)
- Chosen backend framework: FastAPI for local API + TestClient support
- Chosen frontend package manager: npm
- Chosen clip strategy: ffmpeg-generated H.264 MP4 clips, because browser-safe output is mandatory and `ffmpeg` already exists in `flake.nix`
- Guardrails: no DB, no queue system, no auth, no multi-user state, no remote deployment assumptions
- Progress model: coarse phases (`idle/uploading/analyzing/clipping/done/error`) rather than fragile frame-level percentages

## Work Objectives
### Core Objective
Deliver a runnable local MVP that lets a user upload one tennis-serve video, runs the existing serve detector stack, creates one short MP4 clip per detected serve, and reviews clips via a clickable bottom timeline.

### Deliverables
- `web/` Vite React TypeScript app with Tailwind + shadcn baseline
- `web/backend/` FastAPI service importing existing `serve_analyzer` analysis modules
- multipart upload endpoint, status endpoint, static clip serving
- serve-analysis adapter using current video-only detector flow
- clip-extraction service producing browser-playable MP4 clips
- tests for backend contracts and frontend UI states

### Definition of Done (verifiable conditions with commands)
- `nix develop --command bash -lc "node --version && npm --version && python -c 'import fastapi, uvicorn'"`
- `nix develop --command bash -lc "python -m unittest discover -s tests -v"`
- `nix develop --command bash -lc "cd web && npm test -- --run"`
- `nix develop --command bash -lc "cd web && npm run build"`
- `nix develop --command bash -lc "python -m web.backend.app >/tmp/serve-web-api.log 2>&1 & API_PID=$!; sleep 3; curl -sf http://127.0.0.1:8000/api/job; kill $API_PID"`
- `nix develop --command bash -lc "python -m web.backend.app >/tmp/serve-web-api.log 2>&1 & API_PID=$!; sleep 3; curl -sf -F video=@tests/fixtures/web_sample.mp4 http://127.0.0.1:8000/api/analyze; kill $API_PID"`

### Must Have
- all new web-stack files live under `web/`
- backend reuses existing repo analysis logic instead of reimplementing detector heuristics
- backend rejects concurrent upload while a job is active with HTTP 409
- serve clips are temporary artifacts cleaned on startup and on job reset
- clip playback works in Chromium via native `<video>` playback

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- no database, Redis, Celery, websockets, login, or multi-user session tracking
- no remote deployment/hosting work in MVP
- no export/download UI
- no spatial auto-cropping around player/ball in MVP
- no rewrites of core detector heuristics unless needed to expose adapter inputs/outputs cleanly
- no package-management drift outside Nix + npm inside `web/`

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: TDD where practical; Python backend tests with `unittest` + FastAPI `TestClient`, frontend tests with Vitest + React Testing Library
- QA policy: Every task has agent-executed scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Task 1 tooling baseline, Task 2 backend skeleton/contracts, Task 3 frontend scaffold/design system
Wave 2: Task 4 analysis adapter, Task 5 clip generation, Task 6 upload/progress UX, Task 7 results timeline/player

### Dependency Matrix (full, all tasks)
| Task | Depends On | Blocks |
|---|---|---|
| 1 | - | 2,3,4,5,6,7 |
| 2 | 1 | 4,6 |
| 3 | 1 | 6,7 |
| 4 | 1,2 | 5,6,7 |
| 5 | 1,2,4 | 7 |
| 6 | 1,2,3,4 | 7 |
| 7 | 3,5,6 | F1-F4 |

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → business-logic, visual-engineering, general
- Wave 2 → 4 tasks → business-logic, general, visual-engineering
- Final verification → 4 tasks → oracle, unspecified-high, deep

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Expand Nix/tooling baseline for the web stack

  **What to do**: Update `flake.nix` to provision Node.js and npm alongside the existing Python/ffmpeg environment. Keep current Python shell behavior intact. Add only the Python API dependencies needed for the local backend (`fastapi`, `uvicorn`, `python-multipart`, and any small helper deps chosen for schemas/static serving). Ensure `nix develop` is sufficient for both Python and web workflows, with no extra package manager outside npm in `web/`.
  **Must NOT do**: Do not remove existing `.venv` shell behavior for ultralytics. Do not add `pip install` instructions outside the shell hook. Do not introduce pnpm/yarn/bun.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: dependency and environment decisions affect all downstream tasks
  - Skills: [`developing-containers`] - why needed: helps reason about local dev/runtime environment shape even without containerizing
  - Omitted: [`writing-nix-code`] - why not needed: change is small and localized; no module-design work required

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2,3,4,5,6,7 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `flake.nix:14-37` - current dev shell packages Python env + `ffmpeg`; extend this instead of introducing a second environment path
  - Pattern: `flake.nix:38-65` - current shell hook manages `.venv` and ultralytics installs; preserve behavior
  - External: `https://github.com/vitejs/vite/blob/04f974fbae1df6b7b6423d9083130579cdfa06ef/packages/create-vite/template-react-ts/vite.config.ts` - keep generated Vite stack aligned with standard React TS setup

  **Acceptance Criteria** (agent-executable only):
  - [ ] `nix develop --command bash -lc "node --version && npm --version"` exits 0
  - [ ] `nix develop --command bash -lc "python -c 'import fastapi, uvicorn, multipart'"` exits 0
  - [ ] `nix develop --command bash -lc "python -c 'import ultralytics'"` still exits 0 after shell init

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Dev shell exposes web and api tooling
    Tool: Bash
    Steps: run `nix develop --command bash -lc "node --version && npm --version && python -c 'import fastapi, uvicorn, multipart'"`
    Expected: command exits 0 and prints both node/npm versions
    Evidence: .sisyphus/evidence/task-1-tooling.txt

  Scenario: Existing YOLO shell behavior still works
    Tool: Bash
    Steps: run `nix develop --command bash -lc "python -c 'import ultralytics'"`
    Expected: command exits 0 with no ImportError
    Evidence: .sisyphus/evidence/task-1-tooling-yolo.txt
  ```

  **Commit**: YES | Message: `feat(web): expand dev shell for local web stack` | Files: [`flake.nix`]

- [x] 2. Create the local FastAPI backend skeleton under `web/backend`

  **What to do**: Create `web/backend/` as a Python package with `__init__.py`, `app.py`, `schemas.py`, `state.py`, and `paths.py`. Expose a single-process FastAPI app runnable as `python -m web.backend.app`. Define the MVP HTTP contract now: `POST /api/analyze` accepts one multipart `video`; `GET /api/job` returns the current single-job snapshot; `POST /api/job/reset` clears temp artifacts and returns the backend to `idle`; `GET /clips/{filename}` serves generated clips from the temp output directory. Enforce single-job behavior: if status is `uploading|analyzing|clipping`, reject a new analyze request with 409.
  **Must NOT do**: Do not put backend code in `serve_analyzer/`; keep new web-specific orchestration inside `web/backend/`. Do not add DB persistence or historical job lists.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: API contract and state machine are core product behavior
  - Skills: [] - why needed: standard FastAPI/task-state work
  - Omitted: [`developing-rails-background-jobs`] - why not needed: not a Rails app and no durable queueing

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4,6 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `serve_analyzer/serve_attempts.py:424-539` - current CLI already supports video-only output shape; backend contract should align with `selected_serves` and `candidates`
  - Pattern: `serve_analyzer/serve_attempts.py:505-528` - use these result keys as the base response model for analysis results
  - Pattern: `serve_analyzer/analysis.py:1059-1086` - reuse for source video metadata in API payloads where useful
  - External: `https://github.com/tiangolo/fastapi/blob/94643c3b8516928e4cc7fad99912272670a0a990/docs/en/docs/tutorial/background-tasks.md` - background task pattern for non-blocking local processing
  - External: `https://github.com/vitejs/vite/blob/04f974fbae1df6b7b6423d9083130579cdfa06ef/docs/config/server-options.md` - dev server proxy expectations for `/api`

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_web_api_contract -v` passes
  - [ ] `nix develop --command bash -lc "python -m web.backend.app >/tmp/serve-web-api.log 2>&1 & API_PID=$!; sleep 3; curl -sf http://127.0.0.1:8000/api/job; kill $API_PID"` returns JSON with `status`
  - [ ] a second `POST /api/analyze` while the job is active returns HTTP 409 in test coverage

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Idle backend exposes single-job status contract
    Tool: Bash
    Steps: start `python -m web.backend.app`; curl `http://127.0.0.1:8000/api/job`
    Expected: HTTP 200 JSON containing `status`, `phase`, `error`, `clips`, `selected_serves`
    Evidence: .sisyphus/evidence/task-2-api-status.json

  Scenario: Concurrent upload is rejected
    Tool: Bash
    Steps: in automated test, force state to `analyzing`; POST `/api/analyze` with multipart video using TestClient
    Expected: HTTP 409 JSON error payload
    Evidence: .sisyphus/evidence/task-2-api-conflict.txt
  ```

  **Commit**: YES | Message: `feat(web): add local fastapi backend skeleton` | Files: [`web/backend/__init__.py`, `web/backend/app.py`, `web/backend/schemas.py`, `web/backend/state.py`, `web/backend/paths.py`, `tests/test_web_api_contract.py`]

- [x] 3. Scaffold the Vite/React/Tailwind/shadcn frontend in `web/`

  **What to do**: Create a Vite React TypeScript app in `web/`, wire Tailwind CSS, initialize shadcn/ui, and establish the base app shell. Use npm. Add a single-column layout with: top header/title, centered drag-and-drop upload card, progress/status region, and a bottom fixed timeline strip reserved for result clips. Configure Vite dev proxy so frontend calls `/api/*` without hardcoded backend hosts. Include only the minimum shadcn components needed for MVP (`button`, `card`, `progress`, `scroll-area`, `skeleton`, `sonner` or equivalent toast primitive).
  **Must NOT do**: Do not add routing, auth, theme switching, or unnecessary component libraries. Do not hardcode absolute backend URLs in app code.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - Reason: this is UI scaffold + Tailwind/shadcn composition work
  - Skills: [`frontend-ui-ux`] - why needed: ensures the timeline-first MVP layout is clean without overbuilding
  - Omitted: [`browser-debug-tools`] - why not needed: initial scaffold does not require browser automation

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6,7 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - External: `https://github.com/vitejs/vite/blob/04f974fbae1df6b7b6423d9083130579cdfa06ef/packages/create-vite/template-react-ts/src/App.tsx` - baseline Vite React TS app structure
  - External: `https://github.com/vitejs/vite/blob/04f974fbae1df6b7b6423d9083130579cdfa06ef/docs/config/server-options.md` - configure dev proxy for `/api`
  - Pattern: `Context > Original Request` - maintain the requested UX shape: drag-drop upload + bottom timeline clip review

  **Acceptance Criteria** (agent-executable only):
  - [ ] `nix develop --command bash -lc "cd web && npm install"` exits 0
  - [ ] `nix develop --command bash -lc "cd web && npm test -- --run"` passes initial UI tests
  - [ ] `nix develop --command bash -lc "cd web && npm run build"` exits 0

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Frontend build succeeds with proxy config present
    Tool: Bash
    Steps: run `nix develop --command bash -lc "cd web && npm run build"`
    Expected: build exits 0 and emits production assets
    Evidence: .sisyphus/evidence/task-3-web-build.txt

  Scenario: Empty app state renders MVP scaffold
    Tool: Playwright
    Steps: start Vite app; open app root; verify title/header, upload dropzone, disabled timeline placeholder
    Expected: all core empty-state elements visible; no results shown yet
    Evidence: .sisyphus/evidence/task-3-empty-state.png
  ```

  **Commit**: YES | Message: `feat(web): scaffold vite app with tailwind and shadcn` | Files: [`web/package.json`, `web/vite.config.ts`, `web/src/App.tsx`, `web/src/components/*`, `web/src/lib/*`, `web/index.html`, `web/tailwind.config.*`, `web/postcss.config.*`, `web/components.json`, `web/src/**/*.test.tsx`]

- [x] 4. Implement the serve-analysis adapter and single-job runner

  **What to do**: In `web/backend/`, add a service module that imports the existing detector stack directly instead of shelling out. Use `serve_analyzer.serve_attempts.detect_serve_candidates()` plus `serve_analyzer.serve_attempts.select_serves()` for video-only detection, preserving `expected_serves=None` semantics for inferred count. Normalize the result into one internal job payload containing: source video metadata, `candidates`, `selected_serves`, inferred count, phase/status, and a future `clips` list. Add a small progress callback/state updater with coarse phases only: `uploading`, `analyzing`, `clipping`, `done`, `error`. Also create a tiny deterministic MP4 fixture generator for backend tests if no committed fixture exists.
  **Must NOT do**: Do not use timestamp/evaluator flow. Do not move timestamp parsing into backend MVP. Do not fork detector heuristics into new files.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: this task binds existing detector semantics to new API contracts
  - Skills: [] - why needed: direct Python integration task
  - Omitted: [`developing-rspec-tests`] - why not needed: non-Ruby repo

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 5,6,7 | Blocked By: 1,2

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `serve_analyzer/serve_attempts.py:66-139` - detector candidate pipeline entry point
  - Pattern: `serve_analyzer/serve_attempts.py:216-380` - `select_serves` preserves autonomous-count semantics and ranking behavior
  - Pattern: `serve_analyzer/serve_attempts.py:505-528` - exact video-only result payload to mirror in backend normalization
  - Pattern: `serve_analyzer/multi_serve.py:66-139` - underlying YOLO + video metadata behavior that adapter indirectly relies on
  - Pattern: `tests/test_serve_attempts.py:243-308` - detector CLI tests document video-only contract and result-shape expectations
  - Pattern: `tests/test_serve_attempts.py:316-359` - selector quality contract must remain intact

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_web_analysis_service -v` passes
  - [ ] backend tests cover `expected_serves=None` producing `count_inferred=true`
  - [ ] backend tests prove API result payload includes `selected_serves` and `candidates` without evaluator keys

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Backend adapter preserves detector-only schema
    Tool: Bash
    Steps: run `python -m unittest tests.test_web_analysis_service.TestAnalysisServiceShape -v`
    Expected: tests pass and assert absence of `matched`, `target_time_sec`, `delta_sec`, `serve_number` in detector payload
    Evidence: .sisyphus/evidence/task-4-analysis-shape.txt

  Scenario: Autonomous count mode survives API wrapper
    Tool: Bash
    Steps: run `python -m unittest tests.test_web_analysis_service.TestAnalysisServiceCountInference -v`
    Expected: tests pass and payload marks `count_inferred` true when no expected count is forced
    Evidence: .sisyphus/evidence/task-4-analysis-count.txt
  ```

  **Commit**: YES | Message: `feat(web): wrap existing serve detector for api use` | Files: [`web/backend/services/analysis_service.py`, `web/backend/state.py`, `tests/test_web_analysis_service.py`, `tests/fixtures/web_sample.mp4` or generated-test helper]

- [x] 5. Generate temporary H.264 serve clips from selected serves

  **What to do**: Implement `web/backend/services/clip_service.py` that turns each selected serve into one short MP4 clip. Use each selected serve’s `contact_time_sec` to define a deterministic clip window: start at `max(contact_time_sec - 1.5, 0)` and end at `contact_time_sec + 1.0`. Use `ffmpeg` to extract the time range, scale to width 480 preserving aspect ratio, drop audio, encode with H.264 (`libx264`) and `yuv420p`, and write files into the temp clip directory. Name files predictably (`serve-01.mp4`, etc.). Return clip metadata with filename, URL path, serve index, contact time, and duration.
  **Must NOT do**: Do not implement spatial crop tracking in MVP. Do not rely on OpenCV `VideoWriter` for final web-playback clips. Do not keep clips after reset/startup cleanup.

  **Recommended Agent Profile**:
  - Category: `general` - Reason: subprocess orchestration + metadata shaping + temp file lifecycle
  - Skills: [] - why needed: standard Python/ffmpeg integration
  - Omitted: [`developing-containers`] - why not needed: no container/runtime packaging in this task

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 1,2,4

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `flake.nix:32-37` - `ffmpeg` is already in the shell; build on this rather than adding a new encoder path
  - Pattern: `serve_analyzer/analysis.py:1089-1139` - existing code already prefers ffmpeg/H.264 for browser-safe MP4 output; follow that rationale
  - Pattern: `serve_analyzer/serve_attempts.py:519-528` - selected serves already expose the timing data needed for clip windows
  - External: `https://github.com/FFmpeg/FFmpeg/blob/n7.1.1/doc/ffmpeg.texi` - use stable ffmpeg CLI behavior for trim/scale/mp4 encoding

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_web_clip_service -v` passes
  - [ ] clip service tests verify produced metadata uses `/clips/<filename>` URLs
  - [ ] integration test verifies generated MP4 file exists and has non-zero size

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Clip generation produces browser-safe MP4 artifacts
    Tool: Bash
    Steps: run `python -m unittest tests.test_web_clip_service.TestClipGeneration -v`
    Expected: tests pass and output clip files exist with `.mp4` extension and non-zero bytes
    Evidence: .sisyphus/evidence/task-5-clip-generation.txt

  Scenario: Reset removes temporary clips
    Tool: Bash
    Steps: run `python -m unittest tests.test_web_clip_service.TestClipCleanup -v`
    Expected: tests pass and temp clip directory is empty after reset/cleanup
    Evidence: .sisyphus/evidence/task-5-clip-cleanup.txt
  ```

  **Commit**: YES | Message: `feat(web): generate per-serve preview clips` | Files: [`web/backend/services/clip_service.py`, `web/backend/paths.py`, `tests/test_web_clip_service.py`]

- [x] 6. Wire frontend upload, status polling, and error handling

  **What to do**: Implement the frontend app state around the MVP flow. Add a drag-and-drop zone plus standard file picker fallback. Use `XMLHttpRequest` for upload progress because browser upload progress is required. On successful upload start polling `GET /api/job` every 2 seconds until `done` or `error`. Disable the dropzone while a job is active. Surface coarse backend phases and friendly errors (`busy`, invalid file, backend failure). Keep API calls as relative `/api/*` paths. Add tests for empty, uploading, analyzing, error, and done states.
  **Must NOT do**: Do not use websockets/SSE for MVP. Do not introduce global state libraries. Do not allow a second file while the current job is active.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - Reason: user-facing async state/UI interaction work
  - Skills: [`frontend-ui-ux`] - why needed: upload/progress state must stay clean and readable
  - Omitted: [`browser-debug-tools`] - why not needed: unit/integration tests should cover most behavior before browser debugging

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 1,2,3,4

  **References** (executor has NO interview context - be exhaustive):
  - External: `https://github.com/facebook/react/blob/0338278823e879f0b1ecd457aa668be8290f7838/packages/react/src/ReactHooks.js` - use core hooks only (`useState`, `useEffect`, `useRef`) for MVP state
  - External: `https://github.com/vitejs/vite/blob/04f974fbae1df6b7b6423d9083130579cdfa06ef/docs/config/server-options.md` - dev proxy keeps API paths relative
  - Pattern: `web/backend/app.py` - poll only the implemented single-job endpoints; do not invent parallel-job APIs

  **Acceptance Criteria** (agent-executable only):
  - [ ] `nix develop --command bash -lc "cd web && npm test -- --run upload-flow"` passes or equivalent targeted test command
  - [ ] frontend tests verify upload progress rendering from XHR events
  - [ ] frontend tests verify polling stops on `done` and `error`

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Upload and polling states transition correctly
    Tool: Playwright
    Steps: boot API + Vite app; upload test video through file input; observe uploading→analyzing states; wait for done
    Expected: dropzone disables during processing; status text and progress region update; no duplicate submission allowed
    Evidence: .sisyphus/evidence/task-6-upload-flow.png

  Scenario: Busy backend error is surfaced cleanly
    Tool: Playwright
    Steps: force backend to return 409 for `/api/analyze`; upload a file
    Expected: toast/error message appears and app returns to idle without broken state
    Evidence: .sisyphus/evidence/task-6-busy-error.png
  ```

  **Commit**: YES | Message: `feat(web): add upload and polling workflow` | Files: [`web/src/App.tsx`, `web/src/components/upload-dropzone.tsx`, `web/src/hooks/use-analysis-job.ts`, `web/src/lib/api.ts`, `web/src/**/*.test.tsx`]

- [x] 7. Render the serve timeline and selected clip player

  **What to do**: Build the result-review experience after processing completes. Render a primary player for the currently selected clip and a bottom horizontal timeline of serve cards/thumbnails underneath it. Each timeline item must show serve number, contact time, and duration. Clicking a card swaps the primary player source immediately. Auto-select the first clip after a successful run. Use native `<video controls preload="metadata">` for both preview and main playback. Keep the timeline scrollable and keyboard accessible. If no clips were generated, render a clear empty-result state.
  **Must NOT do**: Do not add the original full-video player. Do not autoplay all clips. Do not require manual page refresh after processing.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - Reason: clip review UX is the user-visible MVP outcome
  - Skills: [`frontend-ui-ux`] - why needed: timeline/player composition and empty states matter here
  - Omitted: [`vision-tools`] - why not needed: no screenshot analysis or visual OCR required

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: F1-F4 | Blocked By: 3,5,6

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `web/src/App.tsx` - preserve single-screen layout decided in Task 3
  - Pattern: `web/backend/services/clip_service.py` - consume returned clip metadata exactly; do not recompute clip timing in UI
  - External: `https://github.com/whatwg/html/blob/main/source#the-video-element` - native video element semantics for playback controls

  **Acceptance Criteria** (agent-executable only):
  - [ ] `nix develop --command bash -lc "cd web && npm test -- --run timeline"` passes or equivalent targeted test command
  - [ ] UI tests verify first clip auto-selects on success
  - [ ] UI tests verify clicking a timeline item changes the main player source

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Completed analysis renders clickable clip timeline
    Tool: Playwright
    Steps: stub completed job payload with 3 clips; open app; wait for results UI; click clip 2 in bottom timeline
    Expected: main player source changes to clip 2 and selected state updates visually
    Evidence: .sisyphus/evidence/task-7-timeline.png

  Scenario: No clips result shows empty-result guardrail
    Tool: Playwright
    Steps: stub completed job payload with empty `clips`; open app
    Expected: explicit “no serves found” empty state renders instead of broken player UI
    Evidence: .sisyphus/evidence/task-7-empty-result.png
  ```

  **Commit**: YES | Message: `feat(web): add serve timeline review ui` | Files: [`web/src/components/clip-player.tsx`, `web/src/components/clip-timeline.tsx`, `web/src/App.tsx`, `web/src/**/*.test.tsx`]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit after Wave 1: `feat(web): bootstrap local web app and api skeleton`
- Commit after Wave 2: `feat(web): add serve processing flow and timeline review`
- Commit after final verification fixes: `test(web): finalize mvp verification fixes`

## Success Criteria
- a local developer can enter `nix develop`, start the Python API and Vite UI, upload exactly one video, wait for processing, and play generated serve clips in a bottom timeline
- generated clips are browser-playable MP4 files and are removed with temp cleanup
- existing Python detector/test boundaries remain intact
- frontend build and backend tests pass without manual intervention
