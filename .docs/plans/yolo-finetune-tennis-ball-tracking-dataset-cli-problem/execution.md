# Execution Report

## Source Plan

Source plan: `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/plan.md`

Executed path: Path 1, extending the existing local FastAPI + React web app with a browser-assisted tennis-ball annotation workflow.

## Scope Executed

This execution implemented the first local dataset-building loop: create annotation sessions from uploaded videos, extract sampled frames, optionally pre-label with RJTPP YOLOv8, review frames in the browser, export a YOLO dataset, evaluate the RJTPP baseline on reviewed labels, and check the local training environment.

Actual YOLO fine-tuning was not run in this execution. The implementation creates the reviewed dataset and baseline evaluation scaffolding needed before training.

## Implemented Backend Work

- Added annotation artifact directory helpers in `web/backend/paths.py`.
- Added `web/backend/services/annotation_service.py` for local JSON/file-backed annotation sessions.
- Implemented frame extraction from uploaded video into per-session JPEG files.
- Implemented optional RJTPP pre-labeling using the `RJTPP/tennis-ball-detection` model. For safety, the backend now requires `SERVE_ANALYZER_RJTPP_MODEL_PATH` for a trusted local model file, or explicit `SERVE_ANALYZER_ALLOW_REMOTE_MODEL_DOWNLOAD=1` before downloading a remote `.pt` file from Hugging Face.
- Implemented frame review actions: accept prediction, correct with a manual bounding box, mark absent, skip, and undo.
- Implemented progress tracking across pending, accepted, corrected, absent, skipped, reviewed, and exportable frames.
- Implemented YOLO dataset export with `images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`, and export manifest.
- Implemented empty YOLO label files for reviewed absent frames.
- Implemented RJTPP baseline evaluation against reviewed labels with precision, recall, TP, FP, FN, and TN counts.
- Implemented a training-environment readiness check for `torch`, `ultralytics`, `huggingface_hub`, CUDA/ROCm availability, and recommended device.
- Moved default annotation storage outside the analysis temp root so `/api/job/reset` cannot delete annotation sessions or exports.
- Hardened frame-image path validation with `os.path.commonpath` and changed the local backend runner to bind to `127.0.0.1:8000`.
- Added annotation API response/request schemas in `web/backend/schemas.py`.
- Added `/api/annotation/...` endpoints in `web/backend/app.py` while keeping the existing `/api/analyze` job flow separate.

## Implemented Frontend Work

- Added annotation types in `web/src/lib/types.ts`.
- Added annotation API client helpers in `web/src/lib/api.ts`.
- Added `web/src/components/annotation-workspace.tsx` for the browser review workflow.
- Added an `Annotate Ball` top-level mode in `web/src/App.tsx` while preserving the existing `Analyze Serves` workflow.
- Added upload/session creation, saved-session listing, frame display, prediction/label overlays, click-to-correct, keyboard shortcuts, progress display, YOLO export, RJTPP baseline evaluation, and training-environment check UI.

## Implemented Tests

- Added `tests/test_annotation_service.py` for review actions, progress, YOLO export labels, empty negative labels, and baseline evaluation metrics.
- Added an annotation API contract test in `tests/test_web_api_contract.py` confirming annotation sessions do not mutate the existing analysis job state.
- Added a reset-isolation regression test confirming `/api/job/reset` preserves annotation session artifacts.
- Added frontend coverage in `web/src/App.test.tsx` for switching into annotation mode.
- Updated stale `tests/test_web_analysis_service.py` mocks to match the current detector result shape.
- Updated stale `tests/test_web_api_contract.py` fake-video tests to mock the current duration-estimation step.
- Updated stale `tests/test_web_clip_service.py` to match the current clip-service API with positions and detection frame skip.

## Deferred Work

- Actual YOLO fine-tuning is still pending.
- YOLO26 compatibility has not been proven yet.
- AMD ROCm/Nix training support was not changed; the implementation only reports readiness.
- Advanced active-learning sampling, post-contact oversampling, and multi-video project management remain future improvements.
- Advanced zoom/pan UI is not implemented yet; the first pass uses scaled full-frame display with coordinate-preserving SVG overlay.

## Validation Results

- `python -m unittest discover -s tests -v`: passed, 139 tests.
- `cd web && npm test -- --run`: passed, 2 test files and 9 tests.
- `cd web && rtk npm run build`: passed, TypeScript build and Vite production bundle.
- LSP diagnostics for changed frontend files reported no diagnostics.
- Python LSP diagnostics could not run because `pylsp` is not installed in the environment.

## Artifact Behavior

- Annotation sessions default to `${system temp}/serve_analyzer_annotations`, separate from the analysis upload/clip temp root.
- `SERVE_ANALYZER_ANNOTATION_DIR` can override the annotation root.
- `/api/job/reset` clears analysis artifacts only and preserves annotation artifacts by default.
- Session artifacts are local JSON, copied source video, extracted frame JPEGs, exports, and evaluation metadata.
- Generated videos, frames, labels, datasets, and model weights remain outside the committed source path unless explicitly placed there by the user.

## Post-Review Fixes

The post-implementation review found data-loss and security blockers. Those were addressed before finalizing this execution:

- Annotation data-loss risk fixed by moving the default annotation root out of `SERVE_ANALYZER_TEMP` and adding a reset-preservation regression test.
- Remote model supply-chain risk reduced by requiring a trusted local RJTPP model path unless remote model download is explicitly enabled.
- Frame path containment now uses `os.path.commonpath` rather than string-prefix checking.
- The backend module runner now binds to loopback on port 8000, matching the local-only development contract.

## Current Status

Initial annotation/export/evaluation implementation is complete and validated. The next practical step is to start the backend/frontend locally, open `Annotate Ball`, create a session from the user's serve video, review a small sampled batch, export YOLO labels, and then add the first training command/smoke test once reviewed data exists.
