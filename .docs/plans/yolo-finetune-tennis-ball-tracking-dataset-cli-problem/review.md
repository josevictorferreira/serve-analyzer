# Review

## Problem

Problem folder used: `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/`

The original problem was to improve tennis-ball detection for the user's backyard serve video by building a YOLO fine-tuning workflow that can outperform the RJTPP YOLOv8 baseline on precision and detected visible-ball frames.

## Source Documents

- `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/description.md`
- `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/plan.md`
- `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/execution.md`

## Review Goal

Verify whether the executed work actually solves the original YOLO fine-tuning problem, follows the documented plan, respects project constraints, and passes relevant validation gates.

## Original Problem Alignment

The execution addresses the first major phase of the problem: local browser-assisted frame review, RJTPP-assisted prelabeling, YOLO dataset export, and baseline evaluation scaffolding. It does not yet complete the end-to-end original goal because no YOLO model has been fine-tuned, YOLO26 compatibility has not been proven, AMD/ROCm training has not been configured, and no trained model has been shown to outperform RJTPP.

## Plan Adherence

Execution followed the recommended Path 1 by extending the existing local FastAPI and React app with an annotation mode. Steps for storage, frame extraction, review, export, baseline evaluation, and training-environment reporting were implemented; training smoke tests, real fine-tuning, trained-model evaluation, and second-round data collection were deferred and documented.

## Files Reviewed

- `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/description.md`
- `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/plan.md`
- `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/execution.md`
- `web/backend/paths.py`
- `web/backend/__main__.py`
- `web/backend/schemas.py`
- `web/backend/app.py`
- `web/backend/services/annotation_service.py`
- `web/backend/services/analysis_service.py`
- `web/backend/state.py`
- `tests/test_annotation_service.py`
- `tests/test_web_api_contract.py`
- `tests/test_web_analysis_service.py`
- `tests/test_web_clip_service.py`
- `web/src/lib/types.ts`
- `web/src/lib/api.ts`
- `web/src/App.tsx`
- `web/src/App.test.tsx`
- `web/src/components/annotation-workspace.tsx`
- `web/package.json`
- `.gitignore`

## Quality Gates

### Problem Fit Gate

Status: Fail

Evidence: The annotation/export/evaluation loop exists, but actual YOLO fine-tuning, YOLO26 compatibility, AMD/ROCm training setup, and measured improvement over RJTPP remain pending.

Notes: The completed work is a necessary first phase and is useful, but it is not the original end-to-end outcome.

Required follow-up: Fine-tune a model from the exported reviewed dataset, evaluate it against RJTPP on reviewed validation/test frames, and record metrics.

### Plan Adherence Gate

Status: Pass with warnings

Evidence: Path 1 was implemented in `web/` with annotation sessions, frame review, YOLO export, baseline evaluation, and UI integration. Deferred training work is documented in `execution.md` and `plan.md`.

Notes: Sampling is currently simple uniform sampling, not the planned post-contact/low-confidence active-learning sampler. Splits are contiguous 70/15/15 by sampled frame order, not serve-aware.

Required follow-up: Add post-contact and baseline-miss focused sampling before relying on the dataset for model improvement.

### Constraint Gate

Status: Pass with warnings

Evidence: The workflow remains local, uses the existing Nix/unittest project conventions, stores generated annotation artifacts outside the repo by default, keeps annotation separate from the existing analysis job state, and uses an explicit trust gate for remote RJTPP model downloads.

Notes: Remote `.pt` model download is still possible by explicit opt-in and defaults to revision `main` if no revision is supplied. The safer path is `SERVE_ANALYZER_RJTPP_MODEL_PATH` with a trusted local model file.

Required follow-up: Require or document a pinned `SERVE_ANALYZER_RJTPP_REVISION` for remote opt-in usage.

### Correctness Gate

Status: Fail

Evidence: Backend and frontend tests pass under the documented environment, and the frontend build passes. However, `rtk lint` in `web/` fails with 5 errors and 1 warning.

Notes: Implementation-owned lint findings include `src/components/annotation-workspace.tsx:163:14` unused `_err`, `src/lib/api.ts:126:18` unused `_err`, and `src/components/annotation-workspace.tsx:158:6` missing hook dependencies. The lint run also reports existing or adjacent issues in `src/App.tsx`, `src/components/ui/badge.tsx`, and `src/components/ui/button.tsx`.

Required follow-up: Clean up frontend lint errors or explicitly document which failures are pre-existing before treating this as ready.

### Regression Gate

Status: Pass with warnings

Evidence: `nix develop --command bash -c 'python -m unittest discover -s tests -v'` passed 139 tests. `npm test -- --run` passed 9 frontend tests. `rtk npm run build` passed.

Notes: The working tree includes unrelated/pre-existing modified and untracked files, so attribution is not perfectly clean. The lint failure is covered under the Correctness and Maintainability gates.

Required follow-up: Re-run all gates after lint cleanup and before committing.

### Maintainability Gate

Status: Fail

Evidence: The feature boundaries are understandable, but the frontend lint failure includes changed files. `annotation_service.py` is also large and schema models use broad `Dict[str, Any]` response shapes.

Notes: The current shape is acceptable for a first local prototype, but it should not be treated as fully maintainable until lint is clean and API schemas are narrowed as the contract stabilizes.

Required follow-up: Fix lint, then consider typed Pydantic response models for stable annotation data shapes.

### Safety Gate

Status: Pass with warnings

Evidence: Post-implementation review blockers were addressed: annotation storage is outside the analysis reset root, reset preservation has a regression test, annotation frame path containment uses `os.path.commonpath`, the backend runner binds to `127.0.0.1:8000`, and remote RJTPP model download requires explicit opt-in.

Notes: Export and evaluation responses still include absolute local paths, and some backend HTTP 500 paths expose exception strings. These are less risky for a loopback-only local tool but should be revisited before any non-local deployment.

Required follow-up: Keep the backend local-only, and avoid exposing this unauthenticated API on a network interface.

### Documentation Gate

Status: Pass

Evidence: `execution.md` exists, `plan.md` has execution status, and this review writes `review.md` plus a new review status section in `plan.md`.

Notes: Documentation clearly states deferred fine-tuning, YOLO26, and ROCm/Nix training work.

Required follow-up: Update documentation again after the first actual training/evaluation run.

## Validation Performed

- Command or method: `python -m unittest discover -s tests -v`
- Result: Failed in the ambient shell before real tests ran.
- Evidence: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` from OpenCV import.
- Notes: This is an environment issue outside the documented Nix workflow.

- Command or method: `nix develop --command bash -c 'python -m unittest discover -s tests -v'`
- Result: Passed.
- Evidence: 139 tests ran in 2.461s, OK.
- Notes: This is the relevant Python validation command for the project environment.

- Command or method: `npm test -- --run` in `web/`
- Result: Passed.
- Evidence: 2 test files and 9 tests passed.
- Notes: Covers the existing app tests and the annotation-mode UI smoke test.

- Command or method: `rtk npm run build` in `web/`
- Result: Passed.
- Evidence: `tsc -b && vite build` completed and Vite wrote the production bundle.
- Notes: TypeScript and production build are valid.

- Command or method: `rtk lint` in `web/`
- Result: Failed.
- Evidence: 5 errors and 1 warning in 5 files: `src/App.tsx`, `src/components/annotation-workspace.tsx`, `src/components/ui/badge.tsx`, `src/components/ui/button.tsx`, and `src/lib/api.ts`.
- Notes: This is a blocking review finding for code quality.

- Command or method: LSP diagnostics on `web/src`
- Result: Passed with one unrelated hint.
- Evidence: No errors; one existing unused `React` hint in `web/src/components/ui/scroll-area.tsx`.
- Notes: Not caused by this implementation.

## Validation Not Performed

- Manual browser QA with a real video was not performed during this review.
- Actual RJTPP local-model prelabeling was not run because no trusted local model path was configured during review.
- Actual YOLO fine-tuning was not performed.
- YOLO26 compatibility was not verified.
- AMD/ROCm training readiness was not proven with a real training command.
- Python LSP diagnostics were not available because `pylsp` is not installed.

## Issues Found

- Severity: Blocker
- Description: The original end-to-end fine-tuning goal is not complete.
- Evidence: `execution.md` explicitly defers fine-tuning, YOLO26 compatibility, AMD/ROCm training changes, and trained-model comparison against RJTPP.
- Suggested next step: Run an additional execution phase after a reviewed dataset batch exists.

- Severity: High
- Description: Frontend lint fails, including changed annotation code.
- Evidence: `rtk lint` reports issues in `src/components/annotation-workspace.tsx` and `src/lib/api.ts`, plus additional existing or adjacent frontend lint errors.
- Suggested next step: Fix implementation-owned lint issues and decide whether to clean or suppress pre-existing UI component lint failures.

- Severity: Medium
- Description: Frame sampling does not yet target the hardest planned cases.
- Evidence: `annotation_service.py` samples frames uniformly with `frame_count // max_frames`; it does not yet oversample post-contact, low-confidence, or RJTPP-missed frames.
- Suggested next step: Add sampling informed by serve segments, RJTPP misses, and low-confidence predictions.

- Severity: Medium
- Description: Validation split is not yet serve-aware.
- Evidence: `_assign_split()` uses contiguous sampled-frame index percentages, not held-out serves or independent segments.
- Suggested next step: Move to serve/segment-aware splits before training and reporting model improvements.

- Severity: Low
- Description: Remote RJTPP opt-in still defaults to revision `main`.
- Evidence: `_get_rjtpp_model()` uses `SERVE_ANALYZER_RJTPP_REVISION` defaulting to `main` when `SERVE_ANALYZER_ALLOW_REMOTE_MODEL_DOWNLOAD=1`.
- Suggested next step: Require a pinned revision or clearly document the supply-chain tradeoff.

## Deviations and Risk Assessment

The largest deviation is that execution stopped after the annotation/export/evaluation scaffolding phase rather than completing training and model comparison. This is reasonable as an incremental implementation choice because a reviewed dataset is needed before meaningful training, but it means the original problem is not fixed yet.

The implementation also uses simpler sampling and splitting than the plan described. This is acceptable for an MVP annotation loop, but risky for final model validation because adjacent frames are correlated and post-contact misses were the core failure mode.

## Final Review Status

Not fixed.

The implemented annotation workflow is a strong first phase and passes the main test/build commands under the documented Nix environment. The original end-to-end problem remains unresolved because no fine-tuned model has been trained or shown to beat RJTPP, and the frontend lint quality gate currently fails in changed code. Additional execution is required before this can be marked fixed.

## Lessons Learned

- The problem should be split explicitly into phases: dataset/annotation, training environment, model fine-tuning, and model evaluation.
- Review should include lint early, not only tests and build, because React Compiler lint rules catch issues not surfaced by TypeScript.
- Generated annotation artifacts should stay outside analysis reset directories by default.
- Remote `.pt` model loading must be treated as a trust decision, not a normal dependency fetch.

## Recommended Follow-Up

Additional execution required.

First fix or account for the frontend lint failures. Then run a second execution phase to annotate a real batch, export YOLO labels, train a small model, compare it against RJTPP on reviewed validation frames, and document whether precision and detected-frame count improved.
