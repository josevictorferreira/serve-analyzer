# Plan

## Planning Target

Create an implementation-ready plan for improving tennis-ball detection on the user's backyard serve videos by building a local browser-assisted annotation workflow, generating a YOLO-compatible dataset, fine-tuning a model, and validating that the result improves over the RJTPP YOLOv8 baseline.

This is only a plan. It does not implement production code.

## Source Context

Source of truth: `.docs/plans/yolo-finetune-tennis-ball-tracking-dataset-cli-problem/description.md`

Relevant project context inspected for planning:

- `web/` is an existing Vite React + FastAPI local app.
- `web/backend/app.py` already handles video upload, background processing, and serving generated clips.
- `web/src/App.tsx` currently focuses on serve-analysis upload/results, not annotation.
- `serve_analyzer/compare_detectors.py` already contains RJTPP Hugging Face download/inference logic, currently CPU-oriented.
- `flake.nix` provides the dev environment and currently installs Ultralytics/PyTorch CPU wheels in `.venv` if missing.
- `.gitignore` already ignores videos, images, model weights, frame samples, logs, and nested research repos.

## Confirmed Facts

- The immediate goal is frame-by-frame tennis-ball detection only.
- The target is to outperform `RJTPP/tennis-ball-detection` on the user's video/setup.
- The current best baseline is RJTPP YOLOv8, but it misses too many ball detections.
- The biggest known failure area is post-contact ball flight, though other serve phases also fail.
- Desired improvement metrics are precision and more frames with a detected ball.
- The source video is 4K 60fps from an iPhone 14 Pro Max.
- The video contains 8 backyard serves and more than 4500 frames.
- The camera is side-positioned, below the player, and tilted upward.
- The target generalization domain is similar camera setup, similar distance from the ball, and similar backyard/non-court environment.
- The user is on NixOS with a Radeon 6900XT and 16GB GPU VRAM.
- The user prefers a browser-based review UI.
- The user can record additional videos if that improves fine-tuning.
- The annotation format is open and should be chosen based on the training workflow.
- The assistant should decide later whether no-ball frames are useful.

## Constraints

- Do not treat this as a generic tennis-video detector; optimize first for this narrow recording setup.
- Keep processing local unless the user explicitly approves remote tooling, because videos are personal backyard footage.
- Use the project conventions: no `requirements.txt`, no `pyproject.toml`, no virtualenv instructions outside the existing Nix/.venv pattern, and unittest for Python tests.
- Do not commit videos, extracted frames, labels from personal footage unless the user explicitly wants that; generated artifacts should live in ignored local directories.
- The small, fast, post-contact ball is the hard case and must be represented in sampling, annotation, validation, and metrics.
- The current Nix shell installs CPU PyTorch wheels, so AMD GPU training support is not yet established.
- YOLO26 compatibility must be verified before the plan depends on it.

## Assumptions

- YOLO object detection labels will be bounding boxes with one class, `tennis_ball`, unless the selected training target requires something else.
- User clicks can be used as a correction input, but the tool should convert them into a reviewable bounding box rather than storing only a point.
- A small curated set of no-ball frames is useful for precision and false-positive control, but it should be capped and intentionally sampled, not added indiscriminately.
- Adjacent frames are highly correlated, so validation should not be a naive random frame split across the entire video.
- The first useful model can be produced from a sampled/reviewed subset, not from manually labeling every frame.
- The existing `web/` app is the best home for the browser review UI unless the user strongly prefers a separate tool.
- Fine-tuning from the current RJTPP baseline or another tennis-ball-specific YOLO checkpoint is more likely to beat RJTPP quickly than starting from a generic model.

## Decomposition

Mandatory work:

- Baseline measurement: run RJTPP on sampled frames and establish precision/detected-frame metrics against human labels.
- Frame extraction: sample frames from the video, with extra focus on post-contact and low-confidence/missed RJTPP frames.
- Pre-labeling: run RJTPP predictions to seed annotations.
- Browser annotation UI: show frame, predicted box, confidence, and controls for accept/correct/mark absent/skip/undo.
- Annotation storage: persist labels, review state, frame metadata, prediction metadata, and user corrections locally.
- Dataset export: create YOLO-compatible `images/`, `labels/`, and dataset YAML with train/val/test splits.
- Negative-frame policy: include a small, curated set of no-ball frames with empty labels for precision control.
- Training workflow: fine-tune the selected YOLO model locally, with a CPU fallback and an AMD GPU readiness check.
- Evaluation workflow: compare RJTPP baseline vs fine-tuned model on a held-out reviewed validation/test set.
- Artifact management: keep datasets, runs, and weights out of git and record manifests for reproducibility.

Optional improvements:

- Add temporal navigation around a serve sequence.
- Add zoom/pan for small post-contact ball annotation.
- Add keyboard shortcuts for faster review.
- Add active-learning rounds that prioritize frames where the fine-tuned model still fails.
- Add multi-video project support after the first video works.
- Add training run comparison dashboard.
- Add future tracking/serve-event integration once ball detection is reliable.

## Proposed Solution Paths

### Path 1: Browser-Assisted Active Learning in the Existing Web App

**Summary**

Extend the existing local FastAPI + React app with an annotation mode. A backend job extracts selected frames, runs RJTPP predictions, stores pre-labels, and serves frames to a browser UI. The user accepts/corrects/marks absent labels. The tool exports a YOLO dataset, fine-tunes a model, and compares it against RJTPP on a held-out reviewed set.

**Best when**

Choose this when the goal is a maintainable local workflow that can start small but grow into a better annotation/training/evaluation tool.

**Benefits**

- Matches the user's browser UI preference.
- Reuses the existing `web/` app and backend pattern.
- Keeps data local.
- Supports iterative improvement instead of one-shot labeling.
- Can prioritize post-contact failures and low-confidence frames.
- Produces a reusable dataset workflow rather than one-off labels.

**Risks**

- More implementation work than pure scripts.
- The current app is built around serve analysis, so annotation mode needs clear separation from existing upload/results flow.
- Browser annotation accuracy needs zoom/pan or high-resolution crop support because the ball is small.
- AMD GPU training may still require environment work outside the UI.

**Complexity**

Medium. It adds new backend endpoints, frontend screens, dataset storage, and export logic, but it builds on existing local app infrastructure.

**Key decisions**

- Whether annotation mode lives inside the existing `web/` app as a new route/mode.
- Whether first training uses only the existing video or waits for additional videos.
- Whether the first model target is RJTPP YOLOv8 continuation training, YOLO26, or both.
- Exact validation split: hold out serves/segments rather than randomly mixing adjacent frames.
- Exact no-ball-frame ratio and sampling policy.

**Validation**

- Verify the UI can accurately accept, correct, mark absent, undo, and resume annotations.
- Verify exported YOLO labels are valid and normalized.
- Compare baseline RJTPP and fine-tuned model on the same reviewed validation/test frames.
- Require improved detected-frame count without a precision regression.

### Path 2: Script-First Dataset Builder With Minimal Browser Viewer

**Summary**

Build Python CLIs first for frame extraction, RJTPP pre-labeling, YOLO export, and training. Add only a minimal browser/static review screen for annotations, or use generated HTML/images for review.

**Best when**

Choose this when speed to first dataset matters more than a polished interactive workflow.

**Benefits**

- Faster to implement than a fully integrated web workflow.
- Easier to test individual stages from the command line.
- Keeps the existing serve-analysis web app untouched or minimally touched.
- Good for proving dataset/training value before investing in UI.

**Risks**

- Less aligned with the user's browser-first preference.
- Harder to improve into a comfortable long-term annotation tool.
- Review UX may be too slow for thousands of frames.
- More likely to create disconnected scripts if not carefully organized.

**Complexity**

Medium. The data pipeline is still real work, but the UI surface is smaller.

**Key decisions**

- How minimal the browser review tool can be while still allowing accurate ball annotation.
- Whether script outputs should later be migrated into the existing web app.
- How to prevent one-off scripts from accumulating in the package.

**Validation**

- Validate each CLI stage independently with small frame subsets.
- Manually inspect exported datasets.
- Train a small smoke model and compare against RJTPP on reviewed frames.

### Path 3: External Annotation Tool Plus Local Training

**Summary**

Use an existing annotation tool such as CVAT, Label Studio, or Roboflow to label frames, then export YOLO labels and run local fine-tuning/evaluation in this project.

**Best when**

Choose this when reducing custom UI work is more important than tight integration and local-only control.

**Benefits**

- Mature annotation UI features like zoom, boxes, shortcuts, review workflows, and export formats.
- Less custom frontend/backend work.
- Faster if the external tool already fits the desired workflow.

**Risks**

- May violate local-only/privacy expectations if hosted externally.
- Adds operational overhead if self-hosted.
- Less integrated with RJTPP pre-labeling, custom frame sampling, and later project workflows.
- User explicitly prefers a browser tool that can be improved in this codebase.

**Complexity**

Low to Medium. Low if using hosted tooling, medium if self-hosting and integrating imports/exports.

**Key decisions**

- Whether personal backyard footage can be uploaded to a hosted tool.
- Whether self-hosting is acceptable.
- Whether the project should still build custom pre-label/export/evaluation tools.

**Validation**

- Import/export a small frame subset and verify YOLO label correctness.
- Train/evaluate using exported labels.
- Confirm privacy and reproducibility constraints are acceptable.

### Path 4: Pseudo-Label First, Human Review Later

**Summary**

Use RJTPP predictions directly as pseudo-labels, optionally filter by confidence, fine-tune a model quickly, and only manually review failures afterward.

**Best when**

Choose this only for a fast experiment where label quality is less important than seeing whether training can run at all.

**Benefits**

- Fastest path to a first training run.
- Minimal manual labeling upfront.
- Useful as a training pipeline smoke test.

**Risks**

- Likely to reproduce RJTPP's blind spots because missed post-contact frames will remain unlabeled.
- Can reinforce false positives and label noise.
- Weak evidence of real improvement if evaluation labels are not human-reviewed.
- Poor fit for the user's goal of beating RJTPP on missed frames.

**Complexity**

Low. It avoids the hard annotation workflow, but also provides the weakest learning signal.

**Key decisions**

- Confidence threshold for pseudo-labels.
- Whether to exclude low-confidence or ambiguous frames.
- How much human-reviewed validation is required before trusting results.

**Validation**

- Use only as a smoke test unless paired with a human-reviewed validation set.
- Compare against RJTPP and reject the path if improvements are not supported by reviewed labels.

## Trade-Off Analysis

| Path | Simplicity | Safety | Maintainability | Speed | Reversibility | Fit with constraints |
| --- | --- | --- | --- | --- | --- | --- |
| Path 1: Existing web app active learning | Medium | High | High | Medium | High | Best fit: browser, local, iterative, quality-focused |
| Path 2: Script-first builder | Medium | Medium | Medium | High | High | Good fallback if UI work must be minimized |
| Path 3: External annotation tool | Low to Medium | Medium | Medium | High | Medium | Good UI, weaker local/privacy/project fit |
| Path 4: Pseudo-label first | Low | Low | Low | Highest | High | Useful only as a smoke test, weak fit for beating RJTPP |

Path 1 is the best fit because the user wants a browser workflow that can improve over time, the repository already has a local web app, and the goal requires human-reviewed labels for frames RJTPP misses. Path 4 is tempting but conflicts with the core failure mode: RJTPP misses post-contact frames, so training mostly from RJTPP's own predictions risks copying the baseline's gaps.

## Implementation Questions

### Blocking Questions

1. Should execution use Path 1 as the chosen path?

Why it matters: it determines whether implementation extends the existing web app or builds a separate/script-first tool.

Options:

- A. Use Path 1 and extend the existing `web/` app. Recommended.
- B. Use Path 2 and build script-first tools before UI.
- C. Use Path 3 and rely on an external annotation tool.
- D. Use Path 4 only as a quick smoke test.

2. Which model target should be first?

Why it matters: dataset format is likely compatible across YOLO detectors, but training commands, dependencies, and expected results differ.

Options:

- A. Fine-tune from RJTPP YOLOv8 first, then try YOLO26 after the pipeline works. Recommended.
- B. Use YOLO26 first after a compatibility spike.
- C. Train both RJTPP-derived and YOLO26-derived models and compare.

3. Should the first dataset use only the existing video, or should additional videos be recorded before training?

Why it matters: recording more data improves generalization but delays the first feedback loop.

Options:

- A. Start with the existing video and hold out serves/segments for validation. Recommended for first iteration.
- B. Record 1-3 additional similar videos before building the first training dataset.
- C. Build the tool first, then record and label more videos in round two.

### Non-Blocking Questions

1. What annotation shortcuts should the browser UI prioritize?

This affects review speed. Suggested defaults are accept, correct by click/box, mark absent, skip, previous, next, undo, and save.

2. How strict should the precision target be?

This affects confidence threshold selection. The first plan can evaluate multiple thresholds rather than requiring one upfront.

3. Where should local generated datasets live?

This affects path conventions and `.gitignore`. Suggested default is a local ignored directory such as `datasets/tennis-ball-local/` or a path under `SERVE_ANALYZER_TEMP` for intermediate frames.

4. How much UI polish is required in the first pass?

This affects whether the first browser version includes advanced zoom/pan and sequence navigation or starts with a simpler frame-review loop.

## Recommended Path

Recommended: Path 1, extending the existing local web app with a browser-assisted active-learning annotation workflow.

Recommended model sequence: start by continuing from the RJTPP YOLOv8 tennis-ball model, because it is already the best baseline and the immediate goal is to beat it on the same domain. Treat YOLO26 as a compatibility/benchmark branch after the annotation/export/evaluation pipeline works, or run it in parallel only if the user explicitly wants the extra complexity.

Recommended data sequence: start with the existing video to build the pipeline and get a first measurable comparison. Use a held-out serve/segment validation split to avoid adjacent-frame leakage. Record additional similar videos in a second round if the first model overfits or fails to generalize to another similar recording.

Recommended no-ball-frame policy: include a small curated negative set with empty YOLO label files. Sample negatives from non-serve/background regions and from confusing moments where RJTPP or the fine-tuned model produces false positives. Do not flood training with no-ball frames.

## Execution Outline

1. Establish project boundaries.

Goal: keep annotation/training separate from current serve-analysis behavior.

Actions: define an annotation mode namespace in the web app, local artifact directories, and ignored dataset/run locations.

Expected result: existing serve analyzer flow remains intact while annotation work has a clear home.

Risk or note: avoid mixing serve-event logic into ball annotation; the current immediate target is ball detection only.

2. Add a baseline/evaluation data model.

Goal: make comparisons against RJTPP measurable before training.

Actions: define local records for frame metadata, RJTPP prediction, human-reviewed label, review status, split assignment, and evaluation results.

Expected result: every reviewed frame can be traced from source video to prediction to label to metric.

Risk or note: do not rely on raw detected-frame counts alone; precision requires human-reviewed ground truth.

3. Build frame sampling rules.

Goal: choose frames that teach the model the hard cases without requiring every frame.

Actions: sample across all serves, oversample post-contact frames, include RJTPP misses/low-confidence predictions, include some adjacent temporal context, and include a capped negative set.

Expected result: a review queue focused on improving post-contact detection and precision.

Risk or note: random frame sampling can waste labels on redundant frames and leak adjacent frames into validation.

4. Run RJTPP pre-labeling.

Goal: seed annotations with the best current model.

Actions: run RJTPP on selected frames, store bounding boxes/confidence/missing predictions, and expose this metadata to the browser UI.

Expected result: many frames can be accepted quickly, while misses become correction tasks.

Risk or note: existing code paths currently use CPU inference for RJTPP in `compare_detectors.py`; GPU inference/training support is a separate environment task.

5. Build the browser review workflow.

Goal: let the user efficiently create high-quality labels.

Actions: add a browser screen with frame display, prediction overlay, zoom/crop support, accept/correct/mark absent/skip/undo controls, progress tracking, and save/resume behavior.

Expected result: labels can be reviewed and corrected locally without manually editing files.

Risk or note: the UI must handle 4K frames carefully; serving full-resolution frames may be slow, so zoomed crops or scaled display with correct coordinate mapping may be needed.

6. Export a YOLO dataset.

Goal: produce training-ready files.

Actions: export images, YOLO normalized bounding-box labels, empty label files for selected negative frames, class metadata, and dataset YAML with train/val/test splits.

Expected result: the dataset can be consumed by Ultralytics training commands.

Risk or note: split by serve/video segment rather than naive random frames to reduce leakage.

7. Perform a training environment spike.

Goal: determine whether AMD GPU training works on the user's NixOS/Radeon setup.

Actions: verify PyTorch device support, ROCm availability, Ultralytics training compatibility, and CPU fallback behavior before running long training.

Expected result: the training command and device choice are known before full fine-tuning.

Risk or note: current `flake.nix` installs CPU PyTorch wheels, so GPU training may require a planned Nix/ROCm adjustment.

8. Run a tiny training smoke test.

Goal: catch dataset/export/training issues early.

Actions: train on a very small reviewed subset for a minimal epoch count and confirm that training starts, reads labels, writes metrics, and produces weights.

Expected result: the pipeline works end-to-end before investing annotation/training time.

Risk or note: smoke-test quality is not meaningful; it only validates mechanics.

9. Run the first real fine-tuning pass.

Goal: produce a candidate model that can beat RJTPP.

Actions: fine-tune from the selected starting weights, evaluate on the held-out reviewed set, and record metrics and artifacts.

Expected result: a model candidate with measurable precision and detected-frame count.

Risk or note: if precision drops, threshold tuning or more negative/hard-example labeling may be needed.

10. Compare against RJTPP and decide next round.

Goal: prove whether the fine-tuned model is better for the user's video/setup.

Actions: run RJTPP and the fine-tuned model on the same reviewed validation/test frames, compare precision and detected visible-ball frames, inspect post-contact misses, and choose whether to annotate more data.

Expected result: a clear continue/adjust/stop decision.

Risk or note: a model can detect more frames by producing more false positives; precision prevents that from being counted as success.

11. Optional second round with additional videos.

Goal: improve generalization to similar setups.

Actions: record one or more similar videos, sample and review hard frames, add them to training while holding out at least one segment/video for evaluation.

Expected result: a model less overfit to the original video.

Risk or note: additional data should be similar enough to the target domain but varied enough to teach robustness.

## Validation Strategy

Dataset validation:

- Every exported image has a matching label file.
- YOLO labels are normalized and within `[0, 1]` bounds.
- Positive labels use the intended single class id.
- Negative frames have empty label files only when intentionally selected.
- Train/val/test splits avoid adjacent-frame leakage as much as practical.

Annotation UI validation:

- User can accept a correct prediction.
- User can correct a wrong prediction.
- User can mark a frame as no-ball/absent.
- User can undo and resume after closing/reopening.
- Coordinates remain correct when the browser displays scaled images or zoomed crops.

Baseline/model validation:

- Establish RJTPP metrics on a human-reviewed validation set.
- Evaluate the fine-tuned model on the same frames.
- Report precision, detected visible-ball frames, false positives, false negatives, and post-contact misses.
- Accept improvement only if detected-frame count increases without unacceptable precision loss.

Training validation:

- Run a tiny smoke train before full training.
- Record model weights, dataset manifest, training config, and evaluation summary.
- Check that generated weights are not committed to git.

Project validation:

- Python tests should pass with `python -m unittest discover -s tests -v`.
- Frontend tests should pass with `cd web && npm test -- --run`.
- Frontend build should pass with `cd web && npm run build`.
- Any backend API changes should have focused tests or documented manual checks.

## Rollback / Recovery Strategy

- Keep all generated frames, datasets, runs, predictions, and model weights in ignored local artifact directories so they can be deleted without touching source code.
- Preserve the RJTPP baseline and evaluation manifests so failed fine-tuning runs can be compared or discarded.
- If a training run is bad, delete that run artifact and return to the last known-good dataset/model manifest.
- If annotation data is corrupted, recover from periodic annotation-state snapshots or JSON backups before continuing review.
- If web app changes break the existing serve-analysis workflow, revert only the annotation-related files/route changes from the implementation branch.
- No database migrations are expected in the recommended first path; local JSON/file storage is easier to recover and sufficient for the first iteration.

## Decision Log

Decisions already made:

- Immediate scope is ball detection only.
- Browser UI is preferred.
- Baseline to beat is RJTPP YOLOv8.
- Target metrics are precision and more detected frames.
- Post-contact detection is the highest-priority failure area.
- Similar setup means similar camera setup, similar distance from ball, and similar backyard/non-court environment.
- User can record more videos if useful.

Recommended decisions pending user confirmation:

- Use Path 1 and extend the existing `web/` app.
- Fine-tune from RJTPP YOLOv8 first, then test YOLO26 after the pipeline works.
- Start with the existing video for the first iteration, then record more videos in round two if needed.
- Include a small curated negative-frame set for precision control.

Implementation details still pending:

- Exact artifact directory names.
- Exact frame sampling counts per serve/phase.
- Exact validation split policy.
- Exact browser UI shortcuts and annotation controls.
- Exact Nix/ROCm/PyTorch training configuration.
- Exact YOLO26 compatibility outcome.

## Readiness Assessment

Partially ready for execution.

The planning target is clear, and Path 1 is recommended because it best matches the browser UI, local workflow, and need for high-quality human-reviewed labels around RJTPP failures. Execution should wait for the user to confirm the recommended path, first model target, and whether to start with the existing video or record more data first. Once those are confirmed, implementation can proceed in small phases beginning with baseline measurement and annotation dataset scaffolding.

## Execution Status

Executed on 2026-04-27 using the recommended Path 1 defaults: extend the existing `web/` app, start from the current video workflow, support RJTPP YOLOv8 pre-labeling behind an explicit local-model/remote-download trust gate, and include reviewed absent frames as empty YOLO labels.

Implemented the initial annotation/export/evaluation loop: local annotation sessions, frame extraction, optional RJTPP pre-labels, browser review UI, YOLO dataset export, RJTPP baseline evaluation, and training-environment readiness reporting.

Validation passed with `python -m unittest discover -s tests -v`, `cd web && npm test -- --run`, and `cd web && rtk npm run build`.

Post-review blockers were fixed before completion: annotation artifacts are no longer stored under the analysis reset temp root, reset preservation is covered by a regression test, frame image path containment uses `os.path.commonpath`, remote RJTPP model downloads require explicit opt-in, and the local backend runner binds to `127.0.0.1:8000`.

Fine-tuning itself, YOLO26 compatibility, and Nix/ROCm training-environment changes remain pending follow-up work after a reviewed dataset batch exists. See `execution.md` in this folder for details.

## Review Status

### Status

Not fixed.

### Summary

The review found that the annotation/export/evaluation phase was implemented and validated with the main Python, frontend test, and build commands under the documented environment. The original end-to-end YOLO fine-tuning problem is not complete because no model has been fine-tuned, YOLO26 compatibility and AMD/ROCm training remain unverified, no trained model has been compared against RJTPP, and the frontend lint gate currently fails.

### Quality Gates

- Problem Fit Gate: Fail. The first dataset-building phase is present, but the original fine-tuned-model outcome is still pending.
- Plan Adherence Gate: Pass with warnings. Path 1 was followed, but training steps and active sampling remain deferred.
- Constraint Gate: Pass with warnings. Local workflow and safety constraints are mostly respected, with remaining remote-model revision caution.
- Correctness Gate: Fail. Tests and build pass, but `rtk lint` fails with 5 errors and 1 warning.
- Regression Gate: Pass with warnings. Existing test suites pass under Nix, but the dirty worktree and lint failure leave residual risk.
- Maintainability Gate: Fail. Lint errors in changed frontend files must be cleaned up before treating the result as maintainable.
- Safety Gate: Pass with warnings. Major post-review blockers were fixed, but local-only assumptions and remote model trust choices remain important.
- Documentation Gate: Pass. `execution.md`, `review.md`, and plan status updates document the result and remaining work.

### Issues Found

- Blocker: actual YOLO fine-tuning and improved RJTPP comparison are not done.
- High: frontend lint fails, including changed annotation code in `src/components/annotation-workspace.tsx` and `src/lib/api.ts`.
- Medium: frame sampling is uniform and does not yet prioritize post-contact or RJTPP-missed frames.
- Medium: dataset splitting is contiguous frame-order based, not serve-aware.
- Low: explicit remote RJTPP model opt-in still defaults to revision `main` unless `SERVE_ANALYZER_RJTPP_REVISION` is set.

### Recommended Follow-Up

Additional execution required. First fix or account for frontend lint failures, then annotate/export a real reviewed batch, run a training smoke test, fine-tune a model, and compare it against RJTPP on reviewed validation frames.
