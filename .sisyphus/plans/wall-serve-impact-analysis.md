# Wall Serve Impact Analysis

## TL;DR
> **Summary**: Build a wall-practice serve-analysis pipeline that calibrates a wall/camera setup, detects or manually corrects ball-wall impacts, estimates pre-wall speed with uncertainty, projects equivalent regulation-court landing, and emits JSON/CSV/plots/annotated video. The implementation must explicitly separate measured, inferred, and assumed quantities because monocular 4K/60fps video cannot produce exact 3D truth without constraints.
> **Deliverables**:
> - Wall calibration CLI/script with reusable setup metadata and per-video override/correction support.
> - Wall analysis CLI/script for `videos/wall/*.MOV`.
> - Research-grade geometry/physics module with uncertainty/confidence outputs.
> - JSON, CSV, plots, and annotated MP4 outputs per analyzed video.
> - Synthetic and unit tests using existing `unittest` conventions.
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 6 → Task 8 → Final Verification

## Context

### Original Request
User needs a different algorithm for wall-practice videos in `videos/wall/`: iPhone 14 Pro Max 4K/60fps, camera 1.57m from wall, tilted left/up, serve contact point 6.11m from the wall. The algorithm must find each serve timestamp, infer pre-wall ball speed, identify and overlay wall-hit point, and calculate where the serve would land on a real tennis court. A manual calibration script must annotate a metal hook at 2.45m, chair tops at 1m, wall tilt/lines, and center-line/serve-line references.

### Interview Summary
- Projection model: research-grade.
- Calibration: hybrid reusable setup metadata with per-video override/correction.
- Outputs: JSON, annotated video, CSV, and plots are mandatory.
- Court geometry: regulation tennis court/service-box dimensions by default.
- Serve contact height: manual value required in calibration metadata.
- Impact detection: autonomous detection first; manual correction allowed.
- Camera intrinsics: optional but supported; use chessboard/ChArUco data if provided, otherwise approximate mode with degraded confidence.

### Metis Review (gaps addressed)
Metis required precise definitions for speed, coordinate systems, landing semantics, calibration schema, segmentation strategy, tolerance criteria, and underdetermined-geometry behavior. This plan resolves them as follows:
- **Pre-wall ball speed**: instantaneous 3D speed magnitude immediately before wall impact, estimated from the final clean pre-impact trajectory segment; output units: `m_s`, `km_h`, `mph`.
- **Wall-hit point**: ball-center impact point on calibrated wall plane, reported in pixels and wall meters.
- **Projected landing**: where the ball would land if the wall were absent and the inferred pre-wall trajectory continued from the measured wall-impact state backward/forward under the selected court coordinate model; not rebound-after-wall landing.
- **Coordinate frames**: explicit wall and court frames defined in Task 1.
- **Underdetermined behavior**: refuse projection with structured warnings when calibration/track quality is insufficient; never silently fabricate exact results.

## Work Objectives

### Core Objective
Implement a wall-specific research pipeline that converts calibrated wall-practice videos into auditable per-serve measurements and equivalent regulation-court landing projections.

### Deliverables
- `serve_analyzer/wall_calibration.py` as the wall calibration module/CLI.
- `serve_analyzer/wall_serve.py` as the wall analysis module/CLI.
- `serve_analyzer/wall_outputs.py` for serialization contracts.
- `serve_analyzer/wall_artifacts.py` for annotated video and plot generation.
- Runtime-generated synthetic test fixtures in `tests/wall_test_helpers.py`; no committed binary video fixtures.
- Tests under `tests/` using `unittest` only.
- Output artifacts: JSON, CSV, plots, annotated MP4, and evidence files.

### Definition of Done (verifiable conditions with commands)
- `python -m unittest discover -s tests -v` passes.
- CLI help for new scripts exits successfully and documents required metadata fields.
- Synthetic wall-impact fixture validates impact frame within ±1 frame and wall point within configured pixel/metric tolerance.
- Invalid calibration fixture exits with structured error/warning, not traceback.
- Manual correction fixture deterministically overrides autonomous impact frame/point.
- Approximate intrinsics mode marks confidence as degraded and includes uncertainty fields.

### Must Have
- Hybrid calibration metadata: reusable setup plus per-video override.
- Manual serve contact height required.
- Camera intrinsics optional but schema-supported.
- Explicit regulation court constants.
- Measured/inferred/assumed fields separated in JSON and flattened in CSV.
- Confidence/uncertainty for wall point, speed, and landing projection.
- Annotated output overlays ball track, wall-hit point, projected court landing summary, and warnings.

### Must NOT Have
- Must not present monocular estimates as exact truth.
- Must not train new ML models.
- Must not destabilize existing lateral serve CLI/detectors.
- Must not add pytest, pip/requirements, virtualenv, or non-Nix dependency management.
- Must not require web UI integration.
- Must not rely on human visual confirmation as acceptance.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after with existing `unittest` framework.
- QA policy: Every task has agent-executed scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy

### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Task 1 schema/coordinate definitions, Task 2 calibration primitives, Task 3 output contracts/tests, Task 4 synthetic fixture generator.
Wave 2: Task 5 calibration CLI, Task 6 impact detection/correction, Task 7 trajectory/speed estimator, Task 8 court projection.
Wave 3: Task 9 analysis orchestration/CLI, Task 10 annotated video/plots, Task 11 CSV/JSON finalization, Task 12 real wall video smoke command.
Wave 4: Task 13 integrated tests/docs/help hardening.

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 3, 5, 7, 8, 9, 11.
- Task 2 blocks Tasks 5, 6, 7, 8, 9, 10.
- Task 3 blocks Tasks 9, 11, 13.
- Task 4 blocks Tasks 6, 7, 8, 13.
- Task 5 depends on Tasks 1-2; blocks Task 9.
- Task 6 depends on Tasks 2, 4; blocks Tasks 7, 9, 10.
- Task 7 depends on Tasks 1, 2, 4, 6; blocks Tasks 8, 9, 11.
- Task 8 depends on Tasks 1, 2, 4, 7; blocks Tasks 9, 11.
- Task 9 depends on Tasks 1-8; blocks Tasks 10-13.
- Task 10 depends on Tasks 6, 8, 9.
- Task 11 depends on Tasks 3, 7, 8, 9.
- Task 12 depends on Tasks 5, 9, 10, 11.
- Task 13 depends on Tasks 1-12.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → business-logic, quick, general.
- Wave 2 → 4 tasks → deep, business-logic, general.
- Wave 3 → 4 tasks → general, visual-engineering, business-logic.
- Wave 4 → 1 task → deep.

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Define wall/court coordinate systems and metadata schema

  **What to do**: Create the canonical wall-analysis data model and coordinate-frame definitions. Wall frame: origin at floor/wall intersection directly below the configured center reference; `x_m` horizontal along wall, positive to camera-right after undistortion; `y_m` vertical up from floor; `z_m` positive away from wall toward server. Court frame: same centerline origin projected onto wall/floor, regulation court constants, serve contact at `z_m=6.11` by default with required `serve_contact_height_m`. Define metadata sections for measured, inferred, assumed, intrinsics, calibration references, per-video overrides, and manual impact corrections. Include JSON-serializable validation helpers.
  **Must NOT do**: Do not implement detection/tracking here. Do not make contact height optional.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Requires precise contracts and validation logic.
  - Skills: [] - No specialized skill required.
  - Omitted: [`developing-rspec-tests`] - Project is Python `unittest`, not Rails/RSpec.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2,3,5,7,8,9,11 | Blocked By: none

  **References**:
  - Pattern: `serve_analyzer/analysis.py` - pure math/helper function style.
  - Pattern: `serve_analyzer/serve_evaluation.py` - JSON-oriented CLI/result contracts.
  - Test: `tests/test_evaluator.py` - temp JSON and schema-like assertions.
  - Project convention: `AGENTS.md` - unittest only; Nix-only dependency management.

  **Acceptance Criteria**:
  - [ ] `python -m unittest tests.test_wall_calibration -v` validates required metadata fields, rejects missing `serve_contact_height_m`, rejects fewer than 4 wall-plane points, and accepts optional intrinsics.
  - [ ] Schema/constants expose regulation court dimensions in meters and flattenable output field names for CSV.

  **QA Scenarios**:
  ```
  Scenario: Valid reusable setup metadata
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_calibration.TestWallMetadataSchema.test_valid_reusable_setup -v`.
    Expected: Test passes; metadata includes wall frame, court frame, measured hook/chair references, `serve_contact_distance_m=6.11`, and required `serve_contact_height_m`.
    Evidence: .sisyphus/evidence/task-1-schema-valid.txt

  Scenario: Missing contact height fails
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_calibration.TestWallMetadataSchema.test_missing_contact_height_rejected -v`.
    Expected: Test passes by asserting a structured validation error, not traceback.
    Evidence: .sisyphus/evidence/task-1-schema-error.txt
  ```

  **Commit**: YES | Message: `feat(wall): define calibration metadata schema` | Files: [`serve_analyzer/wall_calibration.py`, `tests/test_wall_calibration.py`]

- [x] 2. Implement wall homography, calibration residuals, and optional intrinsics handling

  **What to do**: Implement deterministic geometry primitives: undistort-if-intrinsics-present, compute wall homography from 4+ known wall points, invert pixel-to-wall coordinates, compute reprojection/calibration residuals, detect collinear/degenerate points, and mark approximate-intrinsics mode as degraded. Intrinsics source values: `none`, `approx_exif`, `opencv_chessboard`, `opencv_charuco`.
  **Must NOT do**: Do not infer full camera pose from sparse hook/chair points unless intrinsics and enough correspondences exist. Do not trust iPhone factory intrinsics as exact.

  **Recommended Agent Profile**:
  - Category: `deep` - Geometry correctness and edge-case handling are critical.
  - Skills: [] - Use OpenCV already in project environment.
  - Omitted: [`research-tools`] - Research has already established homography/intrinsics approach.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5,6,7,8,9,10 | Blocked By: 1

  **References**:
  - Pattern: `serve_analyzer/analysis.py:compute_scale_factor` - simple metric conversion validation style.
  - External: OpenCV `cv2.findHomography`, `cv2.undistort` - use RANSAC where appropriate.
  - Research finding: homography maps wall-plane meters ↔ pixels; distance to camera is not recovered from homography alone.

  **Acceptance Criteria**:
  - [ ] Synthetic square wall points round-trip pixel→wall→pixel within ≤1 px in no-distortion fixture.
  - [ ] Degenerate collinear references produce structured calibration failure.
  - [ ] Approximate/no intrinsics mode sets `confidence.degraded_intrinsics=true`.

  **QA Scenarios**:
  ```
  Scenario: Synthetic homography round trip
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_calibration.TestWallHomography.test_round_trip_synthetic_points -v`.
    Expected: RMS reprojection error ≤1 px and wall-meter coordinates match expected fixture values.
    Evidence: .sisyphus/evidence/task-2-homography.txt

  Scenario: Collinear calibration rejection
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_calibration.TestWallHomography.test_collinear_points_rejected -v`.
    Expected: Structured error includes `calibration_degenerate` and no projection result is produced.
    Evidence: .sisyphus/evidence/task-2-degenerate.txt
  ```

  **Commit**: YES | Message: `feat(wall): add wall homography calibration` | Files: [`serve_analyzer/wall_calibration.py`, `tests/test_wall_calibration.py`]

- [x] 3. Define result contracts for JSON, CSV, plots, and warnings

  **What to do**: Define the output contract before implementation. JSON must separate `measured`, `inferred`, `assumed`, `confidence`, `warnings`, and `artifacts`. CSV must flatten one row per serve with stable columns: video, serve_index, impact_time_sec, impact_frame, wall_x_m, wall_y_m, speed_m_s, speed_km_h, speed_mph, landing_x_m, landing_z_m, in_service_box, confidence_score, warning_codes. Plot artifact names must be deterministic.
  **Must NOT do**: Do not store nested JSON blobs inside CSV cells except warning code list joined deterministically.

  **Recommended Agent Profile**:
  - Category: `quick` - Contract/test task with limited code.
  - Skills: [] - No specialized skill required.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 9,11,13 | Blocked By: 1

  **References**:
  - Pattern: `serve_analyzer/serve_evaluation.py` - JSON output and CLI payload pattern.
  - Test: `tests/test_evaluator.py` - JSON stdout/temp-file assertions.

  **Acceptance Criteria**:
  - [ ] Unit tests assert JSON required keys and CSV exact column order.
  - [ ] Warning codes include at minimum `degraded_intrinsics`, `insufficient_track`, `manual_correction_used`, `projection_refused`, `low_calibration_confidence`.

  **QA Scenarios**:
  ```
  Scenario: JSON and CSV contract serialization
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_outputs.TestWallOutputContracts.test_json_and_csv_contract -v`.
    Expected: Test passes; JSON has measured/inferred/assumed/confidence/warnings/artifacts and CSV has exact stable columns.
    Evidence: .sisyphus/evidence/task-3-output-contract.txt

  Scenario: Warning flattening
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_outputs.TestWallOutputContracts.test_warning_codes_flatten_deterministically -v`.
    Expected: Test passes; warning codes are stable and parseable from CSV.
    Evidence: .sisyphus/evidence/task-3-warning-csv.txt
  ```

  **Commit**: YES | Message: `feat(wall): define wall analysis output contracts` | Files: [`serve_analyzer/wall_outputs.py`, `tests/test_wall_outputs.py`]

- [x] 4. Add synthetic wall-impact fixtures and helper generators

  **What to do**: Add test helpers that generate tiny synthetic OpenCV videos with known ball trajectory, known impact frame, known wall plane, and optional blur/occlusion variants. Follow existing `cv2.VideoWriter` and ffmpeg test patterns; keep fixtures generated at runtime, not committed binary videos.
  **Must NOT do**: Do not depend on real `videos/wall/*.MOV` for unit tests.

  **Recommended Agent Profile**:
  - Category: `general` - Test utility implementation plus OpenCV video generation.
  - Skills: [] - Existing project patterns suffice.
  - Omitted: [`vision-tools`] - Synthetic video creation does not require visual interpretation.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6,7,8,13 | Blocked By: none

  **References**:
  - Test: `tests/test_analysis.py` - creates temporary MP4 with `cv2.VideoWriter` in `setUp`.
  - Test: `tests/test_web_clip_service.py` - ffmpeg-generated tiny MP4 integration pattern.

  **Acceptance Criteria**:
  - [ ] Test helper creates a video readable by `cv2.VideoCapture` with expected frame count/fps.
  - [ ] Helper returns expected impact frame and wall coordinates for assertions.

  **QA Scenarios**:
  ```
  Scenario: Synthetic video generation
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_synthetic.TestSyntheticWallVideo.test_generates_readable_known_impact_video -v`.
    Expected: Video opens, fps is fixture value, expected impact frame is present, temp files cleaned up.
    Evidence: .sisyphus/evidence/task-4-synthetic-video.txt

  Scenario: Blur/occlusion variant metadata
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_synthetic.TestSyntheticWallVideo.test_blur_variant_records_expected_uncertainty -v`.
    Expected: Helper returns increased expected uncertainty and still deterministic ground truth.
    Evidence: .sisyphus/evidence/task-4-synthetic-blur.txt
  ```

  **Commit**: YES | Message: `test(wall): add synthetic impact video fixtures` | Files: [`tests/test_wall_synthetic.py`, `tests/wall_test_helpers.py`]

- [x] 5. Build manual calibration CLI for setup and per-video overrides

  **What to do**: Implement a calibration script/CLI that opens a selected frame, supports manual annotation of hook point (2.45m), chair-top points (1m), wall/floor line, vertical/tilt reference lines, center line, 4+ wall-plane reference points, serve contact height, serve contact distance default `6.11m`, camera-wall distance default `1.57m`, and optional intrinsics file path. Output reusable setup JSON and per-video override JSON. Use OpenCV window/click style consistent with existing interactive calibration.
  **Must NOT do**: Do not require calibration for every video if setup metadata exists. Do not make hook/chair alone pretend to be sufficient for homography unless 4+ wall points exist.

  **Recommended Agent Profile**:
  - Category: `general` - CLI and OpenCV interaction.
  - Skills: [] - Existing CLI patterns are sufficient.
  - Omitted: [`browser-debug-tools`] - No browser UI.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 9,12 | Blocked By: 1,2

  **References**:
  - Pattern: `serve_analyzer/cli.py` - `InteractiveCalibrator`, argparse style, `display_frame == start_frame` guard.
  - Pattern: `serve_analyzer/plot_serve.py` - script-style CLI for non-interactive parameters.

  **Acceptance Criteria**:
  - [ ] `python -m serve_analyzer.wall_calibration --help` exits 0 and documents setup/override modes.
  - [ ] Non-interactive CLI test writes valid calibration JSON from supplied coordinates without opening windows.
  - [ ] Missing required contact height or wall references exits with structured error.

  **QA Scenarios**:
  ```
  Scenario: Non-interactive calibration metadata write
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_calibration_cli.TestWallCalibrationCli.test_noninteractive_writes_valid_setup_json -v`.
    Expected: JSON file exists, validates against schema, contains hook/chair/default distances/contact height.
    Evidence: .sisyphus/evidence/task-5-cli-write.txt

  Scenario: Insufficient wall references fail cleanly
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_calibration_cli.TestWallCalibrationCli.test_insufficient_wall_points_returns_error -v`.
    Expected: Exit code nonzero or structured error payload; no traceback.
    Evidence: .sisyphus/evidence/task-5-cli-error.txt
  ```

  **Commit**: YES | Message: `feat(wall): add manual wall calibration CLI` | Files: [`serve_analyzer/wall_calibration.py`, `tests/test_wall_calibration_cli.py`]

- [x] 6. Implement autonomous wall-impact detection with manual correction overlay

  **What to do**: Implement per-video/per-window wall-impact detection. Start with motion/HSV/frame-difference candidate extraction near the calibrated wall plane, use temporal smoothing and plausibility gates, then select impact frame where pre-impact trajectory reaches wall or discontinuity occurs. Allow manual correction metadata to override impact frame and/or pixel point per serve index. Store both autonomous and final corrected values.
  **Must NOT do**: Do not hide autonomous detection when manual correction is used. Do not require existing v6 serve-contact detector to work unchanged on wall videos.

  **Recommended Agent Profile**:
  - Category: `deep` - Ambiguous CV detection and quality gates.
  - Skills: [] - Reuse current OpenCV utilities.
  - Omitted: [`developing-rails-background-jobs`] - Not a Rails/background-job task.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7,9,10 | Blocked By: 2,4

  **References**:
  - Pattern: `serve_analyzer/analysis.py:track_ball_color`, `track_ball_optical_flow`, `track_ball_yolo` - existing tracking utilities.
  - Pattern: `serve_analyzer/serve_attempts_v6.py` - autonomous candidate windows plus rescue/fine voting concept.
  - Test: `tests/test_serve_attempts.py` - mock-heavy video pipeline tests.

  **Acceptance Criteria**:
  - [ ] Synthetic video detects impact frame within ±1 frame and impact pixel within ±5 px.
  - [ ] Manual correction overrides final frame/point and sets `manual_correction_used` warning/code.
  - [ ] Insufficient track returns structured `insufficient_track` warning and refuses speed/projection.

  **QA Scenarios**:
  ```
  Scenario: Autonomous impact detection on synthetic fixture
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_impact.TestWallImpactDetection.test_detects_known_impact_frame_and_point -v`.
    Expected: Impact frame within ±1 and pixel within ±5 px; no manual correction flag.
    Evidence: .sisyphus/evidence/task-6-impact-auto.txt

  Scenario: Manual impact correction override
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_impact.TestWallImpactDetection.test_manual_correction_overrides_autonomous_detection -v`.
    Expected: Final impact equals correction metadata; autonomous candidate remains recorded; warning includes `manual_correction_used`.
    Evidence: .sisyphus/evidence/task-6-impact-manual.txt
  ```

  **Commit**: YES | Message: `feat(wall): detect and correct wall impacts` | Files: [`serve_analyzer/wall_serve.py`, `tests/test_wall_impact.py`]

- [x] 7. Implement pre-wall trajectory and speed estimator with uncertainty

  **What to do**: Fit the final clean pre-impact segment anchored at wall impact. Estimate instantaneous pre-wall speed magnitude in `m_s`, `km_h`, and `mph`. Use at least 4-6 clean pre-impact positions when available; otherwise return structured `insufficient_track`. Use finite-difference/Savitzky-Golay-style smoothing consistent with existing project dependencies. Include uncertainty from ±1 frame impact ambiguity, homography residuals, and degraded intrinsics mode.
  **Must NOT do**: Do not claim spin/drag-corrected exact speed. Do not output speed when depth/trajectory constraints are insufficient.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Numeric model with clear constraints.
  - Skills: [] - Use numpy/scipy already in project.
  - Omitted: [`research-tools`] - Physics decisions are fixed in this plan.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8,9,11 | Blocked By: 1,2,4,6

  **References**:
  - Pattern: `serve_analyzer/analysis.py:compute_velocity_series` - speed units and smoothing conventions.
  - Research finding: use wall impact as hard anchor; monocular depth component is assumption-sensitive.

  **Acceptance Criteria**:
  - [ ] Synthetic known trajectory speed estimate is within configured tolerance (≤10% for fixture MVP).
  - [ ] Output includes speed units and uncertainty interval.
  - [ ] Too few pre-impact points produces `speed=null` and `insufficient_track` warning.

  **QA Scenarios**:
  ```
  Scenario: Known synthetic speed estimation
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_trajectory.TestPreWallSpeed.test_estimates_known_synthetic_speed -v`.
    Expected: `speed_m_s` within ≤10% of fixture truth and km/h/mph conversions match.
    Evidence: .sisyphus/evidence/task-7-speed-known.txt

  Scenario: Too few frames refuses speed
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_trajectory.TestPreWallSpeed.test_refuses_speed_with_too_few_points -v`.
    Expected: Speed fields are null, warning includes `insufficient_track`, no traceback.
    Evidence: .sisyphus/evidence/task-7-speed-refused.txt
  ```

  **Commit**: YES | Message: `feat(wall): estimate pre-impact speed with uncertainty` | Files: [`serve_analyzer/wall_serve.py`, `tests/test_wall_trajectory.py`]

- [x] 8. Implement regulation-court landing projection with refusal modes

  **What to do**: Implement gravity-first court projection using the inferred pre-wall state and regulation tennis court dimensions. Interpret projected landing as equivalent no-wall ball landing on a real court, not wall rebound. Use manual serve contact height and `serve_contact_distance_m=6.11` default to constrain the court-depth model. Output landing coordinates, service-box classification, assumptions, uncertainty/sensitivity, and refusal warnings when underdetermined.
  **Must NOT do**: Do not model rebound, spin, or trained drag unless explicitly optional and disabled by default. Do not project if speed or calibration is refused.

  **Recommended Agent Profile**:
  - Category: `deep` - Physics and coordinate transformation correctness.
  - Skills: [] - Numeric Python only.
  - Omitted: [`developing-containers`] - No container work.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 9,11 | Blocked By: 1,2,4,7

  **References**:
  - Research finding: gravity-only MVP is preferred; drag/spin unknowns should not dominate first version.
  - Metis directive: define “projected landing” explicitly and classify regulation service box.

  **Acceptance Criteria**:
  - [ ] Deterministic synthetic projectile fixture lands within configured tolerance.
  - [ ] Output states assumptions: no-wall continuation, gravity model, contact height, contact distance, regulation court constants.
  - [ ] Missing speed/calibration refuses projection with `projection_refused` warning.

  **QA Scenarios**:
  ```
  Scenario: Synthetic projectile landing
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_projection.TestCourtProjection.test_projects_known_projectile_landing -v`.
    Expected: Landing x/z coordinates match fixture within tolerance; service-box classification is deterministic.
    Evidence: .sisyphus/evidence/task-8-projection-known.txt

  Scenario: Projection refused without speed
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_projection.TestCourtProjection.test_refuses_projection_without_speed -v`.
    Expected: Landing fields null; warning includes `projection_refused`; no misleading court classification.
    Evidence: .sisyphus/evidence/task-8-projection-refused.txt
  ```

  **Commit**: YES | Message: `feat(wall): project serves onto regulation court` | Files: [`serve_analyzer/wall_serve.py`, `tests/test_wall_projection.py`]

- [x] 9. Build wall analysis orchestration CLI

  **What to do**: Add CLI to analyze one video or glob/batch `videos/wall/*.MOV` using setup metadata plus optional per-video override/corrections. It should run calibration load, impact detection, trajectory/speed estimation, projection, and artifact generation wiring. Provide deterministic output directory and filenames. CLI must expose `--metadata`, `--override`, `--output-dir`, `--no-video`, `--no-plots`, and `--manual-corrections` or equivalent.
  **Must NOT do**: Do not modify existing `serve_analyzer.cli` behavior. Do not require web backend.

  **Recommended Agent Profile**:
  - Category: `general` - Orchestration and CLI integration.
  - Skills: [] - Match existing argparse patterns.
  - Omitted: [`frontend-ui-ux`] - No frontend.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 10,11,12,13 | Blocked By: 1,2,3,5,6,7,8

  **References**:
  - Pattern: `serve_analyzer/plot_serve.py` - script CLI generating output artifacts.
  - Pattern: `serve_analyzer/serve_evaluation.py` - `main([...])` testable CLI style.
  - Videos: `videos/wall/IMG_9340.MOV` through `IMG_9347.MOV` for smoke-only manual/local runs, not unit-test dependency.

  **Acceptance Criteria**:
  - [ ] `python -m serve_analyzer.wall_serve --help` exits 0.
  - [ ] CLI integration test on synthetic video writes JSON, CSV, plot, and annotated video unless disabled.
  - [ ] Batch mode produces one result set per input video.

  **QA Scenarios**:
  ```
  Scenario: Synthetic end-to-end CLI
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_cli.TestWallAnalysisCli.test_synthetic_end_to_end_outputs_all_artifacts -v`.
    Expected: Output dir contains JSON, CSV, plot PNG, annotated MP4; JSON parse succeeds; CSV has one row.
    Evidence: .sisyphus/evidence/task-9-cli-e2e.txt

  Scenario: CLI disables optional heavy artifacts
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_cli.TestWallAnalysisCli.test_no_video_no_plots_flags_skip_artifacts -v`.
    Expected: JSON/CSV exist; annotated video and plots are absent; no error.
    Evidence: .sisyphus/evidence/task-9-cli-flags.txt
  ```

  **Commit**: YES | Message: `feat(wall): add wall analysis CLI` | Files: [`serve_analyzer/wall_serve.py`, `tests/test_wall_cli.py`]

- [x] 10. Generate annotated videos and plots for wall analysis

  **What to do**: Add artifact generation: annotated MP4 with ball track, autonomous impact point, corrected final impact point when applicable, wall coordinate label, speed, projected landing summary, and warning codes. Add plots: speed vs time near impact, wall impact scatter/heatmap per video, and court landing scatter with service-box boundaries.
  **Must NOT do**: Do not require human visual approval for tests. Do not overwrite existing artifacts without deterministic naming or explicit overwrite behavior.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - Visual output quality and clarity.
  - Skills: [] - Matplotlib/OpenCV existing project stack.
  - Omitted: [`browser-debug-tools`] - No browser.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 12,13 | Blocked By: 2,6,8,9

  **References**:
  - Pattern: `serve_analyzer/annotate_video.py` - debug video overlays.
  - Pattern: `serve_analyzer/plot_serve.py` - matplotlib speed graph style.

  **Acceptance Criteria**:
  - [ ] Annotated MP4 is generated and readable by `cv2.VideoCapture` in synthetic test.
  - [ ] Plot PNG files exist and are non-empty.
  - [ ] Warning overlay is included when warnings exist; verified by code path/unit assertion, not visual inspection.

  **QA Scenarios**:
  ```
  Scenario: Annotated MP4 generation
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_artifacts.TestWallArtifacts.test_annotated_video_is_readable -v`.
    Expected: MP4 exists, opens with OpenCV, has frame count > 0.
    Evidence: .sisyphus/evidence/task-10-video-artifact.txt

  Scenario: Plot generation
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_artifacts.TestWallArtifacts.test_plots_are_generated_nonempty_pngs -v`.
    Expected: Speed, wall scatter, and court landing PNGs exist and have nonzero file size.
    Evidence: .sisyphus/evidence/task-10-plot-artifacts.txt
  ```

  **Commit**: YES | Message: `feat(wall): render wall analysis artifacts` | Files: [`serve_analyzer/wall_artifacts.py`, `tests/test_wall_artifacts.py`]

- [x] 11. Finalize JSON/CSV serialization with measured/inferred/assumed separation

  **What to do**: Wire result contracts into orchestration. JSON must contain calibration diagnostics, per-serve autonomous/final impacts, speed estimates, landing projections, uncertainties, assumptions, warnings, and artifact paths. CSV must have one flat row per serve and include warning codes. Add parseability tests.
  **Must NOT do**: Do not omit refused projections; include row/result with null fields and warnings.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Output correctness and stable contracts.
  - Skills: [] - No specialized skill.
  - Omitted: [`xlsx`] - User asked CSV, not spreadsheets.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 12,13 | Blocked By: 3,7,8,9

  **References**:
  - Pattern: `serve_analyzer/serve_evaluation.py` - JSON output shape and CLI output tests.
  - Test: `tests/test_evaluator.py` - temp JSON/timestamp parsing.

  **Acceptance Criteria**:
  - [ ] JSON parse test verifies measured/inferred/assumed/confidence/warnings/artifacts structure.
  - [ ] CSV parse test verifies exact columns and one row per serve.
  - [ ] Refused projection row remains present with `projection_refused` warning.

  **QA Scenarios**:
  ```
  Scenario: Parseable complete outputs
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_outputs.TestWallSerialization.test_parseable_json_and_csv_from_analysis_result -v`.
    Expected: JSON and CSV parse; field values match synthetic result; columns stable.
    Evidence: .sisyphus/evidence/task-11-serialization.txt

  Scenario: Refused projection retained
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_outputs.TestWallSerialization.test_refused_projection_still_writes_row_with_warning -v`.
    Expected: CSV row exists with null landing fields and `projection_refused` warning.
    Evidence: .sisyphus/evidence/task-11-refused-row.txt
  ```

  **Commit**: YES | Message: `feat(wall): serialize wall serve outputs` | Files: [`serve_analyzer/wall_outputs.py`, `tests/test_wall_outputs.py`]

- [x] 12. Add real wall-video smoke command and documented sample workflow

  **What to do**: Add a smoke-testable workflow command in documentation/help or a small sample metadata file strategy showing how to run against `videos/wall/*.MOV` after calibration metadata exists. The automated test must use synthetic video; real wall videos should be referenced for user execution only. Include command examples for calibration, analysis, and output inspection.
  **Must NOT do**: Do not commit large generated artifacts. Do not make tests depend on real wall videos.

  **Recommended Agent Profile**:
  - Category: `writing` - Clear workflow docs and command examples.
  - Skills: [] - Technical writing only.
  - Omitted: [`research-tools`] - No external research needed.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 13 | Blocked By: 5,9,10,11

  **References**:
  - Project README commands for existing CLI usage.
  - Videos: `videos/wall/IMG_9340.MOV`, `IMG_9341.MOV`, `IMG_9342.MOV`, `IMG_9343.MOV`, `IMG_9346.MOV`, `IMG_9347.MOV`.

  **Acceptance Criteria**:
  - [ ] Help text or docs include calibration and analysis commands using `videos/wall/*.MOV`.
  - [ ] Automated test verifies documented synthetic command path remains executable.

  **QA Scenarios**:
  ```
  Scenario: Documented synthetic workflow command
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_cli.TestWallAnalysisCli.test_documented_synthetic_workflow_command -v`.
    Expected: Command path in docs/help remains executable in synthetic fixture mode.
    Evidence: .sisyphus/evidence/task-12-documented-command.txt

  Scenario: Real wall video command is documented not executed
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_cli.TestWallAnalysisCli.test_real_wall_video_examples_are_documented_only -v`.
    Expected: Test confirms examples reference `videos/wall/*.MOV` but unit tests do not open those files.
    Evidence: .sisyphus/evidence/task-12-real-video-docs.txt
  ```

  **Commit**: YES | Message: `docs(wall): document wall analysis workflow` | Files: [`README.md`, `serve_analyzer/wall_calibration.py`, `serve_analyzer/wall_serve.py`, `tests/test_wall_cli.py`]

- [x] 13. Integrated validation and hardening pass

  **What to do**: Run the full test suite, fix integration failures caused by wall modules, ensure all new modules have docstrings for public functions, ensure no existing lateral serve behavior changed, and ensure all warning/refusal edge cases are covered. Add final edge-case tests for variable fps metadata, rotated video metadata if feasible via mocks, conflicting per-video override, nonexistent manual correction serve index/frame, and missing intrinsics.
  **Must NOT do**: Do not broaden scope to web UI, ML training, or arbitrary-camera generalization.

  **Recommended Agent Profile**:
  - Category: `deep` - Cross-module QA and edge-case completion.
  - Skills: [] - Existing project test stack.
  - Omitted: [`developing-rspec-tests`] - Python `unittest` project.

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: Final Verification | Blocked By: 1-12

  **References**:
  - Command: `python -m unittest discover -s tests -v`.
  - Project convention: public functions should have descriptive docstrings.
  - Metis edge cases: close serves, occlusion, outside wall plane, degenerate calibration, override conflicts, variable fps, rotated metadata, missing intrinsics, insufficient pre-impact track.

  **Acceptance Criteria**:
  - [ ] `python -m unittest discover -s tests -v` passes.
  - [ ] Existing tests for `analysis`, `cli`, `serve_attempts*`, `serve_evaluation`, and web contracts still pass.
  - [ ] New edge-case tests pass with structured warnings/errors.

  **QA Scenarios**:
  ```
  Scenario: Full suite regression
    Tool: Bash
    Steps: Run `python -m unittest discover -s tests -v`.
    Expected: All tests pass; no existing module regressions.
    Evidence: .sisyphus/evidence/task-13-full-suite.txt

  Scenario: Edge-case warning coverage
    Tool: Bash
    Steps: Run `python -m unittest tests.test_wall_edge_cases -v`.
    Expected: Variable fps, override conflicts, missing intrinsics, nonexistent correction, and outside-plane cases all produce structured outcomes.
    Evidence: .sisyphus/evidence/task-13-edge-cases.txt
  ```

  **Commit**: YES | Message: `test(wall): harden wall analysis edge cases` | Files: [`tests/test_wall_edge_cases.py`, wall modules as needed]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high (+ annotated video artifact inspection using agent-executed media checks, not user approval)
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit once per task where feasible using the specified messages.
- Do not push unless explicitly requested.
- Do not commit generated videos/plots/CSV/JSON except tiny intentional test fixtures if the executor decides they are appropriate; prefer runtime-generated temp fixtures.

## Success Criteria
- The wall pipeline can be calibrated with reusable setup metadata plus per-video overrides.
- Synthetic fixtures prove impact localization, speed estimation, projection, serialization, and artifact generation.
- Real wall videos have documented commands and can be analyzed after user-created calibration metadata.
- Outputs are auditable: measured/inferred/assumed separated, uncertainty included, and limitations visible.
- Existing serve analyzer functionality remains intact.
