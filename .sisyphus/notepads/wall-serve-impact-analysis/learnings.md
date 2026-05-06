# Learnings: wall-serve-impact-analysis

## Conventions
- Python `unittest` only; run with `python -m unittest discover -s tests -v`.
- No pyproject/requirements/pip — Nix flake is the only dependency manager.
- Synthetic OpenCV videos created at runtime via `cv2.VideoWriter` (see `tests/test_analysis.py`).
- CLI pattern: `main([...])` testable; `argparse`; `display_frame == start_frame` guard in interactive mode.
- Output pattern: JSON to stdout/file, see `serve_analyzer/serve_evaluation.py`.
- Public functions get descriptive docstrings.

## Module layout for this plan
- `serve_analyzer/wall_calibration.py` — schema + homography + CLI for calibration.
- `serve_analyzer/wall_serve.py` — analysis orchestration + impact + speed + projection + CLI.
- `serve_analyzer/wall_outputs.py` — JSON/CSV serialization contracts.
- `serve_analyzer/wall_artifacts.py` — annotated MP4 + plots.
- `tests/wall_test_helpers.py` — synthetic fixtures.
- `tests/test_wall_*.py` — unittest TestCases per module.

## Coordinate system (canonical)
- Wall frame: origin at floor/wall intersection directly under center reference; `x_m` along wall (camera-right positive), `y_m` up, `z_m` away from wall toward server.
- Court frame: same centerline origin; regulation tennis court constants; serve contact at `z_m=6.11` by default.

## Output contracts
- JSON sections: `measured`, `inferred`, `assumed`, `confidence`, `warnings`, `artifacts`.
- CSV columns (stable, exact order): `video, serve_index, impact_time_sec, impact_frame, wall_x_m, wall_y_m, speed_m_s, speed_km_h, speed_mph, landing_x_m, landing_z_m, in_service_box, confidence_score, warning_codes`.
- Warning codes (minimum): `degraded_intrinsics`, `insufficient_track`, `manual_correction_used`, `projection_refused`, `low_calibration_confidence`.

## Regulation constants (meters)
- Court length: 23.77; singles width: 8.23; doubles width: 10.97.
- Service box: 6.40 deep from net x 4.115 wide (half singles).
- Net height (center): 0.914.

## [2026-05-05] Task 1: Schema + validation findings

- `serve_analyzer/wall_calibration.py` created with dataclasses `WallCalibration`, `WallReferencePoint`, `HookReference`, `ChairReference`, `Intrinsics`.
- Custom exception `WallCalibrationError(ValueError)` used for all validation failures; tests assert message contains field name.
- `from_dict()` validates on construction; `to_dict()` round-trips cleanly.
- `tests/test_wall_calibration.py` — 13 tests, all pass. Used `unittest.TestCase` with `subTest` for intrinsics source enumeration.
- `python -m unittest discover -s tests -p "test_wall_calibration.py" -v` is the correct invocation inside `nix develop` (plain `python -m unittest tests.test_wall_calibration` fails because `tests` is not on PYTHONPATH as a package).
- No `__init__.py` in `tests/` — the project relies on `unittest discover` with `-s tests`.
- Evidence saved to `.sisyphus/evidence/task-1-schema.txt`.

## Task 4 Learnings (2026-05-05)

### Synthetic Video Generation
- `cv2.VideoWriter_fourcc(*"mp4v")` produces readable MP4s that `cv2.VideoCapture` can re-open.
- Default parameters must be chosen carefully so `start_x = wall_x_px - speed * impact_frame` stays within frame bounds (≥ ball_radius).
- For tests with `impact_frame=60` and `wall_x_px=240`, `ball_speed_px_per_frame` must be ≤ 4.0 to keep `start_x ≥ 0` (with `ball_radius=6`, need `start_x ≥ 6`).
- Using `impact_frame=30` with `ball_speed_px_per_frame=4.0` gives `start_x = 120`, well within bounds.

### Test Parameters Used
- `width=640, height=480, fps=60, total_frames=90, impact_frame=30, ball_speed_px_per_frame=4.0, wall_x_px=240`
- This yields impact at frame 30, pixel (240, 240), with 32 visible ball positions.

### Import Conventions
- Tests in `tests/` must use `from wall_test_helpers import ...` (not `from tests.wall_test_helpers import ...`) because `tests` is a namespace package without `__init__.py` in the nix develop environment.
- `python -m unittest discover -s tests -v -k wall_synthetic` is the correct invocation pattern.

### Blur Testing
- `blur_sigma=2.0` produces `expected_uncertainty_px = 2.0` (baseline 1.0 + 2.0 * 0.5).
- Gaussian blur with `ksize = ceil(sigma * 3) * 2 + 1` ensures odd kernel size.

## 2026-05-05 — Task 3: Output Contract Definition

### Symbols Created
- `WallAnalysisResult` dataclass with 6 fields: measured, inferred, assumed, confidence, warnings, artifacts
- `to_json()` returns dict with exactly 6 top-level keys — no extras, no missing
- `serve_to_csv_row()` maps serve dict → tuple in `CSV_COLUMNS` order; accepts warnings from either `"warnings"` or `"warning_codes"` keys
- `write_csv()` writes header from `CSV_COLUMNS` then rows using stdlib `csv.writer`
- `PLOT_FILENAMES` dict with 3 deterministic templates: speed, trajectory, wall_impact
- `_flatten_warning_codes()` produces sorted semicolon-joined string (no JSON braces)

### Conventions Followed
- Imported `CSV_COLUMNS` and `WARNING_CODES` from `wall_calibration` — never redefined
- Python stdlib only: `csv`, `json`, `dataclasses`, `pathlib`
- `unittest` (not pytest); `tempfile.TemporaryDirectory()` for CSV write tests
- No modifications to `wall_calibration.py`, `analysis.py`, `cli.py`, or any existing files

### Test Results
- 10 new tests in `TestWallOutputContracts` + 3 supplementary test classes — all passing
- 26 total wall tests (including 13 from wall_calibration, 3 from wall_synthetic) — zero regressions
- Run: `python -m unittest discover -s tests -p 'test_wall_*.py' -v`

## 2026-05-05 — Task 2: Homography calibration primitives

- `serve_analyzer/wall_calibration.py` now exposes `compute_wall_homography`, `compute_reprojection_rms`, `pixel_to_wall`, `wall_to_pixel`, and `undistort_points`.
- Homography maps wall-plane meters → pixels; callers invert it with `np.linalg.inv(H)` for `pixel_to_wall`.
- `compute_wall_homography(..., intrinsics=Intrinsics(source="approx_exif", ...))` sets `residuals["degraded_intrinsics"] = True`.
- Degenerate image or world point sets raise `WallCalibrationError("calibration_degenerate", details)` before OpenCV fitting.
- Test invocation that works with this repo layout: `nix develop --command python -m unittest discover -s tests -p "test_wall_calibration.py" -v` (16 tests after Task 2).
- All wall tests passed with `nix develop --command python -m unittest discover -s tests -p 'test_wall_*.py' -v` (29 tests).
