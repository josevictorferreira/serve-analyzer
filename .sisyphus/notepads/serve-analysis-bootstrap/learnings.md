#HM|# Learnings — serve-analysis-bootstrap
#KM|
#NZ|## flake.nix bootstrap (T1)
#RW|
#VM|- `nixpkgs/nixos-unstable` as of 2026-04-01 has `python3Packages.opencv4` (not `cv2`) and `python3Packages.pillow` (not `Pillow`)
#SN|- `cv2` and `Pillow` are the pip-compatible import names, but nixpkgs attribute names differ — always verify attrs with `nix eval nixpkgs#legacyPackages.<system>.python3Packages.<name>.outPath`
#TR|- `nix develop` auto-generates `flake.lock` on first run; this is expected and should be committed
#PY|- All three verification commands pass: `python --version`, `jupyter --version`, `python -c "import cv2, numpy, matplotlib"`
#MW|- Using `buildInputs` (not `nativeBuildInputs`) is correct for runtime tools like Python, Jupyter, ffmpeg
#NN|- `shellHook` with `PYTHONPATH=$PWD` is a lightweight convenience for notebook/script development
#TJ|
#RJ|## Core analysis module (T2)
#BQ|
#WW|- Package structure: `serve_analyzer/__init__.py`, `analysis.py`, `cli.py` + `__main__.py`
#HX|- Core functions are pure and testable: `compute_scale_factor()`, `compute_velocity_series()`, `track_ball_template()`
#TH|- Velocity computation: frame-to-frame displacement → real-world distance (using scale) → speed in m/s → smoothed with moving average → converted to km/h
#YR|- Ball tracking: template matching with adaptive template updates, confidence threshold (0.5), and bounded search region
#NV|- CLI supports both interactive (click-based) and non-interactive (scriptable) modes
#JR|- All unit tests pass (17 tests covering scale math, velocity computation, and video I/O)
#KY|- CLI --help works and shows limitations clearly
#HB|- Package is importable: `from serve_analyzer.analysis import compute_velocity_series`
#ZP|
#RR|## T2 follow‑up fixes
#KW|
#KY|- Interactive calibration now defaults to `start_frame` when `--display-frame` is not supplied, matching the intended behavior.
#ZP|- Velocity summary `duration_sec` correctly uses frame intervals: `(len(centers) - 1) / fps`.
#WB|- Added tests covering both defaults and duration calculation, confirming the fixes.
#HQ|
#JR|## T4: Minimal usage docs
#ZM|
#HX|- README.md created with dev shell entry, CLI commands, and notebook workflow
#NP|- CLI help verified to match documented options
#NW|- Notebook path set to `notebooks/serve_analysis.ipynb` (to be created in T3)
#MQ|- Docs mention limitations: approximate output, manual calibration dependency, single-view only
#ZS|- Commands are copy-pasteable and reference real repo files
#TB|
#BP|## T3: Jupyter notebook
#MR|
#KB|- Notebook generated via Python JSON (avoided control-char issues in notebook JSON)
#GH|- Notebook uses `DEMO_MODE=True` by default for headless/CI execution with synthetic centers
#QH|- Guarded cells: real-video cells only execute when `DEMO_MODE=False`
#PH|- Uses `matplotlib.use('Agg')` for non-interactive backend
#YJ|- Notebook reuses `serve_analyzer.analysis` functions (compute_scale_factor, compute_velocity_series, track_ball_template, get_video_info)
#NQ|- Synthetic demo generates 48 centers at 240fps simulating ~24 km/h serve (corrected from ~150 km/h)
#KY|- Parameter tuning table included in notebook for iterative workflow
#MH|- Limitations section included; honest about single-view approximation
#HB|- nbconvert execution passes (exit 0) without requiring actual video file
#BP|
#BP|## T3 follow-up fix
#MR|
#KB|- Synthetic demo had speed mismatch: comments claimed ~150 km/h but actual output was ~24 km/h
#GH|- Root cause: scale_factor=0.00333 m/pixel (from 300px=1m calibration) makes 8.3 px/frame ≈ 24 km/h, not 150 km/h
#QH|- To get 150 km/h at this scale would require ~83 px/frame — unrealistic for a real ball
#PH|- Fix: updated comments/markdown to state ~24 km/h instead of ~150 km/h (comments were cleaner to fix than data)

## T4 follow-up: README fixes

- Fixed test command: `python -m pytest tests/` → `python -m unittest discover -s tests -v` (pytest not in dev shell)
- Fixed IMG_9259.MOV comment: removed "(not included)" since file exists in repo
- Both verification commands pass
- Updated CLI docs: `--display-frame` must be omitted or equal to `--start-frame` in interactive mode.
