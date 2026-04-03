# Decisions — serve-analysis-bootstrap

## flake.nix dev shell (T1)

- **nixos-unstable over nixos-24.11**: Chosen because it has more current aarch64-darwin video stack (opencv 4.13 vs older), and this is a greenfield project with no legacy compat constraint
- **opencv4 instead of opencv**: nixpkgs attribute for OpenCV Python bindings is `python3Packages.opencv4` (not `cv2`); the import inside Python remains `import cv2`
- **pillow (lowercase)**: nixpkgs attribute name is `python3Packages.pillow`; Python import is still `PIL`
- **ffmpeg_6 over ffmpeg_7**: opencv4 on aarch64-darwin links against ffmpeg_6 in this nixpkgs revision; pinning avoids a rebuild
- **No pip/venv**: Primary env bootstrap via nixpkgs packages only; pip remains available inside the shell for edge cases
- **Single system (aarch64-darwin)**: Repo is macOS-only for now; cross-platform flake can be added later if needed
- **Packages included**: python3, jupyter, notebook, numpy, scipy, matplotlib, pandas, opencv4, pillow, scikit-image, ffmpeg_6, ipykernel — all practical for video analysis script + notebook MVP

## Core implementation approach (T2)

- **Manual calibration over automatic**: 2-point manual scale chosen for reliability and transparency over error-prone automatic detection
- **Template matching over optical flow/ML**: Simpler, more interpretable, no heavyweight dependencies. Good enough for MVP.
- **Smoothing with moving average (window=3)**: Reduces noise from frame-to-frame velocity jitter while maintaining responsiveness
- **Interactive + non-interactive modes**: CLI supports both click-based calibration and command-line args for scripting
- **Pure functions in analysis.py**: All core logic is testable without video I/O
- **Separate CLI from core**: analysis.py has no CLI dependencies, cli.py handles user interaction
- **Honest limitations**: CLI explicitly states this is approximate, manual, and single-view only

## T2 follow‑up decisions

- Default `--display-frame` to `start_frame` when omitted, preserving interactive override semantics.
- Compute `duration_sec` as `(len(centers) - 1) / fps` to reflect true tracked interval.
- Keep CLI signature unchanged; only internal defaults adjusted.

QP|## T3 notebook decisions
YJ|
ZW|- **DEMO_MODE as default**: Notebook defaults to synthetic data so CI/headless execution works without real video
KM|- **Guarded real-video cells**: Cells that need the actual MOV only run when `DEMO_MODE=False`
QH|- **Reuse over duplication**: Notebook imports from `serve_analyzer.analysis` instead of rewriting logic in cells
NM|- **Headless-safe plotting**: Uses `matplotlib.use('Agg')` backend and saves to /tmp for non-interactive execution
YJ|- **Configuration at top**: All user-tunable parameters in one clearly marked cell
KB|- **Iterative tuning documented**: Parameter tuning table helps users understand what to adjust when results look wrong
- Documented that `--display-frame` must be omitted or equal to `--start-frame` in interactive mode to avoid mismatched frame display.
