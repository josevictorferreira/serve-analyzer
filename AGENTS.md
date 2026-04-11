# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-03
**Commit:** e9e132f
**Branch:** main

## OVERVIEW
Tennis serve velocity estimation from lateral video. Python package using OpenCV template matching + numpy/scipy for tracking and velocity computation.

## STRUCTURE
```
./
├── flake.nix              # Nix dev environment (ONLY dependency management)
├── serve_analyzer/        # Python package
│   ├── __init__.py
│   ├── __main__.py        # Thin wrapper → cli.main()
│   ├── analysis.py        # Core: scale_factor, velocity_series, track_ball_template
│   ├── cli.py             # CLI entry + InteractiveCalibrator
│   └── plot_serve.py      # Speed graph generator script
├── tests/                 # unittest TestCases
├── notebooks/             # Jupyter analysis notebooks
└── .sisyphus/             # Sisyphus tooling (not project code)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Speed graph script | `serve_analyzer/plot_serve.py` | Generates matplotlib speed profile from video |

## CONVENTIONS (THIS PROJECT)
- **No pyproject.toml/setup.cfg** — Nix flake is the ONLY build/dep mechanism
- **unittest** (not pytest) — run with `python -m unittest discover -s tests -v`
- **Descriptive docstrings** on all public functions
- **Tuple unpacking** for multi-return values
- **meters/pixel** scale factor (not pixels/meter)

## ANTI-PATTERNS (THIS PROJECT)
- **DO NOT** use `pip install`, `requirements.txt`, or virtualenvs — use `nix develop`
- **DO NOT** use pytest — this project uses unittest
- **DO NOT** assume 3D reconstruction — single lateral view only (MVP)

## UNIQUE STYLES
- `display_frame` must equal `start_frame` in interactive mode (enforced in `run_analysis()`)
- Template matching uses 0.5 confidence threshold
- Smoothing window default is 3 frames; pass `smoothing_window=1` to disable

## COMMANDS
```bash
# Development shell
nix develop

# Run analysis (interactive)
python -m serve_analyzer.cli video.mp4 --real-distance 1.0

# Generate speed graph (scripted, non-interactive)
python -m serve_analyzer.plot_serve video.mp4 \
    --cal-p1 100 200 \
    --cal-p2 400 200 \
    --real-distance 1.0 \
    --ball-pos 320 240 \
    --start-frame 50 \
    --output speed_graph.png

# Run tests
python -m unittest discover -s tests -v

# Jupyter notebooks
jupyter notebook notebooks/
```

## NOTES
- MVP tool — approximate velocities only from single lateral view
- Accuracy depends on calibration point quality and camera angle
- Ball tracking uses simple template matching (not optical flow or ML)
- Darwin/aarch64 only (hardcoded in flake.nix)
