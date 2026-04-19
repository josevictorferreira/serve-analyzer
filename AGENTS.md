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
│   ├── multi_serve.py     # Multi-serve detector: YOLO+HSV tracking, serve event detection
├── tests/                 # unittest TestCases
├── notebooks/             # Jupyter analysis notebooks
└── .sisyphus/             # Sisyphus tooling (not project code)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Speed graph script | `serve_analyzer/plot_serve.py` | Generates matplotlib speed profile from video |
| Multi-serve analysis | `serve_analyzer/multi_serve.py` | Detects N serves, toss vs post-contact phases |

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

## SESSION LEARNINGS

### numpy types break JSON serialization
**Lesson:** Always wrap numpy int64/float64 in `int()`/`float()` before JSON output.
**Context:** `json.dumps()` silently fails or truncates output on numpy types.
**Verify:** `python -c "import json; json.dumps({'x': int(value)})"` - no TypeError

### 4K video processing needs frame_skip
**Lesson:** For 4K 60fps video (4000+ frames), use `--frame-skip 4` minimum. Process every Nth frame, interpolate gaps.
**Context:** Full-frame processing times out; frame_skip=4 completes in ~5 min for 4133 frames.
**Verify:** `timeout 300 python -m serve_analyzer.multi_serve video.mp4 --frame-skip 4`

### YOLO sports_ball class doesn't detect tennis balls
**Lesson:** YOLO v8 class 32 (sports_ball) fails on tennis balls. Use HSV color masking as primary or train custom model.
**Context:** 0 detections across 4133 frames; HSV yellow mask (18-45, 80-255, 80-255) works reliably.
**Verify:** Check detection rate in output: "Found ball in X/Y frames"

### multi_serve.py usage
**Lesson:** Run via nix develop + venv activation for ultralytics access.
**Context:** YOLO requires ultralytics which is installed in .venv, not nix directly.
**Verify:** `nix develop --command bash -c "source .venv/bin/activate; python -m serve_analyzer.multi_serve -h"`
