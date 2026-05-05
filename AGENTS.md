# PROJECT KNOWLEDGE BASE
**Generated:** 2026-05-05
**Commit:** 4122601
**Branch:** main

## OVERVIEW
Tennis serve velocity estimation from lateral video. Python package using OpenCV template matching + numpy/scipy for tracking and velocity computation.
Web app (Vite React + FastAPI) for interactive analysis.
Multi-serve detector with 6 algorithmic versions (v1-v6).

## STRUCTURE
```
./
├── flake.nix              # Nix dev environment (ONLY dependency management)
├── serve_analyzer/        # Python package
│   ├── __init__.py
│   ├── __main__.py        # Thin wrapper → cli.main()
│   ├── analysis.py        # Core: scale_factor, velocity_series, track_ball_template
│   ├── cli.py             # CLI entry + InteractiveCalibrator
│   ├── plot_serve.py      # Speed graph generator script
│   ├── multi_serve.py     # Multi-serve detector: YOLO+HSV tracking, serve event detection
│   ├── serve_attempts.py  # v1-v6 serve detection algorithms
│   ├── serve_evaluation.py # Benchmark detector output vs manual timestamps
│   └── benchmark_detectors_v2.py # V2+ timing refinement with cached detections
├── tests/                 # unittest TestCases
├── notebooks/             # Jupyter analysis notebooks
├── web/                   # Web app (Vite React + FastAPI backend)
│   ├── backend/           # FastAPI on port 8000
│   │   ├── main.py        # Entry point (non-standard: not app.py)
│   │   ├── app.py         # FastAPI routes
│   │   ├── state.py       # Job state machine
│   │   ├── schemas.py     # Pydantic models
│   │   └── services/      # clip_service.py, detection_services.py
│   ├── src/               # React frontend
│   └── package.json       # npm managed
├── models/                # YOLO training data (roboflow exports)
├── annotation_exports/    # Training run artifacts
└── .sisyphus/             # Sisyphus tooling (not project code)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Speed graph script | `serve_analyzer/plot_serve.py` | Generates matplotlib speed profile from video |
| Multi-serve analysis | `serve_analyzer/multi_serve.py` | Detects N serves, toss vs post-contact phases |
| Serve detector v1-v6 | `serve_analyzer/serve_attempts*.py` | v6 achieves 100% recall on cached benchmark |
| Serve evaluation | `serve_analyzer/serve_evaluation.py` | Compare candidates/selected vs manual timestamps |
| Web app frontend | `web/src/` | React + Vite + Tailwind v4 + shadcn/ui |
| Web app backend | `web/backend/` | FastAPI on port 8000, `python -m web.backend` to start |
| Web clip service | `web/backend/services/clip_service.py` | ffmpeg-based per-serve MP4 extraction |
| Web detection services | `web/backend/services/detection_services.py` | Detector version routing (v1-v6) |

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
- v6 detector is autonomous-only (never accepts `expected_serves`)
- Backend entry point is `web/backend/main.py` (non-standard; typically `app.py`)

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

# Web backend (local FastAPI)
python -m web.backend    # starts on port 8000 (NOT python -m web.backend.app)

# Web frontend dev
cd web && npm run dev    # Vite on port 5173, proxies /api and /clips to :8000
cd web && npm run build  # production build
cd web && npm test -- --run  # vitest
```

## NOTES
- MVP tool — approximate velocities only from single lateral view
- Accuracy depends on calibration point quality and camera angle
- Ball tracking uses simple template matching (not optical flow or ML)
- Darwin/aarch64 only (hardcoded in flake.nix)
- No CI/CD (no .github/workflows, no Makefile)

