# PROJECT KNOWLEDGE BASE

Before starting any implementation, read `AGENTS.md or CLAUDE.md` for project-specific lessons and gotchas.

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
├── web/                   # Web app (Vite React + FastAPI backend)
│   ├── backend/           # FastAPI local API runner
│   ├── src/               # React frontend
│   └── package.json       # npm managed
└── .sisyphus/             # Sisyphus tooling (not project code)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Speed graph script | `serve_analyzer/plot_serve.py` | Generates matplotlib speed profile from video |
| Multi-serve analysis | `serve_analyzer/multi_serve.py` | Detects N serves, toss vs post-contact phases |
| Web app frontend | `web/src/` | React + Vite + Tailwind v4 + shadcn/ui |
| Web app backend | `web/backend/` | FastAPI on port 8000, `python -m web.backend` to start |
| Web clip service | `web/backend/services/clip_service.py` | ffmpeg-based per-serve MP4 extraction |

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

### Keep serve detection and evaluation split
**Lesson:** Keep `serve_analyzer.serve_attempts` detector-only; put timestamp parsing/matching in `serve_analyzer.serve_evaluation` only.
**Context:** Mixing timestamps into detector flow invalidates real inference and caused a full redesign.
**Verify:** `python -m serve_analyzer.serve_attempts video.mov --output out.json` runs without `--timestamps-file`

### Candidate pool is stronger than selector
**Lesson:** Trust multi-profile candidate generation first; tune final ranking/suppression carefully instead of rewriting detection.
**Context:** Real serves were already present in the candidate pool; most failures came from over-aggressive top-K ranking.
**Verify:** Run detector then evaluator and compare `selected_serves` vs post-hoc matched `attempts`

### Frame skip changes pool quality
**Lesson:** Use `frame_skip=4` for practical default speed, but test `frame_skip=2` when count/ranking quality is the blocker.
**Context:** Denser tracking materially improved candidate-pool quality on `video.mov`, even when selector quality still lagged.
**Verify:** Compare `python -m serve_analyzer.serve_attempts video.mov --frame-skip 2|4 --output out.json`

### Autonomous serve count trigger
**Lesson:** Use omitted `--expected-serves` / `expected_serves=None` to enable autonomous count inference; use a positive integer only when forcing an exact count.
**Context:** The detector treats `None` as inference mode, while `0` or negative counts are invalid and do not mean “auto”.
**Verify:** `python -m serve_analyzer.serve_attempts video.mov --output out.json` shows `"count_inferred": true`; adding `--expected-serves 8` flips it to false.

### Ignore nested research repos
**Lesson:** Add cloned research repos or other nested `.git` directories to `.gitignore` before `git add`; if staged accidentally, remove them from the outer index with `git rm --cached -r -f`.
**Context:** Embedded repos trigger add warnings and pollute outer-repo commits without including their contents.
**Verify:** `git status --short` should not show `tennis-analysis-with-cv/` or `tennis_serve_speed/` in the outer repo.

### Validate input BEFORE mutating state
**Lesson:** In FastAPI (or any stateful handler), validate request properties (content type, extension, size) BEFORE calling `reset_state()` or `set_state()`. Rejected requests must not leave state stuck in `uploading`.
**Context:** Invalid upload after `set_state(uploading)` blocked all future uploads with 409 until manual reset.
**Verify:** Send invalid content type to upload endpoint, then `GET /api/job` — should return `idle`, not `uploading`.

### Pydantic response models must match runtime data shapes
**Lesson:** When FastAPI schemas declare `clips: List[str]` but the handler stores `List[Dict]`, `response_model` validation fails at serialization time. Keep schema types aligned with actual runtime data.
**Context:** Done-state `GET /api/job` threw `ResponseValidationError` because `clips` was declared `List[str]` but stored as metadata dicts.
**Verify:** Check `web/backend/schemas.py` field types match what `state.py` and `app.py` actually store.

### Subagent "move" instructions leave duplicate blocks
**Lesson:** When delegating "move validation before state change", explicitly instruct: "Remove the OLD validation block after the state change — do NOT leave duplicates." Subagents often ADD code at new location without REMOVING from old location.
**Context:** Upload validation was duplicated (lines 63-71 AND 76-84) after subagent moved it before state change but forgot to remove the old copy.
**Verify:** After any "move/reorder" delegation, grep for duplicate function calls or validation blocks.

### Nix develop can serve stale derivations
**Lesson:** If `flake.nix` lists a package but `nix develop` can't import it, the nix store derivation may be stale. Pragmatic fallback: `pip install <pkg>` into `.venv` (which already uses `--system-site-packages`). Don't waste time debugging nix cache.
**Context:** `fastapi` was in `flake.nix` pythonEnv but nix store site-packages didn't contain it. `pip install fastapi uvicorn python-multipart` into .venv resolved it immediately.
**Verify:** `source .venv/bin/activate && python -c 'import fastapi'` — no ModuleNotFoundError

### Separate detector version from tracking backend
**Lesson:** Keep web `detector_version` (`v1`, `v2`) separate from low-level tracking `detector` (`yolo`, `tracknetv2`) when adding new serve algorithms.
**Context:** The web API already used `detector` for the tracking backend; overloading it would break labels, estimates, and existing clients.
**Verify:** `GET /api/detectors` lists versions, while `GET /api/job` can include both `detector_version` and `detector`.

### V2 outputs must include frame-indexed tracks
**Lesson:** Any detector service used by the web backend must return JSON-safe, frame-indexed `positions` and `raw_positions`, not just selected serves.
**Context:** `clip_service.generate_clips()` uses those tracks for ball overlay metadata; omitting them breaks clip generation even when serve times are valid.
**Verify:** `python -m unittest discover -s tests -p 'test_web_*' -v` and check v2 output has `positions`/`raw_positions`.

### Benchmark refinements from cached detections first
**Lesson:** For v2+ timing-refinement work, iterate with `--input-detections` cached from a known v1 run before attempting full video inference.
**Context:** Full uncached v2 on 4K video can spend 10+ minutes regenerating the candidate pool; cached runs isolate refinement quality quickly.
**Verify:** `python -m serve_analyzer.benchmark_detectors_v2 video.mov --timestamps-file timestamps_video.txt --input-detections benchmark_outputs/final_yolo_8_match_frame_skip_4/yolo_detections.json --expected-serves 8`
