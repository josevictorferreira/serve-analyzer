# Serve Analyzer

Estimate tennis serve velocity from lateral video.

## Quick Start

```bash
# Enter the development shell
nix develop

# Verify environment
python --version
jupyter --version
```

## CLI Usage

```bash
# Show help
python -m serve_analyzer.cli --help

# Interactive mode (click to calibrate and mark ball)
python -m serve_analyzer.cli video.mp4 --real-distance 1.0

# Non-interactive mode (for scripting)
python -m serve_analyzer.cli video.mp4 \
    --cal-p1 100 200 \
    --cal-p2 400 200 \
    --real-distance 1.0 \
    --start-frame 45 \
    --ball-pos 320 240
```

### Calibration

The tool requires two-point calibration:

1. **Interactive mode**: Click two points with a known real-world distance, then click the ball position
2. **Manual mode**: Use `--cal-p1` and `--cal-p2` with pixel coordinates

### Key CLI Options

| Option | Description |
|--------|-------------|
| `--real-distance` | Real-world distance between calibration points (meters, required) |
| `--cal-p1 X Y` | First calibration point in pixels |
| `--cal-p2 X Y` | Second calibration point in pixels |
| `--ball-pos X Y` | Initial ball position in pixels |
| `--start-frame N` | Frame to start tracking (default: 0) |
| `--display-frame N` | Frame to display for interactive calibration (default: start-frame; must be omitted or equal to --start-frame in interactive mode) |
| `--max-frames N` | Maximum frames to track |
| `--output FILE` | Save results to JSON file |

## Notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Or open the analysis notebook directly
jupyter notebook notebooks/serve_analysis.ipynb
```

## Serve Detector Versions

The multi-serve detector has six historical versions. They all keep inference
separate from evaluation: detector modules output candidate serve times, while
`serve_analyzer.serve_evaluation` compares those candidates to manual timestamp
annotations such as `timestamps_video.txt`.

### Version summary

| Version | Implementation | Main idea | Does well | Tradeoffs |
|---------|----------------|-----------|-----------|-----------|
| `v1` | `serve_analyzer/serve_attempts.py` | Baseline candidate generator and selector built from YOLO/TrackNetV2 ball tracks, trajectory peaks, toss evidence, and post-contact speed estimates. | Simple, debuggable, and keeps a broad candidate pool that usually contains the real serves. | Contact timing often lands late because it scores velocity peaks, not necessarily racket-ball contact; false positives remain in the full pool. |
| `v2` | `serve_analyzer/serve_attempts_v2.py` | Starts from the v1 pool, then adds continuity gating, short-gap interpolation, ball-history scoring, and frame-difference motion cues around candidates. | Rejects obvious tracking jumps and adds body/racket motion evidence. | The tighter selected set can drop true serves; timing is still driven by speed/motion peaks. |
| `v3` | `serve_analyzer/serve_attempts_v3.py` | Reimplements the v2-style refinement with Savitzky-Golay speed smoothing, Kalman smoothing/outlier gating, top-k peak speed, and optional audio onset matching. | Preserves sharper speed peaks than moving averages and records richer diagnostics such as `v3_kalman_stats`, audio matches, and smoothed peak speed. | On sparse detections, Kalman/audio complexity can overfit or shift timing; this cached benchmark still missed three manual serves. |
| `v4` | `serve_analyzer/serve_attempts_v4.py` | Returns to the v1 candidate pool and refines contact time by searching backward for the toss apex/direction change, with optional audio cross-check and recomputed peak speeds. | Best cached mean timing error; fixes the main v1 issue where velocity peak is later than contact. | Still depends on the v1 pool, so it cannot recover serves absent from that pool; apex search can shift too far if the track is noisy. |
| `v5` | `serve_analyzer/serve_attempts_v5.py` | Keeps v1's peak frame as the anchor, uses apex detection only to cap excessive backward shifts, then applies post-selection quality gates and recomputes smoothed peak speed. | Similar recall to v4 with less aggressive timing shifts and near-zero signed bias in the cached run. | Slightly worse max error than v4 in the cached benchmark; no audio/Kalman refinement. |
| `v6` | `serve_analyzer/serve_attempts_v6.py` | **Autonomous-only** detector that never accepts `expected_serves`. It runs a two-stage pipeline: (1) a coarse v1 pass at `frame_skip=4` to find candidate windows, plus motion-HSV rescue windows for uncovered time ranges; (2) a fine per-frame pass inside those windows where YOLO and motion-HSV vote on ball positions (optional TrackNetV2 as third voter). Positions are fused, interpolated, and fed back into the v1 event generator to rebuild the candidate pool. A speed-rescue step adds candidates for uncovered motion-HSV windows with strong velocity spikes. Finally, v5 hybrid contact refinement and peak-velocity recomputation are applied to the full pool before autonomous selection with artifact filtering and wider non-max suppression. | Achieves 100% recall (8/8) on the cached benchmark, including the late 62s serve that all previous versions missed. Selected precision is exact (8 selected, 0 unmatched). No user-supplied serve count needed. | More expensive than v5 because it reopens the video for fine-window voting; selected timing is less precise than candidate-pool timing on the cached benchmark. The internal pool hint is fixed at 12 and not configurable. |

The web backend exposes these versions through
`web/backend/services/detection_services.py`. `SERVE_ANALYZER_DETECTOR_VERSION`
sets the default version, and API callers can request a specific
`detector_version`. The low-level tracking backend (`yolo` or `tracknetv2`) is a
separate choice from the detector version.

### Benchmark against manual timestamps (autonomous mode)

The table below compares detector outputs against `timestamps_video.txt`,
whose manual serve times are `13, 19, 28, 32, 37, 48, 52, 62` seconds.
All versions ran **without** `--expected-serves` (autonomous count inference).
The evaluation uses `serve_analyzer.serve_evaluation.summarize_serve_attempts`
with a 3.0-second matching tolerance against the **candidate pool**.
Lower timing error is better.

Use `--source selected_serves` when benchmarking final detector choices instead
of broad candidate-pool recall.

| Version | Candidates | Selected | Matched / 8 | Recall | Mean abs. error | Max abs. error | Mean signed delta | Missed (s) |
|---------|------------|----------|-------------|--------|-----------------|----------------|-------------------|------------|
| `v1` | 21 | 9 | 7 | 87.5% | 0.464 s | 1.475 s | +0.446 s | 62 |
| `v2` | 9 | 9 | 5 | 62.5% | 1.419 s | 2.497 s | +0.624 s | 13, 32, 62 |
| `v3` | 9 | 9 | 5 | 62.5% | 1.414 s | 2.708 s | +1.011 s | 13, 32, 62 |
| `v4` | 21 | 9 | 7 | 87.5% | 0.394 s | 0.824 s | -0.109 s | 62 |
| `v5` | 21 | 8 | 7 | 87.5% | 0.399 s | 0.958 s | +0.010 s | 62 |
| `v6` | 26 | 8 | 8 | 100.0% | 0.509 s | 1.814 s | +0.208 s | none |

For `v6`, the final `selected_serves` benchmark also matched all 8 manual
timestamps with 8 selected serves and 0 unmatched selected candidates. Its
selected-source timing error was 1.260 s mean absolute error and 2.585 s max
absolute error.

To regenerate comparable results, run a detector-only command and then evaluate
the saved JSON against the manual timestamps:

```bash
# Run detector autonomously (no --expected-serves)
python -m serve_analyzer.serve_attempts video.mov \
    --frame-skip 4 \
    --output run_v1_auto.json

# Evaluate against manual timestamps
python -m serve_analyzer.serve_evaluation \
    --detection-json run_v1_auto.json \
    --timestamps-file timestamps_video.txt \
    --tolerance-sec 3.0 \
    --source candidates \
    --output eval_v1_auto.json
```

Use `serve_analyzer.serve_attempts_v2`, `serve_attempts_v3`,
`serve_attempts_v4`, `serve_attempts_v5`, or `serve_attempts_v6` for later
detector versions. For v2–v6, the cached v1 pool can be reused via
`--input-detections` to skip
the expensive YOLO detection phase:

```bash
# Cache the v1 pool once
python -c "from serve_analyzer.serve_attempts import detect_serve_candidates; \
import json; r = detect_serve_candidates('video.mov', frame_skip=4); \
json.dump(r, open('cached_v1_pool.json','w'), default=str)"

# Run v6 using cached pool, then evaluate final selected serves
python -m serve_analyzer.serve_attempts_v6 video.mov \
    --frame-skip 4 --input-detections cached_v1_pool.json \
    --output run_v6_auto.json

python -m serve_analyzer.serve_evaluation \
    --detection-json run_v6_auto.json \
    --timestamps-file timestamps_video.txt \
    --tolerance-sec 3.0 \
    --source selected_serves \
    --output eval_v6_selected_auto.json
```

## Limitations

This is an MVP tool providing **approximate** velocity estimates:

- Manual calibration required (accuracy depends on point placement)
- Simple template matching for ball tracking (may fail with occlusion or rapid motion)
- Single lateral camera view only (no 3D reconstruction)
- Assumes ball motion is primarily in the calibration plane
- Output speed is post-impact velocity, not peak racquet head speed

## Project Structure

```
.
├── flake.nix              # Nix dev shell
├── serve_analyzer/        # Python package
│   ├── analysis.py        # Core velocity computation
│   └── cli.py             # Command-line interface
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── IMG_9259.MOV          # Sample video
```

## Development

```bash
# Run tests
python -m unittest discover -s tests -v

# Check CLI help alignment
python -m serve_analyzer.cli --help
```

## Wall Serve Analysis

A dedicated pipeline for estimating tennis serve velocity and landing projection from **lateral wall-impact video**. The camera faces a wall from the side; the ball travels toward the wall and makes contact at a known height. The system detects impact frame and pixel, estimates pre-impact speed, and projects the equivalent no-wall landing onto a regulation tennis court using gravity-only physics (no spin or drag).

### Calibration

Before analyzing videos, create a reusable calibration JSON that maps wall reference points from pixels to real-world meters.
You need at least four wall reference points, the serve contact height, and optional hook or chair references for vertical scale.

```bash
# Non-interactive setup with 4 wall points
nix develop --command python -m serve_analyzer.wall_calibration --mode setup \
    --output setup.json \
    --serve-contact-height 2.80 \
    --wall-points "100,500,-4.0,0.0;700,500,4.0,0.0;100,100,-4.0,3.0;700,100,4.0,3.0" \
    --hook-point "400,150" \
    --chair-point "200,450"
```

- `--serve-contact-height` is required and sets the ball contact height above the court surface (meters).
- `--wall-points` takes semicolon-separated groups of four values: `pixel_x,pixel_y,wall_x_m,wall_y_m`.
- `--hook-point` and `--chair-point` are optional vertical references.
- `--serve-contact-distance` and `--camera-wall-distance` default to 6.11 m and 1.57 m.

### Analysis

Run the wall-serve pipeline against one video or a batch glob. The CLI writes per-video `result.json` and `result.csv`, plus an aggregate `all_serves.csv` when batch mode is used.

```bash
# Single video
nix develop --command python -m serve_analyzer.wall_serve \
    --video serve_01.MOV \
    --metadata setup.json \
    --output-dir results/

# Batch mode against real wall videos (user execution only)
nix develop --command python -m serve_analyzer.wall_serve \
    --batch 'videos/wall/*.MOV' \
    --metadata setup.json \
    --output-dir results/
```

Real `videos/wall/*.MOV` files (for example `IMG_9340.MOV` through `IMG_9347.MOV`) are referenced as user examples. Automated tests use synthetic fixtures and never open real wall videos.

### Output Inspection

```bash
# Inspect per-video JSON
cat results/IMG_9340/result.json | jq .

# View aggregate CSV
cat results/all_serves.csv
```

The JSON contains six top-level sections: `measured`, `inferred`, `assumed`, `confidence`, `warnings`, and `artifacts`. The CSV follows the stable column order defined in `serve_analyzer/wall_calibration.py`.

### Optional Flags

| Flag | Description |
|------|-------------|
| `--override` | Per-video override JSON (e.g. different contact height) |
| `--manual-corrections` | JSON mapping `serve_index` to corrected `pixel_x`/`pixel_y` |
| `--no-video` | Skip annotated MP4 generation |
| `--no-plots` | Skip plot PNG generation |
| `--fps` | Override video frame rate |
