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
