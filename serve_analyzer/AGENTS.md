# PACKAGE KNOWLEDGE BASE

## OVERVIEW
Python package is mostly CLI-first scripts around one shared core: `analysis.py` for low-level tracking/math, `multi_serve.py` for full-video serve events, `serve_attempts.py` for detector output, `serve_evaluation.py` for post-hoc matching.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Single-serve interactive flow | `cli.py`, `__main__.py` | `python -m serve_analyzer` lands here |
| Core tracking/math | `analysis.py` | Lowest-level reusable functions |
| Multi-serve event generation | `multi_serve.py` | Upstream detector heuristics + scoring |
| Detector-only JSON output | `serve_attempts.py` | No timestamps in inference |
| Evaluation-only timestamp matching | `serve_evaluation.py` | Reads detector JSON + manual timestamps |
| Video overlays | `annotate_video.py` | Debug/visual inspection |
| Detector comparisons | `compare_detectors.py` | Utility script, not core library |
| Scripted plotting | `plot_serve.py` | Non-interactive graph output |

## MODULE BOUNDARIES
- `analysis.py` = core reusable math/tracking helpers; safest import surface.
- `multi_serve.py` = heavy full-video heuristics; `serve_attempts.py` should reuse, not fork, its event logic.
- `serve_attempts.py` = detector boundary; outputs `candidates` / `selected_serves` only.
- `serve_evaluation.py` = evaluation boundary; owns timestamp parsing, candidate matching, summary JSON.
- `__init__.py` exports only a small subset from `analysis.py`; most modules are script-style and imported directly.

## ENTRY POINT CONVENTION
- Most runnable modules define `build_parser()` + `main(argv: Optional[Sequence[str]] = None)` + `if __name__ == "__main__": raise SystemExit(main())`.
- Keep new CLI modules consistent with that shape.
- `__main__.py` is only a thin wrapper to `cli.main()`; other tools are invoked as `python -m serve_analyzer.<module>`.

## PACKAGE-SPECIFIC CONVENTIONS
- Use absolute package imports when module is meant to run via `python -m serve_analyzer.<module>`.
- Convert numpy scalars before JSON serialization.
- Prefer adding detector metadata fields over changing evaluator schema when debugging selection quality.
- Preserve `Optional[int] expected_serves` semantics: `None` means autonomous count inference, positive int means forced count.

## ANTI-PATTERNS
- Do not mix timestamp parsing/matching back into `serve_attempts.py` inference flow.
- Do not rewrite `multi_serve.py` detection when the candidate pool already contains the real serves; tune selection first.
- Do not treat `expected_serves=0` as autonomous mode; only `None` means infer count.
- Do not add experimental one-off scripts unless they clearly belong in package namespace.

## DEBUGGING CHECKS
```bash
# Detector only
python -m serve_analyzer.serve_attempts video.mov --output out.json

# Post-hoc evaluation only
python -m serve_analyzer.serve_evaluation --detection-json out.json --timestamps-file timestamps_video.txt

# Compare candidate pool vs final picks
python - <<'PY'
import json
data=json.load(open('out.json'))
print(len(data['candidates']), len(data['selected_serves']))
print([round(x['contact_time_sec'],2) for x in data['selected_serves']])
PY
```
