# TESTS KNOWLEDGE BASE

## OVERVIEW
Tests are flat `unittest` files that mirror module boundaries; detector and evaluator contracts are intentionally tested separately.

## FILE MAP
| Test file | Covers | Notes |
|-----------|--------|-------|
| `test_analysis.py` | `analysis.py` | Core math + synthetic video checks |
| `test_cli_defaults.py` | `cli.py` | Interactive calibrator defaults |
| `test_serve_attempts.py` | `serve_attempts.py` | Detector output shape, selector, autonomous count |
| `test_evaluator.py` | `serve_evaluation.py` | Timestamp parsing, matching, summary CLI |

## TEST-WRITING CONVENTIONS
- Use `unittest.TestCase`, not pytest fixtures.
- Keep helper factories file-local (`_make_event`, `_make_candidate`) instead of shared fixture modules.
- For detector pipeline tests, mock the video stack deeply rather than touching real video unless the test is explicitly integration-level.
- Assert output key presence/absence, not just values, when guarding detector/evaluator boundaries.

## DETECTOR / EVALUATOR CONTRACT
- Detector tests must ensure evaluator keys do **not** leak into detector output (`matched`, `target_time_sec`, `delta_sec`, `serve_number`).
- Evaluator tests own timestamp parsing and matching behavior.
- If a change needs manual timestamps to make detector tests pass, the architecture is probably wrong.

## MOCKING PATTERNS
- `test_serve_attempts.py` uses stacked `@patch` decorators for `detect_ball_yolo`, interpolation, velocity computation, event detection, and `analyze_serve`.
- Be careful with mock argument order; it follows decorator stack order.
- Prefer minimal fake candidate/event dicts over large fixtures.

## CLI TEST PATTERNS
- Capture stdout with `io.StringIO`.
- Call `main([...])` directly for most CLI tests.
- Use temp JSON/timestamp files for evaluator CLI tests instead of shelling out.

## ANTI-PATTERNS
- Do not move detector/evaluator assertions into one mixed test file.
- Do not weaken output-shape assertions when changing selector logic.
- Do not replace synthetic fast tests with real-video tests unless validating an integration path.

## COMMANDS
```bash
python -m unittest discover -s tests -v
python -m unittest tests.test_serve_attempts -v
python -m unittest tests.test_evaluator -v
```
