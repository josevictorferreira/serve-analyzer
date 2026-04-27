# Inspect Serve Pipeline — Findings

## Files and Functions Involved (Browser Upload → Selected Serves)

### 1. Web Backend Entry
**File:** `web/backend/app.py`
- `analyze()` (line 58): POST /api/analyze handler. Validates upload, saves temp video, starts background thread.
- `_run_analysis_thread()` (line 99): Background thread that calls `run_analysis()`, then `generate_clips()`.

### 2. Analysis Service Adapter
**File:** `web/backend/services/analysis_service.py`
- `run_analysis()` (line 18): Wraps detector + selector. Always passes `expected_serves=None` (autonomous mode).
- Calls `detect_serve_candidates()` → `select_serves()`.

### 3. Detector (Candidate Generation)
**File:** `serve_analyzer/serve_attempts.py`
- `detect_serve_candidates()` (line 67): Runs YOLO + HSV tracking, merges multi-profile events, builds candidate list.
- Calls into `serve_analyzer/multi_serve.py`:
  - `detect_ball_yolo()` — ball detection per frame
  - `interpolate_missing_detections()` — gap fill
  - `compute_frame_velocities()`, `compute_vertical_velocity()`, `compute_horizontal_velocity()` — motion signals
  - `detect_serve_events()` — event detection with 3 profiles
  - `analyze_serve()` — velocity stats per candidate

### 4. Selector (Final Selection)
**File:** `serve_analyzer/serve_attempts.py`
- `select_serves()` (line 226): Ranks candidates, suppresses overlaps, infers count, returns `selected_serves`.
- `infer_serve_count()` (line 183): Autonomous count inference from quality gap.

### 5. Multi-Serve Core
**File:** `serve_analyzer/multi_serve.py`
- `detect_serve_events()` (line 410): Finds velocity peaks, validates toss, gates on rightward motion, scores events.
- `analyze_serve()` (line 702): Computes post-contact velocities per candidate.

---

## Where `selected_serves` Can Become Empty

### A. Empty candidate pool upstream
If `detect_serve_candidates()` returns `[]`, `select_serves()` returns `[]` immediately (line 237-238).

Causes in `multi_serve.py:detect_serve_events()`:
1. **No velocity peaks** (line 447-452): `signal.find_peaks()` finds 0 peaks above threshold.
2. **All peaks fail toss validation** (line 487-494): `upward_motion` false for every peak.
3. **All peaks fail hard gates** (line 541-611):
   - End-of-video margin rejection (line 549)
   - Drop > 900 (line 578)
   - Contact velocity > 3000 (line 582)
   - Strong leftward with no direction-unreliable excuse (line 603-606)

### B. All candidates filtered in selector
`select_serves()` line 245-261: hard-rejects candidates with:
- `abs(drop) < 10 AND abs(nrd) < 20` (no significant motion)
- `nrd < -500 AND NOT direction_unreliable` (clearly leftward)

If this filters all candidates → returns `[]` (line 263-264).

### C. All candidates have negative rank
Line 445-446: skips candidates with `selector_rank < 0`. If all suppressed candidates have negative rank → `selected` stays empty.

### D. Autonomous inference returns 0
`infer_serve_count()` (line 183):
- Returns 0 if `candidates` empty (line 196)
- Returns 0 if no candidates pass threshold (line 207-208)
- Returns 0 if `above_threshold` length ≤ 1 and that 1 is filtered out by gap rules

Then `k = 0` (line 436), loop never selects anything.

---

## Web Path vs CLI Path — Differences

| Aspect | Web Path | CLI Path (`serve_attempts.py` main) |
|--------|----------|-------------------------------------|
| Entry | `app.py:analyze()` → `_run_analysis_thread()` | `serve_attempts.py:main()` |
| Adapter | `analysis_service.py:run_analysis()` | Direct call in `main()` |
| `expected_serves` | **Always `None`** (autonomous) | `args.expected_serves` (user-set or None) |
| `frame_skip` | **Always default (1)** | `args.frame_skip` (user-set, default 1) |
| `conf_threshold` | **Always default (0.20)** | `args.conf` (user-set, default 0.20) |
| `scale_factor` | **Always default (0.001)** | `args.scale_factor` (user-set, default 0.001) |
| `start_frame` | **Always default (0)** | `args.start_frame` (user-set, default 0) |
| Pool size logic | `pool_size = None` → `expected = 12` in `detect_serve_candidates` | Same if `args.expected_serves is None` |
| Progress callback | `on_progress` lambda updates state | None |
| Result shape | Dict with `selected_serves`, `candidates`, `count_inferred`, `inferred_count` | Same dict shape |

### Key Difference: No parameter tuning from web
The web path **hardcodes all detector parameters** to defaults. The CLI allows `--frame-skip`, `--conf`, `--scale-factor`, `--start-frame` overrides. This means:
- A video that fails with web defaults might succeed with CLI-tuned params (e.g., `--frame-skip 4` for 4K video per AGENTS.md lesson).
- The web has no way to pass `expected_serves` (always autonomous), while CLI can force a count.

### Same code path for core logic
Both paths call the same functions:
- `detect_serve_candidates(video_path, expected_serves=pool_size)`
- `select_serves(candidates, expected_serves=expected_serves)`

The only semantic difference is that the web **always** uses autonomous mode (`expected_serves=None`), while CLI can force it.

---

## Critical Lines for Empty selected_serves

| File | Line | Condition |
|------|------|-----------|
| `serve_attempts.py` | 237-238 | `if not candidates: return []` |
| `serve_attempts.py` | 239-240 | `if expected_serves is not None and expected_serves <= 0: return []` |
| `serve_attempts.py` | 254-255 | `if abs(drop) < 10 and abs(nrd) < 20: continue` (filter) |
| `serve_attempts.py` | 258-259 | `if nrd < -500 and not direction_unreliable: continue` (filter) |
| `serve_attempts.py` | 263-264 | `if not candidates: return []` (post-filter empty) |
| `serve_attempts.py` | 445-446 | `if selector_rank < 0: continue` (skip negative rank) |
| `serve_attempts.py` | 433-436 | `k = infer_serve_count(suppressed)` → can be 0 |
| `serve_attempts.py` | 196 | `if not candidates: return 0` |
| `serve_attempts.py` | 207-208 | `if not above_threshold: return 0` |
| `multi_serve.py` | 578-579 | `if drop > 900.0: continue` (hard gate) |
| `multi_serve.py` | 582-583 | `if cv > 3000.0: continue` (hard gate) |
| `multi_serve.py` | 603-606 | Leftward rejection without direction_unreliable |

---

## Summary

`selected_serves` becomes empty when:
1. **Detector finds nothing** (no peaks, or all peaks fail validation/gating)
2. **Selector filters everything** (motion too weak, too leftward, or all ranks negative)
3. **Inference says 0** (quality threshold too high, no gap in ranks)

Web and CLI use the **same core functions**, but web **always uses default params and autonomous mode**. CLI can tune params or force a count, which may rescue borderline videos.
