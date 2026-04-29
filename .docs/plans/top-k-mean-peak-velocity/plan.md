# Top-K Mean Peak Velocity

**Status:** Proposed
**Backlog ref:** `improvements.md` #20
**Estimated effort:** 1–2 hours (single function change + tests)
**Risk:** Low — isolated change in one statistic; old behavior preserved as a fallback.

---

## 1. Problem

Peak post-contact velocity is currently reported as `np.max(speeds_kmh)` (effectively, via `summary_stats['max_kmh']` in `compute_velocity_series`, `serve_analyzer/analysis.py:107` area, and propagated through `serve_evaluation.py:182` into the `post_contact_max_kmh` field).

`np.max` over a window has two failure modes:

1. **Single-frame outlier domination.** A single spurious detection creating a 3 px → 50 px → 3 px pattern produces a single huge velocity sample. With `gaussian_filter1d(sigma=2)` this gets attenuated, but with `smoothing_window=1` (or with disabled smoothing) it pollutes `max_kmh` directly. Even with smoothing, a 4-frame artifact (motion blur, rolling shutter, half-occluded ball) survives.
2. **Variance across re-runs.** Frame-skip subsampling and detection nondeterminism (YOLO with very-similar bbox confidences across runs) make `max` noisier than it should be — the second-highest sample is often within 5 km/h of the max but ignored.

A **top-K mean** — `mean(sort(speeds)[-K:])` with K=3–5 — gives a peak that:
- Is stable against a single outlier (1 outlier in K=5 contributes 1/5 of the value).
- Captures the genuine peak region rather than a single sample (the real peak in a serve lasts 2–4 frames).
- Is trivial to compute, cheap, and orthogonal to other smoothing choices.

This is the standard practice in sports biomechanics (e.g. baseball pitch tracking systems report a "trimmed peak" rather than instantaneous max).

## 2. Goal & Success Criteria

Add a top-K-mean reduction alongside the existing max in velocity statistics. **Do not remove `max_kmh`** — keep it for backward compat. Add `peak_kmh` (top-K mean) and use it for the headline number in evaluation outputs.

**Verifiable success:**

1. **Synthetic outlier test:** Build velocity series `[180, 182, 181, 350, 179, 183, 180]` (one teleport-induced 350 km/h sample). Old `max_kmh = 350`. New `peak_kmh` (K=5) = `mean([350, 183, 182, 181, 180]) ≈ 215`. With K=3 = 238. Verify both.
2. **Synthetic peak preservation:** Build velocity series with a 4-frame plateau at 200 km/h surrounded by 100 km/h. K=5 returns 200 (top 4 plateau samples + 1 from edge ≈ 180; K=3 returns 200 exactly). Verify K=3 returns ≥199.
3. **Real-video stability:** Run pipeline 5 times on the same video with `frame_skip=4`. Old `max_kmh` standard deviation across runs ≥ X km/h; new `peak_kmh` standard deviation ≤ 0.5 × X. (Document actual numbers in PR.)
4. **Existing tests pass** unchanged (`max_kmh` still present in stats dict; tests asserting it continue to work).
5. **JSON schema additive:** `peak_kmh` added to `summary_stats` and `selected_serves[i]`; no existing fields removed.

## 3. Scope

### In scope

- Modify `compute_velocity_series` in `serve_analyzer/analysis.py` to add `peak_kmh` to the returned stats dict.
- Modify `compute_velocity_series_phase`-style callers in `multi_serve.py` to forward this stat.
- Modify `serve_evaluation.py:182-185` to also include `post_contact_peak_kmh`.
- Add unit tests for the new statistic.
- Update `summary_stats` JSON schema documentation.

### Out of scope (explicit)

- Removing `max_kmh`. Stays as backward-compat field; users can opt into `peak_kmh`.
- Changing how `mean_kmh` (full-window mean) is computed.
- Confidence-weighting of the top-K samples (backlog #25).
- MAD-based outlier rejection (backlog #21 — separate plan).

## 4. Design

### 4.1 Function

Single helper in `analysis.py`:

```python
def _top_k_mean(values: np.ndarray, k: int = 5) -> float:
    """
    Mean of the top-k highest samples in `values`. Returns 0.0 for empty input.
    Falls back to mean of all samples when len(values) < k.
    """
    n = len(values)
    if n == 0:
        return 0.0
    k_eff = min(k, n)
    # np.partition is O(n) and gives us the top-k unsorted; mean is order-invariant.
    top = np.partition(values, n - k_eff)[n - k_eff:]
    return float(np.mean(top))
```

`np.partition` is O(n) — strictly cheaper than `np.sort` (O(n log n)). On the scale of this codebase it's unmeasurable, but the right choice on principle.

### 4.2 Default K

K = 5 by default. Rationale:
- Real serve peaks last ~2–4 frames at 60 fps.
- K = 3 is risky if smoothing already smeared the peak across 5 frames (top-3 might all be on the plateau, but a single outlier inflates the result).
- K = 5 is robust to 1 outlier in the top samples (1/5 = 20% contribution).
- For windows shorter than 5 samples, fall back to mean of available (per the helper).

Configurable via:
- New optional kwarg on `compute_velocity_series`: `peak_top_k: int = 5`.
- New CLI flag `--peak-top-k N` on `serve_attempts.py`, `multi_serve.py`, `plot_serve.py` (default 5).

### 4.3 Where to emit `peak_kmh`

Three reporting layers:

1. **`compute_velocity_series` stats dict** (`analysis.py`):
   ```python
   stats = {
       "max_mps": ...,
       "max_kmh": ...,
       "mean_mps": ...,
       "mean_kmh": ...,
       "peak_mps": _top_k_mean(speeds_mps, k=peak_top_k),  # NEW
       "peak_kmh": _top_k_mean(speeds_kmh, k=peak_top_k),  # NEW
   }
   ```
2. **Per-phase stats in `multi_serve.py`** (`phase_stats` at `multi_serve.py:780-786`):
   ```python
   return stats["max_kmh"], stats["mean_kmh"], stats["peak_kmh"]  # NEW field
   ```
   And callers store it as `post_contact_peak_kmh`.
3. **Evaluation output** (`serve_evaluation.py:182-185`):
   ```python
   record.update({
       ...
       "post_contact_max_kmh": ...,
       "post_contact_mean_kmh": ...,
       "post_contact_peak_kmh": float(candidate.get("post_contact_peak_kmh", 0.0)),  # NEW
       ...
   })
   ```

### 4.4 Interaction with existing smoothing

If `smoothing_window=1` (smoothing disabled), top-K-mean is the primary defense against outliers — a single 350 km/h sample contributes 70 km/h to a K=5 mean.
If smoothing is enabled (Gaussian σ=2 today, Savitzky–Golay after that plan lands), top-K-mean operates on smoothed values; outliers are already attenuated, so top-K-mean and max should agree closely. This is fine — it means the new metric degenerates to ~max in clean data.

### 4.5 Headline number policy

In the JSON output, `max_kmh` and `peak_kmh` both appear. For documentation/README, **recommend `peak_kmh` as the headline**. CLI summary lines (e.g. `print(f"Max speed: {stats['max_kmh']:.1f} km/h")` in `plot_serve.py:351` area) should print *both*:

```
Peak speed: 198.4 km/h (top-5 mean)
Max sample: 215.2 km/h
```

The visible disagreement educates the user that one of them is more trustworthy.

## 5. Implementation Steps

### Phase 1 — Helper + unit tests

1. Add `_top_k_mean` to `serve_analyzer/analysis.py`.
2. Add `tests/test_top_k_mean.py`:
   - `test_outlier_dampened`: input `[180, 182, 181, 350, 179, 183, 180]` → K=5 ≈ 215; K=3 ≈ 238.
   - `test_plateau_preserved`: input `[100, 100, 200, 200, 200, 200, 100, 100]` → K=3 == 200.
   - `test_short_input_falls_back_to_mean`: input `[100, 110]`, K=5 → 105.
   - `test_empty_input`: input `[]` → 0.0.
   - `test_k_one_equals_max`: K=1 → np.max for any input.

**Verify:** `python -m unittest tests.test_top_k_mean -v` — 5 passing.

### Phase 2 — `compute_velocity_series` integration

1. Add `peak_top_k: int = 5` kwarg.
2. Add `peak_mps`, `peak_kmh` to stats dict.
3. Update existing tests in `tests/test_analysis.py` to acknowledge the new keys (don't break, just extend assertions).

**Verify:** `python -m unittest tests.test_analysis -v` — green.

### Phase 3 — Propagate through `multi_serve.py` and `serve_evaluation.py`

1. Update `phase_stats` (`multi_serve.py:780-786`) to return / store `peak_kmh`.
2. Update `compute_serve_event_metrics` (search for it; the function that builds the per-serve dict consumed by `serve_attempts.py`) to include `post_contact_peak_kmh`.
3. Update `serve_evaluation.py:182-185` to include `post_contact_peak_kmh`.
4. Update `web/backend/schemas.py` (per AGENTS.md learning: schemas must match runtime) — add `post_contact_peak_kmh: float` field where peak is reported.

**Verify:**
- `python -m serve_analyzer.serve_attempts video.mov --output out.json` — JSON has new field.
- `python -m unittest discover -s tests -v` — all green.
- Web backend: `python -c "from web.backend.schemas import *"` — no validation errors.

### Phase 4 — CLI flag

1. Add `--peak-top-k N` to `serve_attempts.py`, `multi_serve.py`, `plot_serve.py` parsers.
2. Plumb through to `compute_velocity_series` calls.
3. Default: 5.

**Verify:** `python -m serve_analyzer.serve_attempts video.mov --peak-top-k 3 --output out_k3.json` produces different `peak_kmh` than `--peak-top-k 7`.

### Phase 5 — Real-video stability check

1. Run `serve_attempts` on `video.mov` 5 times with `frame_skip=4`.
2. Collect per-run `max_kmh` and `peak_kmh` for the first selected serve.
3. Compute stddev across runs.
4. **Acceptance:** `stddev(peak_kmh) ≤ 0.5 × stddev(max_kmh)`.
5. Document numbers in PR.

### Phase 6 — Docs

1. AGENTS.md SESSION LEARNINGS:
   ```
   ### Top-K mean as robust peak velocity
   **Lesson:** np.max over a velocity window is fragile to single-frame outliers. Use top-K mean (K=5) as the headline `peak_kmh`; keep `max_kmh` for backward compat.
   **Context:** Helper `_top_k_mean` in analysis.py. CLI flag `--peak-top-k`. Synthetic outlier `[180,182,181,350,179,183,180]` reduces from 350 (max) to 215 (peak K=5).
   **Verify:** stddev of peak_kmh across 5 frame-skip-4 reruns is ≤ 0.5 × stddev of max_kmh.
   ```
2. Update `improvements.md` #20 status with plan path → done.
3. README/CLI help: document `peak_kmh` as the recommended primary metric.

## 6. Test Plan

| Test | Type | Verifies |
|---|---|---|
| `test_outlier_dampened` | unit | Outlier robustness |
| `test_plateau_preserved` | unit | Real peak captured |
| `test_short_input_falls_back_to_mean` | unit | Edge case |
| `test_empty_input` | unit | Edge case |
| `test_k_one_equals_max` | unit | Backward compat |
| Existing `tests/test_analysis.py` | regression | Existing keys unchanged |
| Existing `tests/test_serve_attempts*.py` | regression | Pipeline integration |
| Real-video stddev script | manual | Variance reduction claim |

## 7. Rollback

Single-revert. Remove the new keys and the helper. The change is purely additive to JSON schema; old consumers ignore the new field. Schema removal is a separate concern only if web frontend started reading the field — at that point it'd be a code-coordinated removal, not a rollback.

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pydantic schema drift breaks web backend | Medium | Phase 3 explicitly updates `web/backend/schemas.py` and verifies import |
| Users reading `max_kmh` from JSON see no change and assume nothing happened | Low | Document `peak_kmh` clearly; CLI prints both side-by-side |
| K=5 too aggressive on very-short serves where post-contact window is < 10 frames | Low | Helper falls back to mean-of-all when `len < K`; documented |
| `peak_kmh` < `max_kmh` looks counterintuitive in JSON | Low | Documented in stats dict comment; K=1 escape hatch |

## 9. Open Questions

1. Should the headline be `peak_kmh` or keep `max_kmh` as headline and add `peak_kmh` quietly? **Proposed: keep `max_kmh` as the existing field, add `peak_kmh` as a sibling. Recommend `peak_kmh` in docs but don't break consumers reading `max_kmh`.**
2. Should K scale with window length? **Proposed: no — fixed K=5 with fallback to all-samples is simpler and predictable.**

## 10. Acceptance Checklist

- [ ] Phase 1 helper + 5 unit tests
- [ ] Phase 2 `compute_velocity_series` extended; existing tests green
- [ ] Phase 3 propagation through multi_serve, serve_evaluation, web schemas
- [ ] Phase 4 CLI flag wired
- [ ] Phase 5 real-video stddev numbers documented in PR
- [ ] Phase 6 docs updated
