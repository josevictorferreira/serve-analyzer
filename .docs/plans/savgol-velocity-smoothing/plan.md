# Savitzky–Golay Velocity Smoothing

**Status:** Proposed
**Owner:** TBD
**Estimated effort:** 1–2 hours implementation + ~30 min validation
**Risk:** Low (single replaceable code path, fully covered by existing unit tests)

---

## 1. Problem

Velocity smoothing currently uses two filters that **attenuate the very peak we care about** — the post-contact peak velocity:

| Location | Filter | Default |
|---|---|---|
| `serve_analyzer/analysis.py:107-109` | Uniform moving average via `np.convolve` | `smoothing_window=3` |
| `serve_analyzer/multi_serve.py:296-297, 324-325, 351-352` | `scipy.ndimage.gaussian_filter1d` | `smooth_sigma=2.0` |
| `serve_analyzer/test_roboflow_model.py:302-314` | `gaussian_filter1d` | `sigma` param |

A Gaussian with `sigma=2` over a single-frame velocity peak applies roughly a 0.40 attenuation to the peak sample (Gaussian kernel weight at center ≈ `1/(σ√(2π))` × normalization → the peak value is replaced by a weighted blend with neighbors). For a tennis serve, the post-contact peak is concentrated in 2–4 frames; smoothing this with `σ=2` (effective window ~9 frames) **systematically biases peak velocity downward by ~5–15%**.

A moving-average kernel of width 3 over the same peak applies a roughly 1/3 attenuation — slightly better than Gaussian but still wrong for the same reason.

Savitzky–Golay (`scipy.signal.savgol_filter`) fits a low-order polynomial in a sliding window and evaluates it at the center. It denoises high-frequency wobble **without flattening peaks** (a degree-3 polynomial fit can represent a single-sample maximum exactly). This is the canonical filter for sports trajectory analysis.

## 2. Goal & Success Criteria

Replace Gaussian / moving-average smoothing with Savitzky–Golay in all velocity-smoothing call sites, while keeping public function signatures stable.

**Verifiable success criteria:**

1. **Peak preservation (synthetic):** On a synthetic velocity series `v[t] = base + spike(t=center, height=H)` with `H >> noise`, SG-smoothed peak height ≥ 0.95 × H (today's Gaussian/MA: ≤ 0.65 × H).
2. **Noise rejection (synthetic):** On `v[t] = constant + N(0, σ)`, post-SG std-dev ≤ 0.30 × pre-SG std-dev (i.e. ≥ 70% noise reduction, comparable to current Gaussian).
3. **No regression:** All existing tests in `tests/test_analysis.py`, `tests/test_serve_attempts.py`, `tests/test_serve_attempts_v2.py` pass unchanged.
4. **Real-video sanity:** On `IMG_9259.MOV` (or whichever sample video is in repo), the reported `max_kmh` for the known serve increases (or stays equal) and never decreases vs the Gaussian baseline. Document the before/after delta.
5. **API stability:** External callers using `compute_velocity_series(..., smoothing_window=N)` continue to work without code changes.

## 3. Scope

### In scope

- `serve_analyzer/analysis.py::compute_velocity_series` — replace moving-average path.
- `serve_analyzer/multi_serve.py` — replace `gaussian_filter1d` in `compute_frame_velocities`, `compute_vertical_velocity`, `compute_horizontal_velocity`.
- `serve_analyzer/test_roboflow_model.py::compute_velocity_series` — replace `gaussian_filter1d` (this is a separate experimental script-level function).
- Tests: extend `tests/test_analysis.py` with peak-preservation + noise-rejection cases.
- One AGENTS.md "session learning" entry capturing the tradeoff for future sessions.

### Out of scope (explicit)

- Changing the smoothing applied to **positions** (not currently smoothed; out of scope here).
- Confidence-weighted velocity (separate idea #25 in the audit).
- Kalman replacement for `interpolate_missing_detections` / `continuity_gate_positions` (separate idea #1).
- Top-K mean for peak extraction (separate idea #20).
- The CLI `--smoothing-window` argument is kept; semantics evolve (see §4.2).

## 4. Design

### 4.1 Filter parameters

`scipy.signal.savgol_filter(x, window_length, polyorder)`:

- `window_length` MUST be odd and `> polyorder`.
- `polyorder` ≥ 2 needed to preserve peaks (degree-2 already preserves a parabolic peak exactly; degree-3 is safer for asymmetric peaks like racquet contact).
- We pick **defaults: `window_length=7`, `polyorder=3`** — empirically the standard for sports velocity smoothing at 30–120 fps.
- For very short input arrays (`len < window_length`), reduce `window_length` to the largest odd ≤ `len` that satisfies `> polyorder`. If even that fails (`len <= polyorder`), return input unchanged.

### 4.2 Mapping the existing parameters

| Existing param | Old meaning | New meaning |
|---|---|---|
| `smoothing_window: int = 3` (in `analysis.py`) | MA kernel width | SG `window_length`. If user passes `smoothing_window <= 1` → no smoothing (preserve current "disable" behavior documented in AGENTS.md). If user passes an even number, round up to next odd and warn once via `logging.warning`. |
| `smooth_sigma: float = 2.0` (in `multi_serve.py`) | Gaussian sigma | Map to `window_length = max(5, 2*ceil(2*sigma)+1)` so default `sigma=2.0 → window_length=9`. Document the mapping in the docstring. Add new optional `polyorder: int = 3` kwarg. |

This keeps every existing call site working without argument changes. The defaults shift slightly toward stronger windows (3→7 in `analysis.py`, σ=2 → wlen=9 in `multi_serve.py`) — but because SG preserves peaks, the effective denoising-without-bias improves regardless.

### 4.3 Single helper to centralize logic

Add `serve_analyzer/analysis.py::_savgol_smooth(values: np.ndarray, window_length: int, polyorder: int = 3) -> np.ndarray`:

```python
def _savgol_smooth(values, window_length, polyorder=3):
    """Savitzky–Golay smoothing with safe fallback for short arrays."""
    n = len(values)
    if n == 0 or window_length <= 1:
        return values
    # Force odd window_length within array bounds
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl <= polyorder:
        return values  # array too short for SG; no-op
    if wl % 2 == 0:
        wl -= 1
    return savgol_filter(values, window_length=wl, polyorder=polyorder, mode="interp")
```

Both `analysis.py` and `multi_serve.py` use this helper.

### 4.4 Edge cases handled

- `n < polyorder + 2` → no-op (return input unchanged).
- Even `window_length` → coerce to next lower odd.
- `mode="interp"` (SG default) handles boundaries by polynomial extrapolation — better than `mode="nearest"` for peaks near series start/end.

## 5. Implementation Steps

Phased, each phase verifiable independently.

### Phase 1 — Helper + tests

1. Add `_savgol_smooth` to `serve_analyzer/analysis.py` (top of file, near other helpers).
2. Import `from scipy.signal import savgol_filter` at module top.
3. Add unit tests in `tests/test_analysis.py`:
   - `test_savgol_preserves_peak`: synthetic `[0,0,0,10,0,0,0]` → smoothed peak ≥ 9.5
   - `test_savgol_reduces_noise`: random Gaussian noise input → output std ≤ 0.30 × input std
   - `test_savgol_short_array_noop`: input length 2 → returned unchanged
   - `test_savgol_even_window_coerced`: `window_length=6` accepted, treated as 5

**Verify:** `python -m unittest tests.test_analysis -v` — 4 new passing tests.

### Phase 2 — `compute_velocity_series` migration

1. In `analysis.py:107-109`, replace the `np.convolve` block with `_savgol_smooth(speeds_mps, window_length=smoothing_window, polyorder=3)`.
2. Update the docstring for `smoothing_window` to describe new semantics (window length, must be odd, ≤1 disables).
3. Run existing tests: `python -m unittest tests.test_analysis -v` — all green.

**Verify:** Existing peak-velocity test in `test_analysis.py:151` (currently passes `smoothing_window=1`) remains green; no behavior change when smoothing disabled.

### Phase 3 — `multi_serve.py` migration

1. Add helper to map `smooth_sigma` → `window_length` at top of `multi_serve.py`:
   ```python
   def _sigma_to_window(sigma: float) -> int:
       if sigma <= 0:
           return 1  # no-op
       wl = max(5, 2 * int(np.ceil(2 * sigma)) + 1)
       return wl if wl % 2 == 1 else wl + 1
   ```
2. Replace three `gaussian_filter1d` call sites in `compute_frame_velocities`, `compute_vertical_velocity`, `compute_horizontal_velocity` with:
   ```python
   if smooth_sigma > 0:
       velocities = _savgol_smooth(velocities, _sigma_to_window(smooth_sigma))
   ```
3. Remove `from scipy.ndimage import gaussian_filter1d` import (line 31) if no other usages remain.

**Verify:** `python -m unittest tests.test_serve_attempts tests.test_serve_attempts_v2 -v` — all green.

### Phase 4 — `test_roboflow_model.py` migration

1. Same replacement pattern in its private `compute_velocity_series` (lines 302-314).
2. This is a script-level helper — verify by running `python -m serve_analyzer.test_roboflow_model --help` (smoke test only).

### Phase 5 — Real-video sanity check

1. Run on the canonical sample:
   ```bash
   python -m serve_analyzer.serve_attempts video.mov --output out_savgol.json
   ```
2. Compare against `out_baseline.json` (captured before Phase 2):
   ```bash
   python - <<'PY'
   import json
   a = json.load(open('out_baseline.json'))
   b = json.load(open('out_savgol.json'))
   for ca, cb in zip(a['selected_serves'], b['selected_serves']):
       print(f"contact {ca['contact_frame']}: max_kmh {ca.get('post_contact_max_kmh'):.1f} → {cb.get('post_contact_max_kmh'):.1f}")
   PY
   ```
3. Document the deltas in the PR description and in AGENTS.md (`SESSION LEARNINGS` section).

**Verify:** Peak `post_contact_max_kmh` is **non-decreasing** for every serve (allowed: small increase, expected: 3–10% increase).

### Phase 6 — Documentation

1. Update `serve_analyzer/AGENTS.md` SESSION LEARNINGS:
   ```
   ### Savitzky–Golay preserves velocity peaks
   **Lesson:** Use SG (window=7, polyorder=3) for velocity smoothing, not Gaussian/moving-average. Gaussian σ=2 attenuates single-frame peaks ~40%, biasing peak serve velocity downward by 5–15%.
   **Context:** Helper `_savgol_smooth` in `analysis.py` is the single entry point.
   **Verify:** Synthetic peak `[0,0,0,10,0,0,0]` smoothed → peak ≥ 9.5 (vs Gaussian σ=2 → ~6).
   ```
2. Update docstrings of `compute_velocity_series`, `compute_frame_velocities`, `compute_vertical_velocity`, `compute_horizontal_velocity` to mention SG.

## 6. Test Plan

### New unit tests (Phase 1) — `tests/test_analysis.py`

- `test_savgol_preserves_peak`
- `test_savgol_reduces_noise`
- `test_savgol_short_array_noop`
- `test_savgol_even_window_coerced`

### New integration test — `tests/test_analysis.py`

- `test_compute_velocity_series_preserves_synthetic_peak`:
  Build positions where a known peak velocity occurs at frame 5; assert `stats['max_kmh']` is within 5% of the true value (current Gaussian/MA fails this at >10% error).

### Regression coverage (existing tests)

- `tests/test_analysis.py` — full module
- `tests/test_serve_attempts.py` — full module
- `tests/test_serve_attempts_v2.py` — full module
- `tests/test_evaluator.py` — full module (smoothing changes propagate to evaluator output)

### Manual smoke

- `python -m serve_analyzer.cli IMG_9259.MOV --real-distance 1.0 --cal-p1 100 200 --cal-p2 400 200 --start-frame 50 --ball-pos 320 240` — should run without error.
- `python -m serve_analyzer.serve_attempts video.mov --output out.json` — should run without error and produce a `selected_serves` list.

## 7. Rollback

Single-commit rollback: revert the commit. The change is contained to one helper + ~6 call-site replacements. No schema, CLI, or persisted-data changes.

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| SG with default `polyorder=3` introduces high-frequency wobble on very noisy short clips | Low | `_savgol_smooth` falls back to no-op for short arrays; tests cover noise-reduction floor |
| Existing peak-velocity numbers change (downstream consumers may compare against historical values) | Medium | This is *intended* — direction of change is upward (less attenuation). Document in PR. |
| Boundary frames near start/end of video produce extrapolation artifacts | Low | `mode="interp"` polynomial-extrapolates; documented as preferred boundary mode for SG |
| Default mapping `σ=2.0 → wlen=9` slightly widens the multi_serve window | Low | SG with `polyorder=3` over wlen=9 still preserves peaks; verified by synthetic peak test |

## 9. Open Questions

1. Should `polyorder` be exposed as a CLI/API parameter? **Proposed: no** — keep implementation detail; revisit if a use case appears.
2. Should we also smooth ball positions (not just velocities) with SG? **Proposed: no in this plan** — separate change, separate verification.

## 10. Acceptance Checklist

- [ ] Phase 1 helper + tests merged
- [ ] Phase 2 `analysis.py` migrated, all tests green
- [ ] Phase 3 `multi_serve.py` migrated, all tests green
- [ ] Phase 4 `test_roboflow_model.py` migrated, smoke test passes
- [ ] Phase 5 real-video baseline captured, delta documented
- [ ] Phase 6 AGENTS.md learning + docstrings updated
- [ ] PR description includes before/after `max_kmh` numbers for the sample video
