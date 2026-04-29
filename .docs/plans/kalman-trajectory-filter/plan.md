# Kalman Filter for Ball Trajectory

**Status:** Proposed
**Backlog ref:** `improvements.md` #1
**Estimated effort:** 1 day (helper + integration + tests + tuning on real video)
**Risk:** Medium — replaces two production heuristics; needs careful before/after numeric comparison.

---

## 1. Problem

Two ad-hoc heuristics handle missing/noisy detections today:

| Heuristic | Location | Issue |
|---|---|---|
| `continuity_gate_positions` | `serve_analyzer/serve_attempts_v2.py:39-95` | Fixed `max_jump_px=260` threshold. Treats fast serves at near-camera (where ball can legitimately jump >260 px/frame) the same as static-yellow false positives. |
| `interpolate_missing_detections` | `serve_analyzer/multi_serve.py:212-269` | Linear interpolation between endpoints; forward/backward fill outside that. **Held positions look identical to real detections downstream** — confidence is lost. |

The two interact: continuity-gate marks a frame `None`, then interpolate fills it linearly. Velocity computed across an interpolated stretch is biased toward zero (linear interp = constant velocity = no acceleration peaks).

A 4-state Kalman filter `[x, y, vx, vy]` solves both:
- **Prediction during gaps:** uses learned velocity, not last position or linear interp.
- **Mahalanobis-distance gating:** rejects jumps that are statistically unlikely given current velocity covariance — adapts to ball speed automatically.
- **Confidence per frame:** trace of position covariance → usable as a weight by downstream code (links to backlog #25).

## 2. Goal & Success Criteria

Replace `continuity_gate_positions` and `interpolate_missing_detections` with a single `kalman_track_positions` helper. Public input/output shape stays `Sequence[Optional[(x, y)]] → List[Optional[(x, y)]]` plus a stats dict.

**Verifiable success:**

1. **Detection retention non-decreasing:** On the canonical sample video, `sum(p is not None for p in output)` ≥ baseline output's count. (Kalman should fill more gaps, not fewer.)
2. **Jump rejection equivalent or stricter at low speeds:** Synthetic test — inject a 500 px teleport into otherwise-quiet positions; Kalman rejects it.
3. **Jump rejection more permissive at high speeds:** Synthetic test — a constant 280 px/frame trajectory (above current `max_jump_px=260`) is *retained* by Kalman, *rejected* by current gate.
4. **Peak velocity preserved:** On real video, `post_contact_max_kmh` per serve in `out.json` does not decrease vs baseline. (Linear interp suppresses peaks; Kalman should preserve or slightly increase.)
5. **All existing tests pass** without modification.
6. **New stats field `kalman_innovations` exported** for diagnostic use (Mahalanobis distances per frame).

## 3. Scope

### In scope

- New helper `serve_analyzer/analysis.py::kalman_track_positions` (or a new `serve_analyzer/kalman.py` module if `analysis.py` gets too large — TBD in §4).
- Integration at three call sites:
  - `serve_attempts_v2.py:39` — replace `continuity_gate_positions` body (keep wrapper signature for backward compat).
  - `multi_serve.py:212` — replace `interpolate_missing_detections` body (keep wrapper).
  - Any other caller of the above (verify via grep).
- New unit tests covering 5 scenarios (synthetic).
- Diagnostic JSON field for downstream consumers.

### Out of scope (explicit)

- Smoothing of *velocities* (covered by SG plan).
- Confidence-weighted velocity (#25 — separate plan; this plan provides the confidence signal as an enabler).
- Replacing per-detector tracking (`track_ball_template`, `track_ball_color`, etc.) — those operate at a different layer.
- Non-linear Kalman (EKF/UKF) — constant-velocity model is sufficient for sub-second windows.

## 4. Design

### 4.1 Model

State `x = [px, py, vx, vy]ᵀ`. Constant-velocity transition; observation is position only.

```
F = [[1, 0, dt, 0],
     [0, 1, 0,  dt],
     [0, 0, 1,  0],
     [0, 0, 0,  1]]

H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

`dt` per frame defaults to `1.0`; if `frame_skip > 1` is in use, the caller passes `dt=frame_skip`. (Real-time-based `dt` from ffprobe PTS is a separate plan — backlog #24.)

### 4.2 Noise tuning

- **Process noise `Q`:** discrete white-noise acceleration model with σ_a tunable. Default σ_a = 50 px/frame² (chosen so 95% of frame-to-frame velocity changes fall within ±100 px/frame, matching tennis ball acceleration scale at 4K).
- **Measurement noise `R`:** isotropic σ_z. Default σ_z = 4 px (matches typical YOLO bbox-center precision).

These are exposed as kwargs for tuning.

### 4.3 Gating

For each incoming detection `z`, compute innovation `y = z - H x_pred` and innovation covariance `S = H P_pred Hᵀ + R`. Mahalanobis distance squared:
```
d² = yᵀ S⁻¹ y
```
Reject if `d² > χ²(2 dof, α=0.99) ≈ 9.21` (configurable via `gate_chi2`).

Rejected detections are treated as "no observation": the predict step still runs, the update step is skipped → state continues on predicted trajectory.

### 4.4 Missing-frame handling

If the input frame is `None`:
- Run predict only.
- If `trace(P_position) < trace_threshold` (default 1000 px²), output `(x, y)` from predicted state.
- Else output `None` (uncertainty too high — don't fabricate a position).

This automatically handles long gaps gracefully: short gaps get filled with predictions, long gaps return `None`.

### 4.5 Initialization

Two-pass over leading detections to bootstrap:
- Find the first two consecutive non-`None` detections. Initialize position from frame 1, velocity from finite difference.
- Set initial `P = diag([σ_z², σ_z², (10·σ_a)², (10·σ_a)²])` — large initial velocity uncertainty so first measurements drive convergence.
- For frames before the first detection, output `None` (no backward extrapolation).

### 4.6 Helper signature

```python
def kalman_track_positions(
    detections: Sequence[Optional[Tuple[float, float]]],
    dt: float = 1.0,
    sigma_a: float = 50.0,
    sigma_z: float = 4.0,
    gate_chi2: float = 9.21,
    trace_threshold: float = 1000.0,
) -> Tuple[List[Optional[Tuple[float, float]]], Dict[str, Any]]:
    """
    Smooth + gap-fill ball detections with a 4-state Kalman filter.

    Returns:
        (filtered_positions, stats) where stats contains:
          - input_detections: count of non-None inputs
          - kept_detections: count of accepted (non-rejected) measurements
          - rejected_jumps: count of measurements failing chi² gate
          - filled_gaps: count of None inputs replaced by predictions
          - innovations: per-frame Mahalanobis d² (None where no measurement)
          - track_uncertainty: per-frame trace(P_position)
    """
```

### 4.7 Backward-compat wrappers

Keep `continuity_gate_positions(detections, max_jump_px, max_missing_frames, interpolation_gap)` and `interpolate_missing_detections(detections, max_gap)` as thin wrappers that:
- Call `kalman_track_positions` with mapped parameters.
- Convert old-style stats dict.
- Log a `DeprecationWarning` once per process.

This means *no caller code changes* in this PR — only behavior changes. Removal of the wrappers is a follow-up.

### 4.8 File placement

Decision: **new `serve_analyzer/kalman.py` module**, not `analysis.py`. Rationale:
- `analysis.py` is the documented "core math" module; adding a 150-line filter implementation bloats it.
- AGENTS.md (package): "`analysis.py` = core reusable math/tracking helpers; safest import surface" — Kalman fits but is large enough to warrant its own file.
- New module re-exported from `analysis.py` if needed.

## 5. Implementation Steps

### Phase 1 — Helper + unit tests (no integration)

1. Create `serve_analyzer/kalman.py` with `kalman_track_positions`.
2. Use `numpy` only (no `filterpy` or external dep — keeps Nix flake unchanged).
3. Add `tests/test_kalman.py` with:
   - `test_constant_velocity_no_noise` — input perfect line; output equals input.
   - `test_fills_short_gap_with_prediction` — input has a 5-frame `None` gap mid-track; output fills it within 2 px of true.
   - `test_rejects_outlier_500px_teleport` — single-frame teleport rejected (innovation > gate_chi2).
   - `test_accepts_high_velocity_280px_per_frame` — constant 280 px/frame trajectory accepted (currently fails old `max_jump_px=260`).
   - `test_long_gap_returns_none` — 30-frame `None` gap; trace exceeds threshold; output is `None` from frame ~15 onward.
   - `test_initialization_from_first_two_detections` — first 3 frames `None`; output starts at frame 4 (first valid).

**Verify:** `python -m unittest tests.test_kalman -v` — 6 passing tests.

### Phase 2 — Wire wrappers

1. Replace bodies of `continuity_gate_positions` (`serve_attempts_v2.py:39-95`) and `interpolate_missing_detections` (`multi_serve.py:212-269`) with calls to `kalman_track_positions`.
2. Map old params to new:
   - `max_jump_px=260` → ignored (Kalman uses chi² gate).
   - `max_missing_frames=12` → maps to `trace_threshold` such that 12 frames of pure prediction at σ_a=50 hit the threshold.
   - `interpolation_gap=10` / `max_gap=10` → ignored (Kalman handles arbitrary gaps).
3. Emit one-time `DeprecationWarning` recommending direct use of `kalman_track_positions`.
4. Verify old stats dict shape still produced (`input_detections`, `trusted_detections`, `rejected_jumps`).

**Verify:**
- `python -m unittest discover -s tests -v` — all existing tests pass.
- `python -m serve_analyzer.serve_attempts video.mov --output out_kalman.json` runs without error.

### Phase 3 — Real-video numeric comparison

1. Capture baseline before Phase 2 merge:
   ```bash
   git stash  # or run on main
   python -m serve_analyzer.serve_attempts video.mov --output out_baseline.json
   git stash pop
   ```
2. After Phase 2:
   ```bash
   python -m serve_analyzer.serve_attempts video.mov --output out_kalman.json
   ```
3. Diff script:
   ```python
   import json
   a = json.load(open('out_baseline.json'))
   b = json.load(open('out_kalman.json'))
   for ca, cb in zip(a['selected_serves'], b['selected_serves']):
       print(f"contact {ca['contact_frame']}: max_kmh "
             f"{ca['post_contact_max_kmh']:.1f} → {cb['post_contact_max_kmh']:.1f}, "
             f"mean_kmh {ca['post_contact_mean_kmh']:.1f} → {cb['post_contact_mean_kmh']:.1f}")
   ```
4. **Acceptance:** every serve's `post_contact_max_kmh` is non-decreasing (allowed: small increase from peak preservation).

### Phase 4 — Tuning if needed

If Phase 3 shows regressions (max_kmh drops on any serve):
- Inspect `kalman_innovations` for that serve's window — high values indicate the gate is over-rejecting.
- Tune `sigma_a` upward (50 → 80) to allow more acceleration.
- Re-run Phase 3 until all serves pass.

Document the final tuning in module docstring + AGENTS.md.

### Phase 5 — Documentation

1. Update `serve_analyzer/AGENTS.md` SESSION LEARNINGS with tuning insights from Phase 4.
2. Update `improvements.md` to mark #1 as `🚧 in progress` then `✅ done` with file:line.
3. Add docstring example to `kalman.py` showing canonical usage.

## 6. Test Plan

### New unit tests — `tests/test_kalman.py`

Six tests listed in Phase 1.

### Integration tests

Existing tests in `tests/test_serve_attempts.py` and `tests/test_serve_attempts_v2.py` exercise both wrappers; if they all pass, integration is verified at structural level.

Add one new integration test:
- `test_kalman_preserves_synthetic_serve_peak`: build positions encoding a known 200 km/h post-contact peak; verify `selected_serves[0]['post_contact_max_kmh']` ≥ 195.

### Manual verification

- Sample video before/after diff (Phase 3).
- Inspect `out_kalman.json` for new diagnostic field (`kalman_innovations` per serve, if exposed at that level).

## 7. Rollback

Two-step revert:
1. Restore old bodies of `continuity_gate_positions` and `interpolate_missing_detections`.
2. Delete `serve_analyzer/kalman.py` and `tests/test_kalman.py`.

No schema changes, no CLI changes. Safe.

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Default `sigma_a` wrong for some videos → over/under rejection | Medium | Phase 4 tuning loop with diagnostics. Expose params. |
| Kalman fills more frames than old code → propagates tracking errors into more downstream features | Low | `trace_threshold` caps how long predictions are trusted; misses still return `None`. |
| Performance regression on long videos (Kalman is O(n) but with matrix ops per frame) | Low | Pure numpy with 4×4 matrices is trivially fast (~10⁵ frames/sec); confirmed in Phase 1 micro-benchmark. |
| `DeprecationWarning` spam in test output | Low | Use `warnings.warn(..., stacklevel=2)` once per process via module-level flag. |

## 9. Open Questions

1. Should `kalman_track_positions` also smooth (Rauch-Tung-Striebel backward pass) or forward-only? **Proposed: forward-only initially** — matches old streaming semantics; smoother is a future enhancement.
2. Should we expose `dt` as a callable per-frame for VFR support? **Proposed: no in this plan** — VFR is backlog #24's concern; here `dt` is a scalar.

## 10. Acceptance Checklist

- [ ] Phase 1 helper + 6 unit tests
- [ ] Phase 2 wrappers re-routed; existing tests green
- [ ] Phase 3 baseline-vs-kalman diff captured; no `max_kmh` regression
- [ ] Phase 4 tuning documented (if needed)
- [ ] Phase 5 AGENTS.md + improvements.md updated
- [ ] PR description includes per-serve before/after numbers
