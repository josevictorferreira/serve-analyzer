# Court Line Homography for Position-Dependent Scale

**Status:** Proposed
**Backlog ref:** `improvements.md` #12 (foundation for #29 landing prediction)
**Estimated effort:** 2–3 days (court detection + homography + integration + manual override path)
**Risk:** Medium-high — court detection is fragile on amateur footage; needs an interactive fallback.

---

## 1. Problem

Speed estimation today uses a **single scalar** `scale_factor` (m/px), produced one of two ways:

1. **Manual two-point calibration** (`cli.py` interactive, or `--cal-p1`, `--cal-p2`, `--real-distance` flags).
2. **Ball-diameter auto-estimate** in `multi_serve.py:200`: `scale = 0.067 / median_diameter_px`.

Both assume the ball moves in a single plane parallel to the calibration baseline. In reality, in a lateral-camera setup the ball during toss is closer to the camera than at contact, which is closer than at the back baseline. Using a constant m/px:

- Overestimates speed when the ball is at the back of the court (too few px per real meter near the camera → assumed too few px per real meter everywhere → real-world meters per pixel-displacement underestimated → overestimate).
- Underestimates speed during the toss when the ball is closest to the camera.

For a 200 km/h serve, position-dependent scale errors are typically **10–20% across the trajectory**. Court homography eliminates this by mapping pixel coordinates to a **flat top-down meters frame**, where a fixed scale (1 px = 1 m / known court dimension) applies uniformly.

## 2. Goal & Success Criteria

Add court-detection + homography path. Output: every tracked position is *also* available in court-meters coordinates. Velocity computation can be done in the meters frame directly, eliminating the scalar `scale_factor` for downstream callers that opt in.

**Verifiable success:**

1. **Court detection accuracy:** On 3 test videos with hand-clicked baseline corners, automatically-detected lines yield homography with reprojection RMSE < 5 px on the four court corners.
2. **Speed consistency:** Synthesize (or pick a real video where) the ball travels at known constant speed across regions of the court. Speed measured in the meters frame varies by < 5% across regions; speed measured with the old scalar varies by ≥ 10%.
3. **No regression:** Existing scalar-scale path remains the default; opt-in via flag/config. All existing tests pass.
4. **Manual fallback works:** When auto-detection fails, an interactive 4-click corner-marking flow produces a usable homography.
5. **JSON output extended:** `selected_serves` entries gain `post_contact_max_kmh_homography` (or similar) when homography is available.

## 3. Scope

### In scope

- New module `serve_analyzer/court.py` with:
  - `detect_court_corners(frame) → Optional[np.ndarray (4,2)]` — auto-detect using line segment detector + clustering.
  - `compute_homography(image_corners, court_dimensions_m) → np.ndarray (3,3)` — standard cv2.getPerspectiveTransform.
  - `pixel_to_court_m(point, H) → (x_m, y_m)`.
  - `interactive_corner_picker(frame) → np.ndarray (4,2)` — opencv mouse callback fallback.
- New CLI flag `--use-homography` on `serve_attempts.py`, `multi_serve.py`, `plot_serve.py`.
- Velocity computation alternate path: meters-frame velocity via mapped positions.
- Persistence of homography matrix in JSON output for reproducibility.
- Documentation of expected court layout (singles vs doubles, baselines as references).

### Out of scope (explicit)

- 3D reconstruction (still single-plane; this plan flattens the *court*, not the ball flight).
- Net detection.
- Service-box landing prediction (#29 — this plan provides homography as input to that).
- Replacing the scalar `scale_factor` API. Old code paths unchanged.

## 4. Design

### 4.1 Court geometry reference

Standard tennis singles court (used as default):
- Length (baseline to baseline): 23.77 m
- Singles width: 8.23 m
- Service line distance from net: 6.40 m

Reference frame: top-down, x = sideline direction, y = baseline direction. Origin at near-camera-side service-T (configurable).

The four corners used for homography:
- `near_left`, `near_right`: baseline corners closer to camera.
- `far_left`, `far_right`: baseline corners on far side.

Real-world coordinates of these corners are fixed by the court dimensions.

### 4.2 Auto-detection pipeline

```python
def detect_court_corners(frame: np.ndarray) -> Optional[np.ndarray]:
    """Returns (4, 2) image-space corners or None on failure."""
    # 1. Isolate white pixels (lines): convert to HSV, threshold V > 180, S < 60
    white_mask = _white_mask(frame)
    # 2. Line segment detection
    lsd = cv2.createLineSegmentDetector()
    lines, _, _, _ = lsd.detect(white_mask)
    if lines is None or len(lines) < 4:
        return None
    # 3. Cluster lines by angle into "near-horizontal" (baselines/service lines)
    #    and "near-vertical" (sidelines).
    horizontals, verticals = _cluster_by_angle(lines)
    if len(horizontals) < 2 or len(verticals) < 2:
        return None
    # 4. Pick the two outermost horizontals (baselines) and two outermost verticals
    #    (singles sidelines or doubles sidelines depending on prior).
    baselines = _outermost_pair(horizontals, axis="y")
    sidelines = _outermost_pair(verticals, axis="x")
    # 5. Intersect baselines × sidelines → 4 corners.
    corners = _intersect_quad(baselines, sidelines)
    # 6. Sanity-check: corners should form a convex quadrilateral with reasonable
    #    aspect ratio (court is ~3:1 in lateral view).
    if not _is_plausible_court_quad(corners):
        return None
    return corners
```

**Failure modes** are common: occlusion by player, low contrast lines, indoor red clay with white lines vs blue surface. The pipeline must return `None` cleanly so callers fall back to manual.

### 4.3 Manual fallback: 4-click corner picker

`interactive_corner_picker(frame)` opens an OpenCV window, listens for 4 left-clicks in fixed order (`near_left → near_right → far_right → far_left`), shows running tooltip text. Returns the 4 image-space corners. Reuses the same display infrastructure as `InteractiveCalibrator` in `cli.py`.

### 4.4 Homography computation

```python
COURT_CORNERS_M = np.array([
    [0.0,           0.0    ],   # near_left
    [SINGLES_W,     0.0    ],   # near_right
    [SINGLES_W,     COURT_L],   # far_right
    [0.0,           COURT_L],   # far_left
], dtype=np.float64)

def compute_homography(image_corners: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(
        image_corners.astype(np.float32),
        COURT_CORNERS_M.astype(np.float32),
    )
```

Mapping a pixel point:
```python
def pixel_to_court_m(p: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    src = np.array([[[p[0], p[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])
```

### 4.5 Velocity in court-meters frame

Add a parallel function in `analysis.py`:

```python
def compute_velocity_series_homography(
    pixel_positions: Sequence[Tuple[float, float]],
    fps: float,
    H: np.ndarray,
    smoothing_window: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Velocity series in court-meters frame; returns (speeds_mps, speeds_kmh, stats)."""
    court_positions = np.array([pixel_to_court_m(p, H) for p in pixel_positions])
    diffs = np.diff(court_positions, axis=0)
    speeds_mps = np.linalg.norm(diffs, axis=1) * fps
    # ... apply smoothing (use _savgol_smooth from SG plan if merged)
    speeds_kmh = speeds_mps * 3.6
    stats = {
        "max_mps": float(np.max(speeds_mps)) if len(speeds_mps) else 0.0,
        "max_kmh": float(np.max(speeds_kmh)) if len(speeds_kmh) else 0.0,
        "mean_kmh": float(np.mean(speeds_kmh)) if len(speeds_kmh) else 0.0,
    }
    return speeds_mps, speeds_kmh, stats
```

Crucial caveat: the *ball is not on the court plane* during flight. Mapping a flying-ball pixel through the court homography gives the **ground projection of the ball** — a point on the court directly under the ball. Speed measured this way is the **horizontal ground-projected speed**. For a serve this is approximately the real speed (ball is mostly horizontal during the post-contact phase) but slightly underestimates because vertical component is dropped.

**Document this clearly** in the function docstring and report. Two derived numbers:
- `speed_ground_mps`: from homography (horizontal only).
- `speed_pixel_scaled_mps`: from old scalar (slant-ish).

For a serve at 5° downward, `cos(5°) ≈ 0.996` so the homography-derived speed is ~0.4% lower in the *vertical-component* sense — negligible. The horizontal speed is the meaningful number.

### 4.6 Persistence

Save homography in JSON output for downstream tools:

```json
{
  "homography_matrix": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]],
  "court_corners_image": [[x1, y1], ...],
  "court_corners_meters": [[0, 0], ...],
  "homography_source": "auto" | "manual"
}
```

This means `serve_evaluation.py` and any future trajectory analyzer can re-derive court coordinates without re-running detection.

### 4.7 CLI integration

New flag `--use-homography` on `serve_attempts.py` and `multi_serve.py`:
- Default off → unchanged behavior.
- On → run `detect_court_corners(first_frame)`. On failure, prompt for manual unless `--non-interactive` is also set.
- Once `H` obtained, compute *both* scalar-scaled and homography-scaled velocities; include both in JSON.

## 5. Implementation Steps

### Phase 1 — Court geometry constants + homography math

1. Create `serve_analyzer/court.py` with:
   - `SINGLES_LENGTH_M`, `SINGLES_WIDTH_M`, `DOUBLES_WIDTH_M` constants.
   - `compute_homography`, `pixel_to_court_m`.
2. Add `tests/test_court.py`:
   - `test_homography_identity_when_corners_match_meters`: pass court corners as image corners → H ≈ identity (modulo scale); reprojection error < 0.01 px.
   - `test_pixel_to_court_m_known_quad`: hand-craft a perspective-warped quad → verify mapping reproduces real-world coords within 0.01 m.
   - `test_singleton_point_round_trip`: image → meters → image via H and H⁻¹ within 1 px.

**Verify:** `python -m unittest tests.test_court -v` passes.

### Phase 2 — Manual corner picker (interactive)

1. Add `interactive_corner_picker(frame, window_name="Pick Court Corners")` modeled on `InteractiveCalibrator` in `cli.py`.
2. Smoke-test by running:
   ```bash
   python -c "import cv2, serve_analyzer.court as c; \
     cap=cv2.VideoCapture('video.mov'); _,f=cap.read(); \
     print(c.interactive_corner_picker(f))"
   ```
3. No automated test for this (interactive); manual verification only.

### Phase 3 — Auto-detection

1. Implement `detect_court_corners` (§4.2 pipeline).
2. Add `tests/test_court_detection.py`:
   - `test_auto_detect_synthetic_court`: render a synthetic court (drawn lines on a blank image, perspective-warped) → detection within 5 px of true corners.
   - `test_no_lines_returns_none`: blank image → `None`.
   - `test_real_video_first_frame`: video-gated test on `video.mov` first frame → corners detected (or None acceptable, just no crash).

**Verify:** `python -m unittest tests.test_court_detection -v` passes.

### Phase 4 — Velocity-in-meters

1. Add `compute_velocity_series_homography` to `analysis.py`.
2. Unit test with hand-computed positions on a known H:
   - Build positions tracing 1 m/frame in court coords; verify `max_mps == fps × 1.0`.

### Phase 5 — CLI integration

1. Add `--use-homography` flag to `serve_attempts.py` parser.
2. In the detection flow: if flag set, derive H (auto then manual fallback), compute homography velocities alongside scalar velocities, emit both in JSON.
3. Same for `multi_serve.py` and `plot_serve.py`.

**Verify:**
- `python -m serve_analyzer.serve_attempts video.mov --use-homography --output out_hom.json` runs without error (auto or manual).
- `out_hom.json` contains `homography_matrix` and `post_contact_max_kmh_homography` per serve.

### Phase 6 — Real-video verification

1. Pick a video where ball clearly traverses near and far court regions.
2. Compute scalar `max_kmh` and homography `max_kmh_homography` per serve.
3. Document the per-region differences in PR description.
4. Manually click ground-projection at key frames, verify mapped meters coords are sensible (e.g. ball at far baseline maps to y ≈ 23.77 m).

### Phase 7 — Docs

1. AGENTS.md SESSION LEARNINGS:
   ```
   ### Court homography for position-dependent scale
   **Lesson:** Single scalar scale_factor causes 10-20% speed errors across court. Use serve_analyzer.court.detect_court_corners + compute_homography for ground-projection-based velocity. Auto-detect via LSD + line clustering; fallback to 4-click manual.
   **Context:** Ball-flight trajectories are not on court plane; homography gives horizontal ground-projected speed (≈ real speed for serves at ≤5° downward angle, error <0.5%).
   **Verify:** speed measured at near vs far court regions varies <5% with homography, ≥10% with scalar scale.
   ```
2. Update `improvements.md` #12 status with plan path.
3. README CLI section: document `--use-homography` and the assumption that homography velocity is *horizontal ground-projected*.

## 6. Test Plan

| Test | Type | Verifies |
|---|---|---|
| `test_homography_identity_when_corners_match_meters` | unit | Math correctness |
| `test_pixel_to_court_m_known_quad` | unit | Inverse warp |
| `test_singleton_point_round_trip` | unit | Numerical stability |
| `test_auto_detect_synthetic_court` | unit (synthetic image) | Detection pipeline |
| `test_no_lines_returns_none` | unit | Graceful failure |
| `test_real_video_first_frame` | integration (video-gated) | Real-world detection |
| `test_velocity_homography_known_motion` | unit | Velocity math |
| Existing `tests/test_*` | regression | No break in scalar path |

## 7. Rollback

- Remove `--use-homography` flag → CLI returns to old behavior.
- Delete `serve_analyzer/court.py`, tests, JSON fields.
- No persistent state; trivially reversible.

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Auto-detection fails on >50% of amateur clips | High | Manual 4-click fallback; auto is opt-in |
| Court occlusion by player blocks line detection | Medium | Use frame from before serve (player not on court) for detection; cache H for whole video |
| Wrong corner ordering in manual click breaks H | Medium | Explicit numbered tooltip ("click 1: near_left", etc.); validate convex quad after collection |
| Homography from non-coplanar features (camera shake) drifts over time | Low | Stationary camera assumed; warn if global motion detected (links to #23) |
| Singles vs doubles court ambiguity | Medium | Default to singles; expose `--court doubles` flag |

## 9. Open Questions

1. Cache homography per-video or recompute per-run? **Proposed: per-run by default, with `--save-homography path.json` and `--load-homography path.json` opt-in.** Avoids stale H if camera moved between runs.
2. Should homography velocity be the default if available? **Proposed: no in this PR.** Add as second number; let users compare. Default-switch is a follow-up after verification on more videos.
3. Multiple courts in frame (broadcast camera)? **Proposed: out of scope** — assume single court.

## 10. Acceptance Checklist

- [ ] Phase 1 court math + 3 unit tests
- [ ] Phase 2 manual corner picker (smoke-tested)
- [ ] Phase 3 auto-detection + 3 unit tests
- [ ] Phase 4 velocity-in-meters + unit test
- [ ] Phase 5 CLI integration; JSON has homography fields
- [ ] Phase 6 real-video numbers documented in PR
- [ ] Phase 7 docs updated
- [ ] No regression on existing tests
