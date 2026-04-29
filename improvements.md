# Serve-Analyzer Improvements Backlog

**Generated:** 2026-04-28
**Source:** Audit of `serve_analyzer/` package against `annotations.md` seed ideas.
**Status:** Living document — edit as items move to plans / get implemented.

This file is the canonical registry of proposed improvements. Each item below has been **cross-checked against the current codebase** and confirmed as not-yet-implemented (or only partially implemented and worth deepening).

---

## Audit: What the Codebase Already Does

The following ideas are already implemented and have been **dropped** from the backlog. Do not re-propose without reading the cited code first.

| # | Idea | Status | Where |
|---|------|--------|-------|
| 6 | Custom YOLO fine-tune | ✅ Done — RJTPP HuggingFace model loaded as default | `multi_serve.py:103-112` |
| 7 | TrackNet | ✅ Done — TrackNetV2 with 3 architectures (chgyglin, WASB, generic) | `tracknetv2.py` |
| 13 | Physics-constrained interpolation | ⚠️ Linear-only interpolation exists for small gaps | `multi_serve.py:212-269`, `serve_attempts_v2.py:56-78` |
| 1 | Motion-gated ROI | ⚠️ Partial — `track_ball_template` and `track_ball_color` use `search_radius`; `continuity_gate_positions` rejects jumps > `max_jump_px` | `analysis.py:131,292`, `serve_attempts_v2.py:28` |
| 11 | Forward-backward consistency | ⚠️ Forward+backward fill of missing detections (not full FB tracking) | `multi_serve.py:253-267` |
| 12 | Smoothing | ✅ Gaussian smoothing on velocity (`gaussian_filter1d, sigma=2`) + moving average | `multi_serve.py:296`, `analysis.py:107-109` |
| 16 | Two-pass / multi-profile detection | ✅ `_merge_candidate_events` merges multiple detection profiles | `serve_attempts.py:36-67` |
| 21 | Acceleration spike at contact | ✅ `_refine_contact_frame` uses horizontal acceleration | `multi_serve.py:357-408` |
| 22 | Direction reversal / vy & vx analysis | ✅ `compute_vertical_velocity` + `compute_horizontal_velocity` + `upward_fraction` / `rightward_fraction` features | `multi_serve.py:302,330`, `serve_attempts.py:111-200` |
| 25 | Trophy / serve-phase gating | ⚠️ Phase labels exist (TOSS/HIT/FLIGHT) and toss-evidence gate is strict | `annotate_video.py:147-180`, `serve_attempts.py:140-149` |
| 27 | Phase state machine | ⚠️ Not HMM, but explicit toss → contact → post_contact frames stored | `multi_serve.py:40-63` |
| 30 | Robust peak velocity | ⚠️ Mean over post-contact window stored alongside max | `serve_evaluation.py:182-185` |
| 35 | Multi-hypothesis tracking | ⚠️ Multi-profile candidate pool merging (not full MHT) | `serve_attempts.py:36` |
| 8 | Multi-scale templates | Not done — but YOLO/TrackNet make this irrelevant | — |

**Tracking variants in `analysis.py`** (already exist, user-toggleable):
- `track_ball_template` (template matching)
- `track_ball_color` (HSV)
- `track_ball_csrt` (OpenCV CSRT correlation tracker)
- `track_ball_optical_flow` (Lucas-Kanade pyramidal)
- `track_ball_yolo` (full-frame YOLO)

**Scale factor:** A ball-diameter-based auto-estimator already runs (`0.067 m / median_diameter_px`), so manual calibration is not the only path.

### Partially done — could be deepened

- **Motion cues around contact** (`extract_motion_cues` in `serve_attempts_v2.py`) does frame-differencing inside windows — applied only as a contact-refinement feature, not as a primary detection gate.
- **Velocity smoothing** uses Gaussian, not Savitzky–Golay (SG preserves peaks better — see #19).
- **Multi-detector ensemble:** YOLO + HSV fallback exists in `multi_serve.py:162`, but it's *sequential* (HSV only when YOLO misses), not voting-based.

### Not implemented (still valid)

Nothing from category D-audio, E-homography, F-spin/bounce/landing, G-Kalman, audio sync, court detection.

---

## Backlog: 32 Open Improvement Ideas

Numbered fresh. Grouped by subsystem.

### A. Ball Detection — Robustness Gaps

**1. Kalman filter on top of existing detectors**
Current code rejects jumps > `max_jump_px` and forward-fills missing frames (`continuity_gate_positions`, `interpolate_missing_detections`). Replace both with a 4-state Kalman (x, y, vx, vy) so missing frames get *predicted* (not last-position-held), and jump rejection becomes Mahalanobis-distance-based instead of a fixed pixel threshold. Replaces ~3 ad-hoc heuristics with one principled filter.

**2. MOG2 background subtraction as detection gate**
HSV fallback at `multi_serve.py:162-183` matches yellow on the entire frame → still triggers on static yellow (lines, logos). Add `cv2.createBackgroundSubtractorMOG2()` and intersect its foreground mask with the HSV mask before contour search. Removes static-yellow false positives.

**3. Frame differencing as a primary detection input**
`extract_motion_cues` already computes frame diffs but only to score contact candidates. Promote it: in HSV fallback, AND the HSV mask with `|frame_t − frame_{t-1}|` → ball moves, lines don't.

**4. Hough circle confirmation pass**
When YOLO confidence is low (0.20–0.35 zone), run `cv2.HoughCircles` on the bbox to confirm the round shape. Cheap second opinion that rejects racket-string yellow flashes.

**5. Voting ensemble (YOLO + HSV + TrackNet)**
Today YOLO and HSV are sequential (if YOLO miss → HSV). Run all three already-implemented detectors in parallel and take the position where ≥2 agree within N pixels. Compute is already paid per detector in `compare_detectors.py`; turn that comparison into a runtime ensemble.

**6. Per-video adaptive HSV bounds**
`(18, 80, 80)–(45, 255, 255)` is hardcoded across `multi_serve.py`, `analysis.py`, `test_roboflow_model.py`. Sample HSV at the first confirmed YOLO/TrackNet detection, build per-video tight bounds → ~1.5–2× fewer false positives indoors/in shade.

**7. Sub-pixel ball center via mask-weighted centroid**
Current YOLO path uses bbox center `(x1+x2)/2`. Replace with intensity-weighted centroid over the HSV mask inside the bbox. Sub-pixel precision improves velocity accuracy directly (current 0.3–1.0 px error at 4K → 5–15% at 200 km/h).

**8. Forward-backward tracking pass**
Run track once forward, once backward from a high-confidence late detection, keep only positions agreeing within 5 px. Catches single-frame glitches that survive `continuity_gate_positions`.

### B. Player/Racket Context (none of this exists yet)

**9. MediaPipe Pose for tossing-arm prior**
Run pose in parallel with ball detection. Tossing-wrist y-coordinate constrains where the ball must be during toss frames. Eliminates ball-detected-on-shoes / wristbands false positives during the toss phase.

**10. Racket bbox detector**
Add a small YOLO racket head (or extend the RJTPP model). Contact frame = frame where `ball_bbox ∩ racket_bbox > 0` AND ball direction reverses. Independent signal that the current acceleration-only refinement (`_refine_contact_frame`) lacks.

**11. Player segmentation mask (YOLOv8-seg or SAM2)**
Mask out the player → run HSV fallback only on non-player pixels. Removes the dominant false-positive source (yellow/lime shoes, sponsor logos).

**12. Court line homography**
Detect court lines via Hough on the white channel, compute homography to top-down. Replaces the constant `scale_factor` (or even the ball-diameter-derived one) with position-dependent meters/pixel — the ball at the back baseline and at the contact point use different m/px today, which biases speed.

### C. Serve Event / Impact Detection

**13. Audio-based contact onset detection**
Tennis impact has ~2–4 kHz transient. Use `librosa.onset.onset_detect` on the video's audio track, cross-correlate with `_refine_contact_frame` outputs. Sub-frame timing (±10 ms vs current ±1 frame ≈ 16 ms at 60 fps) and an independent signal that resolves ambiguous direction-reversal frames.

**14. Audio–video disagreement as rejection signal**
If audio onset and visual contact disagree by > 60 ms, demote the candidate. Cheap way to kill non-serve events (player bouncing the ball before service motion) that survive the `has_toss_evidence` gate.

**15. HMM over phase labels**
`annotate_video.py` uses hard threshold-based IDLE/TOSS/HIT/FLIGHT. Replace with a 5-state HMM whose emissions are `[vy, vx, |a|, motion_energy, pose_features]`. Smoother phase boundaries; fewer one-frame misclassifications.

**16. Pose "trophy position" detector as serve gate**
Knee bend + arm extension upward is a strong serve precondition. Use it as a necessary gate: no candidate is accepted unless trophy posture occurs in `[contact - 1.0s, contact - 0.2s]`. Cuts non-serve velocity spikes (player walking, bouncing).

**17. Ball–racket-proximity event**
Add to scoring in `_detect_broad_trajectory_events`: `+200 * (1 if racket_overlap_at_contact else 0)`. Combined with #10.

**18. Pre-serve quiescence detector**
A real serve has a stationary phase right before toss (~300–800 ms). Flag candidates without preceding stillness (player rallying = no serve). Currently `toss_lookback_sec=2.2` gate is unidirectional (only checks upward motion exists, not that motion was stationary before).

### D. Speed Estimation

**19. Savitzky–Golay velocity filter** — *plan written: `.docs/plans/savgol-velocity-smoothing/plan.md`*
Replace `gaussian_filter1d(sigma=2)` with `scipy.signal.savgol_filter(window=7, polyorder=3)`. Gaussian attenuates the peak you care most about (post-contact velocity); SG preserves it.

**20. Top-K frame mean for peak velocity**
`summary_stats['max_kmh']` uses `np.max` → vulnerable to one bad frame. Use `np.mean(np.sort(speeds_kmh)[-5:])` over the top 5 frames in the post-contact window. Robust without losing peak signal.

**21. MAD-based velocity outlier rejection**
Before computing peak: drop frames where `|v - median(v)| > 5 * MAD(v)`. Filters single-frame teleports that survived `continuity_gate_positions`.

**22. Motion-blur-aware displacement**
At 200 km/h ball travels ~28 px/frame at 4K/60fps and smears across more. Estimate blur length from the bbox aspect ratio (motion-blurred ball is elongated) and add half the blur to the displacement. Currently underestimates fast serves by 5–10%.

**23. Camera-shake compensation**
Detect global motion via ORB feature matching between consecutive frames (excluding player/ball pixels), subtract from ball displacement. Handheld phone footage adds 5–15 km/h spurious to current measurements.

**24. Real PTS timestamps via ffprobe**
`1/fps` is wrong for variable-framerate phone videos (iPhones drop frames in low light). Use `ffprobe -show_frames -of json` to get actual presentation timestamps; use real `dt` per frame in `compute_velocity_series`. Fixes a systematic bias the current code can't see.

**25. Confidence-weighted velocity**
Weight each frame's contribution to the peak by detection confidence. Today `interpolate_missing_detections` produces interpolated positions that look identical to real ones in downstream velocity math.

### E. Post-Serve Trajectory

**26. Bounce detection via vy sign-flip + energy loss**
Post-contact tracking continues until `post_contact_end_frame` but doesn't segment bounces. Detect bounces (vy flips with ~0.7× restitution) → enables segmenting "in-flight" velocity (the meaningful number) from post-bounce.

**27. Trajectory parabola RANSAC fit**
Fit `y(t) = y₀ + v_y·t + ½g·t²` to post-contact positions with RANSAC. Output (a) cleaned trajectory, (b) outlier indices for further detection-pipeline tuning.

**28. Spin estimation from drag deviation**
Once you have RANSAC parabola, residuals in the horizontal direction encode Magnus force → estimate slice/topspin/kick magnitude. Differentiates a 180 km/h flat serve from a 180 km/h kick.

**29. Landing point extrapolation**
With #12 (homography), extrapolate trajectory to court Y-plane → predicted bounce location. Useful both as a feature (in/out service box hint) and as a plausibility check to reject implausible candidates whose trajectories would land in the stands.

**30. Trajectory uncertainty bands**
Propagate Kalman covariance (#1) through to a 95% confidence ellipse on the trajectory. Honest uncertainty for an MVP. Today velocity is reported as a single number with no error bar.

### F. Engineering / Verifiability

**31. Synthetic-ball regression fixtures**
Generate Blender or OpenCV-composited videos with known trajectory + speed + spin. Tests today (`tests/test_serve_attempts.py`) only assert structure, not numeric accuracy. A 10-fixture synthetic suite gives end-to-end accuracy regression detection.

**32. Two-stage frame_skip**
Coarse pass at `frame_skip=8` to find serve windows, fine pass at `frame_skip=1` only inside ±1.5s of each candidate. Per AGENTS.md `frame_skip=4` is the practical default; this gives `frame_skip=1` quality at near-`frame_skip=8` cost.

---

## Top-5 Priority

Ordering reflects effort × impact × independence (per AGENTS.md §4 verifiable success criteria).

| Priority | Idea | Plan | Why | Verify |
|----------|------|------|-----|--------|
| 1 | #19 Savitzky–Golay | ✅ `.docs/plans/savgol-velocity-smoothing/plan.md` | One-line core change, measurable peak-velocity improvement | Compare `max_kmh` on a known-speed clip before/after |
| 2 | #1 Kalman replaces continuity_gate + interpolate | ⏳ pending | Replaces 2 ad-hoc components with 1 principled one | Detection rate stable, jump rejection Mahalanobis-distance verified |
| 3 | #13 Audio onset | ⏳ pending | Independent signal, dramatically improves contact-frame accuracy | `|audio_t - visual_t| < 30 ms` on test video |
| 4 | #12 Court homography | ⏳ pending | Foundation for #29 + position-dependent scale | Speed at near-camera vs back-court no longer differs by 15%+ |
| 5 | #20 Top-K mean peak velocity | ⏳ pending | Trivial, removes single-frame outlier sensitivity | Variance of peak across re-runs drops |

---

## Conventions for this file

- One row in "Already implemented" or one entry in the backlog per idea — never both.
- When an idea moves to a plan, add the plan path next to its title (see #19).
- When an idea ships, move it from the backlog to "Already implemented" with the merge commit / file:line reference.
- Do not delete shipped items — historical context is useful.
