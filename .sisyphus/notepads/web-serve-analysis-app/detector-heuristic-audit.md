# Autonomous Serve Detector Heuristic Audit

**Scope**: `serve_analyzer/multi_serve.py` + `serve_analyzer/serve_attempts.py`  
**Goal**: Identify exact scoring, directional, suppression, and velocity calculations that can misclassify upward tosses, aborted motions, wall rebounds, or camera shake as serves, and explain why speeds could be too high.

---

## 1. VELOCITY COMPUTATION

### 1.1 Frame-to-frame pixel velocity
**File**: `serve_analyzer/multi_serve.py:272-299`
```python
dx = np.diff(positions[:, 0])
dy = np.diff(positions[:, 1])
velocities = np.sqrt(dx**2 + dy**2)
velocities = np.concatenate([[velocities[0]], velocities])
if smooth_sigma > 0:
    velocities = gaussian_filter1d(velocities, sigma=smooth_sigma)
```
- **Issue**: Gaussian smoothing (`sigma=2.0`) blurs genuine contact spikes together with adjacent frames, inflating the "contact" velocity if the spike is narrow (1-2 frames). It also spreads energy from a real spike into neighboring frames, creating phantom high-velocity frames.
- **Why speeds too high**: The smoothed velocity at the peak frame is an average of the true peak and its neighbors. If the true peak is very brief, the smoothed value can be higher than the actual instantaneous velocity because the kernel spreads the peak energy while the `np.diff` already overestimates due to interpolation gaps (see below).

### 1.2 Vertical velocity
**File**: `serve_analyzer/multi_serve.py:302-327`
```python
dy = np.diff(positions[:, 1])
dy = np.concatenate([[dy[0]], dy])
if smooth_sigma > 0:
    dy = gaussian_filter1d(dy, sigma=smooth_sigma)
```
- **Issue**: Same smoothing problem. A single-frame camera shake (jitter in y) gets smoothed into a multi-frame "upward" signal, triggering toss detection.

### 1.3 Real-world speed conversion
**File**: `serve_analyzer/analysis.py:54-128`
```python
distances_m = displacements * scale_factor
speeds_mps = distances_m / dt
speeds_mps_smoothed = np.convolve(speeds_mps, kernel, mode='same')
speeds_kmh = speeds_mps_smoothed * 3.6
```
- **Issue**: `mode='same'` in `np.convolve` causes edge effects at the start/end of the phase. The first and last `(smoothing_window // 2)` frames are computed with partial kernels, producing unreliable speeds at boundaries — which are exactly the contact frame and the first post-contact frame.
- **Scale factor risk**: Default `scale_factor=0.001` (1 mm/px) is a wild guess. If auto-estimation fails or is not used, speeds are off by orders of magnitude. Auto-estimation (`multi_serve.py:195-203`) uses median ball diameter from detections; if the detector sees a large blob (shadow, glare, hand), the diameter is wrong and scale is wrong.

---

## 2. CONTACT TIMING DETECTION

### 2.1 Velocity spike detection (contact finder)
**File**: `serve_analyzer/multi_serve.py:358-367`
```python
velocity_threshold = np.percentile(velocities, velocity_spike_percentile)
peaks, properties = signal.find_peaks(
    velocities,
    height=velocity_threshold,
    distance=min_gap_frames,
    prominence=velocity_threshold * 0.3,
)
```
- **Percentile threshold is video-global**: A video with many fast motions (running, ball machine, wall rebounds) raises the global percentile, so real serve contacts may fall below threshold. Conversely, a video with mostly static frames lowers the threshold, so any small motion (camera shake, player arm swing, ball rolling) becomes a "spike."
- **Prominence relative to global threshold**: `prominence=velocity_threshold * 0.3` means prominence scales with the video's overall motion. In a low-motion video, a tiny bump qualifies.
- **No directional requirement for the spike**: `find_peaks` operates on velocity *magnitude*. A fast horizontal wall rebound or a player swinging the racket (not hitting the ball) produces a magnitude peak just as valid as a post-contact ball motion.

### 2.2 Toss validation (upward motion check)
**File**: `serve_analyzer/multi_serve.py:374-409`
```python
search_start = max(0, peak_frame - toss_lookback)
toss_region = vert_velocities[search_start:peak_frame]
upward_fraction = float(np.mean(toss_region < -1)) if len(toss_region) > 0 else 0.0
recent_window = int(0.6 * fps)
recent_region = vert_velocities[max(search_start, peak_frame - recent_window):peak_frame]
recent_upward_fraction = float(np.mean(recent_region < -1)) if len(recent_region) > 0 else 0.0

# Apex
y_coords = [p[1] for p in toss_positions]
apex_idx = np.argmin(y_coords)
apex_frame = search_start + apex_idx
frames_after_apex = peak_frame - apex_frame
drop_after_apex = positions[peak_frame][1] - apex_position[1]

upward_motion = upward_fraction > 0.3 or (
    recent_upward_fraction > 0.35
    and frames_after_apex <= int(0.4 * fps)
    and drop_after_apex > 120
)
```
- **Hardcoded -1 px/frame threshold**: `vert_velocities < -1` means "upward." At 60 fps, -1 px/frame is only -60 px/sec. Any slow drift upward (player lifting the ball, wind, tracking jitter) counts as "toss."
- **upward_fraction > 0.3**: Only 30% of the lookback window needs to be "upward." In a 2-second lookback at 60 fps (120 frames), just 36 frames of slight upward drift satisfies this. A player bouncing the ball before serving, or a wall rebound with slight upward arc, easily qualifies.
- **recent_upward_fraction fallback**: If the long window fails, the recent 0.6 seconds only needs 35% upward. An aborted serve where the player tosses then catches (no contact) can still pass if the abort happens after the recent window.
- **drop_after_apex > 120**: The ball must drop 120 pixels from apex to contact. On a high-resolution video (4K), 120 px is tiny (~3% of frame height). On a low-res video, 120 px might be significant, but the threshold is absolute — not scaled to frame height or ball size. A small camera shake downward after a false apex easily exceeds 120 px.
- **Apex detection is naive**: `np.argmin(y_coords)` finds the highest point in the lookback window. If the lookback includes a prior serve's post-contact trajectory (because `min_serve_gap_sec` was too small or the prior serve was missed), the "apex" could be from the previous serve, and `frames_after_apex` becomes very large or negative, breaking the logic silently.

---

## 3. SCORING (CANDIDATE QUALITY)

### 3.1 Event scoring in `detect_serve_events`
**File**: `serve_analyzer/multi_serve.py:444-484`
```python
score = float(contact_vel)
pre_start = max(0, contact_frame - 5)
if pre_start < contact_frame:
    pre_vel = np.mean(velocities[pre_start:contact_frame])
    score += float(max(0, contact_vel - pre_vel))
score += upward_fraction * 80.0
score += recent_upward_fraction * 120.0
score += float(max(0.0, drop_after_apex) * 0.35)
post_vels = velocities[contact_frame + 1 : post_end]
if len(post_vels) > 0:
    score += float(np.mean(post_vels) * 2.0)
if frames_after_apex <= fps * 0.45:
    score += 120.0
elif frames_after_apex <= fps * 0.75:
    score += 60.0
if frames_after_apex > fps * 0.5:
    score -= float((frames_after_apex - fps * 0.5) / fps * 140.0)
if drop_after_apex < 120:
    score -= float((120.0 - drop_after_apex) * 2.5)
```
- **Score is dominated by raw contact velocity**: `score = contact_vel` starts the score. A fast wall rebound or a player swinging the racket near the camera (large pixel displacement) gets a high base score.
- **Acceleration bonus**: `max(0, contact_vel - pre_vel)` rewards sudden jumps. Camera shake between frames (e.g., tripod bump) produces exactly this pattern: low pre-vel, high instantaneous vel.
- **Post-contact mean bonus**: `np.mean(post_vels) * 2.0` rewards sustained fast motion. A ball rolling on the ground after a missed hit, or a wall rebound with continued motion, scores highly.
- **Proximity bonus (frames_after_apex)**: Contact within 0.45s of apex gets +120. This rewards events where the "contact" is close to a local minimum in y. A player bending down to pick up a ball (y minimum at their hand) followed by standing up (y increasing) can trigger this.
- **Drop penalty is weak**: Drop < 120 px subtracts `(120 - drop) * 2.5`. For drop=0, penalty is 300. But the proximity bonus alone is +120, and contact_vel + post_vel bonus can easily exceed 300 for false positives. The penalty does not dominate.

---

## 4. SUPPRESSION & SELECTION

### 4.1 Overlap filtering in `detect_serve_events`
**File**: `serve_analyzer/multi_serve.py:489-501`
```python
selected_events = []
for event in serve_events:
    overlap = False
    for sel in selected_events:
        if abs(event["contact_frame"] - sel["contact_frame"]) < min_gap_frames:
            overlap = True
            break
    if not overlap:
        selected_events.append(event)
    if len(selected_events) >= expected_serves:
        break
```
- **Greedy by score, not by quality**: After sorting by score descending, the code greedily picks the highest-scoring event, then suppresses nearby events. If the highest-scoring event is a false positive (e.g., wall rebound), all real serves near it are suppressed.
- **No temporal sanity check**: Events can be selected within seconds of each other if they pass `min_gap_frames`. A player bouncing the ball 3 times before serving can produce 3 high-scoring candidates within 2 seconds; the first (bounce) wins, suppressing the real serve.

### 4.2 Multi-profile merging in `serve_attempts.py`
**File**: `serve_analyzer/serve_attempts.py:32-63`
```python
def _merge_candidate_events(...):
    max_merge_gap_frames = max(1, int(max_merge_gap_sec * fps))
    flattened = sorted(..., key=lambda event: int(event["contact_frame"]))
    merged = []
    for event in flattened:
        contact_frame = int(event["contact_frame"])
        if not merged:
            merged.append(event)
            continue
        previous = merged[-1]
        previous_frame = int(previous["contact_frame"])
        if contact_frame - previous_frame > max_merge_gap_frames:
            merged.append(event)
            continue
        previous_score = float(previous.get("score", 0.0))
        current_score = float(event.get("score", 0.0))
        if current_score > previous_score:
            merged[-1] = event
    return merged
```
- **Keeps highest score within 0.75s window**: If one profile detects a false positive at t=10.0s with score 500, and another profile detects the real serve at t=10.3s with score 400, the false positive wins because it has higher score.
- **No cross-profile quality consensus**: A candidate only needs to be best in its local window, not supported by multiple profiles. A single outlier profile can inject false positives.

### 4.3 Selector ranking in `select_serves`
**File**: `serve_analyzer/serve_attempts.py:216-380`

#### 4.3.1 Bonus components
```python
support_bonus = 0.0 if support <= 1 else 0.6 if support == 2 else 1.0
recent_bonus = _clip((recent - 0.25) / 0.30)
after_bonus = _clip((after - 12.0) / 40.0)
contact_bonus = _robust_norm(contacts, contact)
post_bonus = _robust_norm(capped_posts, min(post_value, p90_post))
score_bonus = _robust_norm(capped_scores, min(score_value, p85_score))
```
- **support_bonus**: A candidate supported by 2+ profiles gets +0.6 or +1.0. But because `_merge_candidate_events` already keeps only one event per window, `support_count` counts events from different profiles that fell within the merge window. If all 3 profiles detect the same false positive (e.g., a very obvious wall rebound), support_count=3 and the candidate gets maximum bonus.
- **recent_bonus**: Requires `recent_upward_fraction > 0.25`. As noted in §2.2, this is easy to satisfy with jitter.
- **after_bonus**: Rewards `frames_after_apex > 12`. A false apex from prior motion followed by a late "contact" (e.g., player moving arm down) gets this bonus.
- **contact_bonus / post_bonus / score_bonus**: These are robust-normalized against the candidate pool. If the pool is dominated by false positives with high velocities, the real serves (which may have lower velocities due to tracking errors) get low bonuses. The normalization is *relative to the pool*, not absolute.

#### 4.3.2 Penalty components
```python
early_steep_excess = max(0.0, drop - (120.0 + 8.0 * after))
early_steep_penalty = _clip(early_steep_excess / 220.0)

if after <= 6.0 and drop >= 120.0 and recent < 0.60:
    apex_snap_penalty = 1.0
elif after <= 12.0 and drop >= 220.0 and recent < 0.55:
    apex_snap_penalty = 0.5

post_outlier_penalty = _clip((post_value - p90_post) / max(p90_post - p50_post, 1e-6))
```
- **early_steep_penalty**: Penalizes large drops shortly after apex. But the threshold `120 + 8*after` means at `after=6`, drop must exceed 168 to trigger penalty. At `after=12`, drop must exceed 216. A wall rebound with drop=150 at `after=8` gets zero penalty.
- **apex_snap_penalty**: Only triggers when `recent < 0.60` (or 0.55). If camera shake produces enough "upward" frames to keep `recent >= 0.60`, the penalty is avoided even for impossible drops.
- **post_outlier_penalty**: Penalizes post-contact speed above 90th percentile. But if the entire pool has inflated speeds (due to scale error or tracking on the wrong object), the 90th percentile is also inflated, so outliers are not penalized.

#### 4.3.3 Final rank formula
```python
rank = (
    0.30 * support_bonus
    + 0.22 * recent_bonus
    + 0.14 * after_bonus
    + 0.12 * contact_bonus
    + 0.10 * post_bonus
    + 0.06 * score_bonus
    - 0.70 * early_steep_penalty
    - 0.35 * apex_snap_penalty
    - 0.20 * post_outlier_penalty
)
```
- **Support bonus is the heaviest positive weight (0.30)**. A candidate detected by all 3 profiles gets +0.30 regardless of whether it's a real serve. Multi-profile consensus amplifies false positives if the underlying detector is biased.
- **Penalties are capped at 1.0 and weighted down**: Max penalty contribution is -0.70 (early_steep). A candidate with support_bonus=1.0 and all penalties=1.0 still scores 0.30 - 0.70 - 0.35 - 0.20 = -0.95. But a candidate with support_bonus=1.0, no penalties, and moderate bonuses easily scores > 1.0.

#### 4.3.4 Suppression logic
```python
for candidate in ranked_candidates:
    ...
    dominated = False
    for previous in suppressed:
        gap = current_time - float(previous["contact_time_sec"])
        if (
            0.0 < gap <= 3.5
            and float(previous["selector_rank"]) >= current_rank + 0.10
            and current_steep > 0.10
        ):
            dominated = True
            break
    if not dominated:
        recent_supported = [
            previous for previous in suppressed
            if 0.0 < current_time - float(previous["contact_time_sec"]) <= 8.5
            and int(previous.get("support_count", 1)) >= 2
        ]
        for index, first in enumerate(recent_supported):
            for second in recent_supported[index + 1:]:
                combined_rank = float(first["selector_rank"]) + float(second["selector_rank"])
                if combined_rank >= current_rank + 0.28:
                    dominated = True
                    break
            if dominated:
                break
    if not dominated:
        suppressed.append(candidate)
```
- **Domination by nearby higher-ranked candidate**: If a false positive ranks 0.50 and a real serve 3.5s later ranks 0.35, the real serve is suppressed. The 3.5s gap is arbitrary; real serves in rapid succession (e.g., practice rally) can be 2-3s apart.
- **Pairwise domination**: Two supported candidates within 8.5s can suppress a third. If a player does two practice tosses (both supported by multiple profiles due to repeated motion) and then a real serve, the real serve is suppressed.

#### 4.3.5 Temporal bias
```python
early_bonus = 0.12 * math.exp(-(((candidate_time - 13.0) / 4.5) ** 2))
late_penalty = 0.18 * _clip((candidate_time - 63.0) / 4.0)
```
- **Gaussian bonus centered at 13 seconds**: Candidates near 13s get up to +0.12. This encodes an assumption that serves happen early in the video. If the video starts with warm-up bounces, those get boosted.
- **Late penalty after 63s**: Candidates after 63s are penalized linearly. Long videos with serves throughout are biased against.

### 4.4 Count inference
**File**: `serve_analyzer/serve_attempts.py:173-213`
```python
def infer_serve_count(...):
    threshold = max(min_rank_floor, relative_floor * top_rank)
    above_threshold = [r for r in ranks if r >= threshold]
    ...
    max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    max_gap = gaps[max_gap_idx]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap < 1e-9 or max_gap < 2.0 * mean_gap or max_gap < 0.20:
        return len(above_threshold)
    return max_gap_idx + 1
```
- **relative_floor = 0.55**: A candidate must have rank >= 55% of the top rank. If the top rank is a false positive with rank 2.0, everything above 1.1 is kept. In a video with many false positives, this keeps too many.
- **Elbow detection is fragile**: The gap-based elbow requires `max_gap >= 2.0 * mean_gap` AND `max_gap >= 0.20`. If ranks are [2.0, 1.9, 1.8, 0.5, 0.4], gaps are [0.1, 0.1, 1.3, 0.1], mean=0.4, max_gap=1.3. 1.3 < 2*0.4=0.8 is false, so all 5 are returned. The elbow is missed because the threshold is too strict.

---

## 5. WHY SPEEDS CAN BE TOO HIGH — SUMMARY

| Cause | Mechanism | Location |
|---|---|---|
| **Wrong scale factor** | Auto-estimation uses median ball diameter; false detections (hand, shadow) inflate diameter, reducing scale, inflating speed | `multi_serve.py:195-203` |
| **Interpolation across gaps** | `interpolate_missing_detections` fills up to 15 frames with linear interpolation, then forward-fills indefinitely. Large gaps create unrealistic straight-line displacements | `multi_serve.py:212-269` |
| **Smoothing artifacts** | Gaussian `sigma=2.0` on velocity spreads narrow spikes, inflating contact-frame velocity and creating phantom motion in static regions | `multi_serve.py:272-299`, `analysis.py:107-108` |
| **Tracking the wrong object** | YOLO fallback to HSV color detection picks any yellow blob (shirt, wall, line). The "ball" then moves at player/wall speed, not ball speed | `multi_serve.py:162-183` |
| **Post-contact phase includes pre-contact** | `post_positions = positions[contact : post_end + 1]` includes the contact frame itself. If contact timing is off by a few frames, the pre-contact toss velocity is averaged into post-contact mean | `multi_serve.py:536-537` |
| **Edge effects in convolution** | `mode='same'` in `np.convolve` at phase boundaries produces unreliable speeds at contact frame | `analysis.py:107-108` |

---

## 6. MISCLASSIFICATION VECTORS — SUMMARY

| False Positive Type | Why It Passes | Key Weak Heuristic |
|---|---|---|
| **Upward toss (no contact)** | Toss validation only requires 30% upward frames; no requirement for actual velocity spike at contact | `upward_fraction > 0.3` (`multi_serve.py:402`) |
| **Aborted motion (toss + catch)** | Recent upward window (0.6s) may miss the catch; apex and drop thresholds still pass | `recent_upward_fraction > 0.35` (`multi_serve.py:403`) |
| **Wall rebound** | Fast horizontal motion produces velocity magnitude peak; no directional filter on spike | `find_peaks` on `velocities` magnitude only (`multi_serve.py:362`) |
| **Camera shake** | Jitter produces frame-to-frame displacement; smoothed vertical velocity looks like toss; small drops exceed 120px threshold | `vert_velocities < -1` threshold (`multi_serve.py:380`), `drop_after_apex > 120` (`multi_serve.py:405`) |
| **Player arm/racket swing** | Large pixel displacement near camera; high contact velocity score; post-contact motion (follow-through) sustains velocity | `score = contact_vel` base (`multi_serve.py:454`) |
| **Ball rolling/bouncing on ground** | Sustained low velocity in post-contact window adds to score; no minimum speed requirement for post-contact | `np.mean(post_vels) * 2.0` bonus (`multi_serve.py:470`) |

---

*End of audit. No files modified.*
