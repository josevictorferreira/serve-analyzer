
## 2026-04-22 Toss geometry and floor-drive false-positive rejection

### Real-toss geometry filter (multi_serve.py detect_serve_events)
**Change**: After toss_start/apex computation, added minimum toss geometry gate:
- `toss_rise_px = max(0.0, positions[toss_start][1] - apex_position[1])`
- `toss_duration_frames = max(0, apex_frame - toss_start)`
- Reject when `toss_rise_px < 60` OR `toss_duration_frames < max(6, int(0.12 * fps))`
- Allow through if toss evidence is extremely strong (`upward_fraction > 0.65 and recent_upward_fraction > 0.65 and drop_after_apex > 140`)

**Rationale**: Hand-in-pocket / ball-takeout prep motions have minimal vertical rise and short duration. Real serves have clear toss arcs.

### Early post-contact downward-shape rejection (multi_serve.py detect_serve_events)
**Change**: Added floor-drive detection in gated_events loop:
- `early_post_downward_fraction = mean(vert_velocities[scoring_frame+1:scoring_frame+9] > 1.0)`
- `early_post_net_dy = positions[-1][1] - positions[0][1]` over same window
- Reject when ALL true: `rightward_fraction > 0.45` AND `early_post_downward_fraction > 0.70` AND `early_post_net_dy > 45` AND `drop_after_apex < 80`

**Rationale**: Ball hit into floor to the right shows immediate sharp downward trajectory with weak toss geometry. True serves descend post-contact but not as an immediate drive.

### Metadata passed through to candidates (serve_attempts.py)
**Change**: `detect_serve_candidates` now includes in candidate dict:
- `toss_rise_px`
- `toss_duration_frames`
- `early_post_downward_fraction`
- `early_post_net_dy`

### Selector safety net (serve_attempts.py select_serves)
**Change**: Light backup filter before ranking:
- Reject when `toss_rise_px < 60` AND `toss_duration_frames < 6` AND `early_post_downward_fraction > 0.70` AND `early_post_net_dy > 45`
- Does NOT redesign ranking; only adds small safety net for merged candidates that slip through detector gating

### Clip window expanded to 4.0s (clip_service.py)
**Change**: Window changed from `[contact - 2.0, contact + 1.5]` (3.5s) to `[contact - 2.25, contact + 1.75]` (4.0s).
- Slightly earlier bias to better catch contact moment given imperfect timing anchor.

### Tests added
- `TestTossGeometryAndFloorDriveRejection` in `tests/test_serve_attempts.py`:
  - `test_prep_pocket_false_positive_rejected`: weak toss + short duration + strong downward → rejected
  - `test_floor_hit_false_positive_rejected`: rightward + strong immediate downward → rejected
  - `test_real_serve_with_moderate_descent_not_over_pruned`: strong toss + moderate descent → survives
- `tests/test_web_clip_service.py` updated for 4.0s window

### Verification
- `python -m unittest discover -s tests -v` → 134 tests OK
