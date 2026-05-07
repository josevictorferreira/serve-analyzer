# Grid Calibration Design Decisions

## Oracle Consultation (ses_1ffe4a22cffeGR6ABkGxzS6oD4)
- Default grid: 2.0m wide × 3.0m tall
- Bottom edge = floor/wall line (y=0)
- Origin at bottom-center: bottom-left=(-1,0), bottom-right=(1,0), top-left=(-1,3), top-right=(1,3)
- 4 draggable corner handles → 4 pixel↔world correspondences
- Grid lines rendered through homography (not bilinear interpolation)
- Height markers at 1.0m (chair), 2.45m (hook), 2.80m (contact)
- Both grid width AND grid height must be known
- MVP: 4 draggable corners + grid width/height + contact height
- Must work on mobile (non-secure HTTP context, no crypto.randomUUID)

## Implementation Approach
- Single new component `wall-grid-calibration.tsx` replaces BOTH `wall-calibration-canvas.tsx` and `wall-assumptions-form.tsx`
- Backend unchanged (same API contract, same 4-point homography)
- Grid lines drawn using homography: compute H from world→pixel, then project grid line endpoints
