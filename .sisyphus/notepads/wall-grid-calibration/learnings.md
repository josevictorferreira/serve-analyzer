# Grid Calibration Learnings

## 2026-05-06: Project Context
- This project uses `crypto.randomUUID()` replacement (plain JS uuid helper) for non-secure HTTP contexts
- Backend API contract: `WallCalibrationSetup` requires `wall_reference_points: [{name, pixel: [x,y], wall_m: [x,y]}]`
- Exactly 4 points needed for homography (cv2.getPerspectiveTransform)
- Existing components: `wall-calibration-canvas.tsx` (click-to-place), `wall-assumptions-form.tsx` (manual coord entry)
- Parent workflow: `wall-workflow.tsx` orchestrates both components in calibrate step
- Tests: vitest, located next to components (`.test.tsx`)

## Wall Grid Calibration Component (2026-05-06)
- DLT homography for 4 points: solve 8x8 linear system via Gaussian elimination with partial pivoting, h8=1 normalization
- Coordinate mapping: display pixels → original video pixels using `(displayX / displayWidth) * videoWidth`
- Touch/mouse drag: use window-level event listeners (not canvas-level) for mouseup/touchend to capture releases outside canvas
- Touch events need `e.preventDefault()` to prevent scrolling on mobile
- Canvas `touchAction: 'none'` inline style prevents browser gesture conflicts
- Default corner positions: initialized on video load with 15% margin from edges
- Grid lines every 0.5m: horizontal (y=0..gridHeight) and vertical (x=-halfW..halfW) projected through homography
- Height reference markers: 1.0m (chair/yellow), 2.45m (hook/red), 2.80m (contact/green) with dashed lines
- API payload preserves exact same format: `wall_reference_points` array with `{name, pixel: [x,y], wall_m: [x,y]}`
- Loading existing calibration: derives grid width/height from min/max of wall_m values
