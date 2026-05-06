# Decisions
- Research-grade projection model.
- Hybrid calibration: reusable setup + per-video override.
- Manual contact height required (no default).
- Optional intrinsics: `none|approx_exif|opencv_chessboard|opencv_charuco`.
- Regulation tennis court defaults.
- Autonomous impact detection + manual correction overlay.
- Outputs: JSON + CSV + plots + annotated MP4 (mandatory).
- Gravity-only physics MVP; drag/spin out of scope.
- Tests use synthetic videos only; real `videos/wall/*.MOV` for documented user runs only.

## 2026-05-05 — Task 2

- Homography fitting uses OpenCV RANSAC with a 3.0 px reprojection threshold, matching a conservative manual-calibration tolerance while preserving deterministic synthetic fixtures.
- Approximate EXIF intrinsics are allowed for undistortion but explicitly mark `degraded_intrinsics` in the residuals dict; chessboard/ChArUco sources do not set that flag.
- Collinearity detection uses centered matrix rank rather than first-two-point cross products so duplicate leading points do not hide non-degenerate later references.
