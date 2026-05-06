# Issues — web-wall-serve-analysis

(No issues yet)

## F3 real manual QA - 2026-05-06
- Backend API synthetic wall flow passed end-to-end: upload, calibration, analyze, done job, result.json/result.csv, annotated/review MP4, and plot PNG artifact endpoints returned 200.
- Frontend wall workflow is blocked after upload/calibration: it shows the Configure placeholder with no visible control to continue to Analyze/Results, so a real UI upload → calibrate → analyze → view results flow cannot be completed.
- Screenshots saved under /tmp/nix-shell.weurY0/opencode/: wall-f3-home.png, wall-f3-wall-upload.png, wall-f3-calibrate-blocked.png.

## F3 rerun manual QA - 2026-05-06
- Full wall UI flow advanced Upload → Calibrate → Analyze → Results after saving calibration.
- Results dashboard rendered metrics, court projection, warnings/confidence, review clip, and plot images.
- Blocker remains: annotated video element failed in Chrome with `DEMUXER_ERROR_NO_SUPPORTED_STREAMS` even though `/api/wall/artifacts/..._annotated.mp4` returned 206 `video/mp4`; review clip played successfully. Screenshot evidence saved under `/tmp/nix-shell.weurY0/opencode/wall-f3-rerun-66591d8/`.
