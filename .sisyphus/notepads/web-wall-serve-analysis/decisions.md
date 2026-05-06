# Decisions — web-wall-serve-analysis

## 2026-05-06 Oracle + Metis Decisions
- Backend-staged upload before calibration (not frontend object URL only).
- Direct video serving + frontend canvas overlay for calibration.
- Single global CPU job across serve and wall analysis; 409 if either active.
- Job reset preserves calibration; calibration reset clears calibration.
- Artifact URLs are relative browser URLs; frontend never consumes absolute paths.
- Nested artifact route `GET /api/wall/artifacts/{artifact_path:path}` with path traversal protection.
- Impact-centered review clip using ffmpeg/clip_service pattern around impact_time_sec.
- Manual correction UI out of MVP; show autonomous vs final impact with future note.
- Adapter reads result.json/result.csv after _process_video() completes.
