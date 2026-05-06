# Learnings — web-wall-serve-analysis

## 2026-05-06 Session Start
- Wall pipeline fully implemented: `serve_analyzer/wall_*.py` with 67 wall tests, 248 full suite.
- `_process_video()` returns `serve_row` only; adapter must read `result.json`/`result.csv` from disk.
- Artifact layout: `{video_stem}_annotated.mp4` and `plots/*.png` under output dir.
- Frontend is Vite React + Tailwind v4 + shadcn/ui; backend is FastAPI on port 8000.
- Entry: `python -m web.backend` (not `app.py`).
- Normal serve web plan complete at `.sisyphus/plans/web-serve-analysis-app.md`.
- AGENTS.md lessons: always verify output contracts, don't subclass cv2.VideoCapture, enforce artifact naming in tests.

## 2026-05-06 Task 1 Wall Web Session APIs
- Added wall staging backend with in-memory session state, OpenCV metadata probing, wall temp/output helpers, and FastAPI wall video/metadata/reset/job endpoints.
- Global busy guard now checks normal serve and wall job states with shared state lock; `/api/analyze` and wall upload mutually reject active jobs.
- Verification: targeted `test_wall_web_session.py` passed; full `python -m unittest discover -s tests -v` passed with 257 tests (1 skipped marker for recursive full-suite check).
