# Issues / Gotchas
- Do NOT modify existing `serve_analyzer/cli.py`, `serve_analyzer/analysis.py`, or `serve_attempts*` modules.
- Do NOT depend on real `videos/wall/*.MOV` in unit tests.
- Refused projection rows must still be present in CSV with null fields + warning code.
- Manual correction must NOT erase autonomous detection — both stored.
- CSV warning codes joined deterministically (e.g., `;`-separated, sorted).
