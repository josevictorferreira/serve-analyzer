# Audio-Based Contact Onset Detection

**Status:** Proposed
**Backlog ref:** `improvements.md` #13 (with #14 disagreement-rejection as a follow-up)
**Estimated effort:** 1–2 days (audio extraction plumbing + onset detection + alignment + tests)
**Risk:** Medium — adds a new dependency (`librosa` + `ffmpeg-python` or `soundfile`); audio may be missing from some inputs.

---

## 1. Problem

Contact-frame timing today depends entirely on visual cues:
- `_refine_contact_frame` (`serve_analyzer/multi_serve.py:357-408`) uses horizontal-acceleration spike.
- `extract_motion_cues` (`serve_attempts_v2.py:194`) uses frame differencing in candidate windows.
- Direction reversal (`vy` flip) is the primary trigger.

All three are noisy and frame-quantized: at 60 fps the smallest resolvable error is ~16.7 ms; at 30 fps it's ~33 ms. A 1-frame error at 60 fps with a real serve speed of 200 km/h means the post-contact velocity window starts on a wrong frame, and the peak (which lasts 2–4 frames) can be missed entirely.

Tennis impact produces a sharp acoustic transient (~2–4 kHz, ~5–15 ms duration). Onset-detection libraries (`librosa.onset.onset_detect`) routinely localize this to ±5 ms. Combining audio onset with visual contact gives sub-frame timing accuracy and an *independent* signal that resolves ambiguous direction-reversal frames (e.g. when the ball briefly reverses during a let serve net-clip).

## 2. Goal & Success Criteria

Add `serve_analyzer/audio_contact.py` providing `detect_contact_onsets(video_path) → List[float]` (seconds). Hook it into `serve_attempts.py`'s candidate-scoring loop as an additional feature; do **not** replace visual contact detection (stays as the primary; audio is a refiner).

**Verifiable success:**

1. **Onset detection itself:** On a hand-labeled test video with N known contact times, `mean(|t_audio - t_truth|) < 30 ms` and recall > 0.9.
2. **Refinement effect:** When audio onset is within ±100 ms of a visual contact candidate, snap visual contact to the nearest frame containing the audio onset. Verify post-contact peak velocity stays equal or improves on the test video.
3. **Graceful degradation:** Videos without audio (silent MP4) run without error — audio detector returns `[]`, candidate scoring proceeds unchanged.
4. **No regression:** All existing tests pass.
5. **New diagnostic field in JSON:** `audio_contact_time_sec` (or `null`) per `selected_serves` entry.

## 3. Scope

### In scope

- New module `serve_analyzer/audio_contact.py` with onset extraction.
- ffmpeg-based audio extraction to a temp WAV (no Python audio decoding — ffmpeg is already a runtime dep for clip extraction in `web/backend/services/clip_service.py`).
- Onset detection via `librosa.onset.onset_detect` with tennis-tuned parameters.
- Integration point in `serve_attempts.py` candidate scoring.
- New JSON field in detector output.
- Test fixtures: 2–3 short clips with hand-labeled contact times.

### Out of scope (explicit)

- Audio-video disagreement *rejection* (backlog #14; this plan only adds the signal, doesn't reject on it).
- Real-time / streaming onset detection.
- Crowd-noise / commentary suppression beyond what librosa's onset envelope provides.
- Adding audio to `compare_detectors.py` (separate utility script work).

## 4. Design

### 4.1 Audio extraction

`librosa.load` can decode video audio directly via `soundfile`/`audioread`, but support is patchy on `.mov` files. Cleaner: shell out to ffmpeg:

```python
def _extract_audio_to_wav(video_path: str, out_wav: str, sr: int = 22050) -> bool:
    """Extract mono audio to WAV. Returns False if no audio stream."""
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr),
        out_wav,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and Path(out_wav).stat().st_size > 0
```

Uses `tempfile.NamedTemporaryFile(suffix=".wav")`. Cleaned up in a `finally` block.

### 4.2 Onset detection

```python
def detect_contact_onsets(
    video_path: str,
    min_separation_sec: float = 1.0,
    onset_threshold_db: float = -20.0,
) -> List[float]:
    """
    Detect tennis-impact-like onsets. Returns sorted onset times in seconds.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        if not _extract_audio_to_wav(video_path, wav_path):
            return []
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        # Bandpass to 2-4 kHz where racket impact dominates
        y_band = _bandpass(y, sr, low=2000.0, high=5000.0)
        onset_frames = librosa.onset.onset_detect(
            y=y_band, sr=sr,
            backtrack=True,
            units="frames",
            wait=int(min_separation_sec * sr / 512),  # hop_length default
            pre_avg=10, post_avg=10, pre_max=10, post_max=10,
            delta=0.07,
        )
        return list(librosa.frames_to_time(onset_frames, sr=sr).tolist())
    finally:
        Path(wav_path).unlink(missing_ok=True)
```

Notes:
- Bandpass via `scipy.signal.butter` (already a dep).
- `wait` param enforces minimum separation between onsets (1 s default — serves never come faster than that).
- `backtrack=True` snaps to onset start (more accurate timing than peak).

### 4.3 Tuning

Key parameters and their tennis-specific defaults:

| Param | Default | Rationale |
|---|---|---|
| Bandpass low | 2000 Hz | Below this is mostly speech/crowd |
| Bandpass high | 5000 Hz | Above this is hiss/wind noise |
| `delta` | 0.07 | librosa's relative onset strength threshold; tuned for impact transients |
| `wait` | ~1 s | Minimum inter-serve interval |

All exposed as kwargs.

### 4.4 Integration into candidate scoring

In `serve_attempts.py`, after candidate detection but before `select_serves`:

```python
audio_onsets = detect_contact_onsets(video_path)  # may be empty
for cand in candidates:
    visual_t = cand["contact_time_sec"]
    nearest = _find_nearest(audio_onsets, visual_t, max_delta=0.10)  # 100 ms
    cand["audio_contact_time_sec"] = nearest  # float or None
    if nearest is not None:
        cand["score"] += 50.0  # tunable; small bonus for confirmed audio
```

The 100 ms window is intentional: visual frame error at 60 fps is up to ±16 ms; we allow 6× that for robustness, since false-positive audio (crowd noise) typically falls outside any visual candidate's window anyway.

**Important:** scoring bonus is small (+50 vs the existing 100s-of-points scale in `_detect_broad_trajectory_events`). This plan adds a *signal*, not a *gate*. Promotion to a gate is backlog #14.

### 4.5 Snapping (optional, behind a flag)

When `--snap-to-audio` is passed:
- For each selected serve with a non-None `audio_contact_time_sec` within 100 ms, replace `contact_time_sec` and recompute `contact_frame = round(audio_t * fps)`.
- Recompute `post_contact_max_kmh` using the snapped frame.

Default: off — change to numbers must be opt-in for this PR.

### 4.6 Dependency management

`librosa` is not in the current Nix flake. Two options:

**Option A — Add to flake.nix:**
- Pure-Python, available in nixpkgs as `python3Packages.librosa`.
- Pulls in `numba`, `scipy`, `soundfile` — large transitive closure.

**Option B — Pip install into `.venv`:**
- AGENTS.md SESSION LEARNINGS notes: "Pragmatic fallback: `pip install <pkg>` into `.venv`."
- Faster iteration; can move to flake later.

**Decision: Option A** — `librosa` is a real dependency, not a one-off. Document the addition in PR. Verify with `nix develop --command python -c 'import librosa'`.

If nix fails (per AGENTS.md learning about stale derivations), fall back to Option B and document.

## 5. Implementation Steps

### Phase 1 — Module + unit tests with synthetic audio

1. Create `serve_analyzer/audio_contact.py` with:
   - `_extract_audio_to_wav`
   - `_bandpass`
   - `detect_contact_onsets`
2. Create `tests/test_audio_contact.py`:
   - `test_synthetic_impulse_train_detected`: synthesize WAV with 3 click-like impulses at known times → assert all three returned within ±10 ms.
   - `test_no_audio_returns_empty`: feed a silent WAV → returns `[]`.
   - `test_missing_audio_stream_returns_empty`: feed an MP4 with no audio (use ffmpeg `-an` to construct fixture once) → returns `[]`.
   - `test_minimum_separation_enforced`: synthesize 2 impulses 0.5 s apart with `min_separation_sec=1.0` → only 1 returned.
3. Add `librosa` to `flake.nix` (Option A).

**Verify:** `python -m unittest tests.test_audio_contact -v` — 4 passing.

### Phase 2 — Integration into `serve_attempts.py`

1. Add `detect_contact_onsets` call in `serve_attempts.detect_serve_candidates` (or wherever scoring happens — locate via grep before editing).
2. Add `audio_contact_time_sec` field to candidate dict.
3. Apply +50 score bonus when matched.
4. Add to selected-serves JSON output.
5. CLI flag `--no-audio` to disable (default: enabled if audio present).

**Verify:**
- `python -m serve_analyzer.serve_attempts video.mov --output out.json` — runs, JSON has `audio_contact_time_sec` field.
- All existing tests pass.

### Phase 3 — Hand-labeled fixture + accuracy verification

1. Pick 1 short serve clip (5–10 s) from test videos.
2. Hand-label contact times in a `tests/fixtures/audio_truth.json`:
   ```json
   {"video": "video.mov", "contact_times_sec": [2.34, 4.78]}
   ```
3. Add `tests/test_audio_contact_real.py`:
   - Loads fixture, runs `detect_contact_onsets`.
   - Asserts mean abs error < 30 ms, recall ≥ 0.9 (configurable; tighten later).
4. Mark this test `@unittest.skipUnless(Path('video.mov').exists(), ...)` so CI without the fixture skips.

**Verify:** `python -m unittest tests.test_audio_contact_real -v` passes locally.

### Phase 4 — Optional snapping flag

1. Add `--snap-to-audio` to `serve_attempts.py` CLI parser.
2. When enabled, apply snap logic from §4.5.
3. Document in `--help` and README.

**Verify:** Run with and without flag; confirm `contact_frame` shifts only when flag is set and audio is within 100 ms.

### Phase 5 — Docs

1. AGENTS.md SESSION LEARNINGS:
   ```
   ### Audio onset detection in tennis videos
   **Lesson:** Librosa onset_detect with bandpass 2-5 kHz, delta=0.07, wait=1s reliably finds racket-impact transients within ±10 ms. Crowd noise rarely lies within ±100 ms of a visual contact candidate.
   **Context:** Used as a non-gating signal in serve_attempts.py scoring; snap-to-audio is opt-in via --snap-to-audio.
   **Verify:** tests/test_audio_contact_real.py passes mean_abs_error < 30 ms on hand-labeled fixture.
   ```
2. Update `improvements.md` #13 status with plan path.
3. Update README CLI section.

## 6. Test Plan

| Test | Type | Verifies |
|---|---|---|
| `test_synthetic_impulse_train_detected` | unit | Onset accuracy on synthetic |
| `test_no_audio_returns_empty` | unit | Silent input handled |
| `test_missing_audio_stream_returns_empty` | unit | No audio stream handled |
| `test_minimum_separation_enforced` | unit | `wait` parameter respected |
| `test_audio_contact_real` | integration (video-gated) | Real-video accuracy |
| Existing `tests/test_serve_attempts*.py` | regression | Integration didn't break anything |

## 7. Rollback

- Remove `audio_contact_time_sec` from JSON schema documentation.
- Comment out the `detect_contact_onsets` call in `serve_attempts.py` (remains importable but unused).
- Delete `audio_contact.py` and tests if removing entirely.
- Revert `flake.nix` librosa addition.

Audio addition is fully isolated; no existing behavior changes unless `--snap-to-audio` is set.

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `librosa` adds heavy transitive deps to Nix flake | High | Accept it (real dep); document size in PR |
| Crowd noise in pro-match videos triggers false onsets | Medium | 100 ms matching window keeps noise from affecting candidates without near-time visual evidence; bandpass 2-5 kHz suppresses speech/crowd |
| Phone videos with poor mic quality miss real impacts | Medium | `delta` tunable; recall is the metric to monitor; opt-in `--snap-to-audio` means missed audio doesn't degrade output |
| ffmpeg not in PATH | Low | AGENTS.md notes ffmpeg is already a runtime dep; check at module import; raise actionable error |
| Audio extraction temp files leak on crash | Low | `finally: Path.unlink(missing_ok=True)` |

## 9. Open Questions

1. Should the +50 score bonus be configurable via CLI? **Proposed: not initially** — keep tuning in code; expose if multiple users tune it.
2. Should we also detect *non-tennis* audio events (footstep, ball bounce) and use them to *demote* candidates? **Proposed: out of scope** — that's #14.
3. Should `audio_contact_time_sec` be stored on every candidate or only `selected_serves`? **Proposed: every candidate** — useful for offline tuning of `select_serves`.

## 10. Acceptance Checklist

- [ ] Phase 1 module + 4 unit tests (with `librosa` in flake)
- [ ] Phase 2 integration; JSON has `audio_contact_time_sec`
- [ ] Phase 3 hand-labeled fixture; real-video test passes
- [ ] Phase 4 `--snap-to-audio` flag implemented (optional, can ship later)
- [ ] Phase 5 docs updated
- [ ] No regression on existing `tests/test_serve_attempts*.py`
