"""Audio onset detection for tennis-ball impact times.

Uses librosa.onset.onset_detect on a 2-5 kHz bandpass extract of the video
audio to find sharp transients. Tennis-ball strikes are short, broadband
clicks with significant 2-5 kHz energy and stand out from background voice
or crowd noise.

Used by serve_attempts_v3 to:
  * Cross-validate visual contact frames (+score bonus when audio onset
    matches within `match_tolerance_sec`).
  * Optionally snap contact_frame to nearest audio onset (--snap-to-audio).

The module degrades gracefully when:
  * librosa is not importable (returns []),
  * ffmpeg is missing or audio extraction fails (returns []),
  * the video has no audio stream.
None of these are errors - audio is an opt-in cross-check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

import numpy as np


def _have_librosa() -> bool:
    try:
        import librosa  # noqa: F401

        return True
    except Exception:
        return False


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_audio_to_wav(
    video_path: str,
    sample_rate: int = 22050,
    band_low_hz: float = 2000.0,
    band_high_hz: float = 5000.0,
) -> Optional[str]:
    """Extract mono audio to a temp WAV with a 2-5 kHz bandpass.

    Returns the temp file path on success or None on any failure (no audio,
    ffmpeg missing, etc.). Caller is responsible for deleting the file.
    """
    if not _have_ffmpeg():
        return None

    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="serve_audio_")
    os.close(fd)

    # ffmpeg highpass+lowpass approximate a bandpass; cheap and good enough.
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-af",
        f"highpass=f={int(band_low_hz)},lowpass=f={int(band_high_hz)}",
        wav_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        try:
            os.remove(wav_path)
        except OSError:
            pass
        return None

    if (
        result.returncode != 0
        or not os.path.exists(wav_path)
        or os.path.getsize(wav_path) < 1024
    ):
        try:
            os.remove(wav_path)
        except OSError:
            pass
        return None
    return wav_path


def detect_onsets(
    video_path: str,
    sample_rate: int = 22050,
    delta: float = 0.07,
    wait_sec: float = 0.5,
    band_low_hz: float = 2000.0,
    band_high_hz: float = 5000.0,
) -> List[float]:
    """Return a list of audio-onset timestamps (seconds) detected in the video.

    Empty list on any failure (no librosa, no ffmpeg, no audio, etc.).

    Args:
        delta: peak picking threshold for librosa.onset.onset_detect.
            Higher -> fewer, stronger onsets only.
        wait_sec: minimum spacing between accepted onsets.
        band_low_hz, band_high_hz: bandpass limits applied during extraction.
    """
    if not _have_librosa():
        return []

    wav_path = extract_audio_to_wav(
        video_path,
        sample_rate=sample_rate,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
    )
    if wav_path is None:
        return []

    try:
        import librosa

        y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
        if y is None or len(y) == 0:
            return []

        hop_length = 512
        wait_frames = max(1, int(round(wait_sec * sr / hop_length)))

        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop_length,
            backtrack=False,
            delta=float(delta),
            wait=wait_frames,
            units="frames",
        )
        if onset_frames is None or len(onset_frames) == 0:
            return []
        times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        return [float(t) for t in np.asarray(times).tolist()]
    except Exception:
        return []
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def nearest_onset(
    onsets: List[float],
    target_time_sec: float,
    tolerance_sec: float = 0.10,
) -> Optional[float]:
    """Return the audio onset closest to target_time_sec within tolerance, or None."""
    if not onsets:
        return None
    arr = np.asarray(onsets, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(target_time_sec))))
    nearest = float(arr[idx])
    if abs(nearest - float(target_time_sec)) <= float(tolerance_sec):
        return nearest
    return None
