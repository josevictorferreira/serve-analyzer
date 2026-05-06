"""Wall analysis service — orchestrates background analysis via _process_video()."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from serve_analyzer.wall_calibration import WallCalibration
from serve_analyzer.wall_serve import _process_video
from web.backend.paths import get_wall_output_dir
from web.backend.services.wall_review_clip_service import generate_impact_review_clip

logger = logging.getLogger(__name__)


def _normalize_artifact_paths(
    result: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    """Replace absolute filesystem paths in artifacts with relative browser URLs.

    Top-level artifact keys become ``/api/wall/artifacts/<filename>``.
    Nested plot paths become ``/api/wall/artifacts/plots/<filename>``.
    """
    artifacts = result.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return result

    normalized: Dict[str, Any] = {}
    for key, value in artifacts.items():
        if key == "annotated_video" and isinstance(value, str):
            fname = os.path.basename(value)
            normalized[key] = f"/api/wall/artifacts/{fname}"
        elif key == "plots" and isinstance(value, dict):
            plots: Dict[str, str] = {}
            for plot_key, plot_path in value.items():
                if isinstance(plot_path, str):
                    fname = os.path.basename(plot_path)
                    plots[plot_key] = f"/api/wall/artifacts/plots/{fname}"
            normalized[key] = plots
        else:
            normalized[key] = value

    result = dict(result)
    result["artifacts"] = normalized
    return result


def run_wall_analysis(
    video_path: str,
    calibration: WallCalibration,
    video_id: str,
    video_duration_sec: float,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run wall analysis pipeline and return normalized six-section result.

    Steps:
        1. Create output directory via :func:`get_wall_output_dir`.
        2. Call ``_process_video(video_path, calibration, output_dir)``.
        3. Read ``result.json`` from disk (do not trust ``_process_video`` return).
        4. Generate impact-centered review clip if ``impact_time_sec`` is present.
        5. Normalize artifact paths to relative browser URLs.
        6. Return the full six-section payload.

    Args:
        video_path: Absolute path to the staged wall video.
        calibration: A validated ``WallCalibration`` instance.
        video_id: The staged video identifier (used for output dir).
        video_duration_sec: Total duration of the video in seconds (from session metadata).
        on_progress: Optional callback receiving phase strings (``"artifacting"``).

    Returns:
        Dict with keys ``measured``, ``inferred``, ``assumed``,
        ``confidence``, ``warnings``, ``artifacts``.
    """
    output_dir = Path(get_wall_output_dir(video_id))
    output_dir.mkdir(parents=True, exist_ok=True)

    _process_video(
        video_path,
        calibration,
        output_dir,
        no_video=False,
        no_plots=False,
    )

    if on_progress is not None:
        on_progress("artifacting")

    result_path = output_dir / "result.json"
    if not result_path.is_file():
        raise RuntimeError(
            f"Analysis completed but result.json not found at {result_path}"
        )

    with open(result_path, "r", encoding="utf-8") as f:
        result: Dict[str, Any] = json.load(f)

    result = _normalize_artifact_paths(result, output_dir)

    # --- Impact-centered review clip (best-effort) ---
    measured = result.get("measured", {})
    impact_time_sec = measured.get("impact_time_sec")
    video_stem = measured.get("video")
    if impact_time_sec is not None and video_stem is not None:
        try:
            clip_info = generate_impact_review_clip(
                video_path,
                str(output_dir),
                impact_time_sec,
                video_duration_sec,
                video_stem,
            )
        except Exception as exc:
            logger.warning("Could not generate wall review clip: %s", exc)
            clip_info = None

        if clip_info is not None:
            clip_path, review_meta = clip_info
            fname = os.path.basename(clip_path)
            result["artifacts"] = dict(result.get("artifacts", {}))
            result["artifacts"]["review_clip"] = {
                **dict(review_meta),
                "url": f"/api/wall/artifacts/{fname}",
                "impact_frame": measured.get("impact_frame"),
            }

    return result
