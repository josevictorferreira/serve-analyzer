"""Wall analysis service — orchestrates background analysis via _process_video()."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from serve_analyzer.wall_calibration import WallCalibration
from serve_analyzer.wall_serve import _process_video
from web.backend.paths import get_wall_output_dir


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
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run wall analysis pipeline and return normalized six-section result.

    Steps:
        1. Create output directory via :func:`get_wall_output_dir`.
        2. Call ``_process_video(video_path, calibration, output_dir)``.
        3. Read ``result.json`` from disk (do not trust ``_process_video`` return).
        4. Normalize artifact paths to relative browser URLs.
        5. Return the full six-section payload.

    Args:
        video_path: Absolute path to the staged wall video.
        calibration: A validated ``WallCalibration`` instance.
        video_id: The staged video identifier (used for output dir).
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
    return result
