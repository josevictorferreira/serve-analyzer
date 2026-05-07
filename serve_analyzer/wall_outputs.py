"""Wall-serve analysis output contracts.

Builds the JSON payload (sections measured/inferred/assumed/confidence/warnings/artifacts),
serializes per-serve CSV rows using the LOCKED ``CSV_COLUMNS`` from ``wall_calibration.py``,
and defines deterministic plot artifact names.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from serve_analyzer.wall_calibration import CSV_COLUMNS
from serve_analyzer.wall_calibration import compute_wall_homography
from serve_analyzer.wall_calibration import pixel_to_wall


# ---------------------------------------------------------------------------
# Deterministic plot artifact names
# ---------------------------------------------------------------------------

PLOT_FILENAMES: Dict[str, str] = {
    "speed": "{video_stem}_serve{idx:02d}_speed.png",
    "trajectory": "{video_stem}_serve{idx:02d}_trajectory.png",
    "wall_impact": "{video_stem}_serve{idx:02d}_wall_impact.png",
}
"""Deterministic filename templates for plot artifacts."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class WallAnalysisResult:
    """Container for a single wall-serve analysis run.

    Attributes:
        measured: Directly observed quantities (e.g. impact_time, wall coords).
        inferred: Derived quantities (e.g. speed, landing position).
        assumed: Defaults or fallback values used during analysis.
        confidence: Numeric confidence scores per serve or per field.
        warnings: List of warning code strings.
        artifacts: Dict mapping artifact type to file path(s).
    """

    measured: Dict[str, Any] = field(default_factory=dict)
    inferred: Dict[str, Any] = field(default_factory=dict)
    assumed: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict with exactly the 6 required sections.

        Returns:
            Dict with keys ``measured``, ``inferred``, ``assumed``, ``confidence``,
            ``warnings``, ``artifacts`` — no extras.
        """
        return {
            "measured": self.measured,
            "inferred": self.inferred,
            "assumed": self.assumed,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "artifacts": self.artifacts,
        }


# ---------------------------------------------------------------------------
# Confidence score computation
# ---------------------------------------------------------------------------


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp *value* to the inclusive range [*min_val*, *max_val*]."""
    return max(min_val, min(value, max_val))


def compute_confidence_score(
    speed_m_s: float | None,
    uncertainty_m_s: float,
    degraded_intrinsics: bool,
    has_refusal_warning: bool,
) -> float:
    """Compute an aggregate confidence score in the range [0, 1].

    Formula (documented for reproducibility)::

        base = 1.0
        if speed_m_s is not None and speed_m_s > 0:
            base -= clamp(uncertainty_m_s / speed_m_s, 0, 1) * 0.5
        if degraded_intrinsics:
            base -= 0.3
        if has_refusal_warning:
            base -= 0.2
        score = clamp(base, 0, 1)

    Args:
        speed_m_s: Estimated speed (m/s), or ``None`` when refused.
        uncertainty_m_s: Combined speed uncertainty (m/s).
        degraded_intrinsics: Whether ``degraded_intrinsics`` warning is present.
        has_refusal_warning: Whether any refusal warning (e.g.
            ``projection_refused``) is present.

    Returns:
        Confidence score clamped to [0, 1].
    """
    base = 1.0
    if speed_m_s is not None and speed_m_s > 0:
        base -= _clamp(uncertainty_m_s / speed_m_s, 0.0, 1.0) * 0.5
    if degraded_intrinsics:
        base -= 0.3
    if has_refusal_warning:
        base -= 0.2
    return _clamp(base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def assemble_wall_analysis_result(
    video_path: str | Path,
    calibration: Any,
    impact_result: Any,
    speed_result: Any,
    projection_result: Any,
    *,
    artifact_paths: Dict[str, Any] | None = None,
    per_impact_results: List[Dict[str, Any]] | None = None,
) -> WallAnalysisResult:
    """Compose a ``WallAnalysisResult`` from the 6 pipeline stages.

    This function wires the result contracts into the orchestration pipeline.
    It populates all 6 sections (measured, inferred, assumed, confidence,
    warnings, artifacts) from the constituent result objects.

    Args:
        video_path: Path to the source video (used for stem naming).
        calibration: A ``WallCalibration`` instance (used for intrinsics checks).
        impact_result: A ``WallImpactResult`` with impact frame/pixel data.
        speed_result: A ``PreWallSpeedResult`` with speed estimates.
        projection_result: A ``CourtProjectionResult`` with landing projection.
        artifact_paths: Optional dict like
            ``{"annotated_video": Path | None, "plots": {"speed": Path | None, ...}}``.
            ``None`` values are preserved.
        per_impact_results: Optional list of per-impact dicts, each containing
            ``impact_index``, ``impact_result``, ``speed_result``,
            ``projection_result``.  When provided, a ``measured.impacts``
            array is added to the output.

    Returns:
        A fully populated ``WallAnalysisResult``.
    """
    video_path = Path(video_path)
    video_stem = video_path.stem

    # --- Measured ---
    fps = getattr(speed_result, "metadata", {}).get("fps", None)
    impact_frame = impact_result.impact_frame
    impact_time_sec = None
    if impact_frame is not None and fps is not None and fps > 0:
        impact_time_sec = impact_frame / fps

    measured: Dict[str, Any] = {
        "video": video_stem,
        "serve_index": 0,
        "impact_time_sec": impact_time_sec,
        "impact_frame": impact_frame,
        "autonomous_frame": impact_result.autonomous_frame,
        "autonomous_pixel": impact_result.autonomous_pixel,
        "impact_pixel": impact_result.impact_pixel,
        "wall_x_m": None,
        "wall_y_m": None,
        "calibration_reprojection_rms_px": None,
        "raw_track_samples": len(impact_result.candidate_track),
    }

    # Compute wall-meter coordinates from impact_pixel via calibration homography.
    wall_x_m: float | None = None
    wall_y_m: float | None = None
    H_inv: np.ndarray | None = None
    if (
        impact_result.impact_pixel is not None
        and hasattr(calibration, "wall_reference_points")
        and len(calibration.wall_reference_points) >= 4
    ):
        try:
            image_pts = np.array(
                [p.pixel for p in calibration.wall_reference_points], dtype=np.float64
            )
            world_pts = np.array(
                [p.wall_m for p in calibration.wall_reference_points], dtype=np.float64
            )
            intrinsics = getattr(calibration, "intrinsics", None)
            H, _ = compute_wall_homography(image_pts, world_pts, intrinsics=intrinsics)
            H_inv = np.linalg.inv(H)
            wall_xy = pixel_to_wall(
                H_inv, np.array([list(impact_result.impact_pixel)], dtype=np.float64)
            )
            wall_x_m = float(wall_xy[0, 0])
            wall_y_m = float(wall_xy[0, 1])
        except Exception:
            pass  # Leave None; degraded path
    measured["wall_x_m"] = wall_x_m
    measured["wall_y_m"] = wall_y_m

    # Pull reprojection RMS from speed_result metadata if available.
    homography_residuals = speed_result.metadata.get("homography_residuals", {})
    if homography_residuals:
        measured["calibration_reprojection_rms_px"] = homography_residuals.get(
            "reprojection_rms_px"
        )

    # --- Multi-impact array ---
    measured["impact_count"] = 1
    measured["primary_impact_index"] = 0

    if per_impact_results is not None and len(per_impact_results) > 1:
        measured["impact_count"] = len(per_impact_results)

        impacts_list: List[Dict[str, Any]] = []
        for pi in per_impact_results:
            pi_ir = pi["impact_result"]
            pi_sr = pi["speed_result"]
            pi_pr = pi["projection_result"]
            pi_idx = pi["impact_index"]

            # Compute wall_x_m/wall_y_m for this impact
            pi_wall_x_m: float | None = None
            pi_wall_y_m: float | None = None
            if (
                pi_ir.impact_pixel is not None
                and H_inv is not None
            ):
                try:
                    pi_wall_xy = pixel_to_wall(
                        H_inv,
                        np.array([list(pi_ir.impact_pixel)], dtype=np.float64),
                    )
                    pi_wall_x_m = float(pi_wall_xy[0, 0])
                    pi_wall_y_m = float(pi_wall_xy[0, 1])
                except Exception:
                    pass

            pi_impact_frame = pi_ir.impact_frame
            pi_impact_time_sec = None
            if pi_impact_frame is not None and fps is not None and fps > 0:
                pi_impact_time_sec = pi_impact_frame / fps

            impact_entry: Dict[str, Any] = {
                "impact_index": pi_idx,
                "measured": {
                    "impact_frame": pi_impact_frame,
                    "impact_time_sec": pi_impact_time_sec,
                    "impact_pixel": pi_ir.impact_pixel,
                    "wall_x_m": pi_wall_x_m,
                    "wall_y_m": pi_wall_y_m,
                    "raw_track_samples": len(pi_ir.candidate_track),
                },
                "inferred": {
                    "speed_m_s": pi_sr.speed_m_s,
                    "speed_km_h": pi_sr.speed_km_h,
                    "speed_mph": pi_sr.speed_mph,
                    "landing_x_m": pi_pr.landing_x_m,
                    "landing_z_m": pi_pr.landing_z_m,
                    "in_service_box": pi_pr.in_service_box,
                    "service_box_side": pi_pr.service_box_side,
                },
                "confidence": {
                    "impact_confidence": pi_ir.confidence,
                    "speed_uncertainty_m_s": pi_sr.uncertainty_m_s,
                },
                "warnings": sorted(
                    set(pi_ir.warnings + pi_sr.warnings + pi_pr.warnings)
                ),
            }
            impacts_list.append(impact_entry)

        measured["impacts"] = impacts_list

    # --- Inferred ---
    inferred: Dict[str, Any] = {
        "speed_m_s": speed_result.speed_m_s,
        "speed_km_h": speed_result.speed_km_h,
        "speed_mph": speed_result.speed_mph,
        "speed_uncertainty_m_s": speed_result.uncertainty_m_s,
        "landing_x_m": projection_result.landing_x_m,
        "landing_z_m": projection_result.landing_z_m,
        "in_service_box": projection_result.in_service_box,
        "service_box_side": projection_result.service_box_side,
        "sensitivities": projection_result.uncertainty,
    }

    # --- Assumed ---
    assumed = dict(projection_result.assumptions)

    # --- Warnings (deduplicated, sorted) ---
    all_warnings = sorted(
        set(impact_result.warnings + speed_result.warnings + projection_result.warnings)
    )

    # --- Confidence ---
    degraded_intrinsics = (
        calibration.intrinsics is None
        or calibration.intrinsics.source in ("none", "approx_exif")
    )
    has_refusal = any(
        w in all_warnings for w in ("projection_refused", "insufficient_track")
    )
    confidence_score = compute_confidence_score(
        speed_m_s=speed_result.speed_m_s,
        uncertainty_m_s=speed_result.uncertainty_m_s,
        degraded_intrinsics=degraded_intrinsics,
        has_refusal_warning=has_refusal,
    )

    # Inject degraded_intrinsics warning when intrinsics are missing or low-quality.
    if degraded_intrinsics and "degraded_intrinsics" not in all_warnings:
        all_warnings = sorted(set(all_warnings) | {"degraded_intrinsics"})

    confidence: Dict[str, Any] = {
        "aggregate_score": confidence_score,
        "impact_confidence": impact_result.confidence,
        "speed_uncertainty_m_s": speed_result.uncertainty_m_s,
        "samples_used": speed_result.samples_used,
        "degraded_intrinsics": degraded_intrinsics,
    }

    # --- Artifacts ---
    artifacts: Dict[str, Any] = {"annotated_video": None, "plots": {}}
    if artifact_paths is not None:
        artifacts["annotated_video"] = artifact_paths.get("annotated_video")
        artifacts["plots"] = dict(artifact_paths.get("plots", {}))

    return WallAnalysisResult(
        measured=measured,
        inferred=inferred,
        assumed=assumed,
        confidence=confidence,
        warnings=all_warnings,
        artifacts=artifacts,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

_JSON_SECTIONS = frozenset(
    {"measured", "inferred", "assumed", "confidence", "warnings", "artifacts"}
)


def to_json(result: WallAnalysisResult) -> Dict[str, Any]:
    """Return a JSON-serializable dict with exactly the 6 required sections.

    Args:
        result: A ``WallAnalysisResult`` instance.

    Returns:
        Dict with keys ``measured``, ``inferred``, ``assumed``, ``confidence``,
        ``warnings``, ``artifacts`` — no extras.
    """
    return result.to_json_dict()


# ---------------------------------------------------------------------------
# CSV serialization
# ---------------------------------------------------------------------------

# Build a lookup so we know the column index for each field.
_CSV_COL_INDEX: Dict[str, int] = {col: idx for idx, col in enumerate(CSV_COLUMNS)}


def _flatten_warning_codes(codes: List[str]) -> str:
    """Join warning codes deterministically (sorted, semicolon-delimited).

    The result is a plain string — no JSON braces, no nested structures.

    Args:
        codes: Unordered list of warning code strings.

    Returns:
        A semicolon-joined string of codes in sorted order.
    """
    return ";".join(sorted(set(codes)))


def serve_to_csv_row(serve: Dict[str, Any]) -> Tuple[Any, ...]:
    """Flatten one serve dict into a tuple matching ``CSV_COLUMNS`` order.

    The ``serve`` dict is expected to contain at minimum the keys that map to
    ``CSV_COLUMNS`` (minus ``warning_codes`` which is handled separately via
    ``serve["warnings"]`` or ``serve["warning_codes"]``).  Missing keys are
    filled with ``None``.

    Args:
        serve: Dict with serve-level data.  Warning codes may be under
            ``"warnings"`` (list) or ``"warning_codes"`` (list or str).

    Returns:
        Tuple of values in ``CSV_COLUMNS`` order.
    """
    row: List[Any] = [None] * len(CSV_COLUMNS)

    for col_name in CSV_COLUMNS:
        if col_name == "impact_index":
            row[_CSV_COL_INDEX[col_name]] = serve.get(col_name, 0)
            continue
        if col_name == "warning_codes":
            # Pull from "warnings" or "warning_codes", then flatten.
            raw_codes = serve.get("warnings", serve.get("warning_codes", []))
            if isinstance(raw_codes, str):
                raw_codes = [raw_codes] if raw_codes else []
            row[_CSV_COL_INDEX[col_name]] = _flatten_warning_codes(raw_codes)
            continue

        row[_CSV_COL_INDEX[col_name]] = serve.get(col_name)

    return tuple(row)


def write_csv(path: str | Path, rows: List[Tuple[Any, ...]]) -> None:
    """Write CSV with header from ``CSV_COLUMNS`` followed by data rows.

    Args:
        path: Destination file path.
        rows: List of tuples, each matching ``CSV_COLUMNS`` length and order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
