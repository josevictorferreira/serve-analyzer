"""
Wall/court coordinate systems, metadata schema, and homography calibration
for wall-serve impact analysis.

Coordinate frames
-----------------
    Wall frame:
    Origin at the floor/wall intersection directly under the center reference point.
    x_m  along the wall (camera-right positive after undistort).
    y_m  up (vertical).
    z_m  away from the wall toward the server.

    Court frame:
    Same centerline origin as the wall frame.
    Regulation tennis court dimensions apply.
    Serve contact defaults to z_m = 6.11 m from the wall.

This module owns the canonical data model, validation helpers, regulation
court constants, and wall-plane homography calibration primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Regulation court dimensions (meters)
# ---------------------------------------------------------------------------

COURT_LENGTH_M: float = 23.77
"""Overall court length (baseline to baseline)."""

SINGLES_WIDTH_M: float = 8.23
"""Width of the singles court."""

DOUBLES_WIDTH_M: float = 10.97
"""Width of the doubles court."""

SERVICE_BOX_DEPTH_M: float = 6.40
"""Depth of one service box from the net toward the baseline."""

SERVICE_BOX_WIDTH_M: float = 4.115
"""Width of one service box (half of singles width)."""

NET_HEIGHT_M: float = 0.914
"""Net height at center (meters)."""


# ---------------------------------------------------------------------------
# CSV / output contracts
# ---------------------------------------------------------------------------

CSV_COLUMNS: Tuple[str, ...] = (
    "video",
    "serve_index",
    "impact_time_sec",
    "impact_frame",
    "wall_x_m",
    "wall_y_m",
    "speed_m_s",
    "speed_km_h",
    "speed_mph",
    "landing_x_m",
    "landing_z_m",
    "in_service_box",
    "confidence_score",
    "warning_codes",
)
"""Stable CSV column order for wall-serve impact results."""

WARNING_CODES: frozenset[str] = frozenset(
    {
        "degraded_intrinsics",
        "insufficient_track",
        "manual_correction_used",
        "projection_refused",
        "low_calibration_confidence",
    }
)
"""Minimum set of warning codes emitted by the wall-serve pipeline."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class WallCalibrationError(ValueError):
    """Raised when wall calibration metadata fails validation."""

    pass


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WallReferencePoint:
    """One point on the wall with both pixel and wall-frame coordinates."""

    name: str
    pixel: Tuple[float, float]
    wall_m: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pixel": list(self.pixel),
            "wall_m": list(self.wall_m),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WallReferencePoint":
        return cls(
            name=str(d["name"]),
            pixel=tuple(float(v) for v in d["pixel"]),
            wall_m=tuple(float(v) for v in d["wall_m"]),
        )


@dataclass
class HookReference:
    """Hook (e.g. wall-mounted equipment) used as a vertical height reference."""

    pixel: Tuple[float, float]
    height_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel": list(self.pixel),
            "height_m": float(self.height_m),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HookReference":
        return cls(
            pixel=tuple(float(v) for v in d["pixel"]),
            height_m=float(d["height_m"]),
        )


@dataclass
class ChairReference:
    """Chair or other object used as a secondary vertical height reference."""

    pixel: Tuple[float, float]
    height_m: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel": list(self.pixel),
            "height_m": float(self.height_m),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChairReference":
        return cls(
            pixel=tuple(float(v) for v in d["pixel"]),
            height_m=float(d["height_m"]),
        )


@dataclass
class Intrinsics:
    """Camera intrinsics and distortion coefficients."""

    source: str
    camera_matrix: Optional[List[List[float]]] = None
    dist_coeffs: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"source": self.source}
        if self.camera_matrix is not None:
            result["camera_matrix"] = self.camera_matrix
        if self.dist_coeffs is not None:
            result["dist_coeffs"] = self.dist_coeffs
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Intrinsics":
        return cls(
            source=str(d["source"]),
            camera_matrix=d.get("camera_matrix"),
            dist_coeffs=d.get("dist_coeffs"),
        )


@dataclass
class WallCalibration:
    """Canonical wall-serve calibration and metadata schema.

    Attributes:
        serve_contact_distance_m: Distance from the wall to the serve contact
            point along the court centerline (default 6.11 m).
        camera_wall_distance_m: Distance from the camera to the wall
            (default 1.57 m).
        serve_contact_height_m: Height of the ball at serve contact (meters).
            REQUIRED — no default.
        wall_reference_points: At least 4 points on the wall with pixel and
            wall-frame coordinates.
        hook_reference: Hook used as a vertical height reference.
        chair_references: List of chairs/objects used as secondary height refs.
        video_override: Per-video optional overrides (dict, free-form).
        intrinsics: Optional camera intrinsics.
        manual_corrections: Optional dict keyed by serve_index.
    """

    serve_contact_distance_m: float = 6.11
    camera_wall_distance_m: float = 1.57
    serve_contact_height_m: Optional[float] = None
    wall_reference_points: List[WallReferencePoint] = field(default_factory=list)
    hook_reference: Optional[HookReference] = None
    chair_references: List[ChairReference] = field(default_factory=list)
    video_override: Optional[Dict[str, Any]] = None
    intrinsics: Optional[Intrinsics] = None
    manual_corrections: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the calibration metadata.

        Raises:
            WallCalibrationError: If any required field is missing or invalid.
        """
        if self.serve_contact_height_m is None:
            raise WallCalibrationError("Missing required field: serve_contact_height_m")

        if len(self.wall_reference_points) < 4:
            raise WallCalibrationError(
                f"Expected at least 4 wall_reference_points, got "
                f"{len(self.wall_reference_points)}"
            )

        if self.intrinsics is not None:
            _validate_intrinsics(self.intrinsics)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict suitable for JSON."""
        result: Dict[str, Any] = {
            "setup": {
                "serve_contact_distance_m": self.serve_contact_distance_m,
                "camera_wall_distance_m": self.camera_wall_distance_m,
                "serve_contact_height_m": self.serve_contact_height_m,
                "wall_reference_points": [
                    p.to_dict() for p in self.wall_reference_points
                ],
            },
        }
        if self.hook_reference is not None:
            result["setup"]["hook_reference"] = self.hook_reference.to_dict()
        if self.chair_references:
            result["setup"]["chair_references"] = [
                c.to_dict() for c in self.chair_references
            ]
        if self.video_override is not None:
            result["video_override"] = self.video_override
        if self.intrinsics is not None:
            result["intrinsics"] = self.intrinsics.to_dict()
        if self.manual_corrections is not None:
            result["manual_corrections"] = self.manual_corrections
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WallCalibration":
        """Deserialize from a plain dict with validation.

        Args:
            d: Dictionary with keys ``setup``, optionally ``video_override``,
                ``intrinsics``, and ``manual_corrections``.

        Returns:
            A validated ``WallCalibration`` instance.

        Raises:
            WallCalibrationError: On missing required fields or invalid data.
        """
        setup = d.get("setup", {})

        wall_points: List[WallReferencePoint] = []
        for wp in setup.get("wall_reference_points", []):
            wall_points.append(WallReferencePoint.from_dict(wp))

        hook_ref: Optional[HookReference] = None
        if "hook_reference" in setup:
            hook_ref = HookReference.from_dict(setup["hook_reference"])

        chair_refs: List[ChairReference] = []
        for cr in setup.get("chair_references", []):
            chair_refs.append(ChairReference.from_dict(cr))

        intrinsics: Optional[Intrinsics] = None
        if "intrinsics" in d:
            intrinsics = Intrinsics.from_dict(d["intrinsics"])

        instance = cls(
            serve_contact_distance_m=float(setup.get("serve_contact_distance_m", 6.11)),
            camera_wall_distance_m=float(setup.get("camera_wall_distance_m", 1.57)),
            serve_contact_height_m=setup.get("serve_contact_height_m"),
            wall_reference_points=wall_points,
            hook_reference=hook_ref,
            chair_references=chair_refs,
            video_override=d.get("video_override"),
            intrinsics=intrinsics,
            manual_corrections=d.get("manual_corrections"),
        )
        instance.validate()
        return instance


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_INTRINSICS_SOURCES: frozenset[str] = frozenset(
    {"none", "approx_exif", "opencv_chessboard", "opencv_charuco"}
)


def _validate_intrinsics(intrinsics: Intrinsics) -> None:
    """Validate intrinsics source and required matrix fields."""
    if intrinsics.source not in _INTRINSICS_SOURCES:
        raise WallCalibrationError(
            f"Invalid intrinsics source: {intrinsics.source!r}. "
            f"Expected one of {_INTRINSICS_SOURCES}"
        )
    if intrinsics.source != "none":
        if intrinsics.camera_matrix is None:
            raise WallCalibrationError(
                f"intrinsics.source={intrinsics.source!r} requires camera_matrix"
            )
        if intrinsics.dist_coeffs is None:
            raise WallCalibrationError(
                f"intrinsics.source={intrinsics.source!r} requires dist_coeffs"
            )


# ---------------------------------------------------------------------------
# Homography calibration primitives
# ---------------------------------------------------------------------------


def _check_collinear(points: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if all points in (N, 2) array are (nearly) collinear."""
    if len(points) < 3:
        return True
    centered = points - np.mean(points, axis=0)
    return np.linalg.matrix_rank(centered, tol=tol) < 2


def _validate_point_arrays(
    image_points: np.ndarray, world_points_m: np.ndarray
) -> None:
    """Validate calibration correspondence array shapes."""
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise WallCalibrationError(
            "calibration_degenerate",
            {"detail": "image_points must have shape (N, 2)"},
        )
    if world_points_m.ndim != 2 or world_points_m.shape[1] != 2:
        raise WallCalibrationError(
            "calibration_degenerate",
            {"detail": "world_points_m must have shape (N, 2)"},
        )
    if len(image_points) != len(world_points_m):
        raise WallCalibrationError(
            "calibration_degenerate",
            {"detail": "image_points and world_points_m must have the same length"},
        )


def _intrinsics_residual_flags(intrinsics: Optional[Intrinsics]) -> dict:
    """Return residual flags implied by optional camera intrinsics."""
    return {
        "degraded_intrinsics": bool(
            intrinsics is not None and intrinsics.source == "approx_exif"
        )
    }


def compute_reprojection_rms(
    image_points: np.ndarray, world_points_m: np.ndarray, H: np.ndarray
) -> float:
    """Compute RMS reprojection error for a homography calibration set.

    Args:
        image_points: (N, 2) pixel coordinates.
        world_points_m: (N, 2) wall-plane meters.
        H: 3×3 homography mapping world → pixel.

    Returns:
        RMS reprojection error in pixels.
    """
    projected = cv2.perspectiveTransform(
        world_points_m.reshape(-1, 1, 2).astype(np.float64), H
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points.astype(np.float64), axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def compute_wall_homography(
    image_points: np.ndarray,
    world_points_m: np.ndarray,
    ransac_reproj_threshold: float = 3.0,
    intrinsics: Optional[Intrinsics] = None,
) -> tuple[np.ndarray, dict]:
    """Compute a wall-plane homography from pixel ↔ wall-meter correspondences.

    Uses ``cv2.findHomography`` with RANSAC to reject outliers.

    Args:
        image_points: (N, 2) array of pixel coordinates.
        world_points_m: (N, 2) array of wall-plane coordinates in meters.
        ransac_reproj_threshold: RANSAC inlier threshold in pixels (default 3.0).
        intrinsics: Optional camera intrinsics used to undistort image points
            before fitting. ``source='approx_exif'`` marks residuals degraded.

    Returns:
        ``(H, residuals_dict)`` where *H* is a 3×3 ``np.ndarray`` and
        *residuals_dict* contains:

        - ``reprojection_rms_px`` — RMS reprojection error (pixels).
        - ``inlier_count`` — number of RANSAC inliers.
        - ``point_count`` — total calibration point count.

    Raises:
        WallCalibrationError: If fewer than 4 points are provided or if the
            point set is collinear (degenerate configuration).
    """
    image_points = np.asarray(image_points, dtype=np.float64)
    world_points_m = np.asarray(world_points_m, dtype=np.float64)
    _validate_point_arrays(image_points, world_points_m)

    if len(image_points) < 4:
        raise WallCalibrationError(
            "calibration_degenerate",
            {"detail": f"Need >= 4 points, got {len(image_points)}"},
        )

    if _check_collinear(image_points) or _check_collinear(world_points_m):
        raise WallCalibrationError(
            "calibration_degenerate",
            {"detail": "Calibration points are collinear (degenerate configuration)"},
        )

    fit_image_points = image_points
    residuals = _intrinsics_residual_flags(intrinsics)
    if intrinsics is not None:
        fit_image_points = undistort_points(image_points, intrinsics)

    H, mask = cv2.findHomography(
        world_points_m.reshape(-1, 1, 2),
        fit_image_points.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )

    if H is None:
        raise WallCalibrationError(
            "calibration_degenerate",
            {"detail": "cv2.findHomography returned None (degenerate configuration)"},
        )

    inlier_count = int(mask.sum()) if mask is not None else len(image_points)
    rms = compute_reprojection_rms(fit_image_points, world_points_m, H)

    residuals.update(
        {
            "reprojection_rms_px": rms,
            "inlier_count": inlier_count,
            "point_count": len(image_points),
        }
    )

    return H, residuals


def pixel_to_wall(H_inv: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Project pixel coordinates to wall-plane meters using the inverse homography.

    Args:
        H_inv: 3×3 inverse homography (pixel → wall).
        points_px: (N, 2) pixel coordinates.

    Returns:
        (N, 2) wall-plane coordinates in meters.
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    return result


def wall_to_pixel(H: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    """Project wall-plane meters to pixel coordinates using the homography.

    Args:
        H: 3×3 homography (wall → pixel).
        points_m: (N, 2) wall-plane coordinates in meters.

    Returns:
        (N, 2) pixel coordinates.
    """
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return result


def undistort_points(points_px: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Undistort pixel coordinates using camera intrinsics when available.

    Args:
        points_px: (N, 2) pixel coordinates.
        intrinsics: Camera intrinsics descriptor.

    Returns:
        (N, 2) undistorted pixel coordinates. Approximate-intrinsics warning
        flags are reported by ``compute_wall_homography`` residuals.

    Notes:
        - If ``intrinsics.source == "none"``, input is returned unchanged.
        - If ``intrinsics.source == "approx_exif"``, the returned info dict
          sets ``degraded_intrinsics=True``.
    """
    points_px = np.asarray(points_px, dtype=np.float64)

    if intrinsics.source == "none":
        return points_px

    if intrinsics.camera_matrix is None or intrinsics.dist_coeffs is None:
        return points_px

    K = np.array(intrinsics.camera_matrix, dtype=np.float64)
    dist = np.array(intrinsics.dist_coeffs, dtype=np.float64)

    pts = points_px.reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    undistorted = undistorted.reshape(-1, 2)

    return undistorted
