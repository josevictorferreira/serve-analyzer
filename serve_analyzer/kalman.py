"""4-state constant-velocity Kalman filter for ball-track refinement.

State vector: [x, y, vx, vy]   (pixels, pixels/frame)
Measurement:  [x, y]           (pixels)

Used by serve_attempts_v3 to replace the v2 fixed-jump continuity gate
(`continuity_gate_positions`, max_jump_px=260) and the linear interpolation
in multi_serve.interpolate_missing_detections. The Kalman filter:

  * Predicts the next position from current state, propagating uncertainty.
  * Gates incoming detections via Mahalanobis chi-squared (2 DoF) at alpha=0.99
    (default threshold 9.21). Adapts to current velocity / uncertainty
    instead of using a fixed pixel jump.
  * Fills short detection gaps with predicted positions, marking them as
    `imputed` so downstream code can distinguish real detections.

This module is intentionally numpy-only; no filterpy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


Position = Optional[Tuple[float, float]]

# chi^2(2 DoF) thresholds:
#   alpha=0.95 -> 5.991
#   alpha=0.99 -> 9.210   (default; rejects ~1% of true detections)
#   alpha=0.999 -> 13.816
DEFAULT_GATE_CHI2 = 9.21


@dataclass
class KalmanResult:
    """Output of `smooth_track`: one entry per input frame."""

    positions: List[Position] = field(default_factory=list)
    imputed: List[bool] = field(default_factory=list)
    rejected_jumps: int = 0
    accepted: int = 0
    imputed_count: int = 0


def _build_matrices(
    sigma_a: float, sigma_z: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct F, H, Q, R for a constant-velocity model with dt=1 frame."""
    F = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    # Process noise from a constant-acceleration disturbance, dt=1.
    a2 = float(sigma_a) ** 2
    Q = a2 * np.array(
        [
            [0.25, 0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0],
        ]
    )
    R = (float(sigma_z) ** 2) * np.eye(2)
    return F, H, Q, R


def smooth_track(
    detections: Sequence[Position],
    sigma_a: float = 50.0,
    sigma_z: float = 4.0,
    gate_chi2: float = DEFAULT_GATE_CHI2,
    max_imputed_run: int = 12,
) -> KalmanResult:
    """Run forward Kalman filter over ball detections.

    Args:
        detections: per-frame ball positions; None means no detection.
        sigma_a: stdev of unknown acceleration (px/frame^2). Larger -> filter
            trusts the model less, accepts wilder jumps.
        sigma_z: stdev of measurement noise (pixels).
        gate_chi2: Mahalanobis^2 threshold; detections with d^2 > gate_chi2
            are rejected as outliers.
        max_imputed_run: maximum consecutive frames to impute via prediction
            before declaring the track lost (output stays None until the next
            real detection re-initializes the filter).

    Returns:
        KalmanResult with smoothed positions (one per input frame), imputed
        flags, and counters.
    """
    F, H, Q, R = _build_matrices(sigma_a, sigma_z)

    out_positions: List[Position] = []
    imputed_flags: List[bool] = []
    rejected = 0
    accepted = 0
    imputed_count = 0

    initialized = False
    x = np.zeros(4)
    P = np.eye(4) * 1e6  # huge prior covariance until first detection
    imputed_run = 0

    for det in detections:
        if not initialized:
            if det is None:
                out_positions.append(None)
                imputed_flags.append(False)
                continue
            x = np.array([float(det[0]), float(det[1]), 0.0, 0.0])
            P = np.diag(
                [sigma_z**2, sigma_z**2, (sigma_a * 4) ** 2, (sigma_a * 4) ** 2]
            )
            initialized = True
            imputed_run = 0
            out_positions.append((float(x[0]), float(x[1])))
            imputed_flags.append(False)
            accepted += 1
            continue

        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        if det is None:
            imputed_run += 1
            if imputed_run > max_imputed_run:
                # Track lost; stop emitting until we re-initialize.
                out_positions.append(None)
                imputed_flags.append(False)
                # Keep state but flag uninitialized so the next real det resets.
                initialized = False
                continue
            out_positions.append((float(x[0]), float(x[1])))
            imputed_flags.append(True)
            imputed_count += 1
            continue

        z = np.array([float(det[0]), float(det[1])])
        y = z - H @ x
        S = H @ P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        d2 = float(y @ S_inv @ y)

        if d2 > gate_chi2:
            # Reject as outlier; treat as missing.
            rejected += 1
            imputed_run += 1
            if imputed_run > max_imputed_run:
                out_positions.append(None)
                imputed_flags.append(False)
                initialized = False
                continue
            out_positions.append((float(x[0]), float(x[1])))
            imputed_flags.append(True)
            imputed_count += 1
            continue

        # Update
        K = P @ H.T @ S_inv
        x = x + K @ y
        I = np.eye(4)
        P = (I - K @ H) @ P
        imputed_run = 0
        out_positions.append((float(x[0]), float(x[1])))
        imputed_flags.append(False)
        accepted += 1

    return KalmanResult(
        positions=out_positions,
        imputed=imputed_flags,
        rejected_jumps=rejected,
        accepted=accepted,
        imputed_count=imputed_count,
    )
