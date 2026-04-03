"""
Serve Analyzer - Tennis serve velocity estimation from lateral video.

An MVP tool for estimating post-impact ball velocity from single-camera
lateral recordings using manual 2-point calibration.
"""

__version__ = "0.1.0"
__author__ = "Serve Analyzer"

from .analysis import (
    compute_scale_factor,
    compute_velocity_series,
    track_ball_template,
    extract_ball_centers,
)

__all__ = [
    "compute_scale_factor",
    "compute_velocity_series",
    "track_ball_template",
    "extract_ball_centers",
]
