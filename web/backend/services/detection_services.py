"""Serve detector version registry for the web backend.

New detector versions plug into the web app by implementing
``ServeDetectorService`` and adding an instance to ``_SERVICES``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Protocol

from serve_analyzer.serve_attempts import detect_serve_candidates, select_serves
from serve_analyzer.serve_attempts_v2 import detect_serve_candidates_v2
from serve_analyzer.serve_attempts_v3 import detect_serve_candidates_v3
from serve_analyzer.serve_attempts_v4 import detect_serve_candidates_v4
from serve_analyzer.serve_attempts_v5 import detect_serve_candidates_v5
from serve_analyzer.serve_attempts_v6 import detect_serve_candidates_v6


class ServeDetectorService(Protocol):
    """Contract each web-runnable serve detector version must satisfy."""

    version: str
    label: str
    description: str

    def estimate_seconds_per_sample(self, tracking_detector: str) -> float:
        """Return rough analysis seconds per sampled frame."""

    def run(
        self,
        video_path: str,
        expected_serves: Optional[int],
        frame_skip: int,
        tracking_config: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run detection and return the normalized web detector payload.

        Required result keys are ``detector_version``, ``detector_label``,
        ``detector``, ``selected_serves``, ``candidates``, ``positions``,
        ``raw_positions``, and ``detection_frame_skip``.
        """


class V1ServeDetectorService:
    """Baseline serve detector using the existing v1 candidate selector."""

    version = "v1"
    label = "V1 baseline"
    description = "Existing candidate generator and selector."

    def estimate_seconds_per_sample(self, tracking_detector: str) -> float:
        """Return the current v1 runtime estimate."""
        return 0.7 if tracking_detector == "tracknetv2" else 0.3

    def run(
        self,
        video_path: str,
        expected_serves: Optional[int],
        frame_skip: int,
        tracking_config: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run v1 and normalize its output for the web backend."""
        detection_result = detect_serve_candidates(
            video_path,
            expected_serves=expected_serves,
            detector=tracking_config["detector"] or "yolo",
            model=tracking_config["model"] or "rjtp",
            tracknet_weights=tracking_config["tracknet_weights"],
            tracknet_device=tracking_config["tracknet_device"] or "cpu",
            frame_skip=frame_skip,
        )
        candidates = detection_result["candidates"]
        selected = select_serves(candidates, expected_serves=expected_serves)
        positions = detection_result["positions"]

        return {
            "detector_version": self.version,
            "detector_label": self.label,
            "detector": tracking_config["detector"] or "yolo",
            "selected_serves": selected,
            "candidates": candidates,
            "positions": positions,
            "raw_positions": detection_result.get("raw_positions", positions),
            "detection_frame_skip": detection_result["frame_skip"],
        }


class V2ServeDetectorService:
    """V2 serve detector with continuity, history, and motion-cue refinement."""

    version = "v2"
    label = "V2 continuity refinement"
    description = "V1 candidate pool refined with continuity, history, and motion cues."

    def estimate_seconds_per_sample(self, tracking_detector: str) -> float:
        """Return v2 runtime estimate including refinement overhead."""
        return 0.8 if tracking_detector == "tracknetv2" else 0.38

    def run(
        self,
        video_path: str,
        expected_serves: Optional[int],
        frame_skip: int,
        tracking_config: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run v2 and normalize its output for the web backend."""
        detection_result = detect_serve_candidates_v2(
            video_path,
            expected_serves=expected_serves,
            detector=tracking_config["detector"] or "yolo",
            model=tracking_config["model"] or "rjtp",
            tracknet_weights=tracking_config["tracknet_weights"],
            tracknet_device=tracking_config["tracknet_device"] or "cpu",
            frame_skip=frame_skip,
        )
        selected = detection_result.get("selected_serves", [])
        positions = detection_result.get("positions", [])

        return {
            "detector_version": self.version,
            "detector_label": self.label,
            "detector": detection_result.get("seed_detector")
            or tracking_config["detector"]
            or "yolo",
            "selected_serves": selected,
            "candidates": detection_result.get("candidates", selected),
            "positions": positions,
            "raw_positions": detection_result.get("raw_positions", positions),
            "detection_frame_skip": detection_result.get("frame_skip", frame_skip),
            "v2_continuity": detection_result.get("v2_continuity"),
            "v2_motion_cue_count": detection_result.get("v2_motion_cue_count"),
        }


class V3ServeDetectorService:
    """V3 serve detector with Kalman, SG smoothing, audio onset, and refinement."""

    version = "v3"
    label = "V3 Kalman + SG + Audio"
    description = "V2 refinement plus Kalman filtering, Savitzky-Golay smoothing, and audio onset cross-check."

    def estimate_seconds_per_sample(self, tracking_detector: str) -> float:
        """Return v3 runtime estimate including refinement overhead."""
        return 0.85 if tracking_detector == "tracknetv2" else 0.40

    def run(
        self,
        video_path: str,
        expected_serves: Optional[int],
        frame_skip: int,
        tracking_config: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run v3 and normalize its output for the web backend."""
        detection_result = detect_serve_candidates_v3(
            video_path,
            expected_serves=expected_serves,
            detector=tracking_config["detector"] or "yolo",
            model=tracking_config["model"] or "rjtp",
            tracknet_weights=tracking_config["tracknet_weights"],
            tracknet_device=tracking_config["tracknet_device"] or "cpu",
            frame_skip=frame_skip,
        )
        selected = detection_result.get("selected_serves", [])
        positions = detection_result.get("positions", [])

        return {
            "detector_version": self.version,
            "detector_label": self.label,
            "detector": detection_result.get("seed_detector")
            or tracking_config["detector"]
            or "yolo",
            "selected_serves": selected,
            "candidates": detection_result.get("candidates", selected),
            "positions": positions,
            "raw_positions": detection_result.get("raw_positions", positions),
            "detection_frame_skip": detection_result.get("frame_skip", frame_skip),
            "v3_kalman_stats": detection_result.get("v3_kalman_stats"),
            "v3_motion_cue_count": detection_result.get("v3_motion_cue_count"),
            "v3_audio_onset_count": detection_result.get("v3_audio_onset_count"),
            "v3_audio_matched_serves": detection_result.get("v3_audio_matched_serves"),
        }


class V4ServeDetectorService:
    """V4 serve detector with direction-change refinement and YOLO26n."""

    version = "v4"
    label = "V4 direction-change"
    description = "V1 pipeline with toss apex contact refinement. 21%% better mean error and 26%% better max error than V1."

    def estimate_seconds_per_sample(self, tracking_detector: str) -> float:
        """Return v4 runtime estimate including refinement overhead."""
        return 0.8 if tracking_detector == "tracknetv2" else 0.35

    def run(
        self,
        video_path: str,
        expected_serves: Optional[int],
        frame_skip: int,
        tracking_config: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run v4 and normalize its output for the web backend."""
        detection_result = detect_serve_candidates_v4(
            video_path,
            expected_serves=expected_serves,
            detector=tracking_config["detector"] or "yolo",
            model=tracking_config["model"] or "rjtp",
            tracknet_weights=tracking_config["tracknet_weights"],
            tracknet_device=tracking_config["tracknet_device"] or "cpu",
            frame_skip=frame_skip,
        )
        selected = detection_result.get("selected_serves", [])
        positions = detection_result.get("positions", [])

        return {
            "detector_version": self.version,
            "detector_label": self.label,
            "detector": detection_result.get("seed_detector")
            or tracking_config["detector"]
            or "yolo",
            "selected_serves": selected,
            "candidates": detection_result.get("candidates", selected),
            "positions": positions,
            "raw_positions": detection_result.get("raw_positions", positions),
            "detection_frame_skip": detection_result.get("frame_skip", frame_skip),
            "v4_audio_onset_count": detection_result.get("v4_audio_onset_count"),
            "v4_audio_matched_serves": detection_result.get("v4_audio_matched_serves"),
        }


class V5ServeDetectorService:
    """V5 serve detector with hybrid timing and quality gating."""

    version = "v5"
    label = "V5 hybrid timing"
    description = (
        "V1 pipeline with hybrid contact timing and post-selection quality gating."
    )

    def estimate_seconds_per_sample(self, tracking_detector: str) -> float:
        """Return v5 runtime estimate."""
        return 0.35 if tracking_detector == "tracknetv2" else 0.32

    def run(
        self,
        video_path: str,
        expected_serves: Optional[int],
        frame_skip: int,
        tracking_config: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run v5 and normalize its output for the web backend."""
        detection_result = detect_serve_candidates_v5(
            video_path,
            expected_serves=expected_serves,
            detector=tracking_config["detector"] or "yolo",
            model=tracking_config["model"] or "rjtp",
            tracknet_weights=tracking_config["tracknet_weights"],
            tracknet_device=tracking_config["tracknet_device"] or "cpu",
            frame_skip=frame_skip,
        )
        selected = detection_result.get("selected_serves", [])
        positions = detection_result.get("positions", [])

        return {
            "detector_version": self.version,
            "detector_label": self.label,
            "detector": detection_result.get("seed_detector")
            or tracking_config["detector"]
            or "yolo",
            "selected_serves": selected,
            "candidates": detection_result.get("candidates", selected),
            "positions": positions,
            "raw_positions": detection_result.get("raw_positions", positions),
            "detection_frame_skip": detection_result.get("frame_skip", frame_skip),
        }


class V6ServeDetectorService:
    """V6 serve detector with autonomous two-stage ensemble tracking."""

    version = "v6"
    label = "V6 two-stage ensemble"
    description = "Autonomous v5-style timing with fine-window YOLO/HSV voting."

    def estimate_seconds_per_sample(self, tracking_detector: str) -> float:
        """Return v6 runtime estimate including fine-window rescans."""
        return 0.55 if tracking_detector == "tracknetv2" else 0.45

    def run(
        self,
        video_path: str,
        expected_serves: Optional[int],
        frame_skip: int,
        tracking_config: Dict[str, Optional[str]],
    ) -> Dict[str, Any]:
        """Run v6 autonomously and normalize its output for the web backend."""
        detection_result = detect_serve_candidates_v6(
            video_path,
            detector=tracking_config["detector"] or "yolo",
            model=tracking_config["model"] or "rjtp",
            tracknet_weights=tracking_config["tracknet_weights"],
            tracknet_device=tracking_config["tracknet_device"] or "cpu",
            frame_skip=frame_skip,
        )
        selected = detection_result.get("selected_serves", [])
        positions = detection_result.get("positions", [])

        return {
            "detector_version": self.version,
            "detector_label": self.label,
            "detector": detection_result.get("seed_detector")
            or tracking_config["detector"]
            or "yolo",
            "selected_serves": selected,
            "candidates": detection_result.get("candidates", selected),
            "positions": positions,
            "raw_positions": detection_result.get("raw_positions", positions),
            "detection_frame_skip": detection_result.get("frame_skip", frame_skip),
            "expected_serves": None,
            "count_inferred": True,
            "inferred_count": detection_result.get("inferred_count", len(selected)),
            "v6_windows": detection_result.get("v6_windows", []),
            "v6_fine_detection_count": detection_result.get("v6_fine_detection_count"),
        }


_SERVICES: Dict[str, ServeDetectorService] = {
    "v1": V1ServeDetectorService(),
    "v2": V2ServeDetectorService(),
    "v3": V3ServeDetectorService(),
    "v4": V4ServeDetectorService(),
    "v5": V5ServeDetectorService(),
    "v6": V6ServeDetectorService(),
}


def default_detector_version() -> str:
    """Return the configured default detector version."""
    configured = os.environ.get("SERVE_ANALYZER_DETECTOR_VERSION", "v1").lower()
    return configured if configured in _SERVICES else "v1"


def resolve_detector_version(version: Optional[str]) -> str:
    """Validate and normalize a detector version value."""
    selected = (version or default_detector_version()).lower()
    if selected not in _SERVICES:
        available = ", ".join(sorted(_SERVICES))
        raise ValueError(
            f"Unsupported detector version: {selected}. Available: {available}"
        )
    return selected


def get_detector_service(version: Optional[str]) -> ServeDetectorService:
    """Return a detector service by version."""
    return _SERVICES[resolve_detector_version(version)]


def list_detector_versions() -> List[Dict[str, str]]:
    """Return detector versions exposed to the frontend."""
    return [
        {
            "version": service.version,
            "label": service.label,
            "description": service.description,
        }
        for service in _SERVICES.values()
    ]
