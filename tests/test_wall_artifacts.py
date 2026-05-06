"""Tests for wall-analysis artifact generation."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cv2

from serve_analyzer.wall_calibration import (
    WallCalibration,
    WallReferencePoint,
)
from serve_analyzer.wall_serve import (
    CourtProjectionResult,
    PreWallSpeedResult,
    WallImpactResult,
    detect_wall_impact,
    estimate_pre_wall_speed,
    project_to_court,
)
from wall_test_helpers import generate_wall_impact_video

from serve_analyzer.wall_artifacts import (
    render_annotated_video,
    render_plots,
)


class TestWallArtifacts(unittest.TestCase):
    """Integration-style tests for artifact generation."""

    def _make_calibration(self) -> WallCalibration:
        """Build a minimal 4-point wall calibration matching synthetic video geometry."""
        # Synthetic video: 320x240, wall at x=240, ball at y=120.
        # Map pixel rectangle around the visible area to wall meters.
        return WallCalibration(
            serve_contact_height_m=2.80,
            wall_reference_points=[
                WallReferencePoint("bl", (200.0, 200.0), (-4.0, 0.0)),
                WallReferencePoint("br", (280.0, 200.0), (4.0, 0.0)),
                WallReferencePoint("tl", (200.0, 40.0), (-4.0, 3.0)),
                WallReferencePoint("tr", (280.0, 40.0), (4.0, 3.0)),
            ],
        )

    def _make_results(
        self,
        video_path: str,
        calibration: WallCalibration,
        *,
        add_warnings: bool = False,
    ) -> tuple[WallImpactResult, PreWallSpeedResult, CourtProjectionResult]:
        """Run the full wall pipeline on a synthetic video and return results."""
        impact_result = detect_wall_impact(video_path, calibration)
        fps = 60.0
        speed_result = estimate_pre_wall_speed(
            impact_result, calibration, fps=fps, min_samples=3
        )
        # Enrich speed_result metadata with fps for plot rendering
        speed_result.metadata["fps"] = fps

        projection_result = project_to_court(speed_result, calibration)

        if add_warnings:
            # Inject a warning to exercise warning rendering
            object.__setattr__(
                impact_result,
                "warnings",
                list(impact_result.warnings) + ["test_warning"],
            )

        return impact_result, speed_result, projection_result

    def test_annotated_video_is_readable(self):
        """Generate annotated MP4; verify it opens and has frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "synthetic.mp4")
            gt = generate_wall_impact_video(
                video_path,
                width=320,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=20,
                ball_speed_px_per_frame=8.0,
                wall_x_px=240,
            )
            calibration = self._make_calibration()
            impact_result, speed_result, projection_result = self._make_results(
                video_path, calibration
            )

            output_path = os.path.join(tmpdir, "annotated.mp4")
            result = render_annotated_video(
                video_path,
                impact_result,
                speed_result,
                projection_result,
                calibration,
                output_path,
            )

            self.assertTrue(result.exists())
            cap = cv2.VideoCapture(str(result))
            self.assertTrue(cap.isOpened())
            frame_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frame_count += 1
            cap.release()
            self.assertGreater(frame_count, 0)

    def test_plots_are_generated_nonempty_pngs(self):
        """Call render_plots; assert all 3 PNG paths exist with size > 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "synthetic.mp4")
            generate_wall_impact_video(
                video_path,
                width=320,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=20,
                ball_speed_px_per_frame=8.0,
                wall_x_px=240,
            )
            calibration = self._make_calibration()
            impact_result, speed_result, projection_result = self._make_results(
                video_path, calibration
            )

            plots_dir = os.path.join(tmpdir, "plots")
            paths = render_plots(
                impact_result,
                speed_result,
                projection_result,
                calibration,
                plots_dir,
                video_stem="synthetic",
            )

            self.assertIn("speed", paths)
            self.assertIn("wall_impact", paths)
            self.assertIn("court_landing", paths)

            for key, p in paths.items():
                self.assertTrue(p.exists(), f"{key} plot not found at {p}")
                self.assertTrue(str(p).endswith(".png"), f"{key} plot is not PNG: {p}")
                self.assertGreater(os.path.getsize(str(p)), 0, f"{key} plot is empty")

    def test_warnings_overlay_no_exception(self):
        """Build result with non-empty warnings; assert render_annotated_video succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "synthetic.mp4")
            generate_wall_impact_video(
                video_path,
                width=320,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=20,
                ball_speed_px_per_frame=8.0,
                wall_x_px=240,
            )
            calibration = self._make_calibration()
            impact_result, speed_result, projection_result = self._make_results(
                video_path, calibration, add_warnings=True
            )

            # Verify warnings are present
            self.assertTrue(any("test_warning" in w for w in impact_result.warnings))

            output_path = os.path.join(tmpdir, "warn_annotated.mp4")
            # Should not raise
            render_annotated_video(
                video_path,
                impact_result,
                speed_result,
                projection_result,
                calibration,
                output_path,
            )
            self.assertTrue(Path(output_path).exists())

    def test_overwrite_false_raises_when_exists(self):
        """Pre-create output files; call with overwrite=False; assert FileExistsError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "synthetic.mp4")
            generate_wall_impact_video(
                video_path,
                width=320,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=20,
                ball_speed_px_per_frame=8.0,
                wall_x_px=240,
            )
            calibration = self._make_calibration()
            impact_result, speed_result, projection_result = self._make_results(
                video_path, calibration
            )

            # Test annotated video
            output_path = os.path.join(tmpdir, "exists.mp4")
            Path(output_path).touch()  # pre-create
            with self.assertRaises(FileExistsError):
                render_annotated_video(
                    video_path,
                    impact_result,
                    speed_result,
                    projection_result,
                    calibration,
                    output_path,
                    overwrite=False,
                )

            # Test plots
            plots_dir = os.path.join(tmpdir, "plots2")
            os.makedirs(plots_dir)
            # Pre-create one of the expected plot files
            pre_created = Path(plots_dir) / "synthetic_serve00_speed.png"
            pre_created.touch()
            with self.assertRaises(FileExistsError):
                render_plots(
                    impact_result,
                    speed_result,
                    projection_result,
                    calibration,
                    plots_dir,
                    video_stem="synthetic",
                    overwrite=False,
                )

    def test_plots_refused_projection_renders(self):
        """When projection is refused (None landing), court_landing.png still renders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "synthetic.mp4")
            generate_wall_impact_video(
                video_path,
                width=320,
                height=240,
                fps=60,
                total_frames=90,
                impact_frame=20,
                ball_speed_px_per_frame=8.0,
                wall_x_px=240,
            )
            calibration = self._make_calibration()
            impact_result, speed_result, _ = self._make_results(video_path, calibration)

            # Force a refused projection
            refused_projection = CourtProjectionResult(
                landing_x_m=None,
                landing_z_m=None,
                in_service_box=None,
                service_box_side=None,
                assumptions={"model": "gravity_only", "refused": True},
                uncertainty={
                    "landing_z_sensitivity_m": 0.0,
                    "landing_x_sensitivity_m": 0.0,
                },
                warnings=["projection_refused"],
            )

            plots_dir = os.path.join(tmpdir, "plots_refused")
            paths = render_plots(
                impact_result,
                speed_result,
                refused_projection,
                calibration,
                plots_dir,
                video_stem="synthetic",
            )

            court_path = paths["court_landing"]
            self.assertTrue(court_path.exists())
            self.assertGreater(os.path.getsize(str(court_path)), 0)


if __name__ == "__main__":
    unittest.main()
