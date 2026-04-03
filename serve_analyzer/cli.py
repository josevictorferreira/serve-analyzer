#!/usr/bin/env python3
"""
CLI interface for serve velocity analysis.

Provides interactive and non-interactive modes for:
- Manual 2-point calibration
- Ball tracking initialization
- Velocity estimation

Usage:
    # Interactive mode (click to calibrate and mark ball)
    python -m serve_analyzer.cli video.mp4 --real-distance 1.0

    # Non-interactive mode (for scripting)
    python -m serve_analyzer.cli video.mp4 \
        --cal-p1 100 200 \
        --cal-p2 400 200 \
        --real-distance 1.0 \
        --start-frame 45 \
        --ball-pos 320 240
"""

import argparse
import sys
import cv2
import numpy as np
from typing import Tuple, Optional, List

from .analysis import (
    compute_scale_factor,
    compute_velocity_series,
    track_ball_template,
    get_video_fps,
    get_video_info
)


class InteractiveCalibrator:
    """Interactive click-based calibration and ball position selection."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.calibration_points: List[Tuple[int, int]] = []
        self.ball_position: Optional[Tuple[int, int]] = None
        self.current_frame: Optional[np.ndarray] = None
        self.window_name = "Serve Analyzer - Press 'q' to quit"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for calibration and ball selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.calibration_points) < 2:
                self.calibration_points.append((x, y))
                print(f"Calibration point {len(self.calibration_points)}: ({x}, {y})")
                
                # Draw point on frame
                if self.current_frame is not None:
                    color = (0, 255, 0) if len(self.calibration_points) == 1 else (0, 0, 255)
                    cv2.circle(self.current_frame, (x, y), 5, color, -1)
                    cv2.putText(self.current_frame, f"P{len(self.calibration_points)}", (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.imshow(self.window_name, self.current_frame)
                    
            elif self.ball_position is None:
                self.ball_position = (x, y)
                print(f"Ball position: ({x}, {y})")
                
                # Draw ball position
                if self.current_frame is not None:
                    cv2.circle(self.current_frame, (x, y), 8, (255, 0, 0), -1)
                    cv2.putText(self.current_frame, "BALL", (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.imshow(self.window_name, self.current_frame)
    
    def run_interactive(self, frame_number: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Run interactive calibration session.
        
        Args:
            frame_number: Frame to display for calibration
        
        Returns:
            Tuple of (cal_point1, cal_point2, ball_position)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise IOError(f"Cannot read frame {frame_number}")
        
        self.current_frame = frame.copy()
        
        # Setup window
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== Interactive Calibration ===")
        print("1. Click two points with known real-world distance")
        print("2. Click the ball position to start tracking")
        print("3. Press 'q' when done\n")
        
        cv2.imshow(self.window_name, self.current_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Check if we have all points
            if len(self.calibration_points) == 2 and self.ball_position is not None:
                print("\nAll points collected. Press 'q' to continue.")
        
        cv2.destroyAllWindows()
        
        if len(self.calibration_points) != 2:
            raise ValueError("Need exactly 2 calibration points")
        
        if self.ball_position is None:
            raise ValueError("Need ball position")
        
        return (
            self.calibration_points[0],
            self.calibration_points[1],
            self.ball_position
        )


def run_analysis(
    video_path: str,
    cal_point1: Optional[Tuple[int, int]],
    cal_point2: Optional[Tuple[int, int]],
    real_distance: float,
    ball_position: Optional[Tuple[int, int]],
    start_frame: int = 0,
    template_size: int = 30,
    search_radius: int = 100,
    max_frames: Optional[int] = None,
    interactive: bool = True,
    display_frame: int = 0
) -> dict:
    """
    Run complete velocity analysis pipeline.
    
    Args:
        video_path: Path to video file
        cal_point1: First calibration point (None for interactive)
        cal_point2: Second calibration point (None for interactive)
        real_distance: Real-world distance between calibration points (meters)
        ball_position: Initial ball position (None for interactive)
        start_frame: Frame to start tracking
        template_size: Template size for tracking
        search_radius: Search radius for tracking
        max_frames: Max frames to track (None = until end)
        interactive: Use interactive mode for calibration
        display_frame: Frame to display in interactive mode
    
    Returns:
        Dict with analysis results
    """
    # Validate interactive mode consistency
    if interactive and display_frame is not None and display_frame != start_frame:
        raise ValueError("In interactive mode, --display-frame must be omitted or equal to --start-frame")

    # Get video info
    video_info = get_video_info(video_path)
    print(f"\nVideo: {video_path}")
    print(f"  FPS: {video_info['fps']}")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  Frames: {video_info['frame_count']}")
    print(f"  Duration: {video_info['duration_sec']:.2f}s\n")
    
    # Calibration
    if interactive and (cal_point1 is None or cal_point2 is None or ball_position is None):
        calibrator = InteractiveCalibrator(video_path)
        cal_point1, cal_point2, ball_position = calibrator.run_interactive(display_frame)
    
    # Validate we have all required points
    if cal_point1 is None or cal_point2 is None:
        raise ValueError("Calibration points required (use --cal-p1, --cal-p2, or interactive mode)")
    
    if ball_position is None:
        raise ValueError("Ball position required (use --ball-pos or interactive mode)")
    
    # Compute scale
    scale_factor = compute_scale_factor(cal_point1, cal_point2, real_distance)
    print(f"Scale: {scale_factor:.6f} meters/pixel")
    print(f"      ({1/scale_factor:.2f} pixels/meter)\n")
    
    # Track ball
    print(f"Tracking from frame {start_frame}...")
    centers = track_ball_template(
        video_path,
        start_frame,
        ball_position,
        template_size=template_size,
        search_radius=search_radius,
        max_frames=max_frames
    )
    print(f"Tracked {len(centers)} frames\n")
    
    # Compute velocity
    speeds_mps, speeds_kmh, stats = compute_velocity_series(
        centers,
        video_info['fps'],
        scale_factor
    )
    
    # Display results
    print("=== Velocity Results ===")
    print(f"Max speed:   {stats['max_kmh']:.1f} km/h ({stats['max_mps']:.1f} m/s)")
    print(f"Mean speed:  {stats['mean_kmh']:.1f} km/h ({stats['mean_mps']:.1f} m/s)")
    print(f"Median speed: {stats['median_kmh']:.1f} km/h ({stats['median_mps']:.1f} m/s)")
    print(f"Duration:    {stats['duration_sec']:.3f}s ({stats['frame_count']} frames)\n")
    
    # Limitations notice
    print("NOTE: These are APPROXIMATE velocities from a single lateral view.")
    print("Accuracy depends on:")
    print("  - Quality of manual calibration")
    print("  - Tracking reliability")
    print("  - Camera angle and perspective")
    print("  - Ball motion being primarily in the calibration plane\n")
    
    return {
        'video_info': video_info,
        'calibration': {
            'point1': cal_point1,
            'point2': cal_point2,
            'real_distance': real_distance,
            'scale_factor': scale_factor
        },
        'tracking': {
            'start_frame': start_frame,
            'centers': centers,
            'frame_count': len(centers)
        },
        'velocity': {
            'speeds_mps': speeds_mps.tolist(),
            'speeds_kmh': speeds_kmh.tolist(),
            'stats': stats
        }
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Estimate tennis serve velocity from lateral video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (click to calibrate)
  %(prog)s video.mp4 --real-distance 1.0

  # Non-interactive with manual calibration
  %(prog)s video.mp4 \\
      --cal-p1 100 200 \\
      --cal-p2 400 200 \\
      --real-distance 1.0 \\
      --start-frame 45 \\
      --ball-pos 320 240

  # Track specific frame range
  %(prog)s video.mp4 \\
      --cal-p1 100 200 \\
      --cal-p2 400 200 \\
      --real-distance 1.0 \\
      --start-frame 50 \\
      --max-frames 30

IMPORTANT:
  This is an MVP tool providing APPROXIMATE velocity estimates.
  - Manual calibration is required (2 points with known distance)
  - Ball tracking uses simple template matching
  - Single lateral camera view only (no 3D reconstruction)
  - Accuracy depends on calibration quality and camera angle
        """
    )
    
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--real-distance', type=float, required=True,
                       help='Real-world distance between calibration points (meters)')
    
    # Calibration points (non-interactive mode)
    parser.add_argument('--cal-p1', nargs=2, type=int, metavar=('X', 'Y'),
                       help='First calibration point (x y)')
    parser.add_argument('--cal-p2', nargs=2, type=int, metavar=('X', 'Y'),
                       help='Second calibration point (x y)')
    
    # Tracking parameters
    parser.add_argument('--ball-pos', nargs=2, type=int, metavar=('X', 'Y'),
                       help='Initial ball position (x y)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Frame to start tracking (default: 0)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to track (default: until end)')
    parser.add_argument('--display-frame', type=int, default=None,
                       help='Frame to display for interactive calibration (default: start-frame; must be omitted or equal to --start-frame in interactive mode)')
    
    # Tracking tuning
    parser.add_argument('--template-size', type=int, default=30,
                       help='Template size for tracking in pixels (default: 30)')
    parser.add_argument('--search-radius', type=int, default=100,
                       help='Search radius around previous position (default: 100)')
    
    # Output
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    args = parser.parse_args()
    display_frame = args.display_frame if args.display_frame is not None else args.start_frame
    # Convert calibration points to tuples
    cal_p1 = tuple(args.cal_p1) if args.cal_p1 else None
    cal_p2 = tuple(args.cal_p2) if args.cal_p2 else None
    ball_pos = tuple(args.ball_pos) if args.ball_pos else None
    
    # Determine if interactive mode
    interactive = (cal_p1 is None or cal_p2 is None or ball_pos is None)
    
    try:
        results = run_analysis(
            video_path=args.video,
            cal_point1=cal_p1,
            cal_point2=cal_p2,
            real_distance=args.real_distance,
            ball_position=ball_pos,
            start_frame=args.start_frame,
            template_size=args.template_size,
            search_radius=args.search_radius,
            max_frames=args.max_frames,
            interactive=interactive,
            display_frame=display_frame
        )
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}\n")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
