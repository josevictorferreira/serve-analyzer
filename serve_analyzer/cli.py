#!/usr/bin/env python3
"""
CLI interface for serve velocity analysis.

Provides interactive and non-interactive modes for:
- Manual 2-point calibration
- Ball tracking initialization
- Velocity estimation

Usage:
    # Interactive mode (click to calibrate and mark ball)
    python -m serve_analyzer.cli video.mp4

    # Interactive mode with pre-specified distance
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
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

from .analysis import (
    compute_scale_factor,
    compute_velocity_series,
    track_ball_template,
    get_video_info
)


class InteractiveCalibrator:
    """Interactive click-based calibration and ball position selection."""
    
    def __init__(self, video_path: str, real_distance: Optional[float] = None):
        self.video_path = video_path
        self.calibration_points: List[Tuple[int, int]] = []
        self.ball_position: Optional[Tuple[int, int]] = None
        self.current_frame: Optional[np.ndarray] = None
        self.real_distance: Optional[float] = real_distance
    
    def _read_frame(self, cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
        """Seek to frame and read it. Returns BGR frame."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise IOError(f"Cannot read frame {frame_idx}")
        return frame

    def run_interactive(self, frame_number: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        Run interactive calibration session.
        
        Phase 1: Browse frames with arrow keys to find the right start frame.
        Phase 2: Click 2 calibration points + ball position.
        Phase 3: Terminal prompt for real-world distance (if not provided).
        
        Keyboard controls during browsing:
            Right/D  — next frame
            Left/A   — previous frame
            Shift+Right — skip 10 frames forward
            Shift+Left  — skip 10 frames backward
            Enter    — confirm frame and proceed to calibration
        
        Args:
            frame_number: Initial frame to display
        
        Returns:
            Tuple of (cal_point1, cal_point2, ball_position)
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Read initial frame
        current_idx = frame_number
        frame = self._read_frame(cap, current_idx)
        self.current_frame = frame.copy()
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        print("\n=== Interactive Calibration ===")
        print("Phase 1: Browse frames to find the ball impact frame")
        print("  Right/D: next frame | Left/A: previous frame")
        print("  Shift+Right/Left: skip 10 frames", )
        print("  Enter: confirm frame and proceed to calibration")
        print(f"  Video: {total_frames} frames @ {fps:.1f} fps ({total_frames/fps:.1f}s)\n")
        
        # --- Phase 1: Frame browsing ---
        fig, ax = plt.subplots()
        img_obj = ax.imshow(frame_rgb)
        title = ax.set_title(f"Frame {current_idx}/{total_frames-1}  ({current_idx/fps:.3f}s)  — Use arrow keys, Enter to confirm")
        plt.tight_layout()
        
        browsing = {'active': True}
        
        def on_key(event):
            if not browsing['active']:
                return
            nonlocal current_idx, frame
            step = 1
            if event.key in ('right', 'd'):
                pass  # step = 1
            elif event.key in ('left', 'a'):
                step = -1
            elif event.key == 'shift+right':
                step = 50
            elif event.key == 'shift+left':
                step = -50
            elif event.key == 'enter':
                browsing['active'] = False
                print(f"  Confirmed frame {current_idx} ({current_idx/fps:.3f}s)")
                return
            else:
                return
            
            new_idx = max(0, min(total_frames - 1, current_idx + step))
            if new_idx == current_idx:
                return
            current_idx = new_idx
            frame = self._read_frame(cap, current_idx)
            self.current_frame = frame.copy()
            frame_rgb_new = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img_obj.set_data(frame_rgb_new)
            ax.set_title(f"Frame {current_idx}/{total_frames-1}  ({current_idx/fps:.3f}s)  — Use arrow keys, Enter to confirm")
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Wait for Enter
        while browsing['active']:
            plt.pause(0.05)
        
        # Finalize selected frame
        selected_frame = current_idx
        frame_rgb_final = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        img_obj.set_data(frame_rgb_final)
        
        # --- Phase 2: Calibration points ---
        print("\nPhase 2: Click two points with known real-world distance")
        print("  💡 TIP: You can use the magnifying glass or pan tools on the")
        print("         matplotlib toolbar to zoom in before clicking your points.")
        print("         Make sure to unselect the tool when you are ready to click.")
        ax.set_title(f"Frame {selected_frame} — Click 2 calibration points with known distance")
        plt.draw()
        
        raw_cal = plt.ginput(n=2, timeout=0)
        
        p1 = (int(round(raw_cal[0][0])), int(round(raw_cal[0][1])))
        p2 = (int(round(raw_cal[1][0])), int(round(raw_cal[1][1])))
        self.calibration_points = [p1, p2]
        print(f"Calibration point 1: {p1}")
        print(f"Calibration point 2: {p2}")
        
        # Draw calibration markers and ruler
        ax.scatter([p1[0], p2[0]], [p1[1], p2[1]],
                   c=['green', 'red'], s=80, zorder=5)
        ax.annotate("P1", xy=p1, xytext=(p1[0]+10, p1[1]-10), color='green')
        ax.annotate("P2", xy=p2, xytext=(p2[0]+10, p2[1]-10), color='red')
        
        px_dist = int(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='yellow', linewidth=2)
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(mid_x + 5, mid_y - 10, f"{px_dist} px",
                color='yellow', fontsize=10)
        
        print("\nPhase 3: Click the ball position")
        print("  💡 TIP: You can use the toolbar to zoom precisely on the ball.")
        ax.set_title(f"Frame {selected_frame} — Click the ball position")
        plt.draw()
        
        # --- Ball position ---
        raw_ball = plt.ginput(n=1, timeout=0)
        
        ball = (int(round(raw_ball[0][0])), int(round(raw_ball[0][1])))
        self.ball_position = ball
        print(f"Ball position: {ball}")
        
        ax.scatter([ball[0]], [ball[1]], c=['blue'], s=100, marker='x',
                   linewidths=2, zorder=6)
        ax.annotate("BALL", xy=ball, xytext=(ball[0]+10, ball[1]-10), color='blue')
        plt.draw()
        plt.pause(0.5)
        plt.close(fig)
        cap.release()
        
        if len(self.calibration_points) != 2:
            raise ValueError("Need exactly 2 calibration points")
        
        if self.ball_position is None:
            raise ValueError("Need ball position")
        
        # Prompt for real-world distance if not already known
        if self.real_distance is None:
            while True:
                raw = input("Enter real-world distance between calibration points (meters): ")
                try:
                    value = float(raw)
                    if value <= 0:
                        print("Distance must be positive. Try again.")
                        continue
                    self.real_distance = value
                    break
                except ValueError:
                    print("Invalid number. Try again.")
        
        # Update frame_number to the selected frame
        self.selected_frame = selected_frame
        
        return (
            self.calibration_points[0],
            self.calibration_points[1],
            self.ball_position
        )

def run_analysis(
    video_path: str,
    cal_point1: Optional[Tuple[int, int]],
    cal_point2: Optional[Tuple[int, int]],
    real_distance: Optional[float],
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
        real_distance: Real-world distance between calibration points (meters).
            Required for non-interactive mode. Optional for interactive mode —
            if None, the user will be prompted in the terminal.
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
        calibrator = InteractiveCalibrator(video_path, real_distance=real_distance)
        cal_point1, cal_point2, ball_position = calibrator.run_interactive(display_frame)
        if real_distance is None:
            real_distance = calibrator.real_distance
        # Use the frame the user browsed to (not the original start_frame)
        if hasattr(calibrator, 'selected_frame'):
            start_frame = calibrator.selected_frame
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
    parser.add_argument('--real-distance', type=float,
                       help='Real-world distance between calibration points (meters). Required in non-interactive mode; prompted interactively if omitted.')
    
    # Calibration points (non-interactive mode)
    parser.add_argument('--cal-p1', nargs=2, type=int, metavar=('X', 'Y'),
                       help='First calibration point (x y)')
    parser.add_argument('--cal-p2', nargs=2, type=int, metavar=('X', 'Y'),
                       help='Second calibration point (x y)')
    
    # Tracking parameters
    parser.add_argument('--ball-pos', nargs=2, type=int, metavar=('X', 'Y'),
                       help='Initial ball position (x y)')
    parser.add_argument('--start-frame', type=int, default=550,
                       help='Frame to start browsing/tracking (default: 550)')
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
    
    # Non-interactive mode requires --real-distance
    if not interactive and args.real_distance is None:
        parser.error("--real-distance is required in non-interactive mode")

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
