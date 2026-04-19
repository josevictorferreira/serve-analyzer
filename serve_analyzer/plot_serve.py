#!/usr/bin/env python3
"""
Serve velocity graph generator.

Takes a video file and calibration parameters, then produces a speed graph.

Usage:
    python serve_analyzer/plot_serve.py video.mov \
        --cal-p1 100 200 \
        --cal-p2 400 200 \
        --real-distance 1.0 \
        --ball-pos 320 240 \
        --start-frame 50

The script will:
1. Track the ball using template matching
2. Compute velocity from tracked positions
3. Generate and display/save a speed graph
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from serve_analyzer.analysis import (
    compute_scale_factor,
    compute_velocity_series,
    track_ball_template,
    track_ball_color,
    track_ball_csrt,
    track_ball_optical_flow,
    track_ball_yolo,
    get_video_info,
    generate_annotated_video,
)


def plot_speed_profile(speeds_kmh, fps, output_path=None, show=True):
    """
    Generate and optionally save/display a speed profile plot.

    Args:
        speeds_kmh: Array of speeds in km/h
        fps: Video frame rate
        output_path: If provided, save plot to this path
        show: If True, display the plot interactively
    """
    frames = range(len(speeds_kmh))
    times = np.array(frames) / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Speed vs Frame
    ax1.plot(frames, speeds_kmh, marker="o", linewidth=1, markersize=3, color="#2E86AB")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Post-Impact Ball Speed Profile")
    ax1.grid(True, alpha=0.3)

    # Mark max speed
    max_idx = int(np.argmax(speeds_kmh))
    max_speed = speeds_kmh[max_idx]
    ax1.annotate(
        f"Max: {max_speed:.1f} km/h",
        xy=(max_idx, max_speed),
        xytext=(max_idx + len(frames) * 0.1, max_speed * 0.9),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=10,
    )
    ax1.axhline(y=max_speed, color="red", linestyle="--", alpha=0.3)

    # Speed vs Time
    ax2.plot(times, speeds_kmh, marker="o", linewidth=1, markersize=3, color="#2E86AB")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (km/h)")
    ax2.set_title("Post-Impact Ball Speed Over Time")
    ax2.grid(True, alpha=0.3)

    # Mark max speed on time plot
    ax2.annotate(
        f"Max: {max_speed:.1f} km/h",
        xy=(times[max_idx], max_speed),
        xytext=(times[max_idx] + 0.05, max_speed * 0.9),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=10,
    )
    ax2.axhline(y=max_speed, color="red", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close()

    return max_speed, max_idx


def main():
    parser = argparse.ArgumentParser(
        description="Generate speed graph from serve video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all parameters
  %(prog)s video.mov \\
      --cal-p1 100 200 \\
      --cal-p2 400 200 \\
      --real-distance 1.0 \\
      --ball-pos 320 240 \\
      --start-frame 50

  # Save plot to file
  %(prog)s video.mov \\
      --cal-p1 100 200 \\
      --cal-p2 400 200 \\
      --real-distance 1.0 \\
      --ball-pos 320 240 \\
      --output speed_graph.png

  # Generate annotated video overlay
  %(prog)s video.mov \\
      --cal-p1 100 200 \\
      --cal-p2 400 200 \\
      --real-distance 1.0 \\
      --ball-pos 320 240 \\
      --video-output annotated.mp4
        """,
    )

    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--real-distance",
        type=float,
        required=True,
        help="Real-world distance between calibration points (meters)",
    )
    parser.add_argument(
        "--cal-p1",
        nargs=2,
        type=int,
        required=True,
        metavar=("X", "Y"),
        help="First calibration point (x y)",
    )
    parser.add_argument(
        "--cal-p2",
        nargs=2,
        type=int,
        required=True,
        metavar=("X", "Y"),
        help="Second calibration point (x y)",
    )
    parser.add_argument(
        "--ball-pos",
        nargs=2,
        type=int,
        required=True,
        metavar=("X", "Y"),
        help="Initial ball position (x y)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=550,
        help="Frame to start tracking (default: 550)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to track (default: until end)",
    )
    parser.add_argument(
        "--template-size",
        type=int,
        default=30,
        help="Template size for tracking in pixels (default: 30)",
    )
    parser.add_argument(
        "--search-radius",
        type=int,
        default=300,
        help="Search radius around previous position (default: 300 for fast-moving balls)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Smoothing window size (default: 3, use 1 to disable)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold 0-1 (default: 0.5 for template, 0.1 for YOLO)",
    )
    parser.add_argument(
        "--debug-tracking",
        action="store_true",
        help="Generate debug video showing search regions and tracking status",
    )
    parser.add_argument(
        "--color-tracking",
        action="store_true",
        help="Use HSV color tracking instead of template matching (better for yellow balls)",
    )
    parser.add_argument(
        "--csrt-tracking",
        action="store_true",
        help="Use OpenCV CSRT tracker (robust for fast-moving objects)",
    )
    parser.add_argument(
        "--optical-flow",
        action="store_true",
        help="Use optical flow tracking (best for very fast motion like serves)",
    )
    parser.add_argument(
        "--yolo-tracking",
        action="store_true",
        help="Use YOLO deep learning detector (best accuracy, requires ultralytics)",
    )
    parser.add_argument(
        "--yolo-model",
        default="rjtp",
        help="YOLO model weights file or 'rjtp' for tennis-ball model (default: rjtp)",
    )
    parser.add_argument(
        "--output", "-o", help="Save plot to file instead of displaying"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot (only save if --output is specified)",
    )
    parser.add_argument(
        "--video-output",
        help="Generate annotated video with speed overlay (MP4)",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of positions to show as trajectory trail (default: 20)",
    )
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Include full video instead of just tracked portion (slower)",
    )

    args = parser.parse_args()

    # Convert to tuples
    cal_p1 = tuple(args.cal_p1)
    cal_p2 = tuple(args.cal_p2)
    ball_pos = tuple(args.ball_pos)

    try:
        # Get video info
        print(f"Loading video: {args.video}")
        video_info = get_video_info(args.video)
        fps = video_info["fps"]
        print(f"  FPS: {fps:.2f}")
        print(f"  Resolution: {video_info['width']}x{video_info['height']}")
        print(f"  Frames: {video_info['frame_count']}")
        print(f"  Duration: {video_info['duration_sec']:.2f}s")

        # Compute scale factor
        scale_factor = compute_scale_factor(cal_p1, cal_p2, args.real_distance)
        print(f"\nScale factor: {scale_factor:.6f} m/pixel")
        print(f"  ({1 / scale_factor:.2f} pixels/meter)")

        # Track ball
        print(f"\nTracking from frame {args.start_frame}...")
        if args.yolo_tracking:
            print("Using YOLO deep learning detector")
            # YOLO needs lower confidence threshold than template matching
            yolo_conf = args.confidence if args.confidence != 0.5 else 0.1
            centers = track_ball_yolo(
                args.video,
                args.start_frame,
                ball_pos,
                model_path=args.yolo_model,
                max_frames=args.max_frames,
                conf_threshold=yolo_conf,
                search_radius=args.search_radius,
                debug_output=args.video_output if args.debug_tracking else None,
            )
        elif args.optical_flow:
            print("Using optical flow tracking (optimized for fast motion)")
            centers = track_ball_optical_flow(
                args.video,
                args.start_frame,
                ball_pos,
                search_radius=args.search_radius,
                max_frames=args.max_frames,
                debug_output=args.video_output if args.debug_tracking else None,
            )
        elif args.csrt_tracking:
            print("Using CSRT tracker (discriminative correlation filter)")
            centers = track_ball_csrt(
                args.video,
                args.start_frame,
                ball_pos,
                bbox_size=args.template_size,
                max_frames=args.max_frames,
                debug_output=args.video_output if args.debug_tracking else None,
            )
        elif args.color_tracking:
            print("Using HSV color tracking (yellow ball detection)")
            centers = track_ball_color(
                args.video,
                args.start_frame,
                ball_pos,
                search_radius=args.search_radius,
                max_frames=args.max_frames,
                debug_output=args.video_output if args.debug_tracking else None,
            )
        else:
            centers = track_ball_template(
                args.video,
                args.start_frame,
                ball_pos,
                template_size=args.template_size,
                search_radius=args.search_radius,
                max_frames=args.max_frames,
                confidence_threshold=args.confidence,
                debug_output=args.video_output if args.debug_tracking else None,
            )
        print(f"Tracked {len(centers)} frames")

        # Compute velocity
        speeds_mps, speeds_kmh, stats = compute_velocity_series(
            centers, fps, scale_factor, smoothing_window=args.smoothing_window
        )

        # Print summary
        print("\n" + "=" * 50)
        print("VELOCITY ESTIMATION SUMMARY")
        print("=" * 50)
        print(f"Frames tracked:     {stats['frame_count']}")
        print(f"Duration:           {stats['duration_sec']:.3f} seconds")
        print(f"Smoothing window:   {args.smoothing_window}")
        print()
        print(
            f"Maximum speed:      {stats['max_kmh']:.1f} km/h  ({stats['max_mps']:.2f} m/s)"
        )
        print(
            f"Mean speed:         {stats['mean_kmh']:.1f} km/h  ({stats['mean_mps']:.2f} m/s)"
        )
        print(
            f"Median speed:       {stats['median_kmh']:.1f} km/h  ({stats['median_mps']:.2f} m/s)"
        )
        print("=" * 50)

        # Generate annotated video if requested
        if args.video_output:
            clip = not args.full_video
            mode = "tracked portion only" if clip else "full video"
            print(f"\nGenerating annotated video ({mode}): {args.video_output}")
            generate_annotated_video(
                args.video,
                args.video_output,
                centers,
                speeds_kmh,
                args.start_frame,
                trail_length=args.trail_length,
                clip_to_tracking=clip,
            )
            print(f"Annotated video saved to: {args.video_output}")

        # Generate plot
        print("\nGenerating speed graph...")
        show_plot = not args.no_show
        max_speed, max_idx = plot_speed_profile(
            speeds_kmh, fps, output_path=args.output, show=show_plot
        )

        if args.output:
            print(f"Plot saved to: {args.output}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
