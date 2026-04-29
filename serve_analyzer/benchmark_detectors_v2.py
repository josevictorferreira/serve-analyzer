#!/usr/bin/env python3
"""Benchmark v2 detector-only outputs against post-hoc timestamps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from serve_analyzer.serve_attempts_v2 import detect_serve_candidates_v2
from serve_analyzer.serve_evaluation import (
    load_target_timestamps,
    summarize_serve_attempts,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _evaluation_metrics(evaluation: Dict[str, Any]) -> Dict[str, Optional[float]]:
    deltas = [
        abs(float(row["delta_sec"])) for row in evaluation["attempts"] if row["matched"]
    ]
    if not deltas:
        return {"mean_abs_delta_sec": None, "max_abs_delta_sec": None}
    return {
        "mean_abs_delta_sec": float(sum(deltas) / len(deltas)),
        "max_abs_delta_sec": float(max(deltas)),
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run v2 detector-only inference, then evaluate saved output post hoc."""
    output_dir = Path(args.output_dir)
    started = time.perf_counter()
    detection = detect_serve_candidates_v2(
        args.video,
        expected_serves=args.expected_serves,
        detector=args.detector,
        model=args.model,
        tracknet_weights=args.tracknet_weights,
        tracknet_device=args.tracknet_device,
        scale_factor=args.scale_factor,
        conf_threshold=args.conf,
        frame_skip=args.frame_skip,
        start_frame=args.start_frame,
        input_detections=args.input_detections,
        max_jump_px=args.max_jump_px,
        max_missing_frames=args.max_missing_frames,
        history_sec=args.history_sec,
        search_radius_sec=args.search_radius_sec,
        motion_stride=args.motion_stride,
    )
    runtime_sec = time.perf_counter() - started
    detection["runtime_sec"] = float(runtime_sec)

    detection_path = output_dir / "v2_detections.json"
    _write_json(detection_path, detection)

    selected = detection["selected_serves"]
    target_times = load_target_timestamps(args.timestamps_file)
    evaluation = summarize_serve_attempts(selected, target_times, args.tolerance_sec)
    evaluation.update(
        {
            "detector": "v2",
            "detection_json": str(detection_path),
            "runtime_sec": float(runtime_sec),
            "selected_count": int(len(selected)),
            "candidate_pool_count": int(len(detection["candidates"])),
        }
    )
    evaluation.update(_evaluation_metrics(evaluation))
    evaluation_path = output_dir / "v2_evaluation.json"
    _write_json(evaluation_path, evaluation)

    summary = {
        "video_path": str(args.video),
        "timestamps_file": str(args.timestamps_file),
        "target_count": int(len(target_times)),
        "expected_serves": args.expected_serves,
        "tolerance_sec": float(args.tolerance_sec),
        "results": [evaluation],
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for v2 detector benchmarking."""
    parser = argparse.ArgumentParser(
        description="Run v2 detector-only serve detection and evaluate timestamps post hoc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--timestamps-file", required=True, help="Manual timestamp annotations"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/v2",
        help="Directory for JSON outputs",
    )
    parser.add_argument("--expected-serves", type=int, default=None)
    parser.add_argument("--tolerance-sec", type=float, default=3.0)
    parser.add_argument("--detector", choices=("yolo", "tracknetv2"), default="yolo")
    parser.add_argument("--model", default="rjtp")
    parser.add_argument("--tracknet-weights")
    parser.add_argument("--tracknet-device", default="cpu")
    parser.add_argument("--scale-factor", type=float, default=0.001)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--input-detections", help="Optional detector JSON cache")
    parser.add_argument("--max-jump-px", type=float, default=260.0)
    parser.add_argument("--max-missing-frames", type=int, default=12)
    parser.add_argument("--history-sec", type=float, default=0.35)
    parser.add_argument("--search-radius-sec", type=float, default=0.80)
    parser.add_argument("--motion-stride", type=int, default=2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frame_skip < 1:
        parser.error("frame-skip must be at least 1")
    if args.expected_serves is not None and args.expected_serves < 1:
        parser.error("expected-serves must be at least 1")
    if args.tolerance_sec < 0:
        parser.error("tolerance-sec must be non-negative")
    if args.motion_stride < 1:
        parser.error("motion-stride must be at least 1")

    summary = run_benchmark(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
