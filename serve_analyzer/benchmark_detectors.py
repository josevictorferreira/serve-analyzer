"""Benchmark detector-only serve outputs against post-hoc timestamp annotations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from serve_analyzer.serve_attempts import detect_serve_candidates, select_serves
from serve_analyzer.serve_attempts_v2 import (
    detect_serve_candidates_v2,
    select_serves_v2,
)
from serve_analyzer.serve_evaluation import (
    load_target_timestamps,
    summarize_serve_attempts,
)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _detector_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build detector configurations to run without reading benchmark labels."""
    specs = [
        {
            "name": "yolo",
            "detector": "yolo",
            "model": args.model,
            "tracknet_weights": None,
            "tracknet_device": args.tracknet_device,
            "conf_threshold": args.conf,
        }
    ]
    if args.include_v2:
        specs.append(
            {
                "name": "v2",
                "detector": "v2",
                "model": args.model,
                "tracknet_weights": None,
                "tracknet_device": args.tracknet_device,
                "conf_threshold": args.conf,
            }
        )
    if args.tracknet_weights:
        specs.append(
            {
                "name": "tracknetv2",
                "detector": "tracknetv2",
                "model": args.model,
                "tracknet_weights": args.tracknet_weights,
                "tracknet_device": args.tracknet_device,
                "conf_threshold": args.tracknet_conf,
            }
        )
    return specs


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run detector-only inference, then evaluate saved outputs post hoc."""
    expected_serves = args.expected_serves
    output_dir = Path(args.output_dir)
    detection_outputs: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for spec in _detector_specs(args):
        started = time.perf_counter()
        if spec["detector"] == "v2":
            detection_result = detect_serve_candidates_v2(
                args.video,
                expected_serves=expected_serves,
                model=spec["model"],
                scale_factor=args.scale_factor,
                conf_threshold=spec["conf_threshold"],
                frame_skip=args.frame_skip,
                start_frame=args.start_frame,
            )
        else:
            detection_result = detect_serve_candidates(
                args.video,
                expected_serves=expected_serves,
                detector=spec["detector"],
                model=spec["model"],
                tracknet_weights=spec["tracknet_weights"],
                tracknet_device=spec["tracknet_device"],
                scale_factor=args.scale_factor,
                conf_threshold=spec["conf_threshold"],
                frame_skip=args.frame_skip,
                start_frame=args.start_frame,
            )
        runtime_sec = time.perf_counter() - started

        candidates = detection_result["candidates"]
        if spec["detector"] == "v2":
            selected = select_serves_v2(candidates, expected_serves=expected_serves)
        else:
            selected = select_serves(candidates, expected_serves=expected_serves)
        detection_payload = {
            "video_path": str(args.video),
            "detector": spec["detector"],
            "expected_serves": expected_serves,
            "frame_skip": args.frame_skip,
            "start_frame": args.start_frame,
            "runtime_sec": float(runtime_sec),
            "selected_serves": selected,
            "candidates": candidates,
        }
        detection_path = output_dir / f"{spec['name']}_detections.json"
        _write_json(detection_path, detection_payload)
        detection_outputs.append(
            {
                "spec": spec,
                "detection_path": detection_path,
                "runtime_sec": runtime_sec,
                "selected": selected,
                "candidates": candidates,
            }
        )

    target_times = load_target_timestamps(args.timestamps_file)
    for output in detection_outputs:
        spec = output["spec"]
        selected = output["selected"]
        candidates = output["candidates"]
        runtime_sec = output["runtime_sec"]
        evaluation = summarize_serve_attempts(
            selected, target_times, args.tolerance_sec
        )
        deltas = [
            abs(float(a["delta_sec"])) for a in evaluation["attempts"] if a["matched"]
        ]
        mean_abs_delta = sum(deltas) / len(deltas) if deltas else None
        max_abs_delta = max(deltas) if deltas else None
        evaluation.update(
            {
                "detector": spec["detector"],
                "detection_json": str(output["detection_path"]),
                "runtime_sec": float(runtime_sec),
                "selected_count": int(len(selected)),
                "candidate_pool_count": int(len(candidates)),
                "mean_abs_delta_sec": float(mean_abs_delta)
                if mean_abs_delta is not None
                else None,
                "max_abs_delta_sec": float(max_abs_delta)
                if max_abs_delta is not None
                else None,
            }
        )
        evaluation_path = output_dir / f"{spec['name']}_evaluation.json"
        _write_json(evaluation_path, evaluation)
        results.append(evaluation)

    summary = {
        "video_path": str(args.video),
        "timestamps_file": str(args.timestamps_file),
        "target_count": int(len(target_times)),
        "expected_serves": expected_serves,
        "tolerance_sec": float(args.tolerance_sec),
        "results": results,
    }
    _write_json(output_dir / "benchmark_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for detector benchmarking."""
    parser = argparse.ArgumentParser(
        description="Run detector-only serve detection and evaluate timestamps post hoc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "--timestamps-file", required=True, help="Manual timestamp annotations"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_outputs", help="Directory for JSON outputs"
    )
    parser.add_argument(
        "--expected-serves", type=int, default=None, help="Forced serve count"
    )
    parser.add_argument(
        "--tolerance-sec", type=float, default=3.0, help="Post-hoc match tolerance"
    )
    parser.add_argument("--model", default="rjtp", help="YOLO model path or alias")
    parser.add_argument(
        "--tracknet-weights",
        help="TrackNetV2 weights path; omit to benchmark YOLO only",
    )
    parser.add_argument(
        "--include-v2",
        action="store_true",
        help="Also run the separated v2 fusion detector",
    )
    parser.add_argument(
        "--tracknet-device", default="cpu", help="Torch device for TrackNetV2"
    )
    parser.add_argument(
        "--conf", type=float, default=0.20, help="YOLO confidence threshold"
    )
    parser.add_argument(
        "--tracknet-conf", type=float, default=0.5, help="TrackNetV2 heatmap threshold"
    )
    parser.add_argument(
        "--scale-factor", type=float, default=0.001, help="Meters per pixel fallback"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=1, help="Process every Nth frame"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip frames before this frame number",
    )
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

    summary = run_benchmark(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
