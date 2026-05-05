#!/usr/bin/env python3
"""Post-hoc evaluation: parse timestamp annotations, match detected serve candidates.

This module owns all timestamp-related behaviour — parsing human annotation
files, matching detected candidates to target timestamps, and producing the
combined summary.  It never runs video detection; it consumes detector output
(JSON list of candidate records) plus a timestamps text file.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


def parse_timestamp_token(token: str) -> float:
    """Parse one timestamp token into seconds."""
    parts = token.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
    raise ValueError(f"Invalid timestamp token: {token}")


def parse_timestamp_line(line: str) -> List[float]:
    """Extract all timestamps from one annotation line."""
    content = line.split("#", 1)[0].strip()
    if not content:
        return []

    colon_matches = re.findall(r"\b\d{1,2}:\d{1,2}(?::\d{1,2})?(?:\.\d+)?\b", content)
    if colon_matches:
        return [parse_timestamp_token(token) for token in colon_matches]

    hours = re.findall(r"(\d+(?:\.\d+)?)\s*hour(?:s)?\b", content, flags=re.IGNORECASE)
    minutes = re.findall(
        r"(\d+(?:\.\d+)?)\s*minute(?:s)?\b", content, flags=re.IGNORECASE
    )
    seconds = re.findall(
        r"(\d+(?:\.\d+)?)\s*second(?:s)?\b", content, flags=re.IGNORECASE
    )
    if hours or minutes or seconds:
        total_seconds = (
            sum(float(value) for value in hours) * 3600.0
            + sum(float(value) for value in minutes) * 60.0
            + sum(float(value) for value in seconds)
        )
        return [total_seconds]

    chunks = [chunk.strip() for chunk in re.split(r"[,;]", content) if chunk.strip()]
    timestamps: List[float] = []
    for chunk in chunks:
        numeric_tokens = re.findall(r"\d+(?:\.\d+)?", chunk)
        if not numeric_tokens:
            continue
        if re.search(r"[A-Za-z]", chunk):
            trailing_number = re.search(r"(\d+(?:\.\d+)?)\s*$", chunk)
            if trailing_number:
                timestamps.append(parse_timestamp_token(trailing_number.group(1)))
        else:
            timestamps.extend(parse_timestamp_token(token) for token in numeric_tokens)
    return timestamps


def parse_timestamps_text(text: str) -> List[float]:
    """Parse timestamp annotations into ordered seconds."""
    return parse_timestamp_lines(text.splitlines())


def parse_timestamp_lines(lines: Sequence[str]) -> List[float]:
    """Parse timestamp annotations from text lines."""
    timestamps: List[float] = []
    for line in lines:
        parsed = parse_timestamp_line(line)
        stripped = line.split("#", 1)[0].strip()
        if stripped and not parsed:
            raise ValueError(f"Invalid timestamp line: {line.strip()}")
        timestamps.extend(parsed)
    if not timestamps:
        raise ValueError("No timestamps found")
    return timestamps


def load_target_timestamps(timestamps_file: str) -> List[float]:
    """Load target timestamps from a text file."""
    timestamp_path = Path(timestamps_file)
    if not timestamp_path.exists():
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_file}")
    return parse_timestamp_lines(
        timestamp_path.read_text(encoding="utf-8").splitlines()
    )


# ---------------------------------------------------------------------------
# Target-candidate matching
# ---------------------------------------------------------------------------


def match_targets_to_candidates(
    target_times: Sequence[float],
    candidate_times: Sequence[float],
    tolerance_sec: float,
) -> List[Optional[int]]:
    """Greedy ordered matching from targets to detected candidates."""
    if tolerance_sec < 0:
        raise ValueError("Tolerance must be non-negative")

    matches: List[Optional[int]] = []
    next_candidate_idx = 0

    for target_time in target_times:
        while (
            next_candidate_idx < len(candidate_times)
            and candidate_times[next_candidate_idx] < target_time - tolerance_sec
        ):
            next_candidate_idx += 1

        best_idx: Optional[int] = None
        best_abs_delta = float("inf")
        scan_idx = next_candidate_idx

        while scan_idx < len(candidate_times):
            delta = candidate_times[scan_idx] - target_time
            if delta > tolerance_sec:
                break
            abs_delta = abs(delta)
            if abs_delta <= tolerance_sec and abs_delta < best_abs_delta:
                best_idx = scan_idx
                best_abs_delta = abs_delta
            scan_idx += 1

        matches.append(best_idx)
        if best_idx is not None:
            next_candidate_idx = best_idx + 1

    return matches


def _build_attempt_record(
    serve_number: int,
    target_time: float,
    matched_idx: Optional[int],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one JSON-safe serve attempt record."""
    record: Dict[str, Any] = {
        "serve_number": int(serve_number),
        "target_time_sec": float(target_time),
        "candidate_index": None,
        "detected_time_sec": None,
        "delta_sec": None,
        "matched": False,
        "post_contact_max_kmh": None,
        "post_contact_mean_kmh": None,
        "post_contact_max_mps": None,
        "post_contact_mean_mps": None,
    }

    if matched_idx is None:
        return record

    candidate = candidates[matched_idx]
    detected_time = float(candidate["contact_time_sec"])

    record.update(
        {
            "candidate_index": int(matched_idx),
            "detected_time_sec": detected_time,
            "delta_sec": float(detected_time - target_time),
            "matched": True,
            "post_contact_max_kmh": float(candidate.get("post_contact_max_kmh", 0.0)),
            "post_contact_mean_kmh": float(candidate.get("post_contact_mean_kmh", 0.0)),
            "post_contact_max_mps": float(candidate.get("post_contact_max_mps", 0.0)),
            "post_contact_mean_mps": float(candidate.get("post_contact_mean_mps", 0.0)),
        }
    )
    return record


def summarize_serve_attempts(
    candidates: Sequence[Dict[str, Any]],
    target_timestamps: Sequence[float],
    tolerance_sec: float,
) -> Dict[str, Any]:
    """Match ordered targets to ordered detected candidates."""
    candidate_times = [float(candidate["contact_time_sec"]) for candidate in candidates]
    matched_indices = match_targets_to_candidates(
        target_timestamps,
        candidate_times,
        tolerance_sec,
    )

    attempts = [
        _build_attempt_record(index + 1, target_time, matched_idx, candidates)
        for index, (target_time, matched_idx) in enumerate(
            zip(target_timestamps, matched_indices)
        )
    ]

    matched_candidate_indices = {idx for idx in matched_indices if idx is not None}
    unmatched_candidates = [
        dict(candidate, candidate_index=int(index))
        for index, candidate in enumerate(candidates)
        if index not in matched_candidate_indices
    ]

    return {
        "target_count": int(len(target_timestamps)),
        "candidate_count": int(len(candidates)),
        "matched_count": int(sum(1 for idx in matched_indices if idx is not None)),
        "unmatched_candidate_count": int(len(unmatched_candidates)),
        "tolerance_sec": float(tolerance_sec),
        "attempts": attempts,
        "unmatched_candidates": unmatched_candidates,
    }


# ---------------------------------------------------------------------------
# Post-hoc evaluation from files
# ---------------------------------------------------------------------------


def evaluate_from_files(
    detection_json_path: str,
    timestamps_file: str,
    tolerance_sec: float,
    source: str = "candidates",
) -> Dict[str, Any]:
    """Load detector output JSON plus timestamps file, produce match summary."""
    if source not in {"candidates", "selected_serves"}:
        raise ValueError("source must be 'candidates' or 'selected_serves'")

    with open(detection_json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        candidates = data.get(source) or []
        if source == "candidates" and not candidates:
            candidates = data.get("attempts") or []
    else:
        raise ValueError(f"Unexpected JSON structure in {detection_json_path}")

    target_times = load_target_timestamps(timestamps_file)
    summary = summarize_serve_attempts(candidates, target_times, tolerance_sec)
    summary.update(
        {
            "detection_json": str(detection_json_path),
            "timestamps_file": str(timestamps_file),
            "source": source,
            "targets_sec": [float(value) for value in target_times],
        }
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for the post-hoc evaluator."""
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluator: match detected serves to target timestamps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m serve_analyzer.serve_evaluation \\
        --detection-json candidates.json \\
        --timestamps-file timestamps_video.txt \\
        --tolerance-sec 3.0 \\
        --output evaluation.json
        """,
    )
    parser.add_argument(
        "--detection-json",
        required=True,
        help="JSON file with detected serve candidates (detector output)",
    )
    parser.add_argument(
        "--timestamps-file",
        required=True,
        help="Text file with approximate serve timestamps",
    )
    parser.add_argument(
        "--tolerance-sec",
        type=float,
        default=3.0,
        help="Max target-vs-detected delta for a match in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--source",
        choices=("candidates", "selected_serves"),
        default="candidates",
        help="Detector JSON list to evaluate (default: candidates)",
    )
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tolerance_sec < 0:
        parser.error("Tolerance must be non-negative")

    results = evaluate_from_files(
        args.detection_json,
        args.timestamps_file,
        args.tolerance_sec,
        source=args.source,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
