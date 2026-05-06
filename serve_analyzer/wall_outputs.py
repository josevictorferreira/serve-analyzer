"""Wall-serve analysis output contracts.

Builds the JSON payload (sections measured/inferred/assumed/confidence/warnings/artifacts),
serializes per-serve CSV rows using the LOCKED ``CSV_COLUMNS`` from ``wall_calibration.py``,
and defines deterministic plot artifact names.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from serve_analyzer.wall_calibration import CSV_COLUMNS


# ---------------------------------------------------------------------------
# Deterministic plot artifact names
# ---------------------------------------------------------------------------

PLOT_FILENAMES: Dict[str, str] = {
    "speed": "{video_stem}_serve{idx:02d}_speed.png",
    "trajectory": "{video_stem}_serve{idx:02d}_trajectory.png",
    "wall_impact": "{video_stem}_serve{idx:02d}_wall_impact.png",
}
"""Deterministic filename templates for plot artifacts."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class WallAnalysisResult:
    """Container for a single wall-serve analysis run.

    Attributes:
        measured: Directly observed quantities (e.g. impact_time, wall coords).
        inferred: Derived quantities (e.g. speed, landing position).
        assumed: Defaults or fallback values used during analysis.
        confidence: Numeric confidence scores per serve or per field.
        warnings: List of warning code strings.
        artifacts: Dict mapping artifact type to file path(s).
    """

    measured: Dict[str, Any] = field(default_factory=dict)
    inferred: Dict[str, Any] = field(default_factory=dict)
    assumed: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

_JSON_SECTIONS = frozenset(
    {"measured", "inferred", "assumed", "confidence", "warnings", "artifacts"}
)


def to_json(result: WallAnalysisResult) -> Dict[str, Any]:
    """Return a JSON-serializable dict with exactly the 6 required sections.

    Args:
        result: A ``WallAnalysisResult`` instance.

    Returns:
        Dict with keys ``measured``, ``inferred``, ``assumed``, ``confidence``,
        ``warnings``, ``artifacts`` — no extras.
    """
    return {
        "measured": result.measured,
        "inferred": result.inferred,
        "assumed": result.assumed,
        "confidence": result.confidence,
        "warnings": list(result.warnings),
        "artifacts": result.artifacts,
    }


# ---------------------------------------------------------------------------
# CSV serialization
# ---------------------------------------------------------------------------

# Build a lookup so we know the column index for each field.
_CSV_COL_INDEX: Dict[str, int] = {col: idx for idx, col in enumerate(CSV_COLUMNS)}


def _flatten_warning_codes(codes: List[str]) -> str:
    """Join warning codes deterministically (sorted, semicolon-delimited).

    The result is a plain string — no JSON braces, no nested structures.

    Args:
        codes: Unordered list of warning code strings.

    Returns:
        A semicolon-joined string of codes in sorted order.
    """
    return ";".join(sorted(codes))


def serve_to_csv_row(serve: Dict[str, Any]) -> Tuple[Any, ...]:
    """Flatten one serve dict into a tuple matching ``CSV_COLUMNS`` order.

    The ``serve`` dict is expected to contain at minimum the keys that map to
    ``CSV_COLUMNS`` (minus ``warning_codes`` which is handled separately via
    ``serve["warnings"]`` or ``serve["warning_codes"]``).  Missing keys are
    filled with ``None``.

    Args:
        serve: Dict with serve-level data.  Warning codes may be under
            ``"warnings"`` (list) or ``"warning_codes"`` (list or str).

    Returns:
        Tuple of values in ``CSV_COLUMNS`` order.
    """
    row: List[Any] = [None] * len(CSV_COLUMNS)

    for col_name in CSV_COLUMNS:
        if col_name == "warning_codes":
            # Pull from "warnings" or "warning_codes", then flatten.
            raw_codes = serve.get("warnings", serve.get("warning_codes", []))
            if isinstance(raw_codes, str):
                raw_codes = [raw_codes] if raw_codes else []
            row[_CSV_COL_INDEX[col_name]] = _flatten_warning_codes(raw_codes)
            continue

        row[_CSV_COL_INDEX[col_name]] = serve.get(col_name)

    return tuple(row)


def write_csv(path: str | Path, rows: List[Tuple[Any, ...]]) -> None:
    """Write CSV with header from ``CSV_COLUMNS`` followed by data rows.

    Args:
        path: Destination file path.
        rows: List of tuples, each matching ``CSV_COLUMNS`` length and order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
