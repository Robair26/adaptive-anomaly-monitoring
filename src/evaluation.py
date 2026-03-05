#!/usr/bin/env python3
"""
evaluation.py

Utilities to:
- merge point-level anomaly flags into anomaly events
- compute quick summary metrics (counts + % flagged)
- standardize outputs across detectors

This is intentionally lightweight (Week-2 friendly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionSummary:
    detector: str
    series: str
    n_points: int
    n_flagged_points: int
    pct_flagged_points: float
    n_events: int
    event_gap: str


def merge_anomaly_events(
    timestamps: pd.Series | pd.DatetimeIndex,
    flags: np.ndarray,
    *,
    gap: str = "2h",
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Merge anomaly flags into contiguous "events" based on a time gap.

    If anomalies occur within `gap` of each other, they are grouped into the same event.

    Returns list of (event_start, event_end).
    """
    ts = pd.to_datetime(timestamps)
    if isinstance(ts, pd.DatetimeIndex):
        ts = pd.Series(ts)

    ts = ts.reset_index(drop=True)
    flags = np.asarray(flags, dtype=bool)

    if len(ts) != len(flags):
        raise ValueError(f"timestamps length {len(ts)} != flags length {len(flags)}")

    flagged = ts[flags].reset_index(drop=True)
    if len(flagged) == 0:
        return []

    gap_td = pd.Timedelta(gap)  # use lowercase units, e.g. "2h"
    events: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    start = flagged.iloc[0]
    prev = flagged.iloc[0]

    for t in flagged.iloc[1:]:
        if (t - prev) <= gap_td:
            prev = t
        else:
            events.append((start, prev))
            start = t
            prev = t

    events.append((start, prev))
    return events


def summarize_detection(
    *,
    detector: str,
    series: str,
    timestamps: pd.Series | pd.DatetimeIndex,
    flags: np.ndarray,
    gap: str = "2h",
) -> DetectionSummary:
    """
    Build a standardized summary for a detector's output.
    """
    flags = np.asarray(flags, dtype=bool)
    n_points = len(flags)
    n_flagged = int(flags.sum())
    pct = (n_flagged / n_points) * 100.0 if n_points > 0 else 0.0
    events = merge_anomaly_events(timestamps, flags, gap=gap)

    return DetectionSummary(
        detector=detector,
        series=series,
        n_points=n_points,
        n_flagged_points=n_flagged,
        pct_flagged_points=pct,
        n_events=len(events),
        event_gap=gap,
    )


def summary_to_markdown_table(summaries: list[DetectionSummary]) -> str:
    """
    Convert summaries into a simple markdown table (for README / reports).
    """
    headers = [
        "Detector", "Series", "Points", "Flagged", "% Flagged", "Events", "Merge Gap"
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for s in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    s.detector,
                    s.series,
                    f"{s.n_points:,}",
                    f"{s.n_flagged_points:,}",
                    f"{s.pct_flagged_points:.2f}%",
                    f"{s.n_events:,}",
                    s.event_gap,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test: create fake flags and verify event merging
    ts = pd.date_range("2024-01-01", periods=10, freq="H")
    flags = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=bool)

    events_2h = merge_anomaly_events(ts, flags, gap="2h")
    print("Events (2h gap):", events_2h)

    s = summarize_detection(detector="test", series="demo", timestamps=ts, flags=flags, gap="2h")
    print(s)
    print(summary_to_markdown_table([s]))
