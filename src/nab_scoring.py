#!/usr/bin/env python3
"""
nab_scoring.py

Lightweight NAB evaluation (Week-2 friendly):
- Load NAB labeled anomaly windows from labels/combined_windows.json
- Convert detected point/window flags into merged anomaly events (use evaluation.merge_anomaly_events)
- Score event overlap with labeled windows -> TP/FP/FN + precision/recall

This is NOT NAB's full scoring function, but it's perfect for a capstone baseline comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd


NAB_ROOT = Path("data/raw/NAB")
LABELS_FILE = NAB_ROOT / "labels" / "combined_windows.json"


def _ensure_labels_file() -> None:
    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find NAB labels file:\n  {LABELS_FILE}\n"
            "Expected NAB repo structure under data/raw/NAB.\n"
            "Check: ls data/raw/NAB/labels"
        )


def load_combined_windows() -> Dict[str, List[List[str]]]:
    _ensure_labels_file()
    with open(LABELS_FILE, "r") as f:
        data = json.load(f)
    return data


def guess_nab_key_from_csv_path(csv_path: str, *, windows_dict: Dict[str, List[List[str]]]) -> str:
    """
    NAB combined_windows.json keys look like:
      realKnownCause/ambient_temperature_system_failure.csv

    Our pipeline uses an absolute/local csv path, so we convert it into a NAB-style key.
    If conversion fails, we fall back to matching by filename suffix.
    """
    p = Path(csv_path).expanduser().resolve()

    # Try to get path relative to NAB_ROOT
    try:
        rel = p.relative_to(NAB_ROOT.resolve())
        key = rel.as_posix()
        if key in windows_dict:
            return key
    except Exception:
        pass

    # Fallback: match by suffix on filename
    fname = p.name
    candidates = [k for k in windows_dict.keys() if k.endswith("/" + fname) or k.endswith(fname)]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer keys that include parent folder name if possible
        parent = p.parent.name
        for k in candidates:
            if parent in k:
                return k
        return candidates[0]

    raise ValueError(
        f"Could not find label windows for file: {p}\n"
        f"Tried rel-to-NAB_ROOT and filename match.\n"
        f"Example keys in labels: {list(windows_dict.keys())[:5]}"
    )


def load_label_windows_for_series_key(series_key: str, *, windows_dict: Optional[Dict[str, List[List[str]]]] = None
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return list of (start, end) label windows as pandas Timestamps.
    """
    if windows_dict is None:
        windows_dict = load_combined_windows()

    if series_key not in windows_dict:
        raise ValueError(f"No label windows found for series key: {series_key}")

    windows = []
    for start_str, end_str in windows_dict[series_key]:
        windows.append((pd.Timestamp(start_str), pd.Timestamp(end_str)))
    return windows


def score_events_against_windows(
    detected_events: List[Tuple[pd.Timestamp, pd.Timestamp]],
    label_windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Dict[str, float]:
    """
    Event-level overlap scoring:
    - TP: detected event overlaps at least one labeled window
    - FP: detected event overlaps none
    - FN: labeled window not overlapped by any detected event

    Returns dict with TP/FP/FN + precision/recall.
    """
    tp = 0
    fp = 0
    matched_labels = set()

    for d_start, d_end in detected_events:
        overlapped_any = False
        for i, (l_start, l_end) in enumerate(label_windows):
            # overlap condition
            if (d_start <= l_end) and (d_end >= l_start):
                overlapped_any = True
                matched_labels.add(i)
        if overlapped_any:
            tp += 1
        else:
            fp += 1

    fn = len(label_windows) - len(matched_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp_events": float(tp),
        "fp_events": float(fp),
        "fn_windows": float(fn),
        "precision": float(precision),
        "recall": float(recall),
    }


def scoring_to_markdown_table(rows: List[Dict[str, object]]) -> str:
    """
    rows: list of dicts with keys: Detector, SeriesKey, TP, FP, FN, Precision, Recall
    """
    headers = ["Detector", "SeriesKey", "TP", "FP", "FN", "Precision", "Recall"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["Detector"]),
                    str(r["SeriesKey"]),
                    str(int(r["TP"])),
                    str(int(r["FP"])),
                    str(int(r["FN"])),
                    f'{float(r["Precision"]):.3f}',
                    f'{float(r["Recall"]):.3f}',
                ]
            )
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    d = load_combined_windows()
    print("Loaded combined_windows.json keys:", len(d))
    print("Example key:", list(d.keys())[:1])
