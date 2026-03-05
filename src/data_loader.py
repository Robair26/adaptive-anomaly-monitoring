#!/usr/bin/env python3
"""
data_loader.py

Single, consistent loader for NAB time-series CSV files.

Expected NAB CSV format (typical):
- timestamp,value
- timestamp column is parseable by pandas
- value is numeric

This loader standardizes:
- timestamp parsing
- sorting
- index as DatetimeIndex
- column name normalized to "value"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class LoadedSeries:
    """Container for a loaded time-series."""
    name: str
    path: str
    df: pd.DataFrame  # indexed by timestamp, single column: "value"


def load_nab_series(
    csv_path: str,
    *,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    tz: Optional[str] = None,
) -> LoadedSeries:
    """
    Load a NAB time-series CSV file into a standardized DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to NAB CSV file.
    timestamp_col : str
        Name of the timestamp column.
    value_col : str
        Name of the value column.
    tz : Optional[str]
        If provided, localize timestamps to this timezone.

    Returns
    -------
    LoadedSeries
        Loaded series container: {name, path, df(index=timestamp, col=value)}.
    """
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)

    if timestamp_col not in df.columns:
        raise ValueError(
            f"Expected timestamp column '{timestamp_col}' not found. "
            f"Columns: {list(df.columns)}"
        )
    if value_col not in df.columns:
        # Some datasets might use a different value column name.
        # If there's only 2 columns total, assume the non-timestamp column is value.
        non_ts_cols = [c for c in df.columns if c != timestamp_col]
        if len(non_ts_cols) == 1:
            inferred = non_ts_cols[0]
            df = df.rename(columns={inferred: "value"})
        else:
            raise ValueError(
                f"Expected value column '{value_col}' not found and could not infer it. "
                f"Columns: {list(df.columns)}"
            )
    else:
        df = df.rename(columns={value_col: "value"})

    # Parse timestamps
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    if df[timestamp_col].isna().any():
        bad = df[df[timestamp_col].isna()].head(5)
        raise ValueError(
            "Found unparsable timestamps. Example rows:\n"
            f"{bad.to_string(index=False)}"
        )

    # Set index and sort
    df = df.set_index(timestamp_col).sort_index()

    # Optional timezone localization
    if tz is not None:
        # If timestamps are naive, localize; if aware, convert
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    # Ensure numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    name = p.stem
    return LoadedSeries(name=name, path=str(p), df=df[["value"]].copy())


def pick_default_series() -> str:
    """
    Returns a default NAB series path used throughout quick experiments.
    Adjust here if you want to standardize the project on a different series.
    """
    return "data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv"


if __name__ == "__main__":
    # Quick sanity test
    path = pick_default_series()
    series = load_nab_series(path)
    print(f"Loaded: {series.path}")
    print(f"Name: {series.name}")
    print(f"Rows: {len(series.df):,}")
    print(series.df.head())
    print(series.df.tail())
