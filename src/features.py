#!/usr/bin/env python3
"""
features.py

Feature engineering for univariate time-series anomaly detection.

Produces a feature matrix aligned to timestamps, suitable for classical ML models
(e.g., Isolation Forest) and for diagnostics.

Design goals:
- Simple, fast, reproducible
- Works on a single "value" column DataFrame indexed by timestamp
- Avoids leakage by using only past/rolling windows (no centered windows)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    rolling_window: int = 24          # hours (since NAB series is hourly)
    ewma_span: int = 24               # exponential moving average span
    diff_lags: tuple[int, ...] = (1, 2, 6, 24)
    zscore_window: int = 48           # rolling window for rolling zscore feature


def _safe_rolling_std(s: pd.Series, window: int) -> pd.Series:
    # ddof=0 for numerical stability (and to reduce NaNs early)
    return s.rolling(window=window, min_periods=max(3, window // 4)).std(ddof=0)


def make_features(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Create a feature dataframe from a time-series dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by timestamp, contains `value_col`.
    value_col : str
        Column name of the series values.
    cfg : FeatureConfig
        Feature configuration.

    Returns
    -------
    pd.DataFrame
        Feature dataframe indexed by timestamp. Rows with NaNs are dropped.
    """
    if cfg is None:
        cfg = FeatureConfig()

    if value_col not in df.columns:
        raise ValueError(f"Expected column '{value_col}'. Found: {list(df.columns)}")

    s = df[value_col].astype(float)

    feat = pd.DataFrame(index=df.index)

    # Base
    feat["value"] = s

    # Rolling statistics (past-looking)
    feat["roll_mean"] = s.rolling(window=cfg.rolling_window, min_periods=max(3, cfg.rolling_window // 4)).mean()
    feat["roll_std"] = _safe_rolling_std(s, cfg.rolling_window)
    feat["roll_min"] = s.rolling(window=cfg.rolling_window, min_periods=max(3, cfg.rolling_window // 4)).min()
    feat["roll_max"] = s.rolling(window=cfg.rolling_window, min_periods=max(3, cfg.rolling_window // 4)).max()

    # Exponential moving average (reacts quicker to change)
    feat["ewma"] = s.ewm(span=cfg.ewma_span, adjust=False).mean()

    # Diffs / deltas at multiple lags (captures sudden change & drift)
    for lag in cfg.diff_lags:
        feat[f"diff_{lag}"] = s.diff(lag)
        feat[f"absdiff_{lag}"] = s.diff(lag).abs()

    # Rolling z-score as a feature (not a detector)
    rz_mean = s.rolling(window=cfg.zscore_window, min_periods=max(3, cfg.zscore_window // 4)).mean()
    rz_std = _safe_rolling_std(s, cfg.zscore_window).replace(0.0, np.nan)
    feat["roll_z"] = (s - rz_mean) / rz_std

    # Clean
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    return feat


if __name__ == "__main__":
    # Quick test using the default series
    from src.data_loader import load_nab_series, pick_default_series

    series = load_nab_series(pick_default_series())
    X = make_features(series.df)
    print("Feature rows:", len(X))
    print("Columns:", list(X.columns))
    print(X.head())
