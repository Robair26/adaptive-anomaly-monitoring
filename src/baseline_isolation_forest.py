#!/usr/bin/env python3
"""
baseline_isolation_forest.py

Isolation Forest baseline using standardized:
- src/data_loader.py
- src/features.py
- src/evaluation.py

Outputs:
- figures/05_isolation_forest_detection.png
- Console summary + event count
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from src.data_loader import load_nab_series, pick_default_series
from src.features import make_features, FeatureConfig
from src.evaluation import summarize_detection


# ----------------------------
# Config
# ----------------------------
CSV_PATH = pick_default_series()  # change here if you want a different series
DETECTOR_NAME = "IsolationForest"

CONTAMINATION = 0.01
N_ESTIMATORS = 300
RANDOM_STATE = 42

EVENT_GAP = "2h"   # hourly data -> merge into events across 2 hours

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "05_isolation_forest_detection.png"


# ----------------------------
# Load + features
# ----------------------------
series = load_nab_series(CSV_PATH)
df = series.df

cfg = FeatureConfig(
    rolling_window=24,
    ewma_span=24,
    diff_lags=(1, 2, 6, 24),
    zscore_window=48,
)
Xdf = make_features(df, cfg=cfg)

# Keep timestamps + raw values aligned to feature rows
ts = Xdf.index
vals = Xdf["value"].values

# Model input: all engineered features except the raw "value" column is okay to keep too.
X = Xdf.values

# Standardize features (simple z-normalization)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


# ----------------------------
# Fit + predict
# ----------------------------
model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
)
model.fit(X)

pred = model.predict(X)  # -1 anomaly, +1 normal
flags = pred == -1


# ----------------------------
# Summarize (events)
# ----------------------------
summary = summarize_detection(
    detector=DETECTOR_NAME,
    series=series.name,
    timestamps=ts,
    flags=flags,
    gap=EVENT_GAP,
)

print(f"Loaded: {series.path} | rows={len(df):,}")
print(f"Feature rows used: {len(Xdf):,}")
print(f"Detected anomalies (points): {summary.n_flagged_points}")
print(f"Merged anomaly events: {summary.n_events} (merge gap={EVENT_GAP})")


# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(ts, vals, label="Signal")
plt.scatter(ts[flags], vals[flags], s=12, label="Isolation Forest anomalies")
plt.title(f"Isolation Forest Detection: {series.name} (contamination={CONTAMINATION})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved → {OUT_PATH}")
