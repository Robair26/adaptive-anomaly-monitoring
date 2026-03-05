#!/usr/bin/env python3
"""
baseline_zscore.py

Rolling Z-score baseline using standardized:
- src/data_loader.py
- src/evaluation.py

Outputs:
- figures/04_zscore_detection.png
- Console summary + merged event count
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_nab_series, pick_default_series
from src.evaluation import summarize_detection


# ----------------------------
# Config
# ----------------------------
CSV_PATH = pick_default_series()
DETECTOR_NAME = "RollingZScore"

ROLL_WINDOW = 48       # hours
ZSCORE_THRESHOLD = 3.0 # classic threshold

EVENT_GAP = "2h"

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "04_zscore_detection.png"


# ----------------------------
# Load
# ----------------------------
series = load_nab_series(CSV_PATH)
df = series.df
s = df["value"].astype(float)

# Rolling z-score (past-looking)
roll_mean = s.rolling(window=ROLL_WINDOW, min_periods=max(3, ROLL_WINDOW // 4)).mean()
roll_std = s.rolling(window=ROLL_WINDOW, min_periods=max(3, ROLL_WINDOW // 4)).std(ddof=0).replace(0.0, np.nan)
z = (s - roll_mean) / roll_std

# Flag anomalies
flags = z.abs() > ZSCORE_THRESHOLD
flags = flags.fillna(False).values  # align to array of bool

# Summarize
summary = summarize_detection(
    detector=DETECTOR_NAME,
    series=series.name,
    timestamps=df.index,
    flags=flags,
    gap=EVENT_GAP,
)

print(f"Loaded: {series.path} | rows={len(df):,}")
print(f"Detected anomalies (points): {summary.n_flagged_points}")
print(f"Merged anomaly events: {summary.n_events} (merge gap={EVENT_GAP})")


# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(df.index, s.values, label="Signal")

# Scatter anomalies
anomaly_idx = np.where(flags)[0]
plt.scatter(df.index[anomaly_idx], s.values[anomaly_idx], s=12, label="Z-score anomalies")

plt.title(f"Rolling Z-Score Detection: {series.name} (window={ROLL_WINDOW}, thr={ZSCORE_THRESHOLD})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved → {OUT_PATH}")

