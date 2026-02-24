from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
# Pick a NAB CSV file (swap FILE to explore different series)
FILE = "ambient_temperature_system_failure.csv"

# Where the series lives (change to "realAWSCloudwatch" for AWS series)
SERIES_DIR = Path("data/raw/NAB/realKnownCause")

# NAB combined anomaly labels (time windows)
LABELS_PATH = Path("data/raw/NAB/labels/combined_windows.json")

# Output figure paths
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW = FIG_DIR / "01_raw_timeseries.png"
OUT_ROLLING = FIG_DIR / "02_rolling_stats.png"
OUT_LABELED = FIG_DIR / "03_anomalies_highlighted.png"


# ----------------------------
# Load data
# ----------------------------
file_path = SERIES_DIR / FILE
df = pd.read_csv(file_path)

# Parse and sort timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Loaded: {file_path} | rows={len(df):,}")


# ----------------------------
# Plot 1: Raw time series
# ----------------------------
plt.figure()
plt.plot(df["timestamp"], df["value"])
plt.title(FILE)
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(OUT_RAW, dpi=200)
plt.close()


# ----------------------------
# Plot 2: Rolling statistics
# ----------------------------
rolling_window = 50
rolling_mean = df["value"].rolling(window=rolling_window).mean()
rolling_std = df["value"].rolling(window=rolling_window).std()

plt.figure()
plt.plot(df["timestamp"], df["value"], label="Raw")
plt.plot(df["timestamp"], rolling_mean, label=f"Rolling Mean (w={rolling_window})")
plt.plot(df["timestamp"], rolling_std, label=f"Rolling Std (w={rolling_window})")
plt.legend()
plt.title(f"{FILE} - Rolling Statistics")
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(OUT_ROLLING, dpi=200)
plt.close()


# ----------------------------
# Plot 3: Raw series + labeled anomaly windows
# ----------------------------
if LABELS_PATH.exists():
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    # NAB keys are relative paths like "realKnownCause/<file>.csv"
    series_key = f"{SERIES_DIR.name}/{FILE}"
    anomaly_windows = labels.get(series_key, [])

    plt.figure()
    plt.plot(df["timestamp"], df["value"], label="Raw")

    # Shade labeled anomaly windows
    for start_str, end_str in anomaly_windows:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        plt.axvspan(start, end, alpha=0.3)

    plt.title(f"{FILE} - Labeled Anomaly Windows ({len(anomaly_windows)})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(OUT_LABELED, dpi=200)
    plt.close()

    print(f"Labeled windows: {len(anomaly_windows)} (saved {OUT_LABELED})")
else:
    print(f"WARNING: Labels file not found at {LABELS_PATH}. Skipping labeled plot.")


# ----------------------------
# Summary
# ----------------------------
print(f"Saved plots to {FIG_DIR}/:")
print(f" - {OUT_RAW.name}")
print(f" - {OUT_ROLLING.name}")
if LABELS_PATH.exists():
    print(f" - {OUT_LABELED.name}")
