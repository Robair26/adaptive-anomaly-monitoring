from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Config (swap FILE to try other series)
# ----------------------------
FILE = "ambient_temperature_system_failure.csv"

# Choose which NAB folder to use:
SERIES_DIR = Path("data/raw/NAB/realKnownCause")
# SERIES_DIR = Path("data/raw/NAB/realAWSCloudwatch")

WINDOW = 50
THRESHOLD = 3.0

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "04_zscore_detection.png"

# ----------------------------
# Load data
# ----------------------------
file_path = SERIES_DIR / FILE
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

values = df["value"]

# ----------------------------
# Rolling z-score
# ----------------------------
rolling_mean = values.rolling(window=WINDOW, min_periods=WINDOW).mean()
rolling_std = values.rolling(window=WINDOW, min_periods=WINDOW).std()

z = (values - rolling_mean) / rolling_std
anomalies = np.abs(z) > THRESHOLD

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(df["timestamp"], values, label="Signal")

plt.scatter(
    df.loc[anomalies, "timestamp"],
    df.loc[anomalies, "value"],
    s=12,
    label=f"Anomalies (|z|>{THRESHOLD})"
)

plt.title(f"Rolling Z-Score Anomaly Detection: {SERIES_DIR.name}/{FILE}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Loaded: {file_path} | rows={len(df):,}")
print(f"Detected anomalies: {int(anomalies.sum())}")
print(f"Saved → {OUT_PATH}")
