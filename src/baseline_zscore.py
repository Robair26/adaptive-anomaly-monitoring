from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
FILE = "ambient_temperature_system_failure.csv"
SERIES_DIR = Path("data/raw/NAB/realKnownCause")

WINDOW = 50
THRESHOLD = 3.0   # classic statistical cutoff

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

OUT_PATH = FIG_DIR / "04_zscore_detection.png"

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(SERIES_DIR / FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

values = df["value"]

# ----------------------------
# Rolling statistics
# ----------------------------
rolling_mean = values.rolling(WINDOW).mean()
rolling_std = values.rolling(WINDOW).std()

z_scores = (values - rolling_mean) / rolling_std

# Identify anomalies
anomalies = np.abs(z_scores) > THRESHOLD

print(f"Detected anomalies: {anomalies.sum()}")

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(df["timestamp"], values, label="Raw")

plt.scatter(
    df["timestamp"][anomalies],
    values[anomalies],
    marker="o",
    s=10,
    label="Anomalies"
)

plt.legend()
plt.title(f"Rolling Z-Score Detection (threshold={THRESHOLD})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved → {OUT_PATH}")
