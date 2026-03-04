from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ----------------------------
# Config
# ----------------------------
FILE = "ambient_temperature_system_failure.csv"
SERIES_DIR = Path("data/raw/NAB/realKnownCause")
# SERIES_DIR = Path("data/raw/NAB/realAWSCloudwatch")

WINDOW = 50                    # rolling window for feature creation
CONTAMINATION = 0.01           # expected anomaly fraction (tune later)
RANDOM_STATE = 42

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIG_DIR / "05_isolation_forest_detection.png"

# ----------------------------
# Load data
# ----------------------------
file_path = SERIES_DIR / FILE
df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

values = df["value"].astype(float)

# ----------------------------
# Feature engineering (simple, effective)
# ----------------------------
# Rolling stats as features
df["roll_mean"] = values.rolling(WINDOW, min_periods=WINDOW).mean()
df["roll_std"]  = values.rolling(WINDOW, min_periods=WINDOW).std()
df["diff_1"]    = values.diff()

feat = df[["value", "roll_mean", "roll_std", "diff_1"]].dropna().copy()

# Align timestamps/values to feature rows
df_feat = df.loc[feat.index, ["timestamp", "value"]].copy()

# Normalize features (basic standardization)
X = feat.values
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

# ----------------------------
# Isolation Forest
# ----------------------------
model = IsolationForest(
    n_estimators=300,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
)
model.fit(X)

# Decision function: higher = more normal, lower = more anomalous
scores = model.decision_function(X)
pred = model.predict(X)  # -1 anomaly, +1 normal
anomalies = pred == -1

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(df_feat["timestamp"], df_feat["value"], label="Signal")

plt.scatter(
    df_feat.loc[anomalies, "timestamp"],
    df_feat.loc[anomalies, "value"],
    s=12,
    label="Isolation Forest anomalies"
)

plt.title(f"Isolation Forest Detection: {SERIES_DIR.name}/{FILE} (contamination={CONTAMINATION})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Loaded: {file_path} | rows={len(df):,}")
print(f"Feature rows used: {len(df_feat):,}")
print(f"Detected anomalies: {int(anomalies.sum())}")
print(f"Saved → {OUT_PATH}")
