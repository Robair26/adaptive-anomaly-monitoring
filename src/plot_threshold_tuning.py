#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("reports/week6_threshold_tuning.csv")
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)

out_path = fig_dir / "11_lstm_threshold_tuning.png"

df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 6))
plt.plot(df["threshold_percentile"], df["detected_windows"], marker="o", label="Detected Windows")
plt.plot(df["threshold_percentile"], df["merged_events"], marker="o", label="Merged Events")
plt.title("LSTM Autoencoder Threshold Tuning")
plt.xlabel("Threshold Percentile")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved → {out_path}")
print(df.to_string(index=False))
