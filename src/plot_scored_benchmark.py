#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path("reports/multi_dataset_scored_results.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRECISION = FIG_DIR / "08_precision_by_dataset.png"
OUT_RECALL = FIG_DIR / "09_recall_by_dataset.png"
OUT_AVG = FIG_DIR / "10_average_precision_recall.png"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing scored results file: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Keep dataset names compact
df["dataset"] = df["series"].astype(str)

# -------- Plot 1: Precision by dataset --------
precision_pivot = df.pivot(index="dataset", columns="detector", values="precision")
ax = precision_pivot.plot(kind="bar", figsize=(12, 6))
ax.set_title("Precision by Detector Across NAB Datasets")
ax.set_xlabel("Dataset")
ax.set_ylabel("Precision")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_PRECISION, dpi=200)
plt.close()

# -------- Plot 2: Recall by dataset --------
recall_pivot = df.pivot(index="dataset", columns="detector", values="recall")
ax = recall_pivot.plot(kind="bar", figsize=(12, 6))
ax.set_title("Recall by Detector Across NAB Datasets")
ax.set_xlabel("Dataset")
ax.set_ylabel("Recall")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_RECALL, dpi=200)
plt.close()

# -------- Plot 3: Average precision and recall --------
avg_df = (
    df.groupby("detector")[["precision", "recall"]]
    .mean()
    .reset_index()
    .set_index("detector")
)
ax = avg_df.plot(kind="bar", figsize=(10, 6))
ax.set_title("Average Precision and Recall by Detector")
ax.set_xlabel("Detector")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUT_AVG, dpi=200)
plt.close()

print("Saved:")
print(f" - {OUT_PRECISION}")
print(f" - {OUT_RECALL}")
print(f" - {OUT_AVG}")

print("\nAverage scores:")
print(avg_df.round(3).to_string())
