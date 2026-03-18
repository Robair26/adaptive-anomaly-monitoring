#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV = REPORTS_DIR / "multi_dataset_scored_results.csv"
OUT_CSV = REPORTS_DIR / "multi_dataset_scored_results_with_f1.csv"
OUT_MD = REPORTS_DIR / "multi_dataset_scored_results_with_f1.md"
OUT_BEST = REPORTS_DIR / "best_detector_by_dataset.md"

if not IN_CSV.exists():
    raise FileNotFoundError(f"Missing input file: {IN_CSV}")

df = pd.read_csv(IN_CSV)

def compute_f1(row):
    p = float(row["precision"])
    r = float(row["recall"])
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

df["f1"] = df.apply(compute_f1, axis=1)

df.to_csv(OUT_CSV, index=False)

md_lines = ["# Multi-Dataset Scored Benchmark Results (with F1)", ""]
for series_name, subdf in df.groupby("series"):
    md_lines.append(f"## {series_name}")
    md_lines.append("")
    md_lines.append("| Detector | Precision | Recall | F1 |")
    md_lines.append("|---|---:|---:|---:|")
    for _, r in subdf.iterrows():
        md_lines.append(
            f"| {r['detector']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    md_lines.append("")

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

best_rows = []
for series_name, subdf in df.groupby("series"):
    best = subdf.sort_values(["f1", "precision"], ascending=False).iloc[0]
    best_rows.append(best)

best_df = pd.DataFrame(best_rows)

best_lines = ["# Best Detector by Dataset", ""]
best_lines.append("| Dataset | Best Detector | Precision | Recall | F1 |")
best_lines.append("|---|---|---:|---:|---:|")

for _, r in best_df.iterrows():
    best_lines.append(
        f"| {r['series']} | {r['detector']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
    )

avg_df = (
    df.groupby("detector")[["precision", "recall", "f1"]]
    .mean()
    .reset_index()
    .sort_values("f1", ascending=False)
)

best_lines.append("")
best_lines.append("## Average Scores")
best_lines.append("")
best_lines.append("| Detector | Precision | Recall | F1 |")
best_lines.append("|---|---:|---:|---:|")

for _, r in avg_df.iterrows():
    best_lines.append(
        f"| {r['detector']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
    )

OUT_BEST.write_text("\n".join(best_lines), encoding="utf-8")

print("\nSaved:")
print(OUT_CSV)
print(OUT_MD)
print(OUT_BEST)

print("\nBest detectors:")
print(best_df[["series", "detector", "precision", "recall", "f1"]])
