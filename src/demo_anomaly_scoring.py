#!/usr/bin/env python3

import pandas as pd
from src.anomaly_scoring import add_anomaly_scores

# simulate anomaly scores
data = {
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
    "anomaly_score": [0.1, 0.2, 0.3, 0.9, 1.2, 0.5, 0.6, 1.5, 0.2, 0.1]
}

df = pd.DataFrame(data)

scored = add_anomaly_scores(df, "anomaly_score")

print("\n=== Scored Anomalies ===\n")
print(scored)
