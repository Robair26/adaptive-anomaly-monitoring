# Results Snapshot (NAB quick baselines)

Dataset series:
- `data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv` (7,267 rows)

## Baseline 1 — Rolling Z-Score
- Output: `figures/04_zscore_detection.png`
- Detected anomaly points (z-score): **39**
- Notes: simple statistical baseline; sensitive to window size/threshold.

## Baseline 2 — Isolation Forest (ML baseline)
- Output: `figures/05_isolation_forest_detection.png`
- Feature rows used: **7,218**
- Detected anomaly points: **73**
- Notes: unsupervised ML baseline using rolling features.

## Model 3 — LSTM Autoencoder (Deep model)
- Outputs:
  - `figures/06_lstm_ae_anomalies.png`
  - `figures/07_lstm_ae_error.png`
- Sequence length: 60
- Thresholding: train error percentile = **99.8**
- Window-level anomaly flags: **470**
- Event-level anomalies (merged, gap=2H): **6 events**
- Notes: deep reconstruction-based approach; event merging provides more operationally meaningful alerts.
