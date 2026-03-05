# Week 2 Results Snapshot

## Detector Comparison (event-level)

| Detector | Series | Points | Flagged | % Flagged | Events | Merge Gap |
|---|---|---|---|---|---|---|
| RollingZScore | ambient_temperature_system_failure | 7,267 | 42 | 0.58% | 22 | 2h |
| IsolationForest | ambient_temperature_system_failure | 7,243 | 73 | 1.01% | 23 | 2h |
| LSTM_Autoencoder | ambient_temperature_system_failure | 7,208 | 429 | 5.95% | 7 | 2h |

Series path: `data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv`

## Key Week 2 Deliverables
- Standardized data loading (`src/data_loader.py`) to support reproducible experiments across NAB files.
- Implemented feature engineering for univariate telemetry (`src/features.py`) to support classical ML detectors.
- Implemented event merging + standardized summaries (`src/evaluation.py`) so detectors can be compared operationally (alerts/events instead of raw points).
- Built and compared three detectors:
  - Rolling Z-Score baseline
  - Isolation Forest baseline (feature-based)
  - LSTM Autoencoder (sequence reconstruction)
- Created a single runner (`src/run_all_detectors.py`) that prints a report-ready comparison table in one command.

## Notes
- NAB label-based precision/recall scoring is deferred until `combined_windows.json` is added under:
  `data/raw/NAB/labels/combined_windows.json`.
  The code is designed to skip label scoring gracefully until labels are available.
