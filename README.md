# Adaptive Anomaly Monitoring (Capstone)

An end-to-end anomaly detection system for monitoring time-series infrastructure metrics.
The project compares multiple anomaly detection approaches and converts raw detections into operational **alert events** suitable for monitoring systems.

This capstone focuses on building a **clean, modular anomaly detection pipeline** that can evaluate different detectors on real telemetry datasets.

---

# Project Goals

The goal of this project is to:

• Build a reusable anomaly detection framework
• Compare statistical, machine learning, and deep learning detectors
• Convert anomaly points into **alert events** suitable for operational monitoring
• Provide reproducible experiments using benchmark datasets

The system simulates how anomaly detection systems work in **real production monitoring environments**.

---

# Dataset

This project uses the **Numenta Anomaly Benchmark (NAB)** dataset.

NAB is a well-known benchmark dataset for evaluating anomaly detection algorithms on time-series data.

Example dataset used in Week 2:

ambient_temperature_system_failure.csv

Location:

data/raw/NAB/realKnownCause/

This dataset represents temperature telemetry with a known system failure anomaly.

---

# Implemented Detectors

The following anomaly detection approaches are implemented:

## 1. Rolling Z-Score (Statistical Baseline)

A simple statistical detector that flags values whose deviation from a rolling mean exceeds a threshold.

Used as a **baseline method**.

---

## 2. Isolation Forest (Machine Learning)

A tree-based anomaly detection algorithm that isolates anomalies using random partitioning.

Features used include:

• rolling statistics
• differencing
• EWMA smoothing
• local z-scores

---

## 3. LSTM Autoencoder (Deep Learning)

A sequence-based anomaly detector that learns to reconstruct normal time-series patterns.

Anomalies are detected when the **reconstruction error exceeds a learned threshold**.

---

# Project Structure

adaptive-anomaly-monitoring-capstone/

src/
  data_loader.py
  features.py
  evaluation.py
  nab_scoring.py
  baseline_zscore.py
  baseline_isolation_forest.py
  baseline_lstm_autoencoder.py
  run_all_detectors.py

data/
  raw/
    NAB/

figures/

reports/
  week2_results.md

---

# How the System Works

The anomaly detection pipeline follows these steps:

1. Load time-series data
2. Generate statistical features
3. Run anomaly detection algorithms
4. Convert anomaly points into **alert events**
5. Compare detectors using a standardized summary table

This modular architecture makes it easy to add new detectors later.

---

# Running the Project

From the project root directory run:

python3 src/run_all_detectors.py

This command will:

• run all anomaly detectors
• merge anomaly points into events
• generate a comparison table

Example output:

=== Detector Comparison (event-level) ===

| Detector         | Series                             | Points | Flagged | % Flagged | Events | Merge Gap |
| ---------------- | ---------------------------------- | ------ | ------- | --------- | ------ | --------- |
| RollingZScore    | ambient_temperature_system_failure | 7,267  | 42      | 0.58%     | 22     | 2h        |
| IsolationForest  | ambient_temperature_system_failure | 7,243  | 73      | 1.01%     | 23     | 2h        |
| LSTM_Autoencoder | ambient_temperature_system_failure | 7,208  | 429     | 5.95%     | 7      | 2h        |

---

# Event-Level Detection

Instead of treating anomalies as isolated points, the system merges nearby detections into **events**.

Detected anomaly points are grouped using a configurable time gap.

This approach better reflects how **alerting systems operate in real infrastructure monitoring environments**.

---

# Visualization

Each detector produces plots showing:

• the original time series
• anomaly detections
• flagged anomaly regions

Figures are automatically saved to:

figures/

---

# Week 2 Progress

Completed components:

• Data loading utilities
• Feature engineering pipeline
• Event merging evaluation framework
• Rolling Z-Score baseline detector
• Isolation Forest baseline detector
• LSTM Autoencoder anomaly detector
• Unified detector comparison runner

Results are documented in:

reports/week2_results.md

---

# Future Work (Week 3+)

Planned improvements include:

• NAB ground-truth label scoring (precision / recall)
• evaluation across multiple datasets
• detector hyperparameter tuning
• additional anomaly detection models
• experiment tracking and evaluation automation

---

# Author

Robair Farag
M.S. Applied Artificial Intelligence
University of San Diego
