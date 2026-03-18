# Experiment Plan – Adaptive Anomaly Monitoring

## Objective

The objective of this experiment is to evaluate different anomaly detection approaches for identifying abnormal behavior in telemetry-style time-series data.

The study compares three categories of anomaly detection methods:

1. Statistical baseline  
2. Classical machine learning  
3. Deep learning  

This layered approach allows us to analyze how increasing model complexity affects anomaly detection performance.

---

## Dataset

The experiments use the **Numenta Anomaly Benchmark (NAB)** dataset.

Example dataset used in initial experiments:

data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv

This dataset represents a real-world telemetry signal containing temperature measurements from a system sensor.

Properties:

• Univariate time-series  
• Hourly observations  
• 7,267 data points  
• Contains known anomaly periods  

The NAB dataset is commonly used to evaluate anomaly detection algorithms on streaming telemetry signals.

---

## Feature Engineering

Feature engineering is applied before running machine learning models.

Generated features include:

- Rolling mean  
- Rolling standard deviation  
- Rolling minimum  
- Rolling maximum  
- Exponentially weighted moving average (EWMA)  
- First-order differences  
- Absolute differences  
- Lagged differences  
- Rolling Z-score  

These features capture local temporal structure in the signal.

---

## Models Evaluated

### 1. Rolling Z-Score (Statistical Baseline)

A rolling statistical detector that flags values whose deviation from the rolling mean exceeds a fixed threshold.

The anomaly score is computed using the Z-score formula:

z = (x − μ) / σ

Where:

x = observed value  
μ = rolling mean  
σ = rolling standard deviation  

Observations with |z| > 3 are flagged as anomalies.

Implementation:

src/baseline_zscore.py

---

### 2. Isolation Forest

Isolation Forest is an unsupervised anomaly detection algorithm that isolates anomalies through random partitioning of the feature space.

Implementation:

src/baseline_isolation_forest.py

The model is trained using engineered features derived from the time-series.

Isolation Forest works well for anomaly detection because anomalous observations require fewer splits to isolate.

---

### 3. LSTM Autoencoder

The deep learning model used in this project is an **LSTM Autoencoder**.

Implementation:

src/model_lstm_autoencoder.py

The model learns to reconstruct normal temporal sequences.

Training steps:

1. Extract sliding windows from the time-series  
2. Train the LSTM autoencoder using reconstruction loss  
3. Compute reconstruction error for each window  
4. Flag windows whose error exceeds a chosen percentile threshold  

Because LSTM networks capture sequential dependencies, they are well suited for time-series anomaly detection.

---

## Evaluation Strategy

Each detector produces anomaly points that are converted into **anomaly events**.

Nearby anomalies are merged using a configurable merge gap.

Current merge gap:

2 hours

This approach produces event-level anomaly detections that better represent real monitoring alerts.

---

## Experiment Pipeline

All detectors are executed through a unified pipeline:

python src/run_all_detectors.py

Pipeline steps:

1. Load dataset  
2. Generate features  
3. Run anomaly detectors  
4. Merge anomaly points into events  
5. Generate comparison summary  

---

## Evaluation Metrics

Models are compared using:

• Number of anomaly points detected  
• Percentage of timeline flagged as anomalous  
• Number of anomaly events  

Future evaluation will include precision and recall using labeled anomaly windows from the NAB dataset.

---

## Current Results (Example Dataset)

Dataset: ambient_temperature_system_failure

| Detector | Points | % Flagged | Events |
|---------|--------|----------|-------|
| Rolling Z-Score | 42 | 0.58% | 22 |
| Isolation Forest | 73 | 1.01% | 23 |
| LSTM Autoencoder | 429 | 5.95% | 7 |

These results demonstrate how different model types vary in anomaly sensitivity and detection behavior.

---

## Project Implementation Status

The anomaly detection system currently includes:

• Data loading and preprocessing pipeline  
• Feature engineering module  
• Rolling Z-score baseline detector  
• Isolation Forest machine learning detector  
• LSTM Autoencoder deep learning model  
• Experiment runner for detector comparison  
• Visualization outputs for anomaly detection  

The project architecture supports further experiments across multiple NAB datasets and additional anomaly detection models.
