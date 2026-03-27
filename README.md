# Adaptive Anomaly Monitoring (Capstone)

An end-to-end anomaly detection system for monitoring time-series infrastructure metrics. This project compares statistical, machine learning, and deep learning approaches and converts raw detections into operational alert events suitable for real-world monitoring systems.

This capstone focuses on building a clean, modular anomaly detection pipeline capable of evaluating multiple detectors on real telemetry datasets.

## Project Goals

The goal of this project is to:

• Build a reusable anomaly detection framework  
• Compare statistical, machine learning, and deep learning detectors  
• Convert anomaly points into alert events suitable for operational monitoring  
• Provide reproducible experiments using benchmark datasets  

The system simulates how anomaly detection systems operate in real production environments.

## Dataset

This project uses the Numenta Anomaly Benchmark (NAB) dataset.

NAB is a widely used benchmark for evaluating anomaly detection algorithms on time-series data.

Example dataset:

ambient_temperature_system_failure.csv

Location:

data/raw/NAB/realKnownCause/

This dataset represents temperature telemetry with a known system failure anomaly.

## Implemented Detectors

1. Rolling Z-Score (Statistical Baseline)  
A simple statistical detector that flags values based on deviation from a rolling mean. Used as a baseline method.

2. Isolation Forest (Machine Learning)  
A tree-based anomaly detection algorithm that isolates anomalies using random partitioning.

Features include:

• rolling statistics  
• differencing  
• EWMA smoothing  
• local z-scores  

3. LSTM Autoencoder (Deep Learning)  
A sequence-based anomaly detector that learns to reconstruct normal time-series behavior.

Anomalies are detected when reconstruction error exceeds a learned threshold.

## Project Structure

adaptive-anomaly-monitoring-capstone/

src/  
    data_loader.py  
    features.py  
    evaluation.py  
    nab_scoring.py  
    baseline_zscore.py  
    baseline_isolation_forest.py  
    model_lstm_autoencoder.py  
    run_all_detectors.py  

data/  
    raw/  
        NAB/  

figures/  

reports/  
    week5_experimental_methods.md  
    benchmark_insights.md  
    multi_dataset_scored_results.md  

## How the System Works

The anomaly detection pipeline follows these steps:

1. Load time-series data  
2. Generate statistical features  
3. Run anomaly detection algorithms  
4. Convert anomaly points into alert events  
5. Compare detectors using a standardized summary table  

This modular design allows new detectors to be added easily.

## Running the Project

From the project root directory:

python3 src/run_all_detectors.py

This will:

• run all anomaly detectors  
• merge anomaly points into events  
• generate a comparison table  

## Example Output

=== Detector Comparison (event-level) ===

Detector            Series                                   Points   Flagged   % Flagged   Events   Merge Gap  
RollingZScore       ambient_temperature_system_failure       7,267    42        0.58%       22       2h  
IsolationForest     ambient_temperature_system_failure       7,243    73        1.01%       23       2h  
LSTM_Autoencoder    ambient_temperature_system_failure       7,208    429       5.95%       7        2h  

## Event-Level Detection

Instead of treating anomalies as isolated points, detections are grouped into events.

Nearby anomaly points are merged using a configurable time gap (default: 2 hours).

This reflects how real monitoring systems generate alerts.

## Visualization

Each detector generates plots showing:

• original time series  
• anomaly detections  
• flagged anomaly regions  

Figures are saved to:

figures/

## Current Progress

Completed components:

• Data loading pipeline  
• Feature engineering  
• Event-based evaluation framework  
• Rolling Z-Score baseline  
• Isolation Forest baseline  
• LSTM Autoencoder (deep learning model)  
• Multi-model comparison system  
• Precision / Recall / F1 scoring  
• Multi-dataset benchmarking  

## Key Insights

• Isolation Forest provides stable high-recall performance  
• LSTM Autoencoder achieves higher precision but requires tuning  
• Rolling Z-Score is useful as a baseline but produces more false positives  

## Future Work

• Further model tuning and optimization  
• Hybrid anomaly detection models  
• Real-time streaming support  
• Explainability layer for anomaly reasoning  

## Author

Robair Farag  
M.S. Applied Artificial Intelligence  
University of San Diego
