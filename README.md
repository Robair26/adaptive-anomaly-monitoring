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

Rolling Z-Score (Statistical Baseline)  
A simple statistical detector that flags values based on deviation from a rolling mean. Used as a baseline method.

Isolation Forest (Machine Learning)  
A tree-based anomaly detection algorithm that isolates anomalies using random partitioning.

Features include:

• rolling statistics  
• differencing  
• EWMA smoothing  
• local z-scores  

LSTM Autoencoder (Deep Learning)  
A sequence-based anomaly detector that learns to reconstruct normal time-series behavior.  
Anomalies are detected when reconstruction error exceeds a learned threshold.

Hybrid Detector (Advanced Capstone Enhancement)  
A combined anomaly detection approach using both Isolation Forest and LSTM Autoencoder.  
An anomaly is flagged only when both models agree, reducing false positives and improving precision.

Additional capabilities:

• anomaly severity classification (low / medium / high)  
• confidence scoring based on normalized anomaly scores  

## Additional System Features

Event-Level Detection  
Instead of treating anomalies as isolated points, detections are grouped into events using a configurable time gap (default: 2 hours).

NAB Label Scoring  
The system evaluates performance using:

• Precision  
• Recall  
• F1-score  

Evaluation is performed against ground-truth anomaly windows from NAB.

Anomaly Scoring Module  
A reusable scoring module assigns:

• severity (low / medium / high)  
• confidence (normalized anomaly score)  

This enables prioritization of alerts in monitoring systems.

CLI Tool (Production-Style Interface)

Run any detector using:

python3 src/detect.py --model hybrid --file data.csv

Supported models:

• zscore  
• isolation  
• lstm  
• hybrid  

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
    hybrid_detector.py  
    anomaly_scoring.py  
    detect.py  
    run_all_detectors.py  

data/  
    raw/  
        NAB/  

figures/  

reports/  
    week5_experimental_methods.md  
    week6_model_improvement.md  
    benchmark_insights.md  
    multi_dataset_scored_results.md  
    hybrid_detector_results.md  

## How the System Works

The anomaly detection pipeline follows these steps:

1. Load time-series data  
2. Generate statistical features  
3. Run anomaly detection algorithms  
4. Convert anomaly points into alert events  
5. Score results against ground-truth labels  
6. Compare detectors using standardized metrics  

This modular design allows easy extension and experimentation.

## Running the Project

Run full benchmark:

python3 src/run_all_detectors.py

Run individual models:

python3 src/detect.py --model isolation --file data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv  
python3 src/detect.py --model hybrid --file data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv  

## Example Output

=== Detector Comparison (event-level) ===

Detector            Series                                   Points   Flagged   % Flagged   Events   Merge Gap  
RollingZScore       ambient_temperature_system_failure       7,267    42        0.58%       22       2h  
IsolationForest     ambient_temperature_system_failure       7,243    73        1.01%       23       2h  
LSTM_Autoencoder    ambient_temperature_system_failure       7,208    429       5.95%       7        2h  

## Visualization

Each detector produces plots showing:

• original time series  
• anomaly detections  
• flagged anomaly regions  

Figures are automatically saved to:

figures/

## Key Insights

• Isolation Forest provides stable high-recall performance across datasets  
• LSTM Autoencoder achieves higher precision but requires tuning  
• Hybrid detector reduces false positives by combining models  
• Rolling Z-Score is useful as a baseline but produces more false positives  

## Current Status

Completed components:

• Data pipeline and preprocessing  
• Feature engineering system  
• Three anomaly detection models  
• Hybrid anomaly detection system  
• Event-based evaluation framework  
• NAB label scoring (precision / recall / F1)  
• Multi-dataset benchmarking  
• Threshold tuning experiment  
• Anomaly severity and confidence scoring  
• CLI interface for running models  

## Future Work

• Further hyperparameter tuning  
• Hybrid model improvements  
• Real-time anomaly detection pipeline  
• Explainability for anomaly reasoning  
• Deployment as an API or monitoring service  

## Author

Robair Farag  
M.S. Applied Artificial Intelligence  
University of San Diego
