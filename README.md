# Adaptive Anomaly Monitoring (Capstone)

An end-to-end anomaly detection system for monitoring time-series infrastructure metrics. This project compares statistical, machine learning, and deep learning approaches and converts raw detections into operational alert events suitable for real-world monitoring systems.

This capstone project demonstrates a complete applied machine learning system including data processing, model training, evaluation, optimization, and system-level enhancements, meeting all required components of a full ML pipeline :contentReference[oaicite:1]{index=1}.

---

## Project Goals

The goal of this project is to:

• Build a reusable anomaly detection framework  
• Compare statistical, machine learning, and deep learning detectors  
• Convert anomaly points into alert events suitable for operational monitoring  
• Provide reproducible experiments using benchmark datasets  
• Develop a complete ML system including training, evaluation, and improvement  

---

## Dataset

This project uses the **Numenta Anomaly Benchmark (NAB)** dataset.

NAB is a widely used benchmark for evaluating anomaly detection algorithms on time-series data.

Example dataset:

ambient_temperature_system_failure.csv  

Location:

data/raw/NAB/realKnownCause/

This dataset represents temperature telemetry with a known system failure anomaly.

---

## Implemented Models

### Rolling Z-Score (Statistical Baseline)
A simple statistical detector that flags values based on deviation from a rolling mean.

---

### Isolation Forest (Machine Learning)
A tree-based anomaly detection model that isolates anomalies using random partitioning.

Features include:

• rolling statistics  
• differencing  
• EWMA smoothing  
• local z-scores  

---

### LSTM Autoencoder (Deep Learning)
A sequence-based neural network model built in PyTorch that learns to reconstruct normal time-series behavior.

Anomalies are detected using reconstruction error thresholding.

---

### Hybrid Detector (Capstone Enhancement)

A multi-model anomaly detection system combining:

• Isolation Forest  
• LSTM Autoencoder  

An anomaly is only flagged when both models agree, reducing false positives.

Additional enhancements:

• severity classification (low / medium / high)  
• confidence scoring  
• event-based prioritization  

---

## System Features

### Event-Based Detection
Anomalies are grouped into events instead of individual points using a configurable time gap (default: 2 hours).

---

### Evaluation System
Performance is evaluated using:

• Precision  
• Recall  
• F1-score  

Based on NAB ground-truth anomaly windows.

---

### Multi-Dataset Benchmarking
The system evaluates models across multiple NAB datasets:

• ambient_temperature_system_failure  
• cpu_utilization_asg_misconfiguration  
• ec2_request_latency_system_failure  
• machine_temperature_system_failure  

---

### Model Optimization (Week 6)
LSTM Autoencoder threshold tuning was performed using multiple percentiles:

• 99.5  
• 99.7  
• 99.8  
• 99.9  

This demonstrated tradeoffs between sensitivity and precision.

---

### Anomaly Scoring Module
A reusable module assigns:

• confidence score (normalized anomaly strength)  
• severity label (low / medium / high)  

This simulates real-world alert prioritization.

---

### CLI Tool (Production-Style Interface)

Run detectors using:

python3 src/detect.py --model hybrid --file data.csv

Supported models:

• zscore  
• isolation  
• lstm  
• hybrid  

---

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

---

## Pipeline Overview

The system follows this workflow:

1. Load time-series data  
2. Generate statistical features  
3. Train and apply anomaly detection models  
4. Convert anomaly points into events  
5. Evaluate against labeled anomaly windows  
6. Compare models across datasets  
7. Apply hybrid decision logic and scoring  

---

## Running the Project

Run full benchmark:

python3 src/run_all_detectors.py

Run specific model:

python3 src/detect.py --model isolation --file data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv  
python3 src/detect.py --model hybrid --file data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv  

---

## Example Output

=== Detector Comparison (event-level) ===

Detector            Series                                   Points   Flagged   % Flagged   Events   Merge Gap  
RollingZScore       ambient_temperature_system_failure       7,267    42        0.58%       22       2h  
IsolationForest     ambient_temperature_system_failure       7,243    73        1.01%       23       2h  
LSTM_Autoencoder    ambient_temperature_system_failure       7,208    429       5.95%       7        2h  

---

## Visualization

Each detector generates plots showing:

• original time series  
• anomaly detections  
• flagged anomaly regions  

Saved to:

figures/

---

## Key Insights

• Isolation Forest provides the most stable performance across datasets  
• LSTM Autoencoder achieves higher precision but requires tuning  
• Hybrid model reduces false positives while maintaining strong recall  
• Statistical methods are useful baselines but less precise  

---

## Final System Summary

This project includes all major components of a production-style ML system:

• Data ingestion and preprocessing  
• Feature engineering  
• Model training (ML + deep learning)  
• Model comparison and benchmarking  
• Evaluation using labeled data  
• Model optimization (threshold tuning)  
• Hybrid model design  
• Severity and confidence scoring  
• CLI-based execution interface  

---

## Future Improvements

• Real-time anomaly detection pipeline  
• Deployment as API or monitoring service  
• Advanced ensemble models  
• Explainability for anomaly reasoning  

---

## Author

Robair Farag  
M.S. Applied Artificial Intelligence  
University of San Diego
