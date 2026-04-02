# Adaptive Anomaly Monitoring (Capstone)

An end-to-end anomaly detection system for monitoring time-series infrastructure metrics. This project compares statistical, machine learning, and deep learning approaches and converts raw detections into operational alert events suitable for real-world monitoring systems.

This capstone demonstrates a complete machine learning pipeline including data processing, model training, evaluation, optimization, and system-level enhancements.

---

## Project Goals

• Build a reusable anomaly detection framework  
• Compare statistical, machine learning, and deep learning models  
• Convert anomaly points into alert events  
• Evaluate models using real benchmark datasets  
• Develop a complete end-to-end ML system  

---

## Dataset

This project uses the **Numenta Anomaly Benchmark (NAB)** dataset.

Datasets used:

• ambient_temperature_system_failure  
• cpu_utilization_asg_misconfiguration  
• ec2_request_latency_system_failure  
• machine_temperature_system_failure  

Location:

data/raw/NAB/realKnownCause/

---

## Implemented Models

### Rolling Z-Score (Baseline)
Statistical anomaly detector using rolling mean and standard deviation.

---

### Isolation Forest
Machine learning model that isolates anomalies using random partitioning.

---

### LSTM Autoencoder
Deep learning model (PyTorch) that detects anomalies using reconstruction error.

Includes:

• sequence modeling  
• sliding windows  
• threshold tuning  

---

### Hybrid Detector
Combines:

• Isolation Forest  
• LSTM Autoencoder  

Flags anomalies only when both models agree.

Includes:

• severity classification  
• confidence scoring  

---

## System Features

### Event-Based Detection
Groups anomaly points into events using a time gap (2 hours).

---

### NAB Scoring
Evaluates models using:

• Precision  
• Recall  
• F1-score  

---

### Multi-Dataset Benchmarking
Runs evaluation across multiple NAB datasets.

---

### Threshold Tuning
LSTM Autoencoder optimized using:

• 99.5  
• 99.7  
• 99.8  
• 99.9  

---

### Anomaly Scoring
Assigns:

• confidence score  
• severity (low / medium / high)  

---

### CLI Tool

Run models using:

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

data_loader.py — Load and preprocess NAB time-series data  
features.py — Feature engineering (rolling stats, EWMA, diffs, z-score)  
evaluation.py — Event merging and evaluation utilities  
nab_scoring.py — Precision/recall scoring using NAB labels  

baseline_zscore.py — Rolling z-score anomaly detector  
baseline_isolation_forest.py — Isolation Forest detector  
model_lstm_autoencoder.py — LSTM Autoencoder training and inference  

hybrid_detector.py — Hybrid anomaly detection system  
anomaly_scoring.py — Severity and confidence scoring module  
detect.py — CLI interface for running models  

run_all_detectors.py — Single dataset comparison runner  
run_multi_dataset_benchmark.py — Multi-dataset evaluation  

analyze_scored_benchmark.py — F1 scoring and best model selection  
plot_scored_benchmark.py — Precision/recall visualization  
plot_threshold_tuning.py — Threshold tuning visualization  

data/

raw/NAB/ — NAB datasets  

figures/ — Generated plots and visualizations  

reports/

week5_experimental_methods.md  
week6_model_improvement.md  
benchmark_insights.md  
multi_dataset_scored_results.md  
hybrid_detector_results.md  

---

## Pipeline Overview

1. Load time-series data  
2. Generate statistical features  
3. Train and apply models  
4. Convert anomalies into events  
5. Score against labeled data  
6. Compare across datasets  
7. Apply hybrid detection and scoring  

---

## Running the Project

Run full benchmark:

python3 src/run_all_detectors.py

Run specific model:

python3 src/detect.py --model isolation --file data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv  
python3 src/detect.py --model hybrid --file data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv  

---

## Key Insights

• Isolation Forest provides stable high-recall performance  
• LSTM Autoencoder achieves higher precision but requires tuning  
• Hybrid model reduces false positives  
• Preprocessing and threshold tuning significantly impact results  

---

## AI Use Disclosure

AI tools (such as ChatGPT) were used to assist with brainstorming, structuring explanations, and refining code. All implementations, model logic, evaluation methods, and system design decisions were reviewed, modified, tested, and fully understood by the author before inclusion in this project.

---

## Author

Robair Farag  
M.S. Applied Artificial Intelligence  
University of San Diego
