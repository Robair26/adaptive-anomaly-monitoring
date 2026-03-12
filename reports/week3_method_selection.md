# Week 3 – Machine Learning Method Selection

## Introduction

After exploring the dataset and defining the project objective, the next step in the capstone is selecting the machine learning methods that will be used to detect anomalies in time-series telemetry data. The goal of this project is to identify abnormal system behavior in infrastructure-style signals and compare how different anomaly detection approaches perform on this task. Because anomaly detection often operates in settings where labeled examples are limited or unavailable, the selected methods focus primarily on unsupervised and reconstruction-based approaches.

To provide a meaningful comparison, this project includes three levels of anomaly detection methods: a statistical baseline, a classical machine learning model, and a deep learning model. This structure allows the project to evaluate how model complexity affects anomaly detection performance. The selected methods are Rolling Z-Score, Isolation Forest, and an LSTM Autoencoder.

## Rolling Z-Score Baseline

The first method implemented in this project is a rolling Z-Score detector. This method serves as the statistical baseline and is useful because it is simple, fast, and easy to interpret. The detector calculates a rolling mean and rolling standard deviation over a fixed window of the time-series and flags observations whose deviation from the local mean exceeds a fixed threshold.

In this project, the rolling Z-Score model uses a 48-step rolling window and a threshold of three standard deviations. This method provides a strong baseline because it approximates how simple threshold-based monitoring systems operate in practice. However, it may struggle when the underlying time-series exhibits changing baselines, seasonality, or nonstationary behavior.

## Isolation Forest

The second method implemented is Isolation Forest, which represents the classical machine learning component of the project. Isolation Forest is an unsupervised anomaly detection algorithm that works by randomly partitioning the feature space. Because anomalous points are typically rare and distinct from normal observations, they tend to be isolated more quickly than normal points.

In this project, Isolation Forest is applied to a feature-engineered version of the time-series rather than to the raw values alone. The engineered features include rolling mean, rolling standard deviation, exponentially weighted moving averages, lagged differences, and rolling Z-score features. These transformations allow the model to capture local patterns in the telemetry signal and improve its ability to distinguish between normal and abnormal behavior.

Isolation Forest is well suited for this task because it does not require labeled anomaly examples and can operate effectively on derived tabular features from time-series data.

## LSTM Autoencoder

The third method implemented is an LSTM Autoencoder, which serves as the deep learning model required for the capstone project. This model is designed to learn the temporal structure of normal sequential data and identify anomalies through reconstruction error.

The model architecture consists of an LSTM encoder and an LSTM decoder. The encoder processes a sequence of time-series observations and compresses the sequence into a latent representation. The decoder then reconstructs the original sequence from this compressed representation. During training, the model minimizes reconstruction error between the input sequence and its reconstruction.

In this project, the LSTM Autoencoder is trained on the first portion of the time-series, which is treated as mostly normal behavior. After training, the model calculates reconstruction error across the entire dataset. Windows whose reconstruction error exceeds a selected percentile threshold are classified as anomalies. Because LSTM networks are designed for sequential data, this method is particularly appropriate for time-series anomaly detection.

## Model Training

The model training requirement for the capstone is satisfied through the LSTM Autoencoder. The network is trained using the Adam optimizer and mean squared error loss. Training is performed on sliding windows extracted from the input time-series. This allows the model to learn the typical temporal dynamics of the signal before reconstruction errors are used as anomaly scores.

Although the Rolling Z-Score and Isolation Forest detectors do not require the same type of deep learning training loop, they provide important comparison points that help establish the relative value of the neural network model.

## Evaluation Strategy

The project evaluates each method at both the point level and the event level. Point-level detections indicate which timestamps are flagged as anomalous, while event-level detections merge nearby anomalies into a single event using a configurable time gap. This event-level representation is more operationally meaningful because real monitoring systems are generally concerned with sustained periods of abnormal behavior rather than isolated outlier points.

At this stage, the methods are compared using the number of flagged points, the percentage of the timeline marked anomalous, and the number of merged anomaly events. Future work will incorporate ground-truth anomaly windows from the NAB dataset to calculate event-level precision and recall.

## Summary

The selected methods provide a structured progression from simple baseline detection to more advanced machine learning and deep learning approaches. Rolling Z-Score provides a transparent statistical baseline, Isolation Forest provides a feature-based machine learning method, and the LSTM Autoencoder provides a sequence-based neural network model capable of learning temporal behavior directly from the signal.

Together, these methods create a strong comparative framework for evaluating anomaly detection performance in telemetry-like time-series data while satisfying the capstone requirement that the project include model training and at least one deep learning architecture.

## Implementation Status

The selected methods have already been implemented in code:
- `src/baseline_zscore.py`
- `src/baseline_isolation_forest.py`
- `src/model_lstm_autoencoder.py`
- `src/run_all_detectors.py`

This means the Week 3 method selection is not hypothetical; it reflects the actual methods currently built and tested in the project repository.
