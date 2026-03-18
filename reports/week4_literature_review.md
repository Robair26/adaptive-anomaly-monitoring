# Week 4 – Literature Review (Background Section)

## Introduction

Anomaly detection in time-series data is a critical problem in modern machine learning, particularly in domains such as system monitoring, cybersecurity, and industrial IoT. The goal of anomaly detection is to identify patterns in data that deviate from expected behavior. In real-world telemetry systems, anomalies may indicate system failures, performance degradation, or security threats.

Time-series anomaly detection presents unique challenges due to temporal dependencies, noise, seasonality, and the lack of labeled anomaly data. As a result, many approaches rely on unsupervised or semi-supervised methods that learn patterns of normal behavior and detect deviations from those patterns.

This project focuses on detecting anomalies in telemetry-style time-series data using a combination of statistical, machine learning, and deep learning methods. The selected methods are grounded in existing research and reflect commonly used approaches in anomaly detection systems.

---

## Statistical Methods for Anomaly Detection

One of the most fundamental approaches to anomaly detection is the use of statistical techniques. Methods such as Z-score detection assume that normal data follows a stable distribution and that anomalies can be identified as observations that deviate significantly from the mean.

Rolling statistical methods extend this idea to time-series data by computing statistics over a moving window. These methods are widely used in monitoring systems due to their simplicity, interpretability, and low computational cost. However, they often struggle in scenarios with non-stationary behavior or complex temporal patterns.

In this project, a Rolling Z-Score method is used as a baseline to represent traditional statistical anomaly detection.

---

## Machine Learning Approaches

Machine learning methods have been widely adopted for anomaly detection due to their ability to model complex patterns in data. One of the most well-known algorithms for unsupervised anomaly detection is Isolation Forest.

Isolation Forest, introduced by Liu et al. (2008), operates by randomly partitioning the data space using decision trees. The key idea is that anomalies are easier to isolate than normal points because they differ significantly from the majority of the data. As a result, anomalies tend to have shorter path lengths in the tree structure.

Isolation Forest is particularly effective for high-dimensional datasets and does not require labeled anomaly data. It has been widely used in applications such as fraud detection, network intrusion detection, and system monitoring.

In this project, Isolation Forest is applied to engineered features derived from time-series data, allowing the model to capture local statistical and temporal behavior.

---

## Deep Learning for Time-Series Anomaly Detection

Deep learning methods have become increasingly popular for anomaly detection, particularly for time-series data. Recurrent neural networks (RNNs), and specifically Long Short-Term Memory (LSTM) networks, are well suited for modeling sequential data due to their ability to capture long-term dependencies.

One common deep learning approach for anomaly detection is the use of autoencoders. An autoencoder is a neural network trained to reconstruct its input. When trained on normal data, the model learns a compressed representation of typical patterns. Anomalies can then be detected by measuring reconstruction error, as abnormal inputs are reconstructed poorly.

Malhotra et al. (2015) demonstrated the effectiveness of LSTM-based models for time-series anomaly detection. Their work shows that LSTM networks can learn temporal dependencies and detect anomalies based on deviations from learned patterns.

In this project, an LSTM Autoencoder is implemented to model temporal behavior in telemetry data. The model is trained on normal sequences and uses reconstruction error as an anomaly score.

---

## Benchmarking and Evaluation

Evaluating anomaly detection algorithms is challenging due to the lack of labeled data and the importance of temporal context. The Numenta Anomaly Benchmark (NAB), introduced by Lavin and Ahmad (2015), provides a standardized dataset and evaluation framework for anomaly detection in streaming time-series data.

The NAB dataset includes real-world telemetry signals with labeled anomaly windows, allowing for consistent comparison of anomaly detection algorithms. It is widely used in research and industry for benchmarking anomaly detection performance.

In this project, NAB datasets are used to evaluate multiple anomaly detection methods. The evaluation focuses on event-level detection, where individual anomaly points are grouped into events based on temporal proximity. This approach better reflects real-world alerting systems.

---

## Summary

The literature shows that anomaly detection in time-series data can be approached using a range of methods, from simple statistical techniques to advanced deep learning models. Each approach has strengths and limitations depending on the complexity of the data and the nature of the anomalies.

This project combines three complementary approaches:

• Rolling Z-Score (statistical baseline)  
• Isolation Forest (machine learning method)  
• LSTM Autoencoder (deep learning model)  

By comparing these methods across multiple datasets, the project provides insight into how model complexity impacts anomaly detection performance in real-world telemetry data.
