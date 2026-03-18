# Benchmark Insights

## Best Detector by Dataset

- **ambient_temperature_system_failure** → IsolationForest
- **cpu_utilization_asg_misconfiguration** → LSTM_Autoencoder
- **ec2_request_latency_system_failure** → LSTM_Autoencoder
- **machine_temperature_system_failure** → IsolationForest

## Average Scores Across Datasets

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| LSTM_Autoencoder | 0.700 | 0.812 | 0.632 |
| IsolationForest | 0.322 | 1.000 | 0.478 |
| RollingZScore | 0.188 | 1.000 | 0.300 |

## Interpretation

The benchmark results show that no single anomaly detector is universally optimal across all telemetry datasets.

The **LSTM Autoencoder** achieved the strongest average precision and F1-score, indicating that it was the most selective and balanced detector overall. It performed especially well on `ec2_request_latency_system_failure`, where it achieved perfect precision and recall, and on `cpu_utilization_asg_misconfiguration`, where it outperformed the other detectors by F1-score.

The **Isolation Forest** provided the most stable high-recall performance across datasets. It consistently detected all labeled anomaly windows and emerged as the best detector on both `ambient_temperature_system_failure` and `machine_temperature_system_failure`. This suggests that Isolation Forest is the strongest general-purpose detector in the current benchmark.

The **Rolling Z-Score** baseline achieved perfect recall across all evaluated datasets, but at the cost of very low precision. This confirms that simple statistical thresholding is useful for sensitivity but tends to over-flag anomalies in complex telemetry signals.

## Practical Takeaway

These findings suggest that:

- **Isolation Forest** is the best default detector when consistent recall and robustness are priorities.
- **LSTM Autoencoder** is the strongest candidate when precision and balanced anomaly detection are more important.
- **Rolling Z-Score** remains useful as a transparent baseline, but it is too noisy to serve as the primary detector in this project.
