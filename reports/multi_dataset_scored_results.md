# Multi-Dataset Scored Benchmark Results

## ambient_temperature_system_failure

| Detector | Points | Flagged | % Flagged | Events | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RollingZScore | 7,267 | 42 | 0.58% | 22 | 4 | 18 | 0 | 0.182 | 1.000 |
| IsolationForest | 7,243 | 73 | 1.01% | 23 | 9 | 14 | 0 | 0.391 | 1.000 |
| LSTM_Autoencoder | 7,208 | 546 | 7.57% | 10 | 3 | 7 | 0 | 0.300 | 1.000 |

## cpu_utilization_asg_misconfiguration

| Detector | Points | Flagged | % Flagged | Events | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RollingZScore | 18,050 | 827 | 4.58% | 105 | 9 | 96 | 0 | 0.086 | 1.000 |
| IsolationForest | 18,026 | 181 | 1.00% | 91 | 17 | 74 | 0 | 0.187 | 1.000 |
| LSTM_Autoencoder | 17,991 | 842 | 4.68% | 2 | 1 | 1 | 0 | 0.500 | 1.000 |

## ec2_request_latency_system_failure

| Detector | Points | Flagged | % Flagged | Events | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RollingZScore | 4,032 | 20 | 0.50% | 13 | 5 | 8 | 0 | 0.385 | 1.000 |
| IsolationForest | 4,008 | 41 | 1.02% | 11 | 5 | 6 | 0 | 0.455 | 1.000 |
| LSTM_Autoencoder | 3,973 | 75 | 1.89% | 3 | 3 | 0 | 0 | 1.000 | 1.000 |

## machine_temperature_system_failure

| Detector | Points | Flagged | % Flagged | Events | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RollingZScore | 22,695 | 322 | 1.42% | 102 | 10 | 92 | 0 | 0.098 | 1.000 |
| IsolationForest | 22,671 | 227 | 1.00% | 35 | 9 | 26 | 0 | 0.257 | 1.000 |
| LSTM_Autoencoder | 22,636 | 28 | 0.12% | 1 | 1 | 0 | 3 | 1.000 | 0.250 |
