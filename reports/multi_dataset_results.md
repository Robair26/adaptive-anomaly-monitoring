# Multi-Dataset Benchmark Results

## ambient_temperature_system_failure

| Detector | Points | Flagged Points | % Flagged | Events | Merge Gap |
|---|---:|---:|---:|---:|---|
| RollingZScore | 7,267 | 42 | 0.58% | 22 | 2h |
| IsolationForest | 7,243 | 73 | 1.01% | 23 | 2h |
| LSTM_Autoencoder | 7,208 | 355 | 4.93% | 9 | 2h |

## cpu_utilization_asg_misconfiguration

| Detector | Points | Flagged Points | % Flagged | Events | Merge Gap |
|---|---:|---:|---:|---:|---|
| RollingZScore | 18,050 | 827 | 4.58% | 105 | 2h |
| IsolationForest | 18,026 | 181 | 1.00% | 91 | 2h |
| LSTM_Autoencoder | 17,991 | 717 | 3.99% | 7 | 2h |

## ec2_request_latency_system_failure

| Detector | Points | Flagged Points | % Flagged | Events | Merge Gap |
|---|---:|---:|---:|---:|---|
| RollingZScore | 4,032 | 20 | 0.50% | 13 | 2h |
| IsolationForest | 4,008 | 41 | 1.02% | 11 | 2h |
| LSTM_Autoencoder | 3,973 | 75 | 1.89% | 3 | 2h |

## machine_temperature_system_failure

| Detector | Points | Flagged Points | % Flagged | Events | Merge Gap |
|---|---:|---:|---:|---:|---|
| RollingZScore | 22,695 | 322 | 1.42% | 102 | 2h |
| IsolationForest | 22,671 | 227 | 1.00% | 35 | 2h |
| LSTM_Autoencoder | 22,636 | 28 | 0.12% | 1 | 2h |
