# Multi-Dataset Scored Benchmark Results (with F1)

## ambient_temperature_system_failure

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| RollingZScore | 0.182 | 1.000 | 0.308 |
| IsolationForest | 0.391 | 1.000 | 0.562 |
| LSTM_Autoencoder | 0.300 | 1.000 | 0.462 |

## cpu_utilization_asg_misconfiguration

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| RollingZScore | 0.086 | 1.000 | 0.158 |
| IsolationForest | 0.187 | 1.000 | 0.315 |
| LSTM_Autoencoder | 0.500 | 1.000 | 0.667 |

## ec2_request_latency_system_failure

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| RollingZScore | 0.385 | 1.000 | 0.556 |
| IsolationForest | 0.455 | 1.000 | 0.625 |
| LSTM_Autoencoder | 1.000 | 1.000 | 1.000 |

## machine_temperature_system_failure

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| RollingZScore | 0.098 | 1.000 | 0.179 |
| IsolationForest | 0.257 | 1.000 | 0.409 |
| LSTM_Autoencoder | 1.000 | 0.250 | 0.400 |
