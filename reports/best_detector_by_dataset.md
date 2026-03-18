# Best Detector by Dataset

| Dataset | Best Detector | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| ambient_temperature_system_failure | IsolationForest | 0.391 | 1.000 | 0.562 |
| cpu_utilization_asg_misconfiguration | LSTM_Autoencoder | 0.500 | 1.000 | 0.667 |
| ec2_request_latency_system_failure | LSTM_Autoencoder | 1.000 | 1.000 | 1.000 |
| machine_temperature_system_failure | IsolationForest | 0.257 | 1.000 | 0.409 |

## Average Scores

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| LSTM_Autoencoder | 0.700 | 0.812 | 0.632 |
| IsolationForest | 0.322 | 1.000 | 0.478 |
| RollingZScore | 0.188 | 1.000 | 0.300 |