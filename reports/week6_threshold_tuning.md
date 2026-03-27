# Week 6 – LSTM Threshold Tuning

## Objective

The purpose of this experiment was to improve the LSTM Autoencoder anomaly detector by tuning the reconstruction error threshold percentile.

## Thresholds Tested

- 99.5
- 99.7
- 99.8
- 99.9

## Results on `ambient_temperature_system_failure`

| Threshold Percentile | Detected Windows | Merged Events |
|---|---:|---:|
| 99.5 | 414 | 9 |
| 99.7 | 443 | 12 |
| 99.8 | 490 | 12 |
| 99.9 | 405 | 6 |

## Interpretation

Increasing the threshold percentile generally reduced the number of anomaly alerts by making the detector more selective. Among the tested values, the 99.9th percentile produced the smallest number of merged anomaly events and the lowest number of flagged windows. This suggests that the 99.9 threshold provides the best balance for reducing false positives while still detecting meaningful abnormal periods.

## Final Choice

The final tuned threshold for the LSTM Autoencoder was set to:

`PERCENTILE = 99.9`

This tuned setting will be used in the final project results and analysis.
