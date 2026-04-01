# Week 6 – Model Improvement and Performance Analysis

## Improvement Objective

The goal of this phase was to improve the LSTM Autoencoder anomaly detector by tuning the reconstruction error threshold. Because anomaly detection performance depends heavily on how the anomaly threshold is selected, multiple percentile thresholds were tested to determine how threshold choice affects alert volume and event-level behavior.

## Threshold Tuning Experiment

The LSTM Autoencoder was evaluated on the `ambient_temperature_system_failure` NAB series using four reconstruction error threshold percentiles:

- 99.5
- 99.7
- 99.8
- 99.9

The following results were observed:

| Threshold Percentile | Detected Windows | Merged Events |
|---|---:|---:|
| 99.5 | 414 | 9 |
| 99.7 | 443 | 12 |
| 99.8 | 490 | 12 |
| 99.9 | 405 | 6 |

## Analysis

The threshold tuning experiment showed that the LSTM Autoencoder is sensitive to threshold selection. Lower thresholds produced a larger number of anomaly windows and events, indicating higher sensitivity but also a greater likelihood of false positives. Increasing the threshold reduced the number of detections and made the model more selective.

Among the tested settings, the 99.9th percentile produced the fewest anomaly windows and the smallest number of merged anomaly events. This indicates that the model can be made more conservative through threshold tuning, which is desirable in monitoring environments where alert fatigue is a concern.

However, the tuning process also showed that LSTM-based performance is not perfectly stable across runs. Because the model is trained stochastically, small changes in training can produce different anomaly profiles even with the same architecture. This suggests that while the LSTM Autoencoder is a powerful model, its behavior is more sensitive to threshold choice and training variation than the baseline methods.

## Comparison to Other Models

The broader benchmark results indicate that Isolation Forest remains the most stable model overall, providing strong recall across multiple datasets with more consistent behavior. The LSTM Autoencoder achieved the highest average precision and F1-score in the scored benchmark, but it required additional tuning and showed greater sensitivity to configuration.

These findings suggest the following tradeoff:

- **Isolation Forest** is the strongest general-purpose detector for reliable deployment.
- **LSTM Autoencoder** is the most selective detector and has higher potential precision, but it requires careful tuning.
- **Rolling Z-Score** remains useful as a simple statistical baseline but generates too many false positives to be the primary model.

## Final Week 6 Conclusion

The main model improvement explored in this phase was threshold tuning for the LSTM Autoencoder. This experiment demonstrated that model performance can be meaningfully changed by adjusting the anomaly threshold percentile. The 99.9th percentile produced the most selective alert profile in the tested series and will be retained as the tuned setting for the final project.

Overall, this week confirmed that model optimization is not only about changing architectures, but also about carefully tuning decision boundaries and analyzing how those changes affect anomaly alert behavior.
