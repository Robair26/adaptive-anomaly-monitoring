# Benchmark Insights

## Average Performance Across Evaluated NAB Datasets

- Isolation Forest achieved an average precision of **0.322** and an average recall of **1.000**.
- LSTM Autoencoder achieved the highest average precision at **0.700**, with an average recall of **0.812**.
- Rolling Z-Score achieved an average precision of **0.188** and an average recall of **1.000**.

## Interpretation

These results suggest that the three anomaly detection approaches offer different tradeoffs.

Rolling Z-Score is highly sensitive and successfully identifies all labeled anomaly windows across the evaluated datasets, but it produces a large number of false positives, which lowers precision.

Isolation Forest provides the most balanced high-recall performance across datasets. It consistently detects labeled anomaly windows while maintaining better precision than the statistical baseline.

The LSTM Autoencoder provides the strongest average precision, indicating that it is the most selective detector overall. However, its lower recall suggests that it may miss some anomaly windows when the learned temporal representation does not generalize equally well across all telemetry patterns.

## Practical Takeaway

For high-sensitivity anomaly monitoring, Isolation Forest appears to be the strongest overall candidate in the current benchmark because it combines perfect recall with better precision than the statistical baseline.

For scenarios where reducing false positives is more important than maximizing recall, the LSTM Autoencoder shows strong promise, though additional tuning may be needed to improve consistency across datasets.
