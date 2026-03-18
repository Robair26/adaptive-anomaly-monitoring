# Scored Results Summary

## Key Findings

Across four NAB datasets, Rolling Z-Score, Isolation Forest, and LSTM Autoencoder all demonstrated strong anomaly recall on most time-series. However, their precision varied substantially.

Isolation Forest provided the most consistent balance between recall and precision across datasets. Rolling Z-Score achieved perfect recall in all evaluated cases but generated a high number of false positives, resulting in the lowest precision overall. The LSTM Autoencoder showed the strongest performance on selected datasets, including perfect precision and recall on `ec2_request_latency_system_failure`, but also showed weaker recall on `machine_temperature_system_failure`, indicating that its performance is more sensitive to dataset characteristics and hyperparameter settings.

Overall, the results suggest that statistical baselines are useful for sensitivity, Isolation Forest offers the most stable general-purpose anomaly detector, and the LSTM Autoencoder has the highest potential performance when the signal characteristics align well with sequence reconstruction modeling.
