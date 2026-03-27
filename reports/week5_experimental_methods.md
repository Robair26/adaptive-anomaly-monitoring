# Experimental Methods

## Model Overview

Three anomaly detection approaches were implemented:

- Rolling Z-Score (statistical baseline)
- Isolation Forest (classical machine learning)
- LSTM Autoencoder (deep learning)

The LSTM Autoencoder is the primary model and is designed to learn temporal patterns in time-series data.

---

## Feature Engineering

Isolation Forest uses engineered features including:

- Rolling mean and standard deviation  
- Exponentially weighted moving average (EWMA)  
- Lag-based differences  

The LSTM Autoencoder uses raw time-series data without manual feature engineering.

---

## LSTM Autoencoder Architecture

The model follows an encoder-decoder structure:

- Input: sequences of length 60  
- Encoder: LSTM layer compressing sequence  
- Decoder: LSTM layer reconstructing sequence  
- Output: reconstructed signal  

Reconstruction error (MSE) is used as anomaly score.

---

## Training Procedure

- Sliding window sequence generation  
- Train/test split: 60% training  
- Optimizer: Adam  
- Learning rate: 0.001  
- Epochs: 15  
- Loss: Mean Squared Error  

Training assumes most data is normal.

---

## Anomaly Detection

- Compute reconstruction error  
- Threshold = high percentile of training error (e.g., 99.8)  
- Flag sequences above threshold  

Events are merged using a 2-hour gap.

---

## Evaluation

Models are evaluated using:

- Precision  
- Recall  
- F1-score  

Evaluation is done at the **event level** using NAB labels.

This ensures results reflect real-world alerting behavior.
