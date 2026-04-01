#!/usr/bin/env python3

import numpy as np
import pandas as pd


def normalize_scores(scores):
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def compute_severity(score, low=0.3, high=0.7):
    if score >= high:
        return "high"
    elif score >= low:
        return "medium"
    else:
        return "low"


def add_anomaly_scores(df, score_col):
    df = df.copy()
    
    normalized = normalize_scores(df[score_col].values)
    
    df["confidence"] = normalized
    df["severity"] = [compute_severity(s) for s in normalized]
    
    return df


if __name__ == "__main__":
    print("Anomaly scoring module ready.")
