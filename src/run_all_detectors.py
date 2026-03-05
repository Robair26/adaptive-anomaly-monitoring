#!/usr/bin/env python3
"""
run_all_detectors.py

Week-2 complete runner (robust):
1) Runs 3 detectors
   - Rolling Z-Score
   - Isolation Forest (feature-based)
   - LSTM Autoencoder (reconstruction)

2) Prints comparison table (points/events)

3) OPTIONAL: If NAB labels exist, also prints event-level overlap scoring
   (TP/FP/FN + precision/recall). If labels are missing, it will SKIP cleanly.

This keeps your project runnable even if you only downloaded the CSVs.
"""

from pathlib import Path
import numpy as np

from src.data_loader import load_nab_series, pick_default_series
from src.features import make_features, FeatureConfig
from src.evaluation import (
    summarize_detection,
    summary_to_markdown_table,
    merge_anomaly_events,
)

# Optional scoring (only used if labels file exists)
from src.nab_scoring import (
    LABELS_FILE,
    load_combined_windows,
    guess_nab_key_from_csv_path,
    load_label_windows_for_series_key,
    score_events_against_windows,
    scoring_to_markdown_table,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Shared Config
# ----------------------------
CSV_PATH = pick_default_series()
EVENT_GAP = "2h"   # hourly data


# ----------------------------
# Detector 1: Rolling Z-Score
# ----------------------------
def run_zscore(series):
    s = series.df["value"].astype(float)

    ROLL_WINDOW = 48
    ZSCORE_THRESHOLD = 3.0

    roll_mean = s.rolling(window=ROLL_WINDOW, min_periods=max(3, ROLL_WINDOW // 4)).mean()
    roll_std = s.rolling(window=ROLL_WINDOW, min_periods=max(3, ROLL_WINDOW // 4)).std(ddof=0).replace(0.0, np.nan)
    z = (s - roll_mean) / roll_std

    flags = (z.abs() > ZSCORE_THRESHOLD).fillna(False).values

    summary = summarize_detection(
        detector="RollingZScore",
        series=series.name,
        timestamps=series.df.index,
        flags=flags,
        gap=EVENT_GAP,
    )
    events = merge_anomaly_events(series.df.index, flags, gap=EVENT_GAP)
    return summary, events


# ----------------------------
# Detector 2: Isolation Forest
# ----------------------------
def run_isolation_forest(series):
    cfg = FeatureConfig(
        rolling_window=24,
        ewma_span=24,
        diff_lags=(1, 2, 6, 24),
        zscore_window=48,
    )
    Xdf = make_features(series.df, cfg=cfg)

    X = Xdf.values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    model = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=42,
    )
    model.fit(X)
    pred = model.predict(X)
    flags = (pred == -1)

    summary = summarize_detection(
        detector="IsolationForest",
        series=series.name,
        timestamps=Xdf.index,
        flags=flags,
        gap=EVENT_GAP,
    )
    events = merge_anomaly_events(Xdf.index, flags, gap=EVENT_GAP)
    return summary, events


# ----------------------------
# Detector 3: LSTM Autoencoder
# ----------------------------
class SeqDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = h[-1]
        rep = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        dec, _ = self.decoder(rep)
        return self.out(dec)


def make_windows(arr, seq_len):
    X = []
    for i in range(len(arr) - seq_len + 1):
        X.append(arr[i : i + seq_len])
    return np.stack(X, axis=0)


def reconstruction_errors(model, loader, device):
    model.eval()
    errs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errs.append(mse.detach().cpu().numpy())
    return np.concatenate(errs, axis=0)


def run_lstm_autoencoder(series):
    SEQ_LEN = 60
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 1e-3
    TRAIN_FRACTION = 0.60
    PERCENTILE = 99.8

    device = "cuda" if torch.cuda.is_available() else "cpu"

    values = series.df["value"].astype(float).values.reshape(-1, 1)
    train_end = int(len(values) * TRAIN_FRACTION)

    scaler = StandardScaler()
    scaler.fit(values[:train_end])
    values_scaled = scaler.transform(values)

    X_all = make_windows(values_scaled, SEQ_LEN)
    ts_win = series.df.index[SEQ_LEN - 1 :]

    train_windows_end = int(X_all.shape[0] * TRAIN_FRACTION)
    X_train = X_all[:train_windows_end]

    train_loader = DataLoader(SeqDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    full_loader = DataLoader(SeqDataset(X_all), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAutoencoder(n_features=1, hidden_size=64, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()

    train_errs = reconstruction_errors(model, train_loader, device)
    full_errs = reconstruction_errors(model, full_loader, device)

    thr = np.percentile(train_errs, PERCENTILE)
    flags = full_errs > thr

    summary = summarize_detection(
        detector="LSTM_Autoencoder",
        series=series.name,
        timestamps=ts_win,
        flags=flags,
        gap=EVENT_GAP,
    )
    events = merge_anomaly_events(ts_win, flags, gap=EVENT_GAP)
    return summary, events


def try_print_label_scoring(series, events_map):
    """
    If NAB labels are present, print overlap scoring table.
    Otherwise, print a short note and return.
    """
    if not LABELS_FILE.exists():
        print("\n=== NAB Label Scoring ===\n")
        print("Skipped: NAB label file not found at:")
        print(f"  {LABELS_FILE}")
        print("This is OK for Week 2. You can add labels later for precision/recall scoring.")
        return

    windows_dict = load_combined_windows()
    series_key = guess_nab_key_from_csv_path(series.path, windows_dict=windows_dict)
    label_windows = load_label_windows_for_series_key(series_key, windows_dict=windows_dict)

    scored_rows = []
    for det_name, events in events_map.items():
        m = score_events_against_windows(events, label_windows)
        scored_rows.append(
            {
                "Detector": det_name,
                "SeriesKey": series_key,
                "TP": m["tp_events"],
                "FP": m["fp_events"],
                "FN": m["fn_windows"],
                "Precision": m["precision"],
                "Recall": m["recall"],
            }
        )

    print("\n=== NAB Label Overlap Scoring (event-level) ===\n")
    print(scoring_to_markdown_table(scored_rows))
    print(f"\nLabel windows for series: {len(label_windows)}")


def main():
    series = load_nab_series(CSV_PATH)

    s1, e1 = run_zscore(series)
    s2, e2 = run_isolation_forest(series)
    s3, e3 = run_lstm_autoencoder(series)

    print("\n=== Detector Comparison (event-level) ===\n")
    print(summary_to_markdown_table([s1, s2, s3]))
    print("\nSeries path:", series.path)

    events_map = {
        "RollingZScore": e1,
        "IsolationForest": e2,
        "LSTM_Autoencoder": e3,
    }
    try_print_label_scoring(series, events_map)


if __name__ == "__main__":
    main()
