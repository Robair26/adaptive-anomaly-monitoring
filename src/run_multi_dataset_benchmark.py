#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_nab_series
from src.features import make_features, FeatureConfig
from src.evaluation import summarize_detection, summary_to_markdown_table, merge_anomaly_events

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


EVENT_GAP = "2h"

DATASETS = [
    "data/raw/NAB/realKnownCause/ambient_temperature_system_failure.csv",
    "data/raw/NAB/realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "data/raw/NAB/realKnownCause/ec2_request_latency_system_failure.csv",
    "data/raw/NAB/realKnownCause/machine_temperature_system_failure.csv",
]


def run_zscore(series):
    s = series.df["value"].astype(float)
    roll_window = 48
    thr = 3.0

    roll_mean = s.rolling(window=roll_window, min_periods=max(3, roll_window // 4)).mean()
    roll_std = s.rolling(window=roll_window, min_periods=max(3, roll_window // 4)).std(ddof=0).replace(0.0, np.nan)
    z = (s - roll_mean) / roll_std

    flags = (z.abs() > thr).fillna(False).values

    return summarize_detection(
        detector="RollingZScore",
        series=series.name,
        timestamps=series.df.index,
        flags=flags,
        gap=EVENT_GAP,
    )


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
    flags = pred == -1

    return summarize_detection(
        detector="IsolationForest",
        series=series.name,
        timestamps=Xdf.index,
        flags=flags,
        gap=EVENT_GAP,
    )


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
        X.append(arr[i:i + seq_len])
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
    seq_len = 60
    batch_size = 64
    epochs = 10
    lr = 1e-3
    train_fraction = 0.60
    percentile = 99.8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    values = series.df["value"].astype(float).values.reshape(-1, 1)
    train_end = int(len(values) * train_fraction)

    scaler = StandardScaler()
    scaler.fit(values[:train_end])
    values_scaled = scaler.transform(values)

    X_all = make_windows(values_scaled, seq_len)
    ts_win = series.df.index[seq_len - 1:]

    train_windows_end = int(X_all.shape[0] * train_fraction)
    X_train = X_all[:train_windows_end]

    train_loader = DataLoader(SeqDataset(X_train), batch_size=batch_size, shuffle=True)
    full_loader = DataLoader(SeqDataset(X_all), batch_size=batch_size, shuffle=False)

    model = LSTMAutoencoder(n_features=1, hidden_size=64, num_layers=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()

    train_errs = reconstruction_errors(model, train_loader, device)
    full_errs = reconstruction_errors(model, full_loader, device)

    thr = np.percentile(train_errs, percentile)
    flags = full_errs > thr

    return summarize_detection(
        detector="LSTM_Autoencoder",
        series=series.name,
        timestamps=ts_win,
        flags=flags,
        gap=EVENT_GAP,
    )


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for csv_path in DATASETS:
        print(f"\nRunning detectors on: {csv_path}")
        series = load_nab_series(csv_path)

        summaries = [
            run_zscore(series),
            run_isolation_forest(series),
            run_lstm_autoencoder(series),
        ]

        for s in summaries:
            rows.append(
                {
                    "detector": s.detector,
                    "series": s.series,
                    "points": s.n_points,
                    "flagged_points": s.n_flagged_points,
                    "pct_flagged_points": round(s.pct_flagged_points, 4),
                    "events": s.n_events,
                    "event_gap": s.event_gap,
                }
            )

    df = pd.DataFrame(rows)
    csv_out = reports_dir / "multi_dataset_results.csv"
    md_out = reports_dir / "multi_dataset_results.md"

    df.to_csv(csv_out, index=False)

    md_lines = ["# Multi-Dataset Benchmark Results", ""]
    for series_name, subdf in df.groupby("series"):
        md_lines.append(f"## {series_name}")
        md_lines.append("")
        md_lines.append("| Detector | Points | Flagged Points | % Flagged | Events | Merge Gap |")
        md_lines.append("|---|---:|---:|---:|---:|---|")
        for _, r in subdf.iterrows():
            md_lines.append(
                f"| {r['detector']} | {int(r['points']):,} | {int(r['flagged_points']):,} | {r['pct_flagged_points']:.2f}% | {int(r['events'])} | {r['event_gap']} |"
            )
        md_lines.append("")

    md_out.write_text("\n".join(md_lines), encoding="utf-8")

    print("\nSaved:")
    print(f" - {csv_out}")
    print(f" - {md_out}")
    print("\nPreview:")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
